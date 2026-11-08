"""psk-x1 (EXPERIMENTAL) without hardware: modem round trip, noise-only false frames, firmware parity (gcc compiles
the firmware's own checksum/pulseAt/sampleAt), bench tool on a SYNTHETIC capture. The frozen ecg-v1 is not touched.

   ~/.venvs/galvanic/bin/python -m unittest tests.test_psk -v
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import ecg_ber as B                       # noqa: E402
import ecg_link as L                      # noqa: E402
import ecg_synth as SY                    # noqa: E402
import psk_link as P                      # noqa: E402
import psk_tx as PT                       # noqa: E402
from tests.test_arduino import write_picoscope_csv   # noqa: E402

INO = os.path.join(HERE, "arduino", "watch_psk_tx", "watch_psk_tx.ino")


def recording(p, ppm=3400.0, lag=6.0, noise=0.02, seed=0, dur=29.5):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * 250)) / 250.0
    return t, P.waveform(p, t / (1 + ppm * 1e-6) + lag) + rng.normal(0, noise, len(t))


class TestPsk(unittest.TestCase):
    def test_frozen_decoder_untouched(self):
        self.assertEqual(L.RULES_SHA, B.FROZEN_SHA)

    def test_round_trip(self):
        cases = [(m, r, s) for m in ("dbpsk",) for r in (8, 16, 24, 32) for s in ("rect", "rrc")]
        cases += [("bpsk", 16, "rect"), ("dqpsk", 16, "rrc"), ("dqpsk", 32, "rrc"), ("ask4", 16, "rrc")]
        for mod, bps, shape in cases:
            with self.subTest(mod=mod, bps=bps, shape=shape):
                p = P.plan(mod, bps, shape=shape, seconds=60)
                t, x = recording(p, seed=bps)
                r = P.decode(x, t, p)
                self.assertGreater(r["frames_complete"], 0)
                self.assertEqual(r["frames_valid"], r["frames_complete"], r)
                self.assertEqual(r["errors"], 0)
                self.assertAlmostEqual(r["clock_ppm"], 3400, delta=600)

    def test_x3wl_round_trip(self):
        """the primary receiver on clean signals, every ladder rate it is expected to handle (16 Hz, rrc)"""
        for bps in (16, 24, 32, 40, 50, 64, 100):
            with self.subTest(bps=bps):
                p = P.plan("dbpsk", bps, shape="rrc", seconds=55)
                t, x = recording(p, seed=bps, lag=7.0)
                r = P.decode_wl(x, t, p)
                self.assertGreater(r["frames_complete"], 0)
                self.assertEqual(r["frames_valid"], r["frames_complete"], {k: r.get(k) for k in ("reason", "sync_score")})
                self.assertEqual(r["errors"], 0)

    def test_x3wl_noise_only_gives_no_frames(self):
        t = np.arange(int(29.5 * 250)) / 250.0
        for bps in (16, 32, 64, 100):
            for seed in range(3):
                x = SY.heartbeat(t, 72, np.random.default_rng(seed)) * 0.8 + np.random.default_rng(seed + 9).normal(0, 0.05, len(t))
                r = P.decode_wl(x, t, P.plan("dbpsk", bps, shape="rrc"))
                self.assertEqual(r.get("frames_valid", 0), 0, (bps, seed, r.get("sync_score")))

    def test_noise_and_heartbeat_only_gives_no_frames(self):
        rng = np.random.default_rng(5)
        t = np.arange(int(29.5 * 250)) / 250.0
        for mod in sorted(P.MODS):
            for bps in (16, 32):
                for seed in range(3):
                    x = SY.heartbeat(t, 72, np.random.default_rng(seed)) * 0.8 + rng.normal(0, 0.05, len(t))
                    r = P.decode(x, t, P.plan(mod, bps, shape="rrc"), eq=True)
                    self.assertEqual(r.get("frames_valid", 0), 0, (mod, bps, seed, r.get("sync_score")))

    def _harness(self):
        if not shutil.which("gcc"):
            self.skipTest("no gcc")
        src = open(INO).read()
        grab = lambda sig: re.search(re.escape(sig) + r".*?\n\}", src, re.S).group(0)
        c = ("#include <stdint.h>\n#include <stdio.h>\n#include <math.h>\n#include <string.h>\n"
             "const float SYM_SCALE = 120.0f, PULSE_SCALE = 8192.0f;\n"
             "int8_t symI[4096], symQ[4096]; int16_t pulse[513];\n"
             "uint32_t symUs, fcMilliHz, nSyms, span, pts; float norm;\n"
             "uint32_t loopStart, loopLen, loopReps, nPlay;\n"
             + grab("uint32_t checksum() {") + "\n" + grab("float pulseAt(float tau) {") + "\n"
             + grab("uint32_t stored(uint32_t k) {") + "\n"
             + grab("float sampleAt(double el) {") + "\n"
             "int main(){unsigned i,np,nt; int a,b; unsigned nm; double el;\n"
             " scanf(\"%u %u %u %u %u %u %u %u %u\",&symUs,&fcMilliHz,&nSyms,&span,&pts,&nm,&loopStart,&loopLen,&loopReps);"
             " norm=nm*1e-6f; nPlay=nSyms+loopLen*(loopReps-1);\n"
             " for(i=0;i<nSyms;i++){scanf(\"%d %d\",&a,&b); symI[i]=a; symQ[i]=b;}\n"
             " np=span*pts+1; for(i=0;i<np;i++){scanf(\"%d\",&a); pulse[i]=a;}\n"
             " printf(\"%u\\n\",checksum()); scanf(\"%u\",&nt);\n"
             " for(i=0;i<nt;i++){scanf(\"%lf\",&el); printf(\"%.6f\\n\",sampleAt(el));} return 0;}")
        d = tempfile.mkdtemp(prefix="psk_fw_")
        open(os.path.join(d, "c.c"), "w").write(c)
        subprocess.run(["gcc", "-O1", "-o", os.path.join(d, "c"), os.path.join(d, "c.c"), "-lm"], check=True)
        return os.path.join(d, "c")

    def test_firmware_matches_python_model(self):
        """the firmware's own checksum() and sampleAt() (compiled with gcc) vs psk_tx.checksum / psk_tx.model"""
        exe = self._harness()
        for mod, bps, shape in (("dbpsk", 16, "rect"), ("dbpsk", 32, "rrc"), ("dbpsk", 100, "rrc"), ("dqpsk", 50, "rc"),
                                ("ask4", 24, "rrc")):
            with self.subTest(mod=mod, bps=bps, shape=shape):
                p = P.plan(mod, bps, shape=shape, seconds=20)
                fw = PT.fw_payload(p)
                fc = int(round(p["fc_hz"] * 1000)); nm = int(round(fw["norm"] * 1e6))
                tt = np.sort(np.random.default_rng(1).uniform(0, p["total_s"], 400))
                ls, ll, lr = fw["loop"]
                if mod == "dbpsk":
                    self.assertGreater(ll, 0)                     # DBPSK frames repeat: stored once, looped
                inp = (f"{p['sym_us']} {fc} {len(fw['si'])} {fw['span']} {P.PTS} {nm} {ls} {ll} {lr}\n"
                       + " ".join(f"{a} {b}" for a, b in zip(fw["si"], fw["sq"])) + "\n"
                       + " ".join(map(str, fw["pulse"])) + f"\n{len(tt)}\n" + " ".join(f"{x * 1e6:.3f}" for x in tt))
                out = subprocess.run([exe], input=inp, capture_output=True, text=True, check=True).stdout.split()
                self.assertEqual(int(out[0]), PT.checksum(p["sym_us"], fc, fw["si"], fw["sq"], fw["span"], fw["pulse"],
                                                          loop=fw["loop"]))
                got = np.array([float(v) for v in out[1:]])
                np.testing.assert_allclose(got, PT.model(p, tt, fw), atol=2e-3)

    def bench(self, corrupt=False):
        import psk_bench as PB
        p = P.plan("dbpsk", 16, shape="rect", seconds=30)
        fs, dur = 2000.0, 32.0
        t = np.arange(int(dur * fs)) / fs
        rng = np.random.default_rng(2)
        fw = PT.fw_payload(p)
        if corrupt:                                   # three symbols played with a wrong (90 degree) phase
            fw = dict(fw, aq=fw["aq"].copy())
            fw["aq"][[60, 61, 200]] *= 1j
        v = 1.0 * PT.model(p, (t - 1.0) / (1 + 250e-6), fw) + 1.1747 + rng.normal(0, 0.01, len(t)) \
            + 0.02 * np.sin(2 * np.pi * 60 * t)
        d = tempfile.mkdtemp(prefix="pskb_")
        cap = os.path.join(d, "cap.csv"); write_picoscope_csv(cap, t, v)
        zero = os.path.join(d, "zero.csv"); write_picoscope_csv(zero, t[:4000], rng.normal(0, 0.01, 4000))
        old = B.DATADIR
        B.DATADIR = d
        argv = sys.argv
        try:
            sys.argv = ["psk_bench.py", "frame", cap, "--mod", "dbpsk", "--bps", "16", "--shape", "rect", "--seconds", "30",
                        "--label", "t", "--zero", zero]
            buf = io.StringIO()
            with redirect_stdout(buf):
                PB.main()
        finally:
            sys.argv = argv; B.DATADIR = old
        rep = json.load(open([os.path.join(r, "report.json") for r, _, f in os.walk(d) if "report.json" in f][0]))
        return dict(rep["segments"][0], ready=rep["ready"]), buf.getvalue()

    def test_bench_tool_on_synthetic_capture(self):
        """SYNTHETIC scope capture (firmware model, 2 mVpp, DC, 60 Hz hum) -> psk_bench READY"""
        rep, log = self.bench()
        self.assertTrue(rep["ready"], log)
        self.assertAlmostEqual(rep["fit"]["amp_mvpp"], 2.0, delta=0.05)
        self.assertAlmostEqual(rep["fit"]["clock_ppm"], 250, delta=60)
        self.assertLess(rep["phase_steps"]["max_deg"], 10)

    def test_bench_tool_catches_wrong_phase(self):
        rep, log = self.bench(corrupt=True)
        self.assertFalse(rep["ready"], log)
        self.assertGreater(rep["phase_steps"]["max_deg"], 45)


if __name__ == "__main__":
    unittest.main()


class TestX4(unittest.TestCase):
    """psk-x4 'x4-mb' (psk_rx4.py, EXPERIMENTAL): frozen, decodes through the channel model, no false frames, and
    does not rely on the test transmission's repeated frames (random payload in every frame)"""

    @staticmethod
    def through_channel(p, a=None, ppm=3400.0, lag=7.0, seed=0, signal=True):
        import channel_est as C
        rng = np.random.default_rng(seed)
        t = np.arange(int(29.5 * 250)) / 250.0
        pre = 750
        tt = np.arange(-pre, len(t)) / 250.0
        s = P.waveform(p, tt / (1 + ppm * 1e-6) + lag, a=a) if signal else np.zeros(len(tt))
        y = 0.9 * C.model_filter(s, 1.0, 27.5, 4, "linphase")[pre:]
        return t, y + 0.4 * SY.heartbeat(t, 70, rng) + rng.normal(0, 0.02, len(t))

    def test_frozen(self):
        import psk_ber as PB
        import psk_rx4 as X4
        self.assertEqual((X4.VERSION, X4.RULES_SHA), PB.FROZEN_X4)
        self.assertEqual((P.VERSION, P.RULES_SHA), PB.FROZEN_RECEIVER)      # x3-wl untouched

    def test_round_trip(self):
        import psk_rx4 as X4
        for bps, fc, shape in ((40, 16, "rrc"), (64, 16, "rrc")):
            with self.subTest(bps=bps):
                p = P.plan("dbpsk", bps, fc=fc, shape=shape, seconds=55)
                t, x = self.through_channel(p, seed=bps)
                r = X4.decode(x, t, p)
                self.assertGreater(r["frames_complete"], 3)
                self.assertEqual(r["frames_valid"], r["frames_complete"], r.get("reason"))
                self.assertEqual(r["errors"], 0)
                self.assertAlmostEqual(r["clock_ppm"], 3400, delta=60)

    def test_random_payload_every_frame(self):
        """a different payload in every frame: x4's decided bits must match each frame's own payload"""
        import psk_rx4 as X4
        p = P.plan("dbpsk", 40, fc=16, shape="rrc", seconds=55)
        rng = np.random.default_rng(3)
        _, pl, _ = L.frame(p["bps"])
        ph, a, pays = 0.0, [1.0] * p["lead_symbols"], []
        for _ in range(p["frames"]):
            pays.append(list(rng.integers(0, 2, len(pl))))
            for b in L.HDR + pays[-1] + L.GAP:
                ph += np.pi * b; a.append(np.cos(ph))
        a = np.round(np.array(a + [a[-1]] * p["tail_symbols"])) + 0j
        lag, sc = 7.0, 1 + 3400e-6
        t, x = self.through_channel(p, a=a, lag=lag, seed=4)
        r = X4.decode(x, t, p)
        v = [f for f in r["frames"] if f["valid"]]
        self.assertGreater(len(v), 3)
        T, nf = p["sym_us"] / 1e6, p["frame_symbols"]
        for f in v:
            m = int(round(((f["t_start_s"] / sc + lag) / T - 0.5 - p["lead_symbols"]) / nf))
            self.assertEqual(f["decided"][len(L.HDR):len(L.HDR) + len(pl)], [int(b) for b in pays[m]])

    def test_noise_and_heartbeat_only_gives_no_frames(self):
        import psk_rx4 as X4
        for bps in (32, 64):
            for seed in range(2):
                p = P.plan("dbpsk", bps, shape="rrc", seconds=55)
                t, x = self.through_channel(p, seed=seed, signal=False)
                self.assertEqual(X4.decode(x, t, p).get("frames_valid", 0), 0, (bps, seed))
