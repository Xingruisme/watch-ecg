#!/usr/bin/env python3
"""
Arduino transmitter path (watch_ber_tx.ino + arduino_tx.py + ecg_ber.py --tx arduino + ard_bench.py) without
hardware. Everything here is SYNTHETIC: captures are made from arduino_tx.model_output (a model of the firmware) plus
scope-like noise; the 'board' is a Python fake on a pseudo-terminal. These tests check the software, not the board.

   ~/.venvs/galvanic/bin/python -m unittest tests.test_arduino -v
"""
from __future__ import annotations

import io
import json
import os
import pty
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import arduino_tx as AT                     # noqa: E402
import ard_bench as AB                      # noqa: E402
import ecg_ber as B                         # noqa: E402
import ecg_link as L                        # noqa: E402

INO = os.path.join(HERE, "arduino", "watch_ber_tx", "watch_ber_tx.ino")


def write_picoscope_csv(path, t, mv):
    with open(path, "w") as f:
        f.write("Time,Channel A\n(s),(mV)\n\n")
        for a, b in zip(t, mv):
            f.write(f"{a:.8f},{b:.6f}\n")


def synth_capture(path, plan, board_ppm=-2400.0, clk_ppm=0.0, seconds=30.0, fs=2000.0, start=1.3, seed=1,
                  amp=2.0, noise_mv=0.05, step_mv=0.11, hum_mvpp=1.0):
    """SYNTHETIC PicoScope capture of the divider output while the firmware plays `plan`"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    step_us, reps, hz = AT.steps_of(plan) if plan["kind"] != "cw" else (100_000_000, 1, [plan["hz"]])
    v = AT.model_output(step_us, reps, hz, t, amp_mvpp=amp, board_ppm=board_ppm, clk_ppm=clk_ppm, t_start=start)
    v = v + rng.normal(0, noise_mv, len(t)) + hum_mvpp / 2 * np.sin(2 * np.pi * 60 * t + 0.3)
    v = np.round(v / step_mv) * step_mv
    write_picoscope_csv(path, t - seconds / 2, v)


class FakeBoard(threading.Thread):
    """the firmware's serial protocol (replies only; no waveform) on a pseudo-terminal"""

    def __init__(self):
        super().__init__(daemon=True)
        self.master, slave = pty.openpty()
        self.port = os.ttyname(slave)
        self.cfg = dict(vfull=5.0, rtop=1e6, rbot=470.0, amp=2.0, clk=0.0)
        self.sched = None
        self.lines = []

    def out(self, s):
        os.write(self.master, (s + "\r\n").encode())

    def cfg_line(self):
        c = self.cfg; ratio = c["rbot"] / (c["rtop"] + c["rbot"])
        return (f"# cfg vfull {c['vfull']:.4f} rtop {c['rtop']:.1f} rbot {c['rbot']:.3f} amp_mvpp {c['amp']:.4f} "
                f"dac_vpp {c['amp'] / 1000 / ratio:.4f} dc_mv {c['vfull'] * 2048 / 4095 * ratio * 1000:.4f} "
                f"clk_ppm {c['clk']:.1f} update_us 250 limit_mvpp 10.0")

    def run(self):
        buf = b""
        while True:
            try:
                buf += os.read(self.master, 4096)
            except OSError:
                return
            while b"\n" in buf:
                ln, buf = buf.split(b"\n", 1)
                ln = ln.decode().strip(); self.lines.append(ln)
                w = ln.split()
                if not w:
                    continue
                if w[0] == "id":
                    self.out("# watch_ber_tx v1 board UNO-R4 dac A0 bits 12 update_us 250")
                elif w[0] == "cal":
                    self.cfg.update(vfull=float(w[1]), rtop=float(w[2]), rbot=float(w[3])); self.out(self.cfg_line())
                elif w[0] == "clk":
                    self.cfg["clk"] = float(w[1]); self.out(self.cfg_line())
                elif w[0] == "a":
                    if float(w[1]) > 10:
                        self.out("# error amplitude outside 0-10.0 mVpp"); continue
                    self.cfg["amp"] = float(w[1]); self.out(self.cfg_line())
                elif w[0] == "load":
                    su, rp, mhz = int(w[1]), int(w[2]), [int(x) for x in w[3:]]
                    self.sched = (su, rp, mhz)
                    self.out(f"# loaded n {len(mhz)} sum {AT.checksum(su, rp, mhz)}")
                elif w[0] == "go":
                    self.out(f"# armed in {w[1]} ms")
                    su, rp, mhz = self.sched
                    self.out("# start"); self.out(f"# done events {rp * len(mhz)}")
                elif w[0] == "?":
                    su, rp, mhz = self.sched or (0, 0, [])
                    self.out(self.cfg_line())
                    self.out(f"# state done events {rp * len(mhz)} of {rp * len(mhz)} n {len(mhz)} step_us {su} reps {rp}")
                elif w[0] == "x":
                    self.out("# off")
                elif w[0] == "cw":
                    self.out(f"# cw {float(w[1]):.3f} Hz for {w[2] if len(w) > 2 else 300} s")
                else:
                    self.out(f"# error unknown command: {w[0]}")


class TestArduino(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ardtest_")
        cls._datadir = B.DATADIR
        B.DATADIR = cls.tmp                       # ard_bench / ecg_ber outputs go to the temp folder

    @classmethod
    def tearDownClass(cls):
        B.DATADIR = cls._datadir
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_decoder_still_frozen(self):
        self.assertEqual(L.RULES_SHA, B.FROZEN_SHA)

    def test_checksum_matches_firmware(self):
        """compile the firmware's checksum() with the host gcc and compare with arduino_tx.checksum"""
        if not shutil.which("gcc"):
            self.skipTest("no gcc")
        src = open(INO).read()
        body = re.search(r"uint32_t checksum\(\) \{.*?\n\}", src, re.S).group(0)
        c = ("#include <stdint.h>\n#include <stdio.h>\nuint32_t stepUs, reps, nSteps; uint32_t stepMilliHz[400];\n"
             + body + "\nint main(){unsigned a,b,n,i; while(scanf(\"%u %u %u\",&a,&b,&n)==3){stepUs=a;reps=b;nSteps=n;"
             "for(i=0;i<n;i++)scanf(\"%u\",&stepMilliHz[i]); printf(\"%u\\n\",checksum());} return 0;}")
        d = tempfile.mkdtemp(prefix="cks_")
        open(os.path.join(d, "c.c"), "w").write(c)
        subprocess.run(["gcc", "-O1", "-o", os.path.join(d, "c"), os.path.join(d, "c.c")], check=True)
        cases = []
        for p in (B.ber_plan_for("fsk", 16, "baseline", 60, "arduino"), B.ber_plan_for("ook", 2, "exp", 90, "arduino"),
                  B.arduino_plan(L.tone_plan("sweep", 60))):
            su, rp, hz = AT.steps_of(p)
            cases.append((su, rp, [int(round(f * 1000)) for f in hz]))
        inp = "".join(f"{su} {rp} {len(m)} " + " ".join(map(str, m)) + "\n" for su, rp, m in cases)
        got = subprocess.run([os.path.join(d, "c")], input=inp, capture_output=True, text=True, check=True).stdout.split()
        self.assertEqual([int(g) for g in got], [AT.checksum(su, rp, m) for su, rp, m in cases])
        shutil.rmtree(d, ignore_errors=True)

    def test_plans(self):
        for ts, f1, f0 in (("baseline", 16, 8), ("exp", 12, 8)):
            p = B.ber_plan_for("fsk", 2, ts, 60, "arduino")
            self.assertEqual((p["f1_hz"], p["f0_hz"], p["f1_line_hz"], p["f0_line_hz"]), (f1, f0, f1, f0))
        self.assertEqual(B.ber_plan_for("ook", 2, "exp", 60, "arduino")["f1_hz"], 10)
        self.assertTrue(B.ber_plan_for("ook", 16, "baseline", 60, "arduino")["exploratory"])
        with self.assertRaises(SystemExit):
            B.ber_plan_for("ook", 4, "exp", 60, "arduino")
        with self.assertRaises(SystemExit):
            B.arduino_plan(L.ber_plan("ook", 2, 5.0, None, 60))          # 2.5 cycles per bit
        self.assertNotIn("transmitter", L.ber_plan("ook", 2))             # the TeensyTRX plan is untouched

    def run_bench(self, argv):
        a = AB.argparse.Namespace(**argv)
        with redirect_stdout(io.StringIO()) as out:
            ok = AB.tone(a) if argv["cmd"] == "tone" else AB.frame(a)
        rep = json.load(open(os.path.join(sorted(d for d in [os.path.join(self.tmp, x) for x in os.listdir(self.tmp)]
                                                 if os.path.isdir(d))[-1], "report.json")))
        return ok, rep, out.getvalue()

    def test_bench_tone(self):
        path = os.path.join(self.tmp, "cw16.csv")
        synth_capture(path, dict(kind="cw", hz=16.0), seconds=10, fs=5000, start=-1.0)
        ok, rep, out = self.run_bench(dict(cmd="tone", capture=path, label="syn_t16", zero=None, vfull=None, r1=None,
                                           r2=None, amp_mvpp=None, hz=16.0))
        self.assertAlmostEqual(rep["board_clock_ppm"], -2400, delta=30)
        self.assertAlmostEqual(rep["fit"]["vpp_mv"], 2.0, delta=0.04)
        self.assertIn("MEASURED", rep["source"])                          # the tool's label; the input here is synthetic

    def test_bench_frames(self):
        for mod, bps, ts in (("ook", 2, "exp"), ("fsk", 2, "exp"), ("fsk", 4, "baseline"), ("ook", 8, "baseline")):
            with self.subTest(mod=mod, bps=bps, toneset=ts):
                plan = B.ber_plan_for(mod, bps, ts, 40, "arduino")
                path = os.path.join(self.tmp, f"{mod}{bps}{ts}.csv")
                synth_capture(path, plan, seconds=32)
                ok, rep, out = self.run_bench(dict(cmd="frame", capture=path, label=f"syn_{mod}{bps}", zero=None,
                                                   vfull=None, r1=None, r2=None, amp_mvpp=None, mod=mod, bps=bps,
                                                   toneset=ts))
                dec = rep["offline_decode"]
                self.assertGreaterEqual(dec["frames_valid"], 2, out)
                self.assertEqual(dec["errors"], 0, out)
                self.assertAlmostEqual(rep["timing"]["tx_clock_ppm_vs_scope"], -2400, delta=60, msg=out)
                self.assertAlmostEqual(rep["levels"]["f1_vpp_mv"], 2.0, delta=0.06, msg=out)
                self.assertLess(abs(rep["levels"]["dc_during_1_mv"] - rep["levels"]["dc_during_0_mv"]), 0.02, out)

    def test_bench_frame_corrected_clock(self):
        plan = B.ber_plan_for("ook", 2, "baseline", 40, "arduino")
        path = os.path.join(self.tmp, "ook2corr.csv")
        synth_capture(path, plan, board_ppm=-2400, clk_ppm=-2400, seconds=32)
        ok, rep, out = self.run_bench(dict(cmd="frame", capture=path, label="syn_corr", zero=None, vfull=None,
                                           r1=None, r2=None, amp_mvpp=None, mod="ook", bps=2, toneset="baseline"))
        self.assertTrue(ok, out)
        self.assertLess(abs(rep["timing"]["tx_clock_ppm_vs_scope"]), 60, out)

    def test_driver_and_tx_flow_with_fake_board(self):
        fb = FakeBoard(); fb.start()
        tx = AT.ArduinoTX(fb.port)
        cfg = tx.configure(4.95, 940e3, 470.0, 2.0, -2400)
        self.assertAlmostEqual(cfg["amp_mvpp"], 2.0)
        self.assertAlmostEqual(cfg["dac_vpp"], 2.0e-3 * (940e3 + 470) / 470, places=3)
        with self.assertRaises(RuntimeError):
            tx.ask("a 25", r"^# cfg")
        plan = B.ber_plan_for("fsk", 2, "exp", 30, "arduino")
        su, rp, hz = AT.steps_of(plan)
        up = tx.load(su, rp, hz)
        self.assertEqual(up["n"], 28)
        a = types.SimpleNamespace(label="fake", condition="bench-direct", r1=940e3, r2=470.0, amp_mvpp=2.0,
                                  source_point="vout", source_vpp=None, vout_vpp=None, source_measured=False,
                                  source_dc_mv=None, dc_block=False, watch_vpp_mv=None, watch_dc_mv=None,
                                  watch_instrument=None, vfull=4.95, tx_ppm=-2400)
        st = B.ard_settings(a)
        rec = B.ard_record(a, plan, st, cfg, tx, "bench-direct, ard-exp", "bench")
        self.assertEqual(rec["voltage"]["watch_input"]["status"], "calculated")
        self.assertEqual(rec["voltage"]["a0"]["status"], "unavailable")
        d = os.path.join(self.tmp, "run_fake"); os.makedirs(d)
        plan_short = dict(plan, total_s=0.2)
        import builtins
        orig = builtins.input
        builtins.input = lambda *_: "y"
        try:
            with redirect_stdout(io.StringIO()):
                B.transmit_arduino(tx, d, rec, plan_short)
        finally:
            builtins.input = orig
        saved = json.load(open(os.path.join(d, "tx.json")))
        self.assertEqual(saved["tx_status"]["state"], "done")
        self.assertEqual(saved["tx_status"]["events"], plan["n_steps"])
        self.assertIn("ard-exp", saved["condition"])
        tx.close()
        self.assertEqual(fb.lines[-1], "x")                              # output left at mid-scale


if __name__ == "__main__":
    unittest.main()
