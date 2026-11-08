"""psk_ber.py end to end WITHOUT hardware or a person: a fake transmitter, scripted prompts, SYNTHETIC recordings
written as watch-style PDFs (ecg_synth.write_pdf), then ingest + analyse. Checks the workflow, not the channel.

   ~/.venvs/galvanic/bin/python -m unittest tests.test_psk_ber -v
"""
from __future__ import annotations

import builtins
import datetime
import glob
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import ecg_ber as B                      # noqa: E402
import ecg_synth as SY                   # noqa: E402
import psk_ber as PB                     # noqa: E402
import psk_link as P                     # noqa: E402
import psk_tx as PT                      # noqa: E402


class FakeTX:
    """stands in for psk_tx.PskTX: no serial port, '# done' as soon as asked"""
    ident_override = None

    def __init__(self, port=None):
        self.port = "FAKEPORT"
        self.ident = FakeTX.ident_override or "# watch_psk_tx x2 board FAKE"
        self.dev = types.SimpleNamespace(close=lambda: None)
        self.plan = None

    def configure(self, *a):
        return dict(amp_mvpp=a[3])

    def play(self, p):
        self.plan = p
        fw = PT.fw_payload(p)
        return dict(n_stored=len(fw["si"]), n_played=fw["n_play"], loop=list(fw["loop"]), checksum=0), time.time()

    def readline(self, timeout):
        return f"# done symbols {PT.fw_payload(self.plan)['n_play']}"

    def ask(self, line, want, timeout=3.0):
        return "# off"


def synth(p, lag, seed, signal=True, ppm=3400.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(29.5 * 250)) / 250.0
    kind, n, fh, fl = ("linphase", 4, 1.0, 27.5)
    import channel_est as C
    pre = 750
    tt = np.arange(-pre, len(t)) / 250.0
    s = P.waveform(p, tt / (1 + ppm * 1e-6) + lag) if signal else np.zeros(len(tt))
    y = 0.9 * C.model_filter(s, fh, fl, n, kind)[pre:]
    return y + 0.4 * SY.heartbeat(t, 70, rng) + rng.normal(0, 0.02, len(t))


class TestPskBer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pskber_")
        self.old = (PB.DATADIR, PT.PskTX, builtins.input)
        PB.DATADIR = self.tmp
        PT.PskTX = FakeTX
        FakeTX.ident_override = None
        self.prompts = []

        def answer(prompt=""):
            self.prompts.append(prompt)
            return "y"
        builtins.input = answer

    def tearDown(self):
        PB.DATADIR, PT.PskTX, builtins.input = self.old

    def args(self, **kw):
        a = dict(bps=16.0, label="t16", fc=16.0, shape="rrc", repeats=3, condition="worn-still", safety_review=None,
                 bench_report=None, seconds=40.0, spacing=0.0, port=None, allow_modified_receiver=False)
        a.update({k: None for k in ("vfull", "r1", "r2", "amp_mvpp", "tx_ppm")})
        a.update(kw)
        return types.SimpleNamespace(**a)

    def session(self, signal=True, **kw):
        with redirect_stdout(io.StringIO()):
            PB.run(self.args(**kw))
        sess = glob.glob(os.path.join(self.tmp, "psk_session_*"))[-1]
        pdfs = os.path.join(self.tmp, "pdfs_" + os.path.basename(sess)); os.makedirs(pdfs)
        for i, (t0, d, r) in enumerate(PB.session_runs(sess)):
            t_go = 1_800_000_000.0 + 200 * i                       # as if the runs were spaced 200 s apart
            r["t_go"] = t_go
            json.dump(r, open(os.path.join(d, "tx.json"), "w"), indent=1, default=float)
            lag = 6.0 + i
            x = synth(r["plan"], lag, seed=i, signal=signal)
            SY.write_pdf(os.path.join(pdfs, f"r{i}.pdf"), x, when=datetime.datetime.fromtimestamp(t_go + lag))
        return sess, pdfs

    def refused(self, **kw):
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                PB.run(self.args(**kw))
        self.assertFalse(glob.glob(os.path.join(self.tmp, "psk_session_*")), "a session was created")
        return str(cm.exception)

    def test_run_starts_directly(self):
        with redirect_stdout(io.StringIO()):
            PB.run(self.args(repeats=1))
        self.assertEqual(len(self.prompts), 2, self.prompts)          # Enter to transmit + did the ECG complete
        self.assertIn("press Enter to transmit", self.prompts[0])

    def test_level_and_firmware_limits_kept(self):
        self.assertIn("above the 10 mVpp", self.refused(amp_mvpp=12.0))
        FakeTX.ident_override = "# watch_ber_tx v1"
        self.assertIn("upload arduino/watch_psk_tx", self.refused())

    def test_run_ingest_analyse(self):
        sess, pdfs = self.session()
        buf = io.StringIO()
        with redirect_stdout(buf):
            out, summ = PB.ingest(types.SimpleNamespace(session=sess, files=[pdfs], allow_modified_receiver=False))
        prim = [s for s in summ if s["receiver"] == "x3-wl"][0]
        self.assertEqual(prim["usable"], 3, buf.getvalue())
        self.assertEqual(prim["errors"], 0)
        self.assertGreaterEqual(prim["bits"], 400)
        self.assertGreaterEqual(prim["sync_rate"], 0.95)
        self.assertEqual(prim["verdict"], "reliable", buf.getvalue())
        self.assertEqual(prim["voltage_status"], "calculated")
        self.assertIn("24 bps", PB.next_step(summ, PB.protocol()[0])[0])
        j = json.load(open(os.path.join(out, "summary.json")))
        self.assertEqual(j["receiver"]["rules_sha"], P.RULES_SHA)
        self.assertTrue(os.path.exists(os.path.join(out, "curve.png")))

    def test_no_signal_is_no_bits_not_zero_ber(self):
        sess, pdfs = self.session(signal=False, label="n16")
        with redirect_stdout(io.StringIO()):
            out, summ = PB.ingest(types.SimpleNamespace(session=sess, files=[pdfs], allow_modified_receiver=False))
        prim = [s for s in summ if s["receiver"] == "x3-wl"][0]
        self.assertEqual(prim["bits"], 0)
        self.assertIsNone(prim["ber"])
        self.assertEqual(prim["verdict"], "unreliable")
        self.assertIn("transmission not found", prim["diagnosis"])
        self.assertGreater(prim["frames_transmitted"], 0)

    def test_ambiguous_minute_is_refused(self):
        with redirect_stdout(io.StringIO()):
            PB.run(self.args(label="a16", repeats=2))
        sess = glob.glob(os.path.join(self.tmp, "psk_session_a16_*"))[-1]
        pdfs = os.path.join(self.tmp, "amb"); os.makedirs(pdfs)
        runs = PB.session_runs(sess)
        for i, (t0, d, r) in enumerate(runs):
            r["t_go"] = 1_800_000_000.0 + 10 * i                    # NOT spaced: both runs fit one minute
            json.dump(r, open(os.path.join(d, "tx.json"), "w"), indent=1, default=float)
        x = synth(runs[0][2]["plan"], 6.0, seed=9)
        SY.write_pdf(os.path.join(pdfs, "x.pdf"), x, when=datetime.datetime.fromtimestamp(1_800_000_006.0))
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                PB.ingest(types.SimpleNamespace(session=sess, files=[pdfs], allow_modified_receiver=False))
            except SystemExit:
                pass
        self.assertIn("NOT imported", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
