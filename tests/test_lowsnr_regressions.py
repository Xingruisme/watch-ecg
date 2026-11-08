"""
Regression tests for the two low-SNR failures of the frozen ECG decoder ecg-v1 (found 2026-11-03; README section 10).

Both show on one noise-only recording (tests/fixtures/nosignal_ook4, rebuilt by tests/make_fixture.py):
  F1 clock lock on noise: the clock search returns +688 ppm (true +1227) with a line SNR of 11.2 dB, so the recording
     passes the usability rule (>= 6 dB) although no tone was sent
  F2 false frame acceptance: one frame passes the truth-free acceptance (header r = 0.41 >= 0.3) and contributes
     17 errors / 31 bits to the pooled BER

Three groups:
  TestBenchmarkUnchanged  the decoder is still the frozen file and the saved dry-run benchmark is byte-identical
  TestCurrentBehaviour    pins what ecg-v1 does today on the fixture (must PASS; a change here means the decoder
                          or the synthetic chain changed)
  TestKnownFailures       the desired behaviour, marked expectedFailure. They "fail as expected" today; a decoder
                          that fixes F1/F2 makes them unexpected successes, which unittest reports as a failed run -
                          then move them to the passing group, in a new decoder version (ecg-v1 stays frozen)

   cd ~/xingru/watch-ecg && ~/.venvs/galvanic/bin/python -m unittest discover -s tests -v
Read-only: nothing is written outside a temporary folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

TESTS = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(TESTS)
sys.path.insert(0, HERE); sys.path.insert(0, TESTS)
import ecg_ber as B                                         # noqa: E402
import ecg_io as IO                                         # noqa: E402
import ecg_link as L                                        # noqa: E402
import ecg_synth as Y                                       # noqa: E402

FIX = os.path.join(TESTS, "fixtures")
TRUE_CLOCK_PPM = 1227.0
BENCH = os.path.join(B.DATADIR, "synth_dry1_20261102_155145", "ecgber_synth_dry1_20261102_155159")
BENCH_SHA256 = {                                            # as written on 2026-11-02 15:51
    "per_run.csv": "954853a14e1d92dc67deeccc7bc32efd41eb33a6a55a8bcf3df0b539b84b528a",
    "summary.csv": "70045bdbd0d2e9448e9836b55ac73ab416563ee7ef7cae20fb14d13c88196735",
    "summary.json": "043eaf0bba2455d477dfdf8328d2cdaf9ef2c7056cf2925282fe5649896bb42d",
}
# small seeded noise-only batch (same noise model): pinned counts of today's false acceptances
BATCH_SEED, BATCH_N = 20261104, 20
BATCH_PINNED = dict(usable=7, usable_with_valid_frame=5, valid_frames=8)          # ecg-v1, 2026-11-04


def run_fixture():
    """the fixture through analyse_ber, exactly as `ecg_ber.py analyse` would (in a temporary run folder)"""
    rec = json.load(open(os.path.join(FIX, "nosignal_ook4.json")))
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "tx.json"), "w") as f:
            json.dump(rec, f)
        shutil.copy(os.path.join(FIX, "nosignal_ook4.npz"), os.path.join(d, "ecg.npz"))
        e = IO.load(os.path.join(d, "ecg.npz"))
        res, rep, _ = B.analyse_ber(d, rec, e)
    return rec, res, rep


def noise_batch(n=BATCH_N, seed=BATCH_SEED, mod="ook", bps=4.0):
    rng = np.random.default_rng(seed)
    plan = L.ber_plan(mod, bps)
    out = []
    for _ in range(n):
        t, x, _ = Y.record(plan, rng, no_signal=True)
        rep, _ = L.decode(x, t, plan)
        u = B.usability(types.SimpleNamespace(t=t, mv=x, fs=Y.FS_NOM), rep["line_snr_db"])
        out.append((u["usable"], sum(f["valid"] for f in rep["frames"])))
    return dict(usable=sum(u for u, _ in out), usable_with_valid_frame=sum(u and v > 0 for u, v in out),
                valid_frames=sum(v for u, v in out if u))


class TestBenchmarkUnchanged(unittest.TestCase):
    def test_decoder_is_the_frozen_file(self):
        self.assertEqual(L.RULES_SHA, B.FROZEN_SHA)
        self.assertEqual(L.VERSION, "ecg-v1")

    def test_dry_run_benchmark_files_unchanged(self):
        for name, sha in BENCH_SHA256.items():
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256(open(os.path.join(BENCH, name), "rb").read()).hexdigest(), sha)

    def test_fixture_is_reproducible_from_its_seed(self):
        import make_fixture
        e, plan, info = make_fixture.regenerate()
        f = IO.load(os.path.join(FIX, "nosignal_ook4.npz"))
        self.assertTrue(np.array_equal(e.mv, f.mv) and np.array_equal(e.t, f.t))
        self.assertEqual(json.loads(json.dumps(plan)), json.load(open(os.path.join(FIX, "nosignal_ook4.json")))["plan"])
        self.assertTrue(info["no_signal"])


class TestCurrentBehaviour(unittest.TestCase):
    """ecg-v1 on the noise-only fixture, as found on 2026-11-03 - these values must not drift"""
    @classmethod
    def setUpClass(cls):
        cls.rec, cls.res, cls.rep = run_fixture()

    def test_f1_clock_locks_on_noise(self):
        self.assertAlmostEqual(self.rep["clock_ppm"], 688.23, delta=0.05)
        self.assertAlmostEqual(self.rep["line_snr_db"], 11.184, delta=0.005)
        self.assertTrue(self.res["usability"]["usable"])

    def test_f2_one_false_frame_accepted(self):
        fr = self.rep["frames"]
        self.assertEqual([(f["valid"], f["errors"], f["bits"]) for f in fr], [(True, 17, 31), (False, 17, 31)])
        self.assertAlmostEqual(fr[0]["r"], 0.4067, delta=0.0005)
        b = self.res["blind"]
        self.assertEqual((b["frames_valid"], b["errors"], b["bits"]), (1, 17, 31))

    def test_seeded_noise_batch_false_alarms(self):
        self.assertEqual(noise_batch(), BATCH_PINNED)


class TestKnownFailures(unittest.TestCase):
    """the behaviour a fixed decoder should have; expectedFailure while ecg-v1 is the decoder"""
    @classmethod
    def setUpClass(cls):
        cls.rec, cls.res, cls.rep = run_fixture()

    @unittest.expectedFailure
    def test_f1_no_signal_line_snr_below_usability_threshold(self):
        self.assertLess(self.rep["line_snr_db"], B.MIN_LINE_SNR_DB)

    @unittest.expectedFailure
    def test_f1_no_signal_recording_flagged_unusable(self):
        self.assertFalse(self.res["usability"]["usable"])

    @unittest.expectedFailure
    def test_f1_clock_reported_only_when_recovered(self):
        # a recording that counts as usable must have its clock within the +-200 ppm residual search of the truth
        self.assertTrue(not self.res["usability"]["usable"] or abs(self.rep["clock_ppm"] - TRUE_CLOCK_PPM) < 200)

    @unittest.expectedFailure
    def test_f2_no_frame_accepted_without_signal(self):
        self.assertEqual(self.res["blind"]["frames_valid"], 0)

    @unittest.expectedFailure
    def test_f1_f2_seeded_noise_batch_has_no_false_alarms(self):
        self.assertEqual(noise_batch()["usable_with_valid_frame"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
