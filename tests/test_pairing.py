"""ecg_pairing: recordings that share a 'Recorded on' minute are paired by signal, or reported ambiguous - never by
file order (the 4 bps swap of 2026-11-06). Synthetic recordings only; the decoder is not involved."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ecg_ber as B           # noqa: E402
import ecg_io as IO           # noqa: E402
import ecg_pairing as PA      # noqa: E402

FS, DUR = 250.0, 29.5


def recording(plan, start_tx_s, ppm=3500.0, amp=0.8, noise=0.05, seed=0):
    """what the watch would record: the run's tone sequence from start_tx_s on, watch time base off by ppm. As on the
    real watch (and ecg_synth), a tone sent at f is seen at f / (1 + ppm): transmitter time = t / (1 + ppm)"""
    tb, seq = PA.sent(plan)
    t = np.arange(int(DUR * FS)) / FS
    tx = t / (1 + ppm * 1e-6) + start_tx_s
    idx = np.floor(tx / tb).astype(int)
    f = np.where((idx >= 0) & (idx < len(seq)), seq[np.clip(idx, 0, len(seq) - 1)], 0.0)
    x = 0.5 * amp * np.sin(2 * np.pi * f * tx) + np.random.default_rng(seed).normal(0, noise, len(t))
    return IO.Ecg(t=t, mv=x, fs=FS, meta={})


def run(mod, t_go, name, bps=4):
    plan = B.ber_plan_for(mod, bps, "baseline", 60.0, "arduino")
    return (t_go, name, dict(plan=plan, t_go=t_go, _dir=name))


class Pairing(unittest.TestCase):
    def test_same_minute_resolved_by_signal_not_file_order(self):
        # as on 2026-11-06: OOK go 14:11:51, FSK go 14:12:48, both recordings 'Recorded on 14:12'
        w = 1000.0
        ook, fsk = run("ook", w - 9.0, "ook_run"), run("fsk", w + 48.0, "fsk_run")
        e_ook = recording(ook[2]["plan"], 9.0, seed=1)
        e_fsk = recording(fsk[2]["plan"], 11.0, seed=2)
        readings = [(w, "a_fsk.pdf", e_fsk), (w, "b_ook.pdf", e_ook)]    # file order = the old, wrong order
        got = {readings[i][1]: (d, m) for i, d, m, _ in PA.resolve(readings, [ook, fsk], B.ECG_LATEST_S)}
        self.assertEqual(got["a_fsk.pdf"], ("fsk_run", "signal"))
        self.assertEqual(got["b_ook.pdf"], ("ook_run", "signal"))

    def test_indistinguishable_runs_are_reported_not_guessed(self):
        w = 1000.0
        r1, r2 = run("ook", w - 9.0, "ook_1"), run("ook", w - 8.0, "ook_2")   # same plan, same minute
        readings = [(w, "x.pdf", recording(r1[2]["plan"], 9.0, seed=3)),
                    (w, "y.pdf", recording(r2[2]["plan"], 10.0, seed=4))]
        res = PA.resolve(readings, [r1, r2], B.ECG_LATEST_S)
        self.assertTrue(all(d is None and m == "ambiguous" for _, d, m, _ in res))

    def test_single_candidate_keeps_time_pairing(self):
        w = 1000.0
        ook = run("ook", w - 9.0, "ook_run")
        res = PA.resolve([(w, "o.pdf", recording(ook[2]["plan"], 9.0, seed=5))], [ook], B.ECG_LATEST_S)
        self.assertEqual(res[0][1:3], ("ook_run", "time"))
        self.assertTrue(res[0][3][0]["ok"])

    def test_16bps_with_real_clock_offset_passes(self):
        # v1 mapped watch -> TX time with the wrong clock sign: r ~0.4 at 16 bps on real data. The 0.5 threshold stays.
        w = 1000.0
        for mod in ("ook", "fsk"):
            r_ = run(mod, w - 9.0, f"{mod}16", bps=16)
            ev = PA.check(recording(r_[2]["plan"], 9.0, ppm=3400.0, seed=6), w, r_[2])
            self.assertTrue(ev["ok"], ev)
            self.assertGreater(ev["pattern_r"], 0.8)
            self.assertAlmostEqual(ev["clock_ppm"], 3400, delta=300)

    def test_wrong_rate_is_rejected(self):
        w = 1000.0
        r16, r8 = run("ook", w - 9.0, "ook16", bps=16), run("ook", w - 9.0, "ook8", bps=8)
        ev = PA.check(recording(r16[2]["plan"], 9.0, seed=7), w, r8[2])
        self.assertFalse(ev["ok"], ev)


if __name__ == "__main__":
    unittest.main()
