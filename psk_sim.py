"""SIMULATED evaluation of the experimental psk-x1 modulations through the MEASURED watch channel (added 2026-11-06).

Nothing here is a measured BER. Each simulated recording = the psk-x1 transmitter waveform (exact firmware model)
  -> the watch channel fitted to the 10 real worn recordings (channel_est.py parametric: 1 Hz HP, low-pass ~25-27.5 Hz,
     linear-phase best; minimum-phase 2nd order 22.5 Hz as the alternative)
  -> x the gain measured on one real recording, + THE REAL RESIDUAL of that recording (the wearer's ECG and noise as
     the watch recorded it, after removing the fitted OOK/FSK signal), at a random start and clock (+-5000 ppm).
Pre-registered (before any PSK measurement): thresholds psk_link.R_MIN / G_MIN are checked on residual-only input
('falsealarm'); 'reliable' = frame sync >= 95 % of complete frames AND pooled BER upper 95 % CI < 1 %.

  python psk_sim.py                      -> data_watch_ecg/SIMULATED_psk_<time>/ (results.jsonl, summary.csv, README.txt)
  python psk_sim.py --quick              (fewer cells)
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import itertools
import json
import os

import numpy as np

import channel_est as C
import ecg_ber as B
import ecg_io as IO
import ecg_pairing as PA
import psk_link as P

HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = dict(lin27=("linphase", 4, 1.0, 27.5), min22=("minphase", 2, 1.0, 22.5))


def real_residuals(channel):
    """per real recording: (name, gain, residual mV) under the given channel model"""
    kind, n, fh, fl = CHANNELS[channel]
    out = []
    for s in sorted(glob.glob(os.path.join(B.DATADIR, "session_ard_*"))):
        for t0, d, rec in PA.session_runs(s):
            if not os.path.exists(os.path.join(d, "ecg.npz")):
                continue
            e = IO.load(os.path.join(d, "ecg.npz")); imp = json.load(open(os.path.join(d, "ecg_import.json")))
            ev = PA.check(e, B.parse_recorded(imp["recorded_on"]).timestamp(), rec)
            if not ev["ok"]:
                continue
            sc = 1 + ev["clock_ppm"] * 1e-6
            x = np.asarray(e.mv, float); x = x - x.mean()
            pre = 750
            i = np.arange(-pre, len(x))
            s_ = C.tx_wave(rec["plan"], i / C.FS / sc + ev["start_tx_s"])
            y = C.model_filter(s_, fh, fl, n, kind)[pre:]
            r2, lag, g = C.best_delay_fit(x, y, 40)
            res = x - g * np.roll(y - y.mean(), lag)
            out.append((os.path.basename(d), abs(float(g)), res - res.mean()))
    return out


def simulate(p, channel, gain, residual, rng, amp=1.0, no_signal=False, ppm=None):
    kind, n, fh, fl = CHANNELS[channel]
    N = len(residual)
    t = np.arange(N) / C.FS
    ppm = rng.uniform(-5000, 5000) if ppm is None else ppm
    sc = 1 + ppm * 1e-6
    lag = rng.uniform(1.0, max(1.5, p["total_s"] - N / C.FS - 1.0))
    pre = 750
    tt = np.arange(-pre, N) / C.FS
    s_ = np.zeros(len(tt)) if no_signal else P.waveform(p, tt / sc + lag)
    y = C.model_filter(s_, fh, fl, n, kind)[pre:]
    return t, amp * gain * y + residual, dict(ppm=ppm, lag=lag)


def cp(e, n):
    return B.cp(e, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=20261106)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = os.path.join(B.DATADIR, f"SIMULATED_psk_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out)
    res = {c: real_residuals(c) for c in CHANNELS}
    print(f"real residuals: {len(res['lin27'])} recordings; gains "
          + ", ".join(f"{g:.2f}" for _, g, _ in res["lin27"]) + " (mV per unit peak)")
    mods = ["dbpsk", "bpsk", "dqpsk", "ask4"]
    rates = [8, 16, 24, 32, 50]
    shapes = ["rect", "rrc"] if a.quick else ["rect", "rc", "rrc"]
    rows = []
    fj = open(os.path.join(out, "results.jsonl"), "w")

    def cell(tag, mod, bps, shape, eq, channel, amp, no_signal=False):
        p = P.plan(mod, bps, shape=shape, seconds=60)
        e = n = fv = fc = 0
        snrs, fails = [], {}
        for name, g, r in res[channel]:
            t, x, tr = simulate(p, channel, g, r, rng, amp, no_signal)
            rep = P.decode(x, t, p, eq=eq)
            v = [f for f in rep["frames"] if f["valid"]]
            e += sum(f["errors"] for f in v); n += sum(f["bits"] for f in v)
            fv += len(v); fc += len(rep["frames"])
            if rep.get("symbol_snr_db") is not None:
                snrs.append(rep["symbol_snr_db"])
            why = rep.get("reason") or ("bit errors" if any(f["errors"] for f in v) else "")
            if why:
                fails[why] = fails.get(why, 0) + 1
            fj.write(json.dumps(dict(tag=tag, mod=mod, bps=bps, shape=shape, eq=eq, channel=channel, amp=amp,
                                     no_signal=no_signal, residual=name, truth=tr,
                                     **{k: rep.get(k) for k in ("clock_ppm", "sync_score", "frames_valid",
                                                                "frames_complete", "bits", "errors", "symbol_snr_db",
                                                                "reason")}), default=float) + "\n")
        lo, hi = cp(e, n)
        row = dict(tag=tag, modulation=mod, bps=p["bps"], baud=round(p["baud"], 2), shape=shape, eq=eq,
                   channel=channel, amp_mvpp=2.0 * amp, recordings=len(res[channel]), bits=n, errors=e,
                   ber=e / n if n else None, ci95_lo=lo, ci95_hi=hi, frames_valid=fv, frames_complete=fc,
                   sync_rate=fv / fc if fc else 0.0, payload_bps=round(p["payload_bps"], 2),
                   goodput_bps=round((n - e) / 29.5 / len(res[channel]), 2),
                   snr_db_median=round(float(np.median(snrs)), 1) if snrs else None,
                   failures=";".join(f"{k} x{v}" for k, v in fails.items()),
                   reliable=bool(fc and fv / fc >= 0.95 and n and hi is not None and hi < 0.01))
        rows.append(row)
        ber = "-" if not n else f"{e}/{n}"
        print(f"  {tag:5s} {mod:5s} {bps:3d} {shape:4s} eq={int(eq)} {channel} amp {2*amp:.1f}: frames {fv}/{fc} "
              f"errors {ber:>10s} CI<{'-' if hi is None else format(hi, '.2%')}  snr {row['snr_db_median']}  {'RELIABLE' if row['reliable'] else ''} "
              f"{row['failures']}", flush=True)

    print("\n1) false alarm: REAL residuals only (no PSK transmitted) - every valid frame here is a false frame")
    for mod, bps in itertools.product(mods, rates):
        cell("noise", mod, bps, "rrc", True, "lin27", 1.0, no_signal=True)
    print("\n2) main grid, measured linear-phase channel, 2 mVpp nominal peak-to-peak")
    for mod, bps, shape, eq in itertools.product(mods, rates, shapes, (False, True)):
        cell("main", mod, bps, shape, eq, "lin27", 1.0)
    print("\n3) sensitivity: minimum-phase channel alternative, and half / quarter amplitude (margin)")
    for mod, bps in itertools.product(mods, rates):
        cell("minph", mod, bps, "rrc", True, "min22", 1.0)
        for amp in (0.5, 0.25):
            cell("amp", mod, bps, "rrc", True, "lin27", amp)
    fj.close()
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    open(os.path.join(out, "README.txt"), "w").write(
        "SIMULATED - NOT MEASURED BER.\n" + __doc__ + f"\ndecoder {P.VERSION}; R_MIN {P.R_MIN}, G_MIN {P.G_MIN}; seed {a.seed}\n"
        "channel: data_watch_ecg/channel_est_* (fitted to the real recordings); residuals: the real recordings minus the "
        "fitted OOK/FSK signal.\n")
    print(f"\n-> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
