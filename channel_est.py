"""Estimate the real Arduino -> worn Pixel Watch 2 ECG channel from the existing BER recordings (added 2026-11-06).

The transmitted waveform of every Arduino run is known exactly (tx.json schedule: sine, phase restarted each step).
With the recording aligned by the pairing check (ecg_pairing v2: start time and clock), a least-squares FIR fit
y = h * s + c gives the channel impulse response h - magnitude AND phase (group delay) - plus the residual, which is the
wearer's real ECG and noise. Only frequencies the runs excite are trustworthy (masked by the excitation spectrum).

This is a MEASUREMENT-DERIVED estimate (real recordings), not a simulation. Absolute delay is not identifiable (the
recording start is known only to the minute; the pairing alignment absorbs it), relative (dispersive) delay is.

  python channel_est.py            -> data_watch_ecg/channel_est_<time>/ (h.npz, channel.json, channel.png)
"""
from __future__ import annotations

import datetime
import glob
import json
import os

import numpy as np

import ecg_ber as B
import ecg_io as IO
import ecg_pairing as PA

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 250.0
PRE_S, POST_S = 0.40, 0.60          # taps from -0.4 to +0.6 s around the alignment
RIDGE = 1e-3                        # relative ridge (keeps unexcited frequencies small instead of wild)
FREQS = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 45, 50], float)


def tx_wave(plan, ttx):
    """the Arduino output (relative, peak 1): sine with the phase restarted at every step (watch_ber_tx v1)"""
    tb, seq = PA.sent(plan)
    k = np.floor(ttx / tb).astype(int)
    ins = (k >= 0) & (k < len(seq))
    f = np.where(ins, seq[np.clip(k, 0, len(seq) - 1)], 0.0)
    return np.where(ins, np.sin(2 * np.pi * f * (ttx - k * tb)), 0.0)


def fit(x, s_ext, n0, n1):
    """least squares x[n] = sum_j h[j] s[n - j] + c, j = -n0..n1; s_ext has n1 samples before and n0 after x"""
    N = len(x)
    cols = [s_ext[n1 - j: n1 - j + N] for j in range(-n0, n1 + 1)]
    A = np.vstack(cols + [np.ones(N)]).T
    G = A.T @ A
    lam = RIDGE * np.trace(G) / len(G)
    reg = lam * np.eye(len(G)); reg[-1, -1] = 0
    h = np.linalg.solve(G + reg, A.T @ x)
    return h[:-1], h[-1], x - A @ h


def response(h, taps_s, f):
    return np.array([np.sum(h * np.exp(-2j * np.pi * fi * taps_s)) for fi in f])


def excitation(s, f):
    """relative power of the transmitted waveform at f (where the fit is trustworthy)"""
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    fq = np.fft.rfftfreq(len(s), 1 / FS)
    return np.array([S[(fq > fi - 1) & (fq < fi + 1)].sum() for fi in f])


def main():
    out = os.path.join(B.DATADIR, f"channel_est_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out)
    n0, n1 = int(PRE_S * FS), int(POST_S * FS)
    taps = np.arange(-n0, n1 + 1) / FS
    rows, H, EX, resid = [], [], [], {}
    for s in sorted(glob.glob(os.path.join(B.DATADIR, "session_ard_*"))):
        for t0, d, rec in PA.session_runs(s):
            if not os.path.exists(os.path.join(d, "ecg.npz")):
                continue
            e = IO.load(os.path.join(d, "ecg.npz"))
            imp = json.load(open(os.path.join(d, "ecg_import.json")))
            ev = PA.check(e, B.parse_recorded(imp["recorded_on"]).timestamp(), rec)
            if not ev["ok"]:
                print(f"  skip {os.path.basename(d)}: pairing check not consistent"); continue
            scale = 1 + ev["clock_ppm"] * 1e-6
            x = np.asarray(e.mv, float); x = x - x.mean()
            i = np.arange(-n1, len(x) + n0)
            sx = tx_wave(rec["plan"], i / FS / scale + ev["start_tx_s"])
            h, c, res = fit(x, sx, n0, n1)
            r2 = 1 - np.var(res) / np.var(x)
            Hf = response(h, taps, FREQS); ex = excitation(sx, FREQS)
            H.append(Hf); EX.append(ex)
            name = os.path.basename(d)
            resid[name] = res
            rows.append(dict(run=name, modulation=rec["plan"]["modulation"], bps=rec["plan"]["bps"],
                             tones=PA.tones(rec["plan"]), explained=round(float(r2), 4),
                             residual_rms_mv=round(float(np.std(res)), 4), signal_rms_mv=round(float(np.std(x - res)), 4)))
            print(f"  {name:45s} explained {r2:6.1%}  residual (ECG + noise) {np.std(res)*1e3:6.1f} uV rms")
    H, EX = np.array(H), np.array(EX)
    # pool: per frequency, weight each recording by its excitation there; trust only well-excited frequencies
    W = EX / EX.max(axis=1, keepdims=True)
    Hp = (W * H).sum(0) / np.maximum(W.sum(0), 1e-12)
    ref = Hp[FREQS == 16][0]
    trusted = W.max(0) > 1e-3
    phase = np.unwrap(np.angle(Hp / ref))
    gd = -np.gradient(phase, 2 * np.pi * FREQS)                  # group delay relative, s
    spread = np.abs(H / H[:, [list(FREQS).index(16)]])            # per-recording |H|/|H(16)|
    table = [dict(f_hz=float(f), rel_db=round(float(20 * np.log10(abs(hp / ref))), 2),
                  rel_phase_deg=round(float(np.degrees(ph)), 1), group_delay_rel_ms=round(float(g * 1e3), 1),
                  trusted=bool(tr), rel_db_spread=round(float(20 * np.log10(np.nanmax(sp[W[:, k] > 1e-3]) /
                                                                  np.nanmin(sp[W[:, k] > 1e-3]))), 2)
                  if (W[:, k] > 1e-3).sum() > 1 else None)
             for k, (f, hp, ph, g, tr, sp) in enumerate(zip(FREQS, Hp, phase, gd, trusted, spread.T))]
    print(f"\n{'f [Hz]':>7} {'|H| re 16 Hz':>12} {'phase':>8} {'group delay re':>15} {'spread':>7}  trusted")
    for r in table:
        print(f"{r['f_hz']:7g} {r['rel_db']:+10.1f} dB {r['rel_phase_deg']:+7.0f}° {r['group_delay_rel_ms']:+12.0f} ms "
              f"{'' if r['rel_db_spread'] is None else format(r['rel_db_spread'], '5.1f') + ' dB':>7}  {'yes' if r['trusted'] else 'NO'}")
    # pooled time-domain h for simulation: average the per-recording h fits weighted like above at 16 Hz
    np.savez(os.path.join(out, "h.npz"), taps_s=taps, freqs=FREQS, H_pooled=Hp, H_each=H, excitation=EX,
             residual_runs=np.array(list(resid)), **{f"res_{i}": v for i, v in enumerate(resid.values())})
    json.dump(dict(created=datetime.datetime.now().isoformat(timespec="seconds"), kind="measurement-derived",
                   method="ridge LS FIR fit of each real recording against its known transmitted waveform, aligned "
                          "by ecg_pairing v2; pooled per frequency weighted by excitation",
                   pairing=PA.VERSION, runs=rows, response=table,
                   caveats=["absolute delay not identifiable (minute timestamps); relative group delay is",
                            "frequencies marked trusted=false were not excited by any run",
                            "TX output assumed = divider output; amplitude not measured on the worn watch"]),
              open(os.path.join(out, "channel.json"), "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        tr = np.array([r["trusted"] for r in table])
        ax[0].plot(FREQS, [r["rel_db"] for r in table], "o-", color="#1f5fbf")
        ax[0].plot(FREQS[~tr], np.array([r["rel_db"] for r in table])[~tr], "o", mfc="white", color="#1f5fbf")
        ax[0].set_xlabel("Hz"); ax[0].set_ylabel("|H| re 16 Hz [dB]"); ax[0].set_title("magnitude (open = not excited)")
        ax[1].plot(FREQS[tr], np.array([r["group_delay_rel_ms"] for r in table])[tr], "o-", color="#e66100")
        ax[1].set_xlabel("Hz"); ax[1].set_ylabel("relative group delay [ms]"); ax[1].set_title("group delay (excited only)")
        fig.suptitle("Arduino -> worn Pixel Watch 2 channel, estimated from real BER recordings")
        fig.tight_layout(); fig.savefig(os.path.join(out, "channel.png"), dpi=110)
    except Exception as ex:
        print("plot skipped:", ex)
    print(f"\n-> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()


# ================================================================ parametric model (physically constrained)
# watch = gain x high-pass (1st order, fh) x low-pass (order n, corner fl) x delay; low-pass either minimum-phase
# Butterworth (dispersive group delay) or linear-phase (constant delay, same magnitude). Shape shared by all
# recordings; gain, delay and DC per recording. Score = pooled explained variance.
def model_filter(s, fh, fl, n, kind):
    from scipy import signal
    y = signal.sosfilt(signal.butter(1, fh, "highpass", fs=FS, output="sos"), s)
    sos = signal.butter(n, fl, "lowpass", fs=FS, output="sos")
    if kind == "minphase":
        return signal.sosfilt(sos, y)
    # linear phase with the same |H|: zero-phase |H| (sosfiltfilt squares |H|, so use half the order twice)
    if n % 2:
        return None
    return signal.sosfiltfilt(signal.butter(n // 2, fl, "lowpass", fs=FS, output="sos"), y)


def best_delay_fit(x, y, max_lag):
    """gain and integer-sample delay (|lag| <= max_lag) minimising |x - g y(t - lag) - c|"""
    xc, yc = x - x.mean(), y - y.mean()
    best = (-1, 0, 0)
    for lag in range(-max_lag, max_lag + 1):
        ys = np.roll(yc, lag)
        g = (xc @ ys) / (ys @ ys)
        r2 = 1 - np.var(xc - g * ys) / np.var(xc)
        if r2 > best[0]:
            best = (r2, lag, g)
    return best


def parametric(recs):
    """recs: [(name, x, s_aligned)] -> table of model scores"""
    res = []
    for kind in ("minphase", "linphase"):
        for n in (2, 4, 6, 8):
            for fh in (0.5, 1.0, 2.0):
                for fl in np.arange(20, 62.5, 2.5):
                    tot, num, lags = 0.0, 0.0, []
                    for name, x, s in recs:
                        y = model_filter(s, fh, fl, n, kind)
                        if y is None:
                            break
                        r2, lag, g = best_delay_fit(x, y, 40)
                        num += r2 * np.var(x); tot += np.var(x); lags.append(lag)
                    else:
                        res.append(dict(kind=kind, order=n, fh=fh, fl=float(fl), explained=num / tot,
                                        lags_ms=[round(l * 4.0) for l in lags]))
    return sorted(res, key=lambda r: -r["explained"])
