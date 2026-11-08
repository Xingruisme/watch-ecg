#!/usr/bin/env python3
"""
PicoScope bench validation of the Arduino transmitter (watch_ber_tx.ino) at the divider output, BEFORE the watch.
Every number this prints comes from a PicoScope capture (MEASURED) or from the firmware settings (CALCULATED); the
tone-set simulation (data_watch_ecg/exploratory_tonesim_*) is not used here. README section 12 has the procedure.

   python ard_bench.py tone  CAPTURE.csv --hz 16 --label t16 [--zero ZERO.csv] [calibration]
        1. carrier frequency (-> board clock error, ppm) and amplitude      2. DC offset and waveform quality
   python ard_bench.py frame CAPTURE.csv --mod ook --bps 2 --toneset exp --label f1 [--zero ZERO.csv] [calibration]
        3. OOK / FSK switching transients      4. symbol timing and clock drift
        5. offline decoding of the recorded frames with the FROZEN ecg-v1 decoder (capture resampled to 250 S/s)
calibration (what the firmware was told; for the CALCULATED expectation): --vfull --r1 --r2 --amp-mvpp
Writes data_watch_ecg/bench_ard_<label>_<time>/ (report.json, plot, the capture). Exit code 1 if a check fails.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import att_check as AC                     # noqa: E402  (PicoScope CSV reader and sine fit)
import arduino_tx as AT                    # noqa: E402
import ecg_ber as B                        # noqa: E402
import ecg_link as L                       # noqa: E402

SCOPE_TB_PPM = 100                         # PicoScope 2204A timebase accuracy (datasheet: +-100 ppm)
EDGE_MS = 5.0                              # transient window each side of a bit boundary


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def load(path, zero=None):
    t, y, info = AC.load_any(path)
    z = None
    if zero:
        _, yz, _ = AC.load_any(zero)
        z = dict(file=os.path.abspath(zero), mean_mv=float(yz.mean()), rms_mv=float(yz.std()))
        y = y - z["mean_mv"]
    return t, y, info, z


def expected(a):
    st = dict(vfull=a.vfull or AT.DEFAULTS["vfull"], r1=a.r1 or AT.DEFAULTS["r1"], r2=a.r2 or AT.DEFAULTS["r2"],
              amp_mvpp=a.amp_mvpp or AT.DEFAULTS["amp_mvpp"])
    ratio = st["r2"] / (st["r1"] + st["r2"])
    return dict(st, ratio=ratio, dc_mv=st["vfull"] * AT.MID / 4095 * ratio * 1e3,
                nominal=[k for k in ("vfull", "r1", "r2") if getattr(a, k) is None])


def outdir(label):
    d = os.path.join(B.DATADIR, f"bench_ard_{label}_{stamp()}")
    os.makedirs(d)
    return d


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, value, limit, warn=False):
        self.rows.append(dict(check=name, result="PASS" if ok else ("WARN" if warn else "FAIL"), value=value, limit=limit))

    def show(self):
        for r in self.rows:
            print(f"  {r['result']:4s}  {r['check']:38s} {r['value']:34s} limit {r['limit']}")
        return not any(r["result"] == "FAIL" for r in self.rows)


# ================================================================ 1 + 2: steady tone
def tone(a):
    t, y, info, z = load(a.capture, a.zero)
    ex = expected(a)
    r = AC.fit(t, y, a.hz)
    ppm = (r["f_hz"] / a.hz - 1) * 1e6
    print(f"MEASURED (PicoScope) {os.path.basename(a.capture)}: {r['n']} samples, {r['fs_hz']:.0f} S/s, "
          f"{r['seconds']:.1f} s, step {r['step_mv'] or 0:.4f} mV" + ("" if z else "   (no --zero: DC not zero-corrected)"))
    print(f"  1. carrier   f {r['f_hz']:.5f} Hz (set {a.hz:g}) -> board clock {ppm:+.0f} ppm vs the scope "
          f"(scope timebase +-{SCOPE_TB_PPM} ppm) | tone {r['vpp_mv']:.4f} +- {r['vpp_sigma_mv']:.4f} mVpp")
    print(f"  2. DC {r['dc_mv']:+.4f} mV | noise {r['resid_rms_mv']:.4f} mVrms | H2 {r['h2_pct']:.2f} % H3 "
          f"{r['h3_pct']:.2f} % | SNR {r['snr_db'] or float('nan'):.1f} dB | mains "
          + " ".join(f"{k} Hz {v:.3f} mVpp" for k, v in r["mains_mvpp"].items()))
    print(f"  CALCULATED expectation: {ex['amp_mvpp']:.4f} mVpp, DC {ex['dc_mv']:.3f} mV (vfull {ex['vfull']:g} V, "
          f"R1 {ex['r1']:g}, R2 {ex['r2']:g}" + (f"; nominal, not measured: {', '.join(ex['nominal'])}" if ex["nominal"] else "") + ")")
    c = Checks()
    c.add("frequency (board clock)", abs(ppm) <= 5000, f"{ppm:+.0f} ppm", "|ppm| <= 5000 (decoder search)")
    c.add("amplitude vs calculated", abs(r["vpp_mv"] / ex["amp_mvpp"] - 1) <= 0.10, f"{r['vpp_mv']:.4f} mVpp",
          f"{ex['amp_mvpp']:.3f} mVpp +- 10 %")
    c.add("amplitude <= 10 mVpp", r["vpp_mv"] <= 10.0, f"{r['vpp_mv']:.4f} mVpp", "<= 10 mVpp")
    if z:
        c.add("DC vs calculated (zero-corrected)", abs(r["dc_mv"] - ex["dc_mv"]) <= 0.5, f"{r['dc_mv']:+.4f} mV",
              f"{ex['dc_mv']:.3f} +- 0.5 mV")
    else:
        c.add("DC (not zero-corrected)", False, f"{r['dc_mv']:+.4f} mV", "give --zero", warn=True)
    for h in ("h2_pct", "h3_pct"):
        c.add(f"distortion {h[:2].upper()}", (r[h] or 0) <= 3.0, f"{r[h]:.2f} %", "<= 3 %")
    c.add("capture >= 10 cycles", r["cycles"] >= 10, f"{r['cycles']:.0f}", ">= 10")
    c.add("fit precision", r["vpp_sigma_mv"] <= 0.03 * max(r["vpp_mv"], 1e-9), f"+-{r['vpp_sigma_mv']:.4f} mVpp",
          "<= 3 % of Vpp", warn=True)
    print("checks:")
    ok = c.show()
    d = outdir(a.label)
    shutil.copy2(a.capture, d)
    rep = dict(kind="tone", label=a.label, created=datetime.datetime.now().isoformat(timespec="seconds"),
               source="MEASURED: PicoScope capture at the divider output (no watch)", capture=os.path.abspath(a.capture),
               zero=z, set_hz=a.hz, fit=r, board_clock_ppm=ppm, scope_timebase_ppm=SCOPE_TB_PPM,
               calculated=ex, checks=c.rows, ok=ok,
               note="use --tx-ppm {:.0f} for the transmitter (ecg_ber.py / arduino_tx.py) if this capture was "
                    "taken with --tx-ppm 0".format(ppm))
    json.dump(rep, open(os.path.join(d, "report.json"), "w"), indent=1, default=float)
    print(("READY" if ok else "NOT ready") + f"  -> {os.path.relpath(d, HERE)}")
    print(f"  clock correction for later runs (if this tone was played with --tx-ppm 0): --tx-ppm {ppm:.0f}")
    return ok


# ================================================================ 3 + 4 + 5: frames
def model(t, s0, K, Tb, bits, f1, f0):
    """unit-amplitude waveform the firmware plays (phase restarted every bit): bit n starts at s0 + n Tb K"""
    u = (t - s0) / (Tb * K)
    n = np.floor(u).astype(np.int64)
    b = np.asarray(bits)[n % len(bits)]
    tau = (u - n) * Tb                                         # seconds since the bit started (transmitter time)
    s1 = np.where(b == 1, np.sin(2 * np.pi * f1 * tau), 0.0)
    s0_ = np.where(b == 0, np.sin(2 * np.pi * f0 * tau), 0.0) if f0 else np.zeros_like(t)
    return s1, s0_, n, b, tau


def lin_fit(y, s1, s0_):
    X = np.c_[np.ones_like(y), s1] if not np.any(s0_) else np.c_[np.ones_like(y), s1, s0_]
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    return c, y - X @ c


def frame(a):
    from scipy.optimize import minimize
    from scipy.signal import resample_poly
    t, y, info, z = load(a.capture, a.zero)
    ex = expected(a)
    plan = B.ber_plan_for(a.mod, a.bps, a.toneset, 90.0, "arduino")
    bits, truth, _ = L.frame(plan["bps"])
    nb, Tb, f1, f0 = len(bits), plan["bit_us"] / 1e6, plan["f1_hz"], plan["f0_hz"] or 0.0
    fs = 1 / float(np.median(np.diff(t)))
    print(f"MEASURED (PicoScope) {os.path.basename(a.capture)}: {len(t)} samples, {fs:.0f} S/s, {t[-1] - t[0]:.1f} s"
          f"; plan {plan['modulation'].upper()} {plan['bps']:g} bps, '1' {f1:g} Hz, '0' "
          + (f"{f0:g} Hz" if f0 else "off") + f", tone set {plan['toneset']}, frame {nb} bits = {plan['frame_s']:.2f} s")
    if t[-1] - t[0] < plan["frame_s"] * 1.5:
        sys.exit(f"capture too short: need >= {plan['frame_s'] * 1.5:.0f} s (two frames); 30 s recommended")

    # 5. offline decode, frozen decoder, at the watch's rate (polyphase anti-alias filter)
    fr = Fraction(250.0 / fs).limit_denominator(2000)
    x = resample_poly(y - y.mean(), fr.numerator, fr.denominator)
    tw = np.arange(len(x)) * (fr.denominator / fr.numerator) / fs        # 250 S/s on the scope's time base
    rep, tr = L.decode(x, tw, plan)
    fv = [f for f in rep["frames"] if f["valid"]]
    E, N = sum(f["errors"] for f in fv), sum(f["bits"] for f in fv)
    dec = dict(decoder=rep["decoder"], rules_sha=rep["rules_sha"], frames_complete=rep["frames_complete"],
               frames_valid=len(fv), bits=N, errors=E, ber=E / N if N else None, clock_ppm=rep["clock_ppm"],
               residual_ppm=rep["residual_ppm"], line_snr_db=rep["line_snr_db"], es_n0_db=rep["es_n0_db"],
               resampled_hz=fs * fr.numerator / fr.denominator, ratio=f"{fr.numerator}/{fr.denominator}")

    # 4. symbol timing: start from the decoder's lock, refine on the full-rate capture
    K0 = (1 + rep["clock_ppm"] * 1e-6) * (1 + rep["residual_ppm"] * 1e-6)
    s00 = t[0] + rep["tau_s"] * (1 + rep["clock_ppm"] * 1e-6)
    k = max(1, int(round(fs / 2000)))                         # fit at ~2 kS/s (block means)
    m = len(y) // k * k
    tf, yf = t[:m].reshape(-1, k).mean(1), y[:m].reshape(-1, k).mean(1)
    span = (tf >= s00) & (tf < s00 + rep["frames_complete"] * nb * Tb * K0) if rep["frames_complete"] else \
        np.ones_like(tf, bool)

    def cost(p):
        s1, s0_, *_ = model(tf[span], s00 + p[0] * 1e-3, K0 * (1 + p[1] * 1e-6), Tb, bits, f1, f0)
        return float(np.sum(lin_fit(yf[span], s1, s0_)[1] ** 2))
    best = min((minimize(cost, [dx, 0.0], method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-12, maxiter=400))
                for dx in (-2.0, 0.0, 2.0)), key=lambda r: r.fun)
    s0, K = s00 + best.x[0] * 1e-3, K0 * (1 + best.x[1] * 1e-6)
    tx_ppm = (1 / K - 1) * 1e6                                 # transmitter clock vs the scope (+ = fast)
    per = []                                                   # each complete frame on its own: start time
    nfr = int((tf[span][-1] - s0) // (nb * Tb * K)) if span.any() else 0
    for j in range(nfr):
        a0, b0 = s0 + j * nb * Tb * K, s0 + (j + 1) * nb * Tb * K
        sel = (tf >= a0) & (tf < b0)
        if sel.sum() < 100:
            continue
        g = lambda dx: float(np.sum(lin_fit(yf[sel], *model(tf[sel], s0 + dx * 1e-3, K, Tb, bits, f1, f0)[:2])[1] ** 2))
        from scipy.optimize import minimize_scalar
        rr = minimize_scalar(g, bounds=(-3, 3), method="bounded", options=dict(xatol=1e-4))
        per.append(dict(frame=j, start_s=a0, offset_us=float(rr.x * 1e3)))
    offs = np.array([p["offset_us"] for p in per])
    jitter = float(offs.std()) if len(offs) > 1 else None
    drift = float(np.polyfit([p["start_s"] for p in per], offs, 1)[0]) if len(offs) > 2 else None   # us per s = ppm

    # 2 + 3. levels, DC, transients on the full-rate data inside the fitted span
    sel = (t >= s0) & (t < s0 + nfr * nb * Tb * K) if nfr else (t >= s0)
    s1, s0_, n, b, tau = model(t[sel], s0, K, Tb, bits, f1, f0)
    c, res = lin_fit(y[sel], s1, s0_)
    amp1 = 2 * abs(c[1]); amp0 = 2 * abs(c[2]) if len(c) > 2 else None
    ba = np.asarray(bits)
    change = ((tau < EDGE_MS * 1e-3) & (b != ba[(n - 1) % nb])) | ((tau > Tb - EDGE_MS * 1e-3) & (b != ba[(n + 1) % nb]))
    mid = (tau > 0.25 * Tb) & (tau < 0.75 * Tb)
    rms_edge = float(res[change].std()) if change.any() else None
    rms_mid = float(res[mid].std())
    peak_edge = float(np.max(np.abs(res[change]))) if change.any() else None
    dc1 = float(np.mean((y[sel] - c[1] * s1 - (c[2] * s0_ if len(c) > 2 else 0))[b == 1]))
    dc0 = float(np.mean((y[sel] - c[1] * s1 - (c[2] * s0_ if len(c) > 2 else 0))[b == 0]))
    noise = z["rms_mv"] if z else None

    print(f"  5. offline decode (FROZEN {rep['decoder']} {rep['rules_sha']}, capture resampled to 250 S/s): "
          f"{len(fv)}/{rep['frames_complete']} frames, {E} errors / {N} bits"
          + (f" (BER {E / N:.2%})" if N else "") + f", line SNR {rep['line_snr_db']:.1f} dB, "
          f"Es/N0 {rep['es_n0_db'] if rep['es_n0_db'] is None else round(rep['es_n0_db'], 1)} dB")
    print(f"  4. timing: bit {Tb * K * 1e6:.1f} us (nominal {Tb * 1e6:.0f}) -> transmitter clock {tx_ppm:+.0f} ppm vs "
          f"the scope (+-{SCOPE_TB_PPM}); per-frame start jitter "
          + ("-" if jitter is None else f"{jitter:.1f} us rms") + ", drift within the capture "
          + ("-" if drift is None else f"{drift:+.1f} ppm") + f" over {len(per)} frames")
    print(f"  2. levels: '1' tone {amp1:.4f} mVpp" + (f", '0' tone {amp0:.4f} mVpp" if amp0 is not None else "")
          + f" | DC during '1' {dc1:+.4f} mV, during '0' {dc0:+.4f} mV (step {dc1 - dc0:+.4f} mV)"
          + ("" if z else "  [not zero-corrected]"))
    print(f"  3. transients: residual within {EDGE_MS:g} ms of a bit change {rms_edge if rms_edge is None else round(rms_edge, 4)} "
          f"mVrms (peak {peak_edge if peak_edge is None else round(peak_edge, 4)} mV) vs mid-bit {rms_mid:.4f} mVrms"
          + (f"; scope noise (zero capture) {noise:.4f} mVrms" if noise else ""))
    ch = Checks()
    ch.add("offline decode: frames", len(fv) >= 1 and len(fv) == rep["frames_complete"],
           f"{len(fv)}/{rep['frames_complete']}", "all complete frames valid")
    ch.add("offline decode: errors", N > 0 and E == 0, f"{E}/{N}", "0 errors")
    ch.add("transmitter clock", abs(tx_ppm) <= 1000, f"{tx_ppm:+.0f} ppm", "|ppm| <= 1000 (else set --tx-ppm)", warn=True)
    if jitter is not None:
        ch.add("frame start jitter", jitter <= 0.02 * Tb * 1e6, f"{jitter:.1f} us rms", f"<= 2 % of a bit ({0.02 * Tb * 1e6:.0f} us)")
    ch.add("'1' amplitude vs calculated", abs(amp1 / ex["amp_mvpp"] - 1) <= 0.10, f"{amp1:.4f} mVpp",
           f"{ex['amp_mvpp']:.3f} +- 10 %")
    if amp0 is not None:
        ch.add("FSK tones equal", abs(amp0 / amp1 - 1) <= 0.05, f"ratio {amp0 / amp1:.3f}", "+- 5 %")
    ch.add("DC step '1' vs '0'", abs(dc1 - dc0) <= 0.05, f"{dc1 - dc0:+.4f} mV", "<= 0.05 mV")
    if rms_edge is not None:
        ch.add("switching transients", rms_edge <= 1.5 * rms_mid + 0.01, f"{rms_edge:.4f} vs {rms_mid:.4f} mVrms",
               "edge rms <= 1.5 x mid-bit rms")
    print("checks:")
    ok = ch.show()
    d = outdir(a.label)
    shutil.copy2(a.capture, d)
    plot_frame(os.path.join(d, "frame.png"), t, y, s0, K, Tb, bits, f1, f0, c, plan, res, sel)
    rep_out = dict(kind="frame", label=a.label, created=datetime.datetime.now().isoformat(timespec="seconds"),
                   source="MEASURED: PicoScope capture at the divider output (no watch)", capture=os.path.abspath(a.capture),
                   zero=z, plan=plan, calculated=ex, offline_decode=dec,
                   timing=dict(bit_us=Tb * K * 1e6, nominal_bit_us=Tb * 1e6, tx_clock_ppm_vs_scope=tx_ppm,
                               scope_timebase_ppm=SCOPE_TB_PPM, frame_start_jitter_us=jitter,
                               drift_ppm_within_capture=drift, frames=per),
                   levels=dict(f1_vpp_mv=amp1, f0_vpp_mv=amp0, dc_during_1_mv=dc1, dc_during_0_mv=dc0,
                               zero_corrected=bool(z)),
                   transients=dict(window_ms=EDGE_MS, edge_rms_mv=rms_edge, edge_peak_mv=peak_edge, mid_rms_mv=rms_mid,
                                   scope_noise_rms_mv=noise),
                   checks=ch.rows, ok=ok)
    json.dump(rep_out, open(os.path.join(d, "report.json"), "w"), indent=1, default=float)
    print(("READY" if ok else "NOT ready") + f"  -> {os.path.relpath(d, HERE)}")
    return ok


def plot_frame(path, t, y, s0, K, Tb, bits, f1, f0, c, plan, res, sel):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nb = len(bits)
    fig, ax = plt.subplots(3, 1, figsize=(13, 9))
    w = (t >= s0) & (t < s0 + min(nb, 14) * Tb * K)
    s1, s0_, *_ = model(t[w], s0, K, Tb, bits, f1, f0)
    fit = c[0] + c[1] * s1 + (c[2] * s0_ if len(c) > 2 else 0)
    ax[0].plot(t[w] - s0, y[w], color="#999", lw=0.5, label="PicoScope (measured)")
    ax[0].plot(t[w] - s0, fit, color="#d62728", lw=1.0, label="fitted firmware waveform")
    for j in range(min(nb, 14) + 1):
        ax[0].axvline(j * Tb * K, color="#ccc", lw=0.6)
        if j < min(nb, 14):
            ax[0].text((j + 0.5) * Tb * K, ax[0].get_ylim()[1] if j else 0, str(bits[j]), ha="center", va="bottom", fontsize=8)
    ax[0].set_title(f"{plan['modulation'].upper()} {plan['bps']:g} bps, tone set {plan['toneset']}: first bits of a frame")
    ax[0].legend(loc="upper right", fontsize=8)
    tt = t[sel] - s0
    ax[1].plot(tt, res, color="#555", lw=0.4)
    ax[1].set_title("residual (measured - fitted) over the analysed frames"); ax[1].set_xlabel("s from the first frame start")
    # zoom on the first 0->1 and 1->0 changes
    ch = [j for j in range(1, nb) if bits[j] != bits[j - 1]][:2]
    for j in ch:
        tc = s0 + j * Tb * K
        z = (t >= tc - 0.06) & (t < tc + 0.06)
        s1, s0_, *_ = model(t[z], s0, K, Tb, bits, f1, f0)
        ax[2].plot((t[z] - tc) * 1e3, y[z], lw=0.6, label=f"{bits[j - 1]}->{bits[j]} measured")
        ax[2].plot((t[z] - tc) * 1e3, c[0] + c[1] * s1 + (c[2] * s0_ if len(c) > 2 else 0), "--", lw=0.8,
                   label=f"{bits[j - 1]}->{bits[j]} fitted")
    ax[2].axvline(0, color="#ccc"); ax[2].set_xlabel("ms from the bit change"); ax[2].legend(fontsize=8)
    for a_ in ax[:2]:
        a_.set_ylabel("mV")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("tone", "frame"):
        p = sub.add_parser(name)
        p.add_argument("capture"); p.add_argument("--label", required=True); p.add_argument("--zero")
        p.add_argument("--vfull", type=float); p.add_argument("--r1", type=float); p.add_argument("--r2", type=float)
        p.add_argument("--amp-mvpp", type=float)
        if name == "tone":
            p.add_argument("--hz", type=float, required=True)
        else:
            p.add_argument("--mod", choices=("ook", "fsk"), required=True)
            p.add_argument("--bps", type=float, required=True)
            p.add_argument("--toneset", choices=("baseline", "exp"), default="baseline")
    a = ap.parse_args()
    ok = tone(a) if a.cmd == "tone" else frame(a)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
