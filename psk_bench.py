"""PicoScope validation of the EXPERIMENTAL watch_psk_tx x2 output (bench, watch NOT connected). Added 2026-11-06.

  python psk_bench.py sweep CAPTURE.csv --record data_watch_ecg/benchsweep_<label>_<time>.json [--zero ZERO.csv]
        one capture of a whole `psk_tx.py sweep`: the bursts are found, matched in order to the record, each analysed
  python psk_bench.py frame CAPTURE.csv --bps 16 [--mod dbpsk --shape rect --fc 16 --seconds 45] --label d16
                      [--zero ZERO.csv] [calibration]          one `psk_tx.py frame` capture

Per segment, on the captured divider output (crown node - GND):
  1. alignment from the KNOWN transmitted waveform (cross-correlation, then start + clock refined): independent of
     the receiver, so a receiver failure cannot hide a transmitter fault or vice versa
  2. waveform fidelity: amplitude, DC, clock, residual vs the exact firmware model, residual at symbol boundaries
  3. phase transitions: every symbol's complex amplitude by least squares (known pulse, no decisions) -> EVM and the
     error of every phase step
  4. receiver test: psk-x3 decode (x3-wl for DBPSK) of the capture resampled to the watch's 250 S/s: frames, bit errors (must be 0)
  5. MODEL only (not a measurement): the captured waveform through the watch channel fitted from the real recordings,
     decoded - an expectation before any worn test
Output: data_watch_ecg/bench_psk_<label>_<time>/ (report.json, one PNG per segment).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import optimize, signal, sparse
from scipy.sparse.linalg import lsqr

import ard_bench as AB
import channel_est as C
import psk_link as P
import psk_tx as PT

FS_FIT = 1000.0


def resample(t, y, fs_new):
    fs = 1 / np.median(np.diff(t))
    from fractions import Fraction
    fr = Fraction(fs_new / fs).limit_denominator(2000)
    z = signal.resample_poly(y, fr.numerator, fr.denominator)
    return t[0] + np.arange(len(z)) / fs_new, z


def symbol_ls(p, fw, t, y, lag, scale):
    """least-squares complex amplitude of every played symbol (known pulse, carrier, timing)"""
    T = p["sym_us"] / 1e6
    u_tx = t / scale + lag
    fc = int(round(p["fc_hz"] * 1000)) / 1000.0
    car = np.exp(2j * np.pi * fc * u_tx)
    n = fw["n_play"]; span = fw["span"]
    rows, cols, vals = [], [], []
    k0 = np.floor(u_tx / T).astype(int)
    for dk in range(-(span // 2 + 1), span // 2 + 2):
        k = k0 + dk
        ok = (k >= 0) & (k < n)
        w = P.pulse_at(u_tx / T - k - 0.5, span, fw["tq"]) * ok
        nz = np.flatnonzero(w)
        g = w[nz] * car[nz]
        rows += [nz, nz]; cols += [2 * k[nz], 2 * k[nz] + 1]; vals += [g.real, -g.imag]
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(t), 2 * n))
    A = sparse.hstack([A, sparse.csr_matrix(np.ones((len(t), 1)))]).tocsr()
    sol = lsqr(A, y, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    c = sol[0:2 * n:2] + 1j * sol[1:2 * n:2]
    seen = np.zeros(n, bool)
    seen[np.unique(np.concatenate(cols)) // 2] = True
    return c, seen


def align(p, fw, t1, y1):
    """(lag, ppm) of the capture against the known waveform: FFT cross-correlation at the power-line clock"""
    t250, y250 = resample(t1, y1 - np.median(y1), 250.0)
    sc0, _ = P.coarse_clock(y250, 250.0, p)
    ppm0 = (sc0 - 1) * 1e6
    fs1 = 1 / np.median(np.diff(t1))
    tm = np.arange(-2.0, p["total_s"] + 2.0, 1 / fs1)
    m = PT.model(p, tm, fw)
    xc = signal.fftconvolve(y1 - np.mean(y1), m[::-1], mode="full")
    k = int(np.argmax(np.abs(xc)))
    shift = (k - (len(m) - 1)) / fs1                      # capture time of TX time tm[0]
    return tm[0] - shift, ppm0


def fit_model(p, fw, t1, y1, lag0, ppm0):
    def resid(q):
        lag, ppm = q
        mdl = PT.model(p, t1 / (1 + ppm * 1e-6) + lag, fw)
        A = np.vstack([mdl, np.ones_like(mdl)]).T
        coef, *_ = np.linalg.lstsq(A, y1, rcond=None)
        return y1 - A @ coef, coef
    T = p["sym_us"] / 1e6
    best = min((np.var(resid((lag0 + d, ppm0))[0]), lag0 + d) for d in np.arange(-T, T + 1e-9, T / 8))
    o = optimize.minimize(lambda q: np.var(resid(q)[0]), [best[1], ppm0], method="Nelder-Mead",
                          options=dict(xatol=1e-7, fatol=1e-14, maxiter=600))
    r, (g, c0) = resid(o.x)
    return o.x[0], o.x[1], r, g, c0


def segment(t, y, z, p, exp, out, name):
    """analyse one transmission (t, y: capture slice in s, mV). Returns the report dict."""
    fw = PT.fw_payload(p)
    ck = AB.Checks()
    q = max(1, int(round((1 / np.median(np.diff(t))) / FS_FIT)))    # pick samples, NO filter (keeps abrupt steps)
    t1, y1 = t[::q] - t[0], y[::q]
    fs1 = 1 / np.median(np.diff(t1))
    T = p["sym_us"] / 1e6
    lag0, ppm0 = align(p, fw, t1, y1)
    lag, ppm, r, g, c0 = fit_model(p, fw, t1, y1, lag0, ppm0)
    sc = 1 + ppm * 1e-6
    u_tx = t1 / sc + lag
    on = np.abs(PT.model(p, u_tx, fw)) > 1e-9
    amp_pp = 2 * abs(g)
    nrmse = np.std(r[on]) / (abs(g) / np.sqrt(2))
    pos = np.mod(u_tx / T, 1.0)
    edge = on & ((pos < 0.04) | (pos > 0.96))
    mid = on & (np.abs(pos - 0.5) < 0.2)
    edge_ratio = np.std(r[edge]) / max(np.std(r[mid]), 1e-12) if edge.sum() > 10 and mid.sum() > 10 else float("nan")
    floor = (z["rms_mv"] / (abs(g) / np.sqrt(2))) if z else None
    lim = max(0.05, 1.5 * floor) if floor else 0.05
    ck.add("amplitude vs calculated", abs(amp_pp / exp["amp_mvpp"] - 1) <= 0.10, f"{amp_pp:.4f} mVpp measured", "+-10 % of calc")
    ck.add("peak <= 10 mVpp", amp_pp <= 10, f"{amp_pp:.4f} mVpp", "<= 10 mVpp")
    if z:
        ck.add("DC vs calculated (zero-corrected)", abs(c0 - exp["dc_mv"]) <= 0.5, f"{c0:+.4f} mV (calc {exp['dc_mv']:.3f})", "+-0.5 mV")
    ck.add("board clock", abs(ppm) <= 1000, f"{ppm:+.0f} ppm", "|ppm| <= 1000", warn=True)
    ck.add("waveform residual vs model", nrmse <= lim, f"{nrmse:.2%}" + (f" (scope noise {floor:.2%})" if floor else ""),
           f"<= {lim:.1%}", warn=not z)
    ck.add("residual at symbol boundaries", not (edge_ratio > 1.5), f"{edge_ratio:.2f} x mid-symbol", "<= 1.5")
    cs, seen = symbol_ls(p, fw, t1, y1, lag, sc)
    sent = fw["aq"]
    use = seen & (np.abs(sent) > 0)
    use[:2] = False
    gain = np.sum(cs[use] * np.conj(sent[use])) / np.sum(np.abs(sent[use]) ** 2)
    evm_db = float(10 * np.log10(np.mean(np.abs(cs[use] / gain - sent[use]) ** 2) / np.mean(np.abs(sent[use]) ** 2)))
    steps, err = None, None
    if p["modulation"] in ("dbpsk", "dqpsk"):
        k = np.flatnonzero(use[1:] & use[:-1]) + 1
        got = np.angle(cs[k] * np.conj(cs[k - 1])); want = np.angle(sent[k] * np.conj(sent[k - 1]))
        err = np.degrees(np.angle(np.exp(1j * (got - want))))
        steps = dict(n=int(len(k)), flips=int(np.sum(np.abs(want) > 1)), rms_deg=float(np.sqrt(np.mean(err ** 2))),
                     max_deg=float(np.max(np.abs(err))))
        ck.add("phase-step error (max)", steps["max_deg"] <= 10, f"{steps['max_deg']:.2f} deg ({steps['n']} steps)", "<= 10 deg")
    ck.add("constellation error (EVM)", evm_db <= -25, f"{evm_db:.1f} dB", "<= -25 dB")
    t250, y250 = resample(t1, y1 - np.median(y1), 250.0)
    rep = P.decode_wl(y250, t250 - t250[0], p) if p["modulation"] == "dbpsk" else P.decode(y250, t250 - t250[0], p, eq=True)
    v = [f for f in rep["frames"] if f["valid"]]
    nb = sum(f["bits"] for f in v); ne = sum(f["errors"] for f in v)
    ck.add("frames recovered (receiver)", len(v) == len(rep["frames"]) and len(v) > 0, f"{len(v)}/{len(rep['frames'])}", "all")
    ck.add("bit errors (receiver, no watch)", bool(nb) and ne == 0, f"{ne}/{nb}" if nb else "no bits", "0")
    kind, n_, fh, fl = ("linphase", 4, 1.0, 27.5)
    ym = C.model_filter(y250, fh, fl, n_, kind) * 0.9
    rm = P.decode_wl(ym, t250 - t250[0], p) if p["modulation"] == "dbpsk" else P.decode(ym, t250 - t250[0], p, eq=True)
    vm = [f for f in rm["frames"] if f["valid"]]
    model_note = (f"MODEL only (fitted watch channel, no ECG/noise): {len(vm)}/{len(rm['frames'])} frames, "
                  + (f"{sum(f['errors'] for f in vm)}/{sum(f['bits'] for f in vm)} errors" if vm else "no frames"))
    ok = ck.show()
    print(f"  {model_note}\n  -> {'READY' if ok else 'NOT READY'}")
    res = dict(name=name, plan=p, fit=dict(amp_mvpp=amp_pp, dc_mv=c0, clock_ppm=ppm, lag_s=lag, nrmse=nrmse,
                                           boundary_ratio=edge_ratio), evm_db=evm_db, phase_steps=steps,
               decode={k: rep.get(k) for k in ("decoder", "clock_ppm", "sync_score", "frames_valid", "frames_complete",
                                               "bits", "errors", "symbol_snr_db", "reason")},
               model_through_watch=model_note, checks=ck.rows, ready=ok)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        i0 = np.flatnonzero(on)[0] + int(fs1 * p["lead_symbols"] * T)
        sl = slice(i0, i0 + int(fs1 * 12 * T))
        ax[0].plot(t1[sl], y1[sl], color="#1f5fbf", lw=1, label="capture")
        ax[0].plot(t1[sl], (PT.model(p, u_tx, fw) * g + c0)[sl], color="#e66100", lw=0.8, ls="--", label="firmware model")
        ax[0].legend(fontsize=8); ax[0].set_xlabel("s"); ax[0].set_ylabel("mV"); ax[0].set_title("12 symbols")
        cc = cs[use] / gain
        ax[1].plot(cc.real, cc.imag, ".", ms=3, color="#1f5fbf"); ax[1].set_aspect("equal")
        ax[1].set_title(f"symbols from the capture (EVM {evm_db:.1f} dB)")
        if err is not None:
            ax[2].hist(err, bins=60, color="#1f5fbf"); ax[2].set_xlabel("phase-step error [deg]")
        fig.suptitle(f"{name}: {p['modulation']} {p['bps']:g} bps fc {p['fc_hz']:g} Hz {p['shape']} (bench, no watch)")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"{name}.png"), dpi=100); plt.close(fig)
    except Exception as ex:
        print("  plot skipped:", ex)
    return res


def bursts(t, y, n_expected):
    """active regions (transmission) separated by >= 1 s of mid-scale silence"""
    fs = 1 / np.median(np.diff(t))
    x = y - np.median(y)
    w = max(1, int(0.1 * fs))
    env = np.sqrt(np.convolve(x ** 2, np.ones(w) / w, mode="same"))
    act = env > 0.25 * np.percentile(env, 95)
    gap = max(1, int(1.0 * fs))
    regions, i, n = [], 0, len(act)
    while i < n:
        if act[i]:
            j, last = i, i
            while j < n and (act[j] or j - last < gap):
                if act[j]:
                    last = j
                j += 1
            regions.append((i, last))
            i = j
        else:
            i += 1
    regions = [r for r in regions if (r[1] - r[0]) / fs > 1.0]
    if len(regions) != n_expected:
        raise SystemExit(f"found {len(regions)} transmission bursts, the record has {n_expected}: check the capture "
                         "covers the whole sweep (and nothing else was transmitted)")
    pad = int(0.5 * fs)
    return [(max(0, a - pad), min(n, b + pad)) for a, b in regions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("sweep", "frame")); ap.add_argument("capture")
    ap.add_argument("--record"); ap.add_argument("--label"); ap.add_argument("--zero")
    ap.add_argument("--mod", choices=sorted(P.MODS), default="dbpsk"); ap.add_argument("--bps", type=float)
    ap.add_argument("--shape", choices=("rect", "rc", "rrc"), default="rect"); ap.add_argument("--fc", type=float, default=16.0)
    ap.add_argument("--seconds", type=float, default=45.0)
    for k in ("vfull", "r1", "r2", "amp_mvpp"):
        ap.add_argument("--" + k.replace("_", "-"), type=float)
    a = ap.parse_args()
    t, y, info, z = AB.load(a.capture, a.zero)
    if a.mode == "sweep":
        if not a.record:
            raise SystemExit("sweep needs --record (printed by psk_tx.py sweep)")
        rec = json.load(open(a.record))
        for k in ("vfull", "r1", "r2", "amp_mvpp"):
            if getattr(a, k) is None and k not in rec.get("nominal", []):
                setattr(a, k, rec["cal"][k])
        plans = [s["plan"] for s in rec["segments"]]
        label = a.label or rec["label"]
        cuts = bursts(t, y, len(plans))
    else:
        if a.bps is None or not a.label:
            raise SystemExit("frame needs --bps and --label")
        plans = [P.plan(a.mod, a.bps, fc=a.fc, shape=a.shape, seconds=a.seconds)]
        label, cuts = a.label, [(0, len(t))]
    exp = AB.expected(a)
    out = os.path.join(AB.B.DATADIR, f"bench_psk_{label}_{AB.stamp()}")
    os.makedirs(out)
    print(f"capture {os.path.basename(a.capture)}: {len(t)} samples, {1/np.median(np.diff(t)):.0f} S/s, "
          f"{t[-1]-t[0]:.1f} s; calculated {exp['amp_mvpp']:g} mVpp, DC {exp['dc_mv']:.3f} mV "
          f"(nominal: {', '.join(exp['nominal']) or 'none'})")
    res = []
    for k, ((i0, i1), p) in enumerate(zip(cuts, plans)):
        name = f"s{k + 1}_{p['modulation']}{p['bps']:g}_fc{p['fc_hz']:g}_{p['shape']}"
        print(f"\n[{name}] {p['bps']:g} bps = {p['baud']:g} baud, {p['cycles_per_symbol']:.2f} carrier cycles/symbol")
        res.append(segment(t[i0:i1], y[i0:i1], z, p, exp, out, name))
    print("\nsummary (bench, watch NOT connected; 'measured' = this scope capture of the divider output)")
    print(f"{'segment':34s} {'measured mVpp':>13s} {'clock ppm':>9s} {'EVM dB':>7s} {'max step':>9s} {'frames':>7s} {'errors':>9s}  result")
    for r in res:
        d = r["decode"]
        print(f"{r['name']:34s} {r['fit']['amp_mvpp']:13.4f} {r['fit']['clock_ppm']:+9.0f} {r['evm_db']:7.1f} "
              f"{(r['phase_steps'] or {}).get('max_deg', float('nan')):8.2f}° {d.get('frames_valid') or 0:>3}/{d.get('frames_complete') or 0:<3} "
              f"{(str(d.get('errors')) + '/' + str(d.get('bits'))) if d.get('bits') else 'no bits':>9s}  "
              f"{'READY' if r['ready'] else 'NOT READY'}")
    allok = all(r["ready"] for r in res)
    json.dump(dict(capture=os.path.abspath(a.capture), record=a.record, zero=z, info=info, calculated=exp,
                   decoder=P.VERSION, firmware=PT.FW_ID, segments=res, ready=allok),
              open(os.path.join(out, "report.json"), "w"), indent=1, default=float)
    print(("ALL READY" if allok else "NOT ALL READY") + f"  -> {out}")


if __name__ == "__main__":
    main()
