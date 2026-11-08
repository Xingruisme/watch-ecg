#!/usr/bin/env python3
"""
Low-level TeensyTRX output for the fixed-tone watch test (README section 11): size the attenuator, then verify it from
PicoScope CSV exports before the watch is connected. No transmission, no decoder.

   python att_check.py design --source-vpp 892.2 --source-dc 385.1          # R2 for 2 / 5 / 10 mVpp, R1 = 2 x 160 kOhm
   python att_check.py check --label a1 teensy.csv --zero zero.csv --ref gen_2mV.csv --loaded teensy_1k.csv --rl 1e3
   python att_check.py check --label w1 watch.pdf --ref watch_gen.pdf        # the tone in a watch ECG export

check fits DC + the tone (+ its 2nd/3rd harmonics and 50/60 Hz) by least squares over the whole capture, so Vpp is the
tone's own amplitude, not the scope's max-min (at 2 mVpp that is mostly noise). Every value it prints is from the
capture; the design numbers are calculations and are labelled so.
Writes data_watch_ecg/attcheck_<label>_<stamp>/ (inputs copied in, nothing overwritten). Bench data, not BER data.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecg_ber as B                                         # divider(), DATADIR, stamp() - unchanged
import ecg_io as IO

E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5,
       8.2, 9.1]
E96 = [round(10 ** (i / 96), 2) for i in range(96)]
WATCH_ZIN_MIN = 1e6                 # assumed lower bound of the watch ECG input impedance (not measured)
UNIT = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "μs": 1e-6, "ns": 1e-9,
        "V": 1e3, "mV": 1.0, "uV": 1e-3, "µV": 1e-3, "μV": 1e-3}
WATCH_EXT = (".pdf", ".npz", ".json")
MAX_FIT_SAMPLES = 500_000
WATCH_BAND = 0.05                   # tone search +-5 % in a watch export: ~9 noise bins in 30 s, 12 dB -> < 0.1 % false alarms


# ================================================================ design (calculated)
def series(ser, lo=1.0, hi=1e7):
    return [v * 10 ** d for d in range(0, 8) for v in ser if lo <= v * 10 ** d <= hi]


def nearest(x, ser):
    return min(series(ser), key=lambda v: abs(v - x))


def pair(x):
    """two E24 resistors in series closest to x"""
    best = None
    for a in series(E24):
        if a > x:
            break
        b = nearest(x - a, E24) if x - a >= 1 else 0.0
        if best is None or abs(a + b - x) < abs(sum(best) - x):
            best = (a, b)
    return best


def level(vs_mv, r1, r2, rs, c, f):
    """mVpp at the R1/R2 junction, open circuit: divider x the C1 high-pass"""
    mv = vs_mv * r2 / (r1 + r2 + rs)
    if c:
        mv /= math.sqrt(1 + (1 / (2 * math.pi * f * c * (r1 + r2 + rs))) ** 2)
    return mv


def design(a):
    vs, rs, r1, c, f = a.source_vpp, a.rs, a.r1, a.c1, a.f
    print(f"source {vs:g} mVpp, DC {a.source_dc:g} mV (as measured), Rs {rs:g} Ohm, R1 {r1:g} Ohm, "
          f"C1 {c * 1e6:g} uF, tone {f:g} Hz - all numbers below are CALCULATED")
    fc = 1 / (2 * math.pi * c * (r1 + rs)) if c else 0                 # R2 << R1
    print(f"C1 high-pass {fc:.3f} Hz ({100 * (1 - 1 / math.sqrt(1 + (fc / f) ** 2)):.2f} % loss at {f:g} Hz); "
          f"settles in 5 tau = {5 * c * (r1 + rs):.0f} s after power-up or reconnecting" if c else "no C1")
    print(f"\n{'target':>7s} {'R2 exact':>9s} {'E96':>16s} {'E24':>16s} {'E24 pair':>24s} {'DC no C1':>9s} {'Rout':>6s}")
    rows = []
    for m in a.targets:
        k = math.sqrt(1 + (fc / f) ** 2) if c else 1.0
        r2 = m * k * (r1 + rs) / (vs - m * k)
        e96, e24, p = nearest(r2, E96), nearest(r2, E24), pair(r2)
        lv = {n: level(vs, r1, v, rs, c, f) for n, v in (("e96", e96), ("e24", e24), ("pair", sum(p)))}
        dc0 = a.source_dc * e96 / (r1 + e96 + rs)
        rout = e96 * (r1 + rs) / (r1 + rs + e96)
        rows.append(dict(target_mvpp=m, r2_exact_ohm=r2, r2_e96_ohm=e96, mvpp_e96=lv["e96"], r2_e24_ohm=e24,
                         mvpp_e24=lv["e24"], r2_pair_ohm=list(p), mvpp_pair=lv["pair"], dc_without_c1_mv=dc0,
                         rout_ohm=rout, **B.divider(vs / 1e3, r1, e96, rs)))
        print(f"{m:5.3g} mV {r2:7.1f} Ohm {e96:6.0f} -> {lv['e96']:5.3f} mV {e24:6.0f} -> {lv['e24']:5.3f} mV "
              f"{p[0]:6.0f} + {p[1]:4.3g} -> {lv['pair']:5.3f} mV {dc0:7.3f} mV {rout:5.0f}")
    print(f"\nRout ~ R2 (the generator's is 50 Ohm). A watch input of >= {WATCH_ZIN_MIN / 1e6:g} MOhm (assumed, not measured) "
          f"loads it by < {100 * rows[0]['rout_ohm'] / WATCH_ZIN_MIN:.2f} %.")
    print(f"Peak current into the watch at {rows[0]['target_mvpp']:g} mVpp: {rows[0]['normal_current_ua_peak']:.2f} uA; "
          f"VOUT stuck at 3.3 V with C1 shorted: {rows[0]['fault_dc_current_ua']:.1f} uA.")
    print("Use 1 % resistors and measure R1 and R2 with a DMM; then verify the result with 'check'.")
    return rows


# ================================================================ captures
def load_scope(path, channel=None):
    """PicoScope 6/7 CSV/TXT export ('Time,Channel A,...' / '(ms),(mV)' / data). -> t s, y mV, info"""
    rows = open(path, encoding="utf-8-sig", errors="replace").read().splitlines()
    sep = "\t" if "\t" in rows[0] else (";" if ";" in rows[0] else ",")
    head = [c.strip() for c in rows[0].split(sep)]
    units = [c.strip().strip("()") for c in rows[1].split(sep)]
    if not head or not head[0].lower().startswith("time") or not all(u in UNIT for u in units[:2]):
        raise SystemExit(f"{path}: not a PicoScope CSV export (expected 'Time,Channel ...' then '(unit),(unit)')")
    col = 1 if channel is None else next(i for i, h in enumerate(head) if h.lower().endswith(channel.lower()))
    ts, ys = UNIT[units[0]], UNIT[units[col]]
    t, y, over = [], [], 0
    for r in rows[2:]:
        c = r.split(sep)
        if len(c) <= col or not c[0].strip():
            continue
        if sep == ";":
            c = [v.replace(",", ".") for v in c]
        try:
            tv = float(c[0])
        except ValueError:
            continue
        s = c[col].strip()
        if "∞" in s or s.lower() in ("inf", "-inf", "+inf", "nan", ""):
            over += 1
            continue
        t.append(tv * ts); y.append(float(s) * ys)
    t, y = np.array(t), np.array(y)
    return t, y, dict(source="picoscope_csv", channel=head[col], units=[units[0], units[col]], overrange=over)


def load_any(path, channel=None):
    if path.lower().endswith(WATCH_EXT):
        e = IO.load(path)
        return np.asarray(e.t, float), np.asarray(e.mv, float), dict(source="watch_ecg", overrange=0,
                                                                    fs_nominal=float(e.fs))
    return load_scope(path, channel)


def fit(t, y, f_hint, fixed=False, band=0.2, mains=(50.0, 60.0)):
    """least squares DC + tone (+2f, 3f, mains), tone searched within f_hint +- band. fixed=True: not searched"""
    from scipy.optimize import minimize_scalar
    t = t - t[0]
    n0, raw_ptp, u = len(y), float(y.max() - y.min()), np.unique(y)
    if len(y) > MAX_FIT_SAMPLES:                                # block mean, keeps >> 60 Hz bandwidth
        k = int(math.ceil(len(y) / MAX_FIT_SAMPLES))
        m = len(y) // k * k
        t, y = t[:m].reshape(-1, k).mean(1), y[:m].reshape(-1, k).mean(1)
    dt = float(np.median(np.diff(t)))
    T, nyq = float(t[-1] - t[0] + dt), 0.5 / dt

    def cols(f):
        fs = [f] + [h * f for h in (2, 3) if h * f < 0.9 * nyq] + [m for m in mains if m < 0.9 * nyq
                                                                       and abs(m - f) > 2 / T and abs(m - 3 * f) > 2 / T
                                                                       and abs(m - 2 * f) > 2 / T]
        X = [np.ones_like(t)]
        for g in fs:
            X += [np.cos(2 * np.pi * g * t), np.sin(2 * np.pi * g * t)]
        return fs, np.array(X).T

    def solve(f):
        fs, X = cols(f)
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        return fs, X, b

    if not fixed:
        w = np.hanning(len(y))
        nf = 1 << int(math.ceil(math.log2(len(y) * 8)))
        sp = np.abs(np.fft.rfft((y - y.mean()) * w, nf))
        fr = np.fft.rfftfreq(nf, dt)
        sel = (fr > (1 - band) * f_hint) & (fr < (1 + band) * f_hint)
        f0 = float(fr[sel][np.argmax(sp[sel])])
        r = minimize_scalar(lambda f: -np.hypot(*solve(f)[2][1:3]), bounds=(f0 - 1 / T, f0 + 1 / T),
                            method="bounded", options=dict(xatol=1e-7))
        f_hint = float(r.x)
    fs, X, b = solve(f_hint)
    res = y - X @ b
    sig = float(res.std())
    amp = {g: float(np.hypot(b[1 + 2 * i], b[2 + 2 * i])) for i, g in enumerate(fs)}
    a1 = amp[fs[0]]
    return dict(f_hz=f_hint, vpp_mv=2 * a1, vpp_sigma_mv=2 * sig * math.sqrt(2 / len(y)),
                floor_vpp_mv=2 * sig * math.sqrt(math.pi / len(y)), dc_mv=float(b[0]), raw_ptp_mv=raw_ptp,
                resid_rms_mv=sig, h2_pct=100 * amp.get(2 * fs[0], float("nan")) / a1 if a1 else None,
                h3_pct=100 * amp.get(3 * fs[0], float("nan")) / a1 if a1 else None,
                mains_mvpp={f"{m:g}": 2 * amp[m] for m in mains if m in amp},
                n=n0, fs_hz=1 / dt, seconds=T, cycles=T * f_hint,
                step_mv=float(np.min(np.diff(u))) if len(u) > 1 else None, levels=int(len(u)),
                snr_db=20 * math.log10(2 * a1 / (2 * sig * math.sqrt(math.pi / len(y)))) if sig > 0 and a1 else None)


def describe(role, r, info):
    m = " ".join(f"{k} Hz {v:.3f}" for k, v in r["mains_mvpp"].items())
    h = f"H2 {r['h2_pct']:.1f} % H3 {r['h3_pct']:.1f} %" if (r["snr_db"] or 0) >= 20 else "H2/H3 -"
    print(f"  {role:7s} f {r['f_hz']:.4f} Hz | tone {r['vpp_mv']:.4f} +- {r['vpp_sigma_mv']:.4f} mVpp "
          f"(scope max-min {r['raw_ptp_mv']:.3f}) | DC {r['dc_mv']:+.4f} mV | noise {r['resid_rms_mv']:.4f} mVrms | "
          f"{h} | mains mVpp {m or '-'}")
    print(f"  {'':7s} {r['n']} samples, {r['fs_hz']:.0f} S/s, {r['seconds']:.1f} s = {r['cycles']:.0f} cycles, "
          f"{r['levels']} levels (step {r['step_mv'] or 0:.4f} mV), overrange {info['overrange']}, "
          f"SNR {r['snr_db'] or float('nan'):.1f} dB over the noise floor")


def check(a):
    caps, out = {}, os.path.join(a.outdir, f"attcheck_{a.label}_{B.stamp()}")
    os.makedirs(out)
    inputs = {}
    for role, p in (("dut", a.capture), ("zero", a.zero), ("ref", a.ref), ("loaded", a.loaded)):
        if not p:
            continue
        dst = os.path.join(out, f"{role}_{os.path.basename(p)}")
        shutil.copy2(p, dst)
        inputs[role] = dict(path=os.path.abspath(p), copy=os.path.basename(dst),
                            sha256=hashlib.sha256(open(p, "rb").read()).hexdigest())
        caps[role] = load_any(p, a.channel)
    watch = caps["dut"][2]["source"] == "watch_ecg"
    print(f"{'watch ECG export' if watch else 'scope capture'}: {a.capture}")
    fits = {}
    band = WATCH_BAND if watch else 0.2                      # watch clock is within 0.5 %; a scope tone may be anywhere
    fits["dut"] = fit(caps["dut"][0], caps["dut"][1], a.f, band=band)
    if "ref" in caps:
        fits["ref"] = fit(caps["ref"][0], caps["ref"][1], a.f, band=band)
    if "zero" in caps:
        fits["zero"] = fit(caps["zero"][0], caps["zero"][1], fits["dut"]["f_hz"], fixed=True)
    if "loaded" in caps:
        fits["loaded"] = fit(caps["loaded"][0], caps["loaded"][1], fits["dut"]["f_hz"], fixed=True)
    for role, r in fits.items():
        describe(role, r, caps[role][2])

    d, chk = fits["dut"], []

    def add(name, ok, value, limit, note="", warn=False):
        st = "PASS" if ok else ("WARN" if warn else "FAIL")
        chk.append(dict(check=name, status=st, value=value, limit=limit, note=note))

    zdc = fits["zero"]["dc_mv"] if "zero" in fits else 0.0
    dc = d["dc_mv"] - zdc
    if "ref" in fits:
        r = fits["ref"]
        tgt_v, tgt_dc, tgt_src = r["vpp_mv"], r["dc_mv"] - zdc, "generator capture (--ref)"
    else:
        tgt_v, tgt_dc, tgt_src = a.target_mvpp, 0.0, "--target-mvpp"
    ftol = a.ftol if not watch else max(a.ftol, 0.02)       # watch clock: allow 2 %
    add("frequency", abs(d["f_hz"] / a.f - 1) <= ftol, f"{d['f_hz']:.4f} Hz", f"{a.f:g} Hz +- {100 * ftol:g} %")
    if watch:
        add("tone detected", (d["snr_db"] or 0) >= a.detect_db, f"{d['snr_db']:.1f} dB", f">= {a.detect_db:g} dB")
        if "ref" in fits:
            add("level vs generator recording", abs(d["vpp_mv"] / tgt_v - 1) <= 0.25,
                f"{d['vpp_mv']:.3f} vs {tgt_v:.3f} mVpp (ECG units)", "+- 25 %", warn=True)
    else:
        add("amplitude", abs(d["vpp_mv"] / tgt_v - 1) <= a.vtol, f"{d['vpp_mv']:.4f} mVpp",
            f"{tgt_v:.4f} mVpp +- {100 * a.vtol:g} % ({tgt_src})")
        add("DC offset", abs(dc - tgt_dc) <= a.dc_tol_mv, f"{dc:+.4f} mV" + (" (zero-corrected)" if "zero" in fits else
                                                                         " (NOT zero-corrected: add --zero)"),
            f"{tgt_dc:+.4f} mV +- {a.dc_tol_mv:g} mV ({tgt_src})")
        add("no clipping", caps["dut"][2]["overrange"] == 0, caps["dut"][2]["overrange"], 0,
            "overrange samples: use a larger range")
        add("capture length", d["cycles"] >= 10, f"{d['cycles']:.0f} cycles", ">= 10", warn=True)
        add("scope resolution", d["step_mv"] is not None and d["step_mv"] <= d["vpp_mv"] / 8,
            f"step {d['step_mv'] or 0:.4f} mV", f"<= Vpp/8 = {d['vpp_mv'] / 8:.4f} mV",
            "smallest range, x1 probe, resolution enhancement / 12+ bit mode", warn=True)
        add("fit precision", d["vpp_sigma_mv"] <= 0.03 * d["vpp_mv"], f"+- {d['vpp_sigma_mv']:.4f} mVpp",
            "<= 3 % of Vpp", "longer capture or more resolution", warn=True)
        mx = max(d["mains_mvpp"].values(), default=0.0)
        mref = max(fits["ref"]["mains_mvpp"].values(), default=0.0) if "ref" in fits else None
        add("mains pickup", mx <= max(0.5 * d["vpp_mv"], 2 * (mref or 0)), f"{mx:.4f} mVpp",
            f"<= {max(0.5 * d['vpp_mv'], 2 * (mref or 0)):.4f} mVpp" + (" (2 x the generator's)" if mref else ""),
            "shorten/twist the leads, laptop on battery", warn=True)
        if "zero" in fits:
            z = fits["zero"]["vpp_mv"]
            add("ground (tone with tip on the ground node)", z <= 0.02 * d["vpp_mv"], f"{z:.4f} mVpp",
                f"<= 2 % of {d['vpp_mv']:.4f}", "clip and tip both on W-; a tone here means the ground lead carries it")
        if "loaded" in fits and a.rl:
            vl = fits["loaded"]["vpp_mv"]
            rpar = a.r2 * (a.r1 + a.rs) / (a.r1 + a.rs + a.r2) if a.r1 and a.r2 else None
            if a.c_out:                         # cap in series with the output (TeensyTRX Ra/Rb -> 10 uF -> E+)
                zs = rpar + 1 / (2j * math.pi * d["f_hz"] * a.c_out) if rpar else None
                pred, ratio = (abs(a.rl / (a.rl + zs)) if zs is not None else None), vl / d["vpp_mv"]
                add("output impedance (load step)", pred is None or abs(ratio / pred - 1) <= 0.15,
                    f"loaded/open {ratio:.3f}", f"{pred:.3f} +- 15 % (R2 || R1 + {a.c_out * 1e6:g} uF in series)"
                    if pred else "(give --r1 --r2 to compare)",
                    f"|Zs| {abs(zs):.0f} Ohm: a {WATCH_ZIN_MIN / 1e6:g} MOhm watch input would lower the level by "
                    f"{100 * abs(zs) / WATCH_ZIN_MIN:.2f} %" if zs is not None else "")
            else:
                rout = a.rl * (d["vpp_mv"] / vl - 1)
                add("output impedance", rpar is None or abs(rout / rpar - 1) <= 0.3, f"{rout:.0f} Ohm",
                    f"{rpar:.0f} Ohm +- 30 % (R2 || R1)" if rpar else "(give --r1 --r2 to compare)",
                    f"a {WATCH_ZIN_MIN / 1e6:g} MOhm watch input would lower the level by {100 * rout / WATCH_ZIN_MIN:.3f} %")
    print("\nchecks:")
    for c in chk:
        print(f"  {c['status']:4s}  {c['check']:44s} {str(c['value']):34s} limit {c['limit']}"
              + (f"  - {c['note']}" if c["note"] and c["status"] != "PASS" else ""))
    fails = [c["check"] for c in chk if c["status"] == "FAIL"]
    verdict = ("tone present in the watch recording" if watch and not fails else
               "tone NOT confirmed in the watch recording" if watch else
               "READY for the watch (all required checks pass)" if not fails else
               "NOT ready: " + ", ".join(fails))
    print(f"\n{verdict}")
    json.dump(dict(label=a.label, created=datetime.datetime.now().isoformat(timespec="seconds"),
                   kind="watch_recording" if watch else "scope_verification", measured=True,
                   note=a.note, setup=dict(f_hz=a.f, target_mvpp=a.target_mvpp, r1_ohm=a.r1, r2_ohm=a.r2, rs_ohm=a.rs,
                                           rl_ohm=a.rl, channel=a.channel),
                   inputs=inputs, info={k: v[2] for k, v in caps.items()}, fits=fits,
                   dc_zero_corrected_mv=None if watch else dc, checks=chk, verdict=verdict),
              open(os.path.join(out, "check.json"), "w"), indent=1, default=float)
    print(f"-> {os.path.relpath(out) if out.startswith(HERE) else out}")
    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("design", help="R2 for the target levels (calculated)")
    p.add_argument("--source-vpp", type=float, required=True, help="mVpp at the attenuator input (AD9833 VOUT), measured")
    p.add_argument("--source-dc", type=float, default=0.0, help="DC mean at the attenuator input, mV")
    p.add_argument("--rs", type=float, default=B.AD9833_ROUT, help="source resistance (AD9833 VOUT 200 Ohm)")
    p.add_argument("--r1", type=float, default=320e3, help="series resistor (2 x 160 kOhm, README section 3)")
    p.add_argument("--c1", type=float, default=10e-6, help="series DC-blocking film capacitor, F (0 = none)")
    p.add_argument("--f", type=float, default=5.0)
    p.add_argument("--targets", type=lambda s: [float(v) for v in s.split(",")], default=[2.0, 5.0, 10.0])
    p = sub.add_parser("check", help="verify PicoScope captures (or the tone in a watch export)")
    p.add_argument("capture", help="PicoScope CSV of the attenuator output (or a watch .pdf/.npz)")
    p.add_argument("--label", required=True)
    p.add_argument("--zero", help="capture with probe tip AND ground clip on W- (scope offset, ground check)")
    p.add_argument("--ref", help="the function generator at the same point, same load and scope settings")
    p.add_argument("--loaded", help="capture with --rl across the output (output impedance)")
    p.add_argument("--rl", type=float, help="test load resistor for --loaded, Ohm (measured; 1 kOhm suggested)")
    p.add_argument("--r1", type=float, default=320e3); p.add_argument("--r2", type=float)
    p.add_argument("--rs", type=float, default=B.AD9833_ROUT)
    p.add_argument("--c-out", type=float, default=0.0,
                   help="capacitor in series with the OUTPUT, F (TeensyTRX board: 10e-6 between the Ra/Rb tap and E+)")
    p.add_argument("--f", type=float, default=5.0); p.add_argument("--target-mvpp", type=float, default=2.0)
    p.add_argument("--vtol", type=float, default=0.10); p.add_argument("--ftol", type=float, default=0.01)
    p.add_argument("--dc-tol-mv", type=float, default=0.2); p.add_argument("--detect-db", type=float, default=12.0)
    p.add_argument("--channel", help="scope channel letter if the CSV has several (default: the first)")
    p.add_argument("--note", default="")
    p.add_argument("--outdir", default=B.DATADIR, help="parent folder (default data_watch_ecg)")
    a = ap.parse_args()
    if a.cmd == "design":
        design(a)
    else:
        sys.exit(1 if check(a) else 0)


if __name__ == "__main__":
    main()
