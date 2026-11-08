#!/usr/bin/env python3
"""
Bench isolation: why the Teensy's 16 Hz tone is not seen while the function generator works. NO PERSON in the circuit
(watch off the wrist, nobody touching the leads). Nothing here changes the ECG decoder or any earlier result.

Part 1 (automatic, A0 only)
  a) output off: A0 DC and noise
  b) sine ladder 16 Hz ... 160 kHz (cw): tone Vpp at A0 per frequency -> where the VOUT -> E+ -> A0 chain stops passing
  c) 16 Hz SQUARE: the step response at A0. A high-pass anywhere before A0 turns the flat tops into decays; the decay
     time constant gives its corner directly (fc = 1 / (2 pi tau))
  A0 sees the chain only up to A0's own input, so part 1 can show THAT low frequencies are lost before A0, not WHERE
  (E+ drive or A0's input network). Part 2 answers that.

Part 2 (--scope, operator readings)
  The script holds each tone while you read the scope and type the values; anything you skip is stored as
  'not measured'. Points: VOUT (AD9833 pin), E+ (board output wire), ATT (crown node = across R2, what the watch
  sees), and the function generator at the same crown-back point as the reference that works.

   ~/.venvs/galvanic/bin/python bench_isolate.py --label b1            # part 1
   ~/.venvs/galvanic/bin/python bench_isolate.py --label b1 --scope    # part 1 + 2
Results: data_watch_ecg/bench_<label>_<time>/ (bench.json, a0_captures.npz). README section 9 has the procedure.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecg_ber as B                                         # noqa: E402  (parse_mono, StentTX, ADC scale)

STX = B.STX
LADDER = [16, 50, 160, 500, 1600, 5000, 16000, 50000, 160000]
REF_BAND = (1600, 50000)                                     # the board's working range: the gain reference
DETECT_DB = 10.0


def capture_plan(f):
    if f <= 500:
        return 10000, 2.0
    if f <= 16000:
        return 100000, 0.5
    return 1000000, 0.2


def capture(t, sps, secs):
    """A0 via the 'mono' stream -> counts (the first 50 ms dropped)"""
    t.dev.reset_input_buffer()
    t.send(f"mono {sps}")
    need, buf, end = int(2 * sps * (secs + 0.06)), b"", time.time() + secs + 5
    while len(buf) < need and time.time() < end:
        buf += t.dev.read(1 << 16)
    t.send("x")
    time.sleep(0.3)
    t.dev.reset_input_buffer()
    y = B.parse_mono(buf[int(2 * sps * 0.05):])
    return y[:int(sps * secs)]


def tone(y_mv, sps, f):
    """Vpp of the line near f (Hann, zero-padded FFT, peak within +-max(0.5 Hz, 0.3 %)) and its SNR over the median"""
    v = y_mv - y_mv.mean()
    w = np.hanning(len(v))
    n = 8 * len(v)
    X = np.abs(np.fft.rfft(v * w, n))
    fr = np.fft.rfftfreq(n, 1 / sps)
    span = max(0.5, 0.003 * f)
    sel = (fr > f - span) & (fr < f + span)
    if not sel.any():
        return dict(vpp_mv=None, snr_db=None)
    k = np.flatnonzero(sel)[np.argmax(X[sel])]
    vpp = 2 * 2 * X[k] / w.sum()
    floor = np.median(X[(fr > 1) & ~((fr > f - 10 * span) & (fr < f + 10 * span))])
    return dict(vpp_mv=float(vpp), snr_db=float(20 * np.log10(X[k] / (floor + 1e-12))), peak_hz=float(fr[k]))


def step_response(y_mv, sps, f):
    """16 Hz square at A0: fold one period, compare the level just after an edge with the level just before the next
    edge. ratio = exp(-span / tau) for a first-order high-pass"""
    nb = 400
    ph = (np.arange(len(y_mv)) / sps * f) % 1.0
    idx = np.minimum((ph * nb).astype(int), nb - 1)
    wv = np.bincount(idx, y_mv, nb) / np.maximum(np.bincount(idx, minlength=nb), 1)
    d = wv - wv.mean()
    jump = np.diff(np.r_[wv, wv[0]])
    ir = int(np.argmax(jump)) + 1; ifl = int(np.argmin(jump)) + 1
    h = float(jump.max() - jump.min()) / 2                    # edge height
    res = dict(edge_mv=h, period_s=1 / f)
    noise = float(np.std(np.diff(wv)) )
    if h < 5 * noise or h < 2 * B.ADC_LSB_MV:
        return dict(res, verdict="no square edges at A0")
    a, b = (ir + 3) % nb, (ifl - 3) % nb                       # 3 bins inside the plateau
    span = ((b - a) % nb) / nb / f
    d0, d1 = d[a], d[b]
    res.update(after_edge_mv=float(d0), before_next_mv=float(d1), span_s=span)
    if d0 <= 0:
        return dict(res, verdict="edge shape unclear")
    r = d1 / d0
    if r >= 0.95:
        return dict(res, ratio=float(r), verdict=f"flat tops: no high-pass above {-np.log(0.95) / (2 * np.pi * span):.2f} Hz")
    if r <= 0.02:
        return dict(res, ratio=float(r), fc_hz_min=float(np.log(50) / (2 * np.pi * span)),
                    verdict=f"spikes only: high-pass corner above {np.log(50) / (2 * np.pi * span):.0f} Hz")
    tau = -span / np.log(r)
    return dict(res, ratio=float(r), tau_s=float(tau), fc_hz=float(1 / (2 * np.pi * tau)),
                verdict=f"decaying tops: high-pass, tau {tau * 1e3:.1f} ms, corner {1 / (2 * np.pi * tau):.1f} Hz")


def actual(reply):
    m = re.search(r"# cw ([0-9.]+)", reply)
    return float(m.group(1)) if m else None


def part1(t, raw):
    mv = lambda c: c * B.ADC_LSB_MV
    out = {}
    print(t.off())
    y = capture(t, 10000, 1.0); raw["off"] = y.astype(np.uint16)
    out["off"] = dict(dc_mv=float(mv(y).mean()), std_mv=float(mv(y).std()), ptp_mv=float(np.ptp(mv(y))))
    print(f"  output off: A0 DC {out['off']['dc_mv']:.0f} mV, noise {out['off']['std_mv']:.2f} mV rms")
    t.configure(wave="sine")
    lad = []
    print(f"  {'f [Hz]':>8s} {'AD9833 [Hz]':>12s} {'A0 Vpp [mV]':>12s} {'SNR [dB]':>9s} {'DC [mV]':>8s} {'clip':>5s}")
    for f in LADDER:
        sps, secs = capture_plan(f)
        fa = actual(t.tone(f)) or f
        time.sleep(0.3)
        y = capture(t, sps, secs); raw[f"sine_{f}"] = y.astype(np.uint16)
        r = dict(f_hz=f, ad9833_hz=fa, sps=sps, seconds=secs, **tone(mv(y), sps, fa),
                 dc_mv=float(mv(y).mean()), clip_frac=float(np.mean((y <= 0) | (y >= 4095))))
        r["detected"] = r["snr_db"] is not None and r["snr_db"] >= DETECT_DB
        lad.append(r)
        print(f"  {f:8g} {fa:12.3f} {r['vpp_mv']:12.2f} {r['snr_db']:9.1f} {r['dc_mv']:8.0f} {r['clip_frac']:5.0%}"
              + ("" if r["detected"] else "   (not detected)"))
    print(t.off())
    ref = [r["vpp_mv"] for r in lad if REF_BAND[0] <= r["f_hz"] <= REF_BAND[1] and r["detected"]]
    vref = float(np.median(ref)) if ref else None
    for r in lad:
        r["gain_vs_ref"] = r["vpp_mv"] / vref if vref else None
        g = r["gain_vs_ref"]
        r["fc_hz_first_order"] = float(r["f_hz"] * np.sqrt(1 / g ** 2 - 1)) if g and 0 < g < 0.9 and r["detected"] else None
    out.update(ladder=lad, ref_vpp_mv=vref)
    t.configure(wave="square")
    fa = actual(t.tone(16)) or 16.0
    time.sleep(0.5)
    y = capture(t, 10000, 3.0); raw["square_16"] = y.astype(np.uint16)
    print(t.off()); t.configure(wave="sine")
    out["square16"] = dict(ad9833_hz=fa, dc_mv=float(mv(y).mean()), ptp_mv=float(np.ptp(mv(y))), **step_response(mv(y), 10000, fa))
    print(f"  16 Hz square at A0: {out['square16']['ptp_mv']:.0f} mV p-p, {out['square16']['verdict']}")
    out["verdict"] = verdict1(out)
    print(f"\n  PART 1: {out['verdict']}")
    return out


def verdict1(o):
    lad = {r["f_hz"]: r for r in o["ladder"]}
    if not o["ref_vpp_mv"]:
        return ("A0 shows no tone at 1.6-50 kHz either: the AD9833 output is not reaching A0 at all (or A0 is not "
                "connected). This is not a low-frequency effect - check VOUT on the scope first (part 2, point VOUT)")
    g16 = lad[16]["gain_vs_ref"] if lad[16]["detected"] else 0.0
    if g16 >= 0.7:
        return (f"A0 passes 16 Hz ({lad[16]['vpp_mv']:.0f} mVpp vs {o['ref_vpp_mv']:.0f} at 1.6-50 kHz): the AD9833 and "
                "the path up to A0 are fine at 16 Hz; look after E+ (attenuator, leads, crown/back contact)")
    fc = [r["fc_hz_first_order"] for r in o["ladder"] if r["fc_hz_first_order"]]
    sq = o["square16"]
    return (f"A0 sees the tone at 1.6-50 kHz ({o['ref_vpp_mv']:.0f} mVpp) but {'not at all' if not lad[16]['detected'] else f'only {g16:.1%}'} "
            f"at 16 Hz: a high-pass between the AD9833 and A0"
            + (f", first-order corner ~{np.median(fc):.0f} Hz from the ladder" if fc else "")
            + f"; square: {sq['verdict']}. The AD9833 itself is producing output (same DAC at every frequency). "
            "A0 cannot tell whether the block is in the E+ drive (the watch then gets nothing) or only in A0's own input "
            "network (the watch may get the tone) - part 2, points E+ and ATT")


def ask(q):
    s = input(f"    {q} (Enter = not measured): ").strip()
    try:
        return float(s) if s else None
    except ValueError:
        print("      not a number - stored as not measured"); return None


def point(name, where, hint):
    print(f"\n  [{name}] {where}\n    {hint}")
    vpp, dc = ask("Vpp [mV]"), ask("DC (mean) [mV]")
    note = input("    note (probe x1/x10, V/div, averaging, load) : ").strip()
    return dict(point=where, vpp_mv=vpp, dc_mv=dc, note=note, status="measured" if vpp is not None else "not measured")


def part2(t):
    print("\nPART 2 - scope. DC coupling, 20 MHz bandwidth limit, scope ground on Teensy GND (= the back lead), laptop on "
          "battery. Nobody touching the watch or the leads.")
    att_from = input("  The attenuator input is wired to: [v]OUT or [e]+ ? ").strip().lower()[:1]
    att_from = {"v": "AD9833 VOUT", "e": "E+"}.get(att_from, "not stated")
    print(t.tone(16))
    m = dict(attenuator_input=att_from, tone_16hz={}, tone_160khz={})
    m["tone_16hz"]["vout"] = point("1/6", "AD9833 VOUT pin, 16 Hz sine", "expect ~600 mVpp around ~+300 mV (x10 probe, 100 mV/div)")
    m["tone_16hz"]["eplus"] = point("2/6", "E+ output wire (as used for the maXTouch), 16 Hz sine", "expect: unknown - this is the question")
    m["tone_16hz"]["att"] = point("3/6", "attenuator output = crown node, across R2 (crown - back), 16 Hz sine",
                                  "x1 probe, 1-2 mV/div, averaging 16-64; ~1 mVpp if fed from a working source")
    print(t.tone(160000))
    m["tone_160khz"]["vout"] = point("4/6", "AD9833 VOUT pin, 160 kHz sine", "the design frequency of the E+ drive")
    m["tone_160khz"]["eplus"] = point("5/6", "E+ output wire, 160 kHz sine", "reference: E+ at its design frequency")
    print(t.off())
    input("\n  [6/6] Now connect the FUNCTION GENERATOR exactly as in the ECG test that worked (same leads, same setting)"
          " and press Enter ")
    m["generator"] = point("6/6", "function generator at the crown - back point (the working reference)",
                           "same scope settings as point 3; also note the generator's set amplitude and frequency")
    m["generator"]["setting"] = input("    generator setting (e.g. '16 Hz sine 1 mVpp, 50 Ohm out') : ").strip()
    m["verdict"] = verdict2(m)
    print("\n  PART 2:"); [print(f"   - {s}") for s in m["verdict"]]
    return m


def verdict2(m):
    v = []
    g = lambda k, p: m[k][p]["vpp_mv"]
    vo, ep, at, ep_hi, gen = g("tone_16hz", "vout"), g("tone_16hz", "eplus"), g("tone_16hz", "att"), g("tone_160khz", "eplus"), m["generator"]["vpp_mv"]
    if vo is None:
        v.append("VOUT not measured: the AD9833 16 Hz output is unverified")
    elif vo < 100:
        v.append(f"VOUT {vo:g} mVpp at 16 Hz: the AD9833 is NOT producing the expected ~600 mVpp sine - fix this first")
    else:
        v.append(f"VOUT {vo:g} mVpp at 16 Hz: the AD9833 produces the 16 Hz sine")
    if ep is not None and vo:
        v.append(f"E+ / VOUT at 16 Hz = {ep / vo:.3g}" + (f"; at 160 kHz E+ = {ep_hi:g} mVpp" if ep_hi is not None else "")
                 + (": the E+ drive blocks 16 Hz" if ep < 0.1 * vo else ": the E+ drive passes 16 Hz"))
    if ep is not None and ep >= 0.1 * (vo or ep):
        v.append("E+ carries the tone, so a missing tone on A0 is an A0 monitoring limitation, not a missing signal")
    if at is not None:
        v.append(f"crown - back with the Teensy: {at:g} mVpp (MEASURED, fed from {m['attenuator_input']})"
                 + (f"; with the generator: {gen:g} mVpp -> ratio {at / gen:.2g}" if gen else ""))
    elif gen is not None:
        v.append(f"generator at crown - back: {gen:g} mVpp (MEASURED); the Teensy's crown - back level was not measured")
    v.append("anything marked 'not measured' stays unknown - the divider arithmetic in ecg_ber is a calculation, not a "
             "substitute for these readings")
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True); ap.add_argument("--port")
    ap.add_argument("--scope", action="store_true", help="also part 2: guided scope readings")
    ap.add_argument("--no-a0", action="store_true", help="skip part 1")
    a = ap.parse_args()
    print("Bench only: watch OFF the wrist, nobody touching the watch, the leads or the attenuator.")
    out = os.path.join(B.DATADIR, f"bench_{a.label}_{B.stamp()}")
    t = STX.StentTX(a.port)
    rec = dict(label=a.label, created=datetime.datetime.now().isoformat(timespec="seconds"),
               transmitter=t.configure(wave="sine"), a0_point=B.A0_POINT, a0_lsb_mv=B.ADC_LSB_MV)
    raw = {}
    try:
        os.makedirs(out)
        if not a.no_a0:
            print("\nPART 1 - A0 (the board's E+ node through its RX path; not the watch input)")
            rec["a0"] = part1(t, raw)
        if a.scope:
            rec["scope"] = part2(t)
    except KeyboardInterrupt:
        rec["stopped"] = "Ctrl+C"; print("\nstopped (Ctrl+C)")
    finally:
        t.close()
        if os.path.isdir(out):
            json.dump(rec, open(os.path.join(out, "bench.json"), "w"), indent=1, default=float)
            if raw:
                np.savez_compressed(os.path.join(out, "a0_captures.npz"), **raw)
            print(f"-> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
