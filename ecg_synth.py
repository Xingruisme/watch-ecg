#!/usr/bin/env python3
"""
Synthetic Pixel Watch ECG recordings for the receiver-independent dry run (no hardware, no person).

Channel model (every number is an assumption to be replaced by the bench passband sweep):
  transmitter  TeensyTRX schedule exactly as the firmware plays it: AD9833 sine at the quantised word frequency,
               phase restarted at every step, word 0 = mid-scale (= 0 after the DC block)
  attenuator   amplitude at the watch electrodes in mVpp (--amp; bench start 1 mVpp)
  watch        first-order high-pass 1 Hz + 5th-order Butterworth low-pass 31.5 Hz (fit to the harmonics of
               WatchLink's real Pixel Watch 2 export pixel_1_Hz.pdf, rms misfit 3 dB below 70 Hz), sample clock
               250 Hz x (1 + clock_ppm), default +1227 ppm (same export), recording 30 s of which the PDF shows the
               last 29.5 s, values kept to 0.2 uV (the PDF's 0.01-unit resolution)
  noise        white (uV rms at the output), 50 Hz mains (uV after the watch filter), baseline wander (uV), optional
 
                             heartbeats (a human-contact case; off for the bench), contact dropouts
Outputs a Fitbit-style vector PDF (same structure as the real export: 4 row forms, 1 mm grid of 5 units, 1 mV pulse,
one polyline point per sample) so the dry run goes through the same PDF importer as real data.
"""
from __future__ import annotations

import datetime
import math

import numpy as np
from scipy import signal

import ecg_link as L

FS_NOM = 250.0
HP_HZ, LP_HZ, LP_ORDER = 1.0, 31.5, 5
CLOCK_PPM = 1227.0
REC_S, HIDDEN_S = 30.0, 0.5
FS_SIM = 2000.0


def tx_waveform(plan, t):
    """electrode voltage (relative, amplitude 1 = peak of the sine) at TX times t (s from the first step)"""
    sched = L.schedule(plan)
    step_us, reps, words = sched[0]
    step = step_us / 1e6
    hz = np.array([w * L.STX.MCLK / 2 ** 28 for w in words])
    k = np.floor(t / step).astype(int)
    inside = (k >= 0) & (k < reps * len(words))
    f = np.where(inside, hz[np.clip(k, 0, None) % len(words)], 0.0)
    return np.where(inside, np.sin(2 * np.pi * f * (t - k * step)), 0.0)


def watch_filter(fs):
    hp = signal.butter(1, HP_HZ, "highpass", fs=fs, output="sos")
    lp = signal.butter(LP_ORDER, LP_HZ, "lowpass", fs=fs, output="sos")
    return np.vstack([hp, lp])


def heartbeat(t, bpm, rng):
    """sum-of-Gaussians PQRST (mV) - only for the human-contact scenario"""
    x = np.zeros_like(t)
    beat = 60.0 / bpm
    tb = t[0] + rng.uniform(0, beat)
    while tb < t[-1] + 1:
        for a, mu, s in ((0.12, -0.20, 0.025), (-0.10, -0.03, 0.01), (1.0, 0.0, 0.012), (-0.25, 0.03, 0.01),
                         (0.3, 0.25, 0.04)):
            x += a * np.exp(-0.5 * ((t - tb - mu) / s) ** 2)
        tb += beat * (1 + rng.normal(0, 0.03))
    return x


def record(plan, rng, amp_mvpp=1.0, white_uv=10.0, mains_uv=5.0, wander_uv=30.0, clock_ppm=CLOCK_PPM,
           start_s=None, heart_bpm=None, dropout_s=0.0, no_signal=False):
    """one 30 s recording (t from 0 at the first shown sample, mV), and where it started in the TX schedule"""
    if start_s is None:
        start_s = rng.uniform(3.0, max(3.5, plan["total_s"] - REC_S - 3.0))
    pre = 3.0                                                 # let the watch filter settle as on a real watch
    ts = np.arange(start_s - pre, start_s + REC_S + 0.2, 1 / FS_SIM)
    v = np.zeros_like(ts) if no_signal else 0.5 * amp_mvpp * tx_waveform(plan, ts)
    if heart_bpm:
        v = v + heartbeat(ts, heart_bpm, rng)
    v = signal.sosfilt(watch_filter(FS_SIM), v)
    fs_true = FS_NOM * (1 + clock_ppm * 1e-6)
    n = int(round(REC_S * FS_NOM))
    tk = start_s + np.arange(n) / fs_true
    x = np.interp(tk, ts, v)
    x += rng.normal(0, white_uv * 1e-3, n)
    x += mains_uv * 1e-3 * np.sin(2 * np.pi * 50.0 * tk + rng.uniform(0, 2 * np.pi))
    w = np.cumsum(rng.normal(0, 1, n)); w = signal.sosfiltfilt(signal.butter(2, 0.3, fs=FS_NOM, output="sos"), w)
    x += wander_uv * 1e-3 * w / (np.std(w) + 1e-12)
    if dropout_s > 0:                                         # contact lost: rail then slow recovery
        i0 = int(rng.uniform(0.2, 0.7) * n); i1 = min(n, i0 + int(dropout_s * FS_NOM))
        x[i0:i1] = 2.5
    k0 = int(round(HIDDEN_S * FS_NOM))
    x = np.round(x[k0:] * 5000) / 5000                        # 0.2 uV (0.01 PDF unit)
    return np.arange(len(x)) / FS_NOM, x, dict(start_s=float(start_s + HIDDEN_S), clock_ppm=clock_ppm,
                                                amp_mvpp=amp_mvpp, white_uv=white_uv, mains_uv=mains_uv,
                                                wander_uv=wander_uv, heart_bpm=heart_bpm, dropout_s=dropout_s,
                                                no_signal=no_signal)


# ================================================================ Fitbit-style PDF
def write_pdf(path, x_mv, when=None, label="SYNTHETIC"):
    """x_mv = the samples shown (29.5 s from 0.5 s). Layout as the real export (rows of 8 s, 0.5 s hidden by the pulse)"""
    u_s, u_mv, dx, y0 = 125.0, 50.0, 0.5, 99.0
    when = when or datetime.datetime.now()
    rows = []
    k = 0
    for r in range(4):
        x_start = 2.0 + (HIDDEN_S * u_s if r == 0 else 0.0)
        n_row = int(round((1002.0 - x_start) / dx)) + 1          # includes the boundary sample at x = 1002
        seg = x_mv[k:k + n_row]
        if len(seg) == 0:
            break
        rows.append((x_start, seg))
        k += n_row - 1                                            # next row repeats the boundary sample
    forms = []
    for r, (xs, seg) in enumerate(rows):
        c = ["q 0 198 m 0 0 l 1004 0 l 1004 198 l h W n", "1 1 1 rg 0 0 1004 198 re f",
             "0.906 0.914 0.922 RG 0.5 w"]
        c += [f"q 1 0 0 1 {1.5 + 5 * i:g} 174 cm 0 0 m 0 -150 l S Q" for i in range(201)]
        c += [f"q 1 0 0 1 2 {24 + 5 * i:g} cm 0 0 m 1000 0 l S Q" for i in range(31)]
        c += ["0.812 0.827 0.839 RG 1 w"]
        c += [f"q 1 0 0 1 {2 + 25 * i:g} 174 cm 0 0 m 0 -150 l S Q" for i in range(41)]
        c += ["q 2 174 m 2 24 l 1002 24 l 1002 174 l h W n"]
        if r == 0:
            c += ["0 0 0 RG 3 w 1 j 1 J q 1 0 0 1 2 99 cm 0 0 m 10 0 l 10 50 l 35 50 l 35 0 l 45 0 l S Q"]
        ya = y0 + seg * u_mv
        pts = [f"{dx * i:.1f} {ya[i] - ya[0]:.2f} l" for i in range(1, len(seg))]
        c += ["0.22 0.498 0.635 RG 3 w 1 j 1 J", f"q 1 0 0 1 {xs:g} {ya[0]:.2f} cm 0 0 m"] + pts + ["S Q", "Q Q"]
        forms.append("\n".join(c).encode())
    txt = [(18.57, 748, 20, "Heart Rhythm Assessment"), (18.57, 729, 10, f"Recorded on {when:%b} {when.day}, "
           f"{when:%Y} {when.hour % 12 or 12}:{when:%M} {'AM' if when.hour < 12 else 'PM'}"),
           (18.57, 685, 16, "Inconclusive"), (18.57, 186.64, 8,
           "[ 25mm/s, 10mm/mV, 250 Hz ] : [ Qualitatively similar to a Lead I ECG ]"),
           (300, 30, 7, f"Feature Version: [{label}] Hardware: [{label} - not a watch recording] Wear OS Version: [none]")]
    page = []
    for s, (xx, yy, sz, t) in enumerate(txt):
        t = t.replace("(", "\\(").replace(")", "\\)")
        page.append(f"BT /F1 {sz} Tf {xx} {yy} Td ({t}) Tj ET")
    for r, yy in enumerate((549.16, 434.32, 319.48, 204.64)[:len(forms)]):
        page.append(f"q 0.58 0 0 0.58 16.57 {yy} cm 0 g 0 G 1 w /F1 10 Tf /Fm{r} Do Q")
    page = "\n".join(page).encode()
    objs = {1: b"<</Type/Catalog/Pages 2 0 R>>", 2: b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
            4: b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"}
    xo = " ".join(f"/Fm{r} {6 + r} 0 R" for r in range(len(forms)))
    objs[3] = (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 5 0 R/Resources<</Font<</F1 4 0 R>>"
               f"/XObject<<{xo}>>>>>>").encode()
    import zlib
    def stream(head, data):
        z = zlib.compress(data)
        return head + f"/Filter/FlateDecode/Length {len(z)}>>stream\n".encode() + z + b"\nendstream"
    objs[5] = stream(b"<<", page)
    for r, f in enumerate(forms):
        objs[6 + r] = stream(b"<</Type/XObject/Subtype/Form/FormType 1/BBox[0 0 1004 198]/Matrix[1 0 0 1 0 0]", f)
    out = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    offs = {}
    for n in sorted(objs):
        offs[n] = len(out)
        out += f"{n} 0 obj\n".encode() + objs[n] + b"\nendobj\n"
    xref = len(out)
    N = max(objs) + 1
    out += f"xref\n0 {N}\n0000000000 65535 f \n".encode()
    for n in range(1, N):
        out += f"{offs[n]:010d} 00000 n \n".encode()
    out += f"trailer\n<</Size {N}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n".encode()
    open(path, "wb").write(bytes(out))
