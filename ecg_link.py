#!/usr/bin/env python3
"""
Stent/TeensyTRX -> Pixel Watch 2 ECG link: frames, transmit schedules and the bit decoder (FROZEN - do not edit; the sha in ecg_ber.FROZEN_SHA checks it).

Frames (why not the maXTouch v4 frame): the ECG app records 30 s, and the v4 frame (16 + 11 + 127 + 16 = 170 bits) lasts
21 s at 8 bps and 85 s at 2 bps, so most recordings would hold no complete frame. ECG frame = PREAMBLE 1010 + Barker-7
SFD 1110010 (11 header bits) + PRBS payload (order chosen per rate, the longest whose frame lasts <= FRAME_MAX_S, so a
29.5 s recording always holds >= 1 complete frame, usually 2) + 2 zeros. The transmitter repeats the frame without
pause for --seconds, so the recording can start anywhere in it (blind sync on the header, as v4 does).

Tones: every bit is one TeensyTRX schedule step, and the firmware restarts the AD9833 phase at every step
('off -> FREQ0 -> on'). With an integer number of tone cycles per bit the restart falls on a zero crossing and the
waveform is continuous; the spectral line then sits at exactly (cycles per bit) x bps on the Teensy crystal, whatever
the 0.093 Hz AD9833 word quantisation. OOK '0' = frequency word 0 in sine mode = DAC at mid-scale = the sine's own
mean: no DC step between bits (square mode would step 0 <-> 1.65 V at every OOK edge and must not be used).

Decoder rules (v4's rules 3-6 with a header argument; the bits are only used to count errors):
  1. Clock: the watch's sample clock is not exactly 250 Hz (WatchLink's own Pixel export: a 1 Hz source read as
     0.99878 Hz, i.e. ~+1200 ppm). delta = scheduled line / observed line - 1, from the summed power of the scheduled
     tone lines on a +-CLOCK_PPM grid (bit-free). Time is corrected t <- t / (1 + delta).
  2. Timing feature per sample: p_f = |half-bit boxcar of x * exp(-j 2 pi f t)|^2; OOK log10(1 + p1/s1),
     FSK log10((p1 + s1) / (p0 + s0)); s_f = NOISE_Q (25th) percentile of p_f over the recording - in band and
     bit-free (about half of all bits leave each tone off; an off-tone reference picks up OOK sidebands, and at
     16 bps every bin orthogonal to the tone is outside the watch's passband).
  3. Timing: v4 rule 3 (tau over one frame period in 4 ms steps, then +-6 ms x +-200 ppm) with this frame's header,
     maximising the MEAN header correlation over the frames whose header lies inside the recording. v4 maximises
     the SUM; with ~40 frames per maXTouch run that is the same thing, but a 30 s ECG window holds 2 or 3 frames
     depending on tau, and the sum then prefers a weaker lock with one frame more (synthetic FSK 4 bps, seed 1:
     3 x r 0.68 = 2.03 beat the true 2 x r 1.00 = 2.00, a 20-bit false lock). reduce='sum' gives v4's score.
  4. Bit statistic: tone energy of the bit's samples more than GUARD from both edges (GUARD = max(2 ms, GUARD_FRAC Tb):
     the watch's ~31 Hz low-pass smears edges over tens of ms), E_f = |mean x exp(-j 2 pi f t)|^2;
     OOK v = log10(1 + E1/n1), FSK v = log10((E1 + n) / (E0 + n)), n_f = NOISE_Q percentile of E_f over the bits.
  5. Threshold per frame halfway between the means of its known 1 and 0 bits (header + gap) - v4 rule 5.
  6. Frame accepted when its header correlation r >= 0.3 (truth-free). Frames expected = the complete frame periods
     inside the recording at the found timing. v4's extra 'frame BER < 0.35' condition uses
     the payload truth, so it is reported separately ('v4_accounting') for comparison with the maXTouch numbers only.
Es/N0 (evaluation only, uses the sent bits): (E of a tone in bits that sent it - E of that tone in bits that did not)
/ the latter. For white noise and bit means this is stent_modcomp's k P1 / (2 sigma^2); here the 'off' energy also
holds filter tails (ISI) and, for FSK, leakage of the other tone, so it is a lower bound.
Line SNR (bit-free): power of the scheduled lines (after the clock correction) over the same lines shifted 2.5 Hz.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAXTOUCH = os.path.join(os.path.dirname(HERE), "maxtouch-screen-main")
sys.path.insert(0, MAXTOUCH)
import stent_tx as STX                          # fword, MCLK, StentTX (TeensyTRX host driver) - unchanged

VERSION = "ecg-v1"
FROZEN_AT = "2026-11-02 16:34:17"                   # frozen at the user's request before the first worn-watch BER session
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]

PREAMBLE = [1, 0, 1, 0]
SFD = [1, 1, 1, 0, 0, 1, 0]                     # Barker-7
HDR = PREAMBLE + SFD
GAP = [0, 0]
FRAME_MAX_S = 14.0
PRBS_TAPS = {3: (3, 2), 4: (4, 3), 5: (5, 3), 6: (6, 5), 7: (7, 6)}     # x^n + x^k + 1, maximal length
CLOCK_PPM, CLOCK_STEP_PPM = 5000, 20
EPS_PPM = np.arange(-200, 201, 10)              # v4 rule 3
TAU_STEP, FINE_HALF, FINE_STEP = 0.004, 0.006, 0.00025
GUARD_MS, GUARD_FRAC = 2.0, 0.10
NOISE_Q = 25
R_LOCAL, BER_INVALID = 0.3, 0.35                # v4 / cap_ook_rx
MOD_NAMES = dict(ook="OOK", fsk="FSK")

# Initial tones (Hz): inside the 3-30 Hz flat band estimated from WatchLink's Pixel export (pixel_1_Hz.pdf: first-order
# high-pass ~1 Hz, flat within ~1 dB 3-25 Hz, -2 dB 31 Hz, -8 dB 41 Hz, -25 dB 51 Hz), an integer number of cycles per
# bit at 2/4/8/16 bps, FSK pair a multiple of the bit rate apart (orthogonal). Re-pick after the passband sweep.
TONES = {2: dict(ook=16.0, fsk=(16.0, 8.0)), 4: dict(ook=16.0, fsk=(16.0, 8.0)),
         8: dict(ook=16.0, fsk=(16.0, 8.0)), 16: dict(ook=16.0, fsk=(16.0, 32.0))}
SWEEP_HZ = [2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 100]
SWEEP_STEP_S = 1.25                             # 20 tones = 25 s cycle: a 29.5 s recording sees every tone whole
SWEEPLOW_HZ = [0.5, 0.75, 1, 1.5, 2, 3]         # the ~1 Hz high-pass needs >= 2 cycles per window: 4 s steps
SWEEPLOW_STEP_S = 4.0
ALIAS_HZ = [16, 130, 170, 200, 240, 245, 255, 260, 375, 490, 510, 1000, 4000, 159585]
ALIAS_STEP_S = 1.25


# ================================================================ frames and plans
def prbs(order):
    a, b = PRBS_TAPS[order]
    s, out = [1] * order, []
    for _ in range((1 << order) - 1):
        out.append(s[-1])
        s = [s[a - 1] ^ s[b - 1]] + s[:-1]
    return out


def frame(bps):
    """(frame bits, payload bits, prbs order): the longest PRBS payload whose frame lasts <= FRAME_MAX_S"""
    for order in (7, 6, 5, 4, 3):
        pl = prbs(order)
        bits = HDR + pl + GAP
        if len(bits) / bps <= FRAME_MAX_S:
            return bits, pl, order
    raise ValueError(f"{bps} bps: even a PRBS3 frame is longer than {FRAME_MAX_S} s")


def actual_hz(hz):
    return STX.fword(hz) * STX.MCLK / 2 ** 28 if hz else 0.0


def line_hz(hz, bps):
    """where the spectral line of a keyed tone really is: phase restarts every bit, so with ~integer cycles per bit
    the waveform repeats every bit and the line sits at round(cycles) * bps (Teensy crystal)"""
    if not hz:
        return 0.0
    c = actual_hz(hz) / bps
    return round(c) * bps if abs(c - round(c)) < 0.05 else actual_hz(hz)


def ber_plan(mod, bps, f1=None, f0=None, seconds=90.0):
    t = TONES.get(int(bps), TONES[16])
    if mod == "ook":
        f1, f0 = f1 or t["ook"], 0.0
    else:
        f1, f0 = (f1, f0) if f1 and f0 else t["fsk"]
    bits, pl, order = frame(bps)
    bit_us = int(round(1e6 / bps))
    frame_s = len(bits) * bit_us / 1e6
    reps = math.ceil(seconds / frame_s)
    cyc = [actual_hz(f) / bps for f in (f1, f0) if f]
    return dict(kind="ber", modulation=mod, bps=1e6 / bit_us, bit_us=bit_us, f1_hz=f1, f0_hz=f0,
                f1_actual_hz=actual_hz(f1), f0_actual_hz=actual_hz(f0), f1_line_hz=line_hz(f1, bps),
                f0_line_hz=line_hz(f0, bps), cycles_per_bit=[round(c, 3) for c in cyc],
                phase_continuous=all(abs(c - round(c)) < 0.05 for c in cyc), header=HDR, gap=GAP, prbs_order=order,
                frame_bits=len(bits), payload_bits=len(pl), frame_s=frame_s, frames=reps, total_s=reps * frame_s,
                n_steps=reps * len(bits), wave="sine")


def tone_plan(kind, seconds=60.0):
    hz, step = dict(sweep=(SWEEP_HZ, SWEEP_STEP_S), sweeplow=(SWEEPLOW_HZ, SWEEPLOW_STEP_S),
                    alias=(ALIAS_HZ, ALIAS_STEP_S))[kind]
    step_us = int(round(step * 1e6))
    cycle = len(hz) * step_us / 1e6
    reps = math.ceil(seconds / cycle)
    return dict(kind=kind, tones_hz=list(hz), tones_actual_hz=[actual_hz(f) for f in hz], step_s=step_us / 1e6,
                cycle_s=cycle, frames=reps, total_s=reps * cycle, n_steps=reps * len(hz), wave="sine")


def schedule(plan):
    """TeensyTRX schedule [(step_us, reps, words)] - one sseg, as stent_modcomp.schedule"""
    if plan["kind"] == "ber":
        bits, _, _ = frame(plan["bps"])
        words = [STX.fword(plan["f1_hz"] if b else plan["f0_hz"]) for b in bits]
        return [(plan["bit_us"], plan["frames"], words)]
    return [(int(round(plan["step_s"] * 1e6)), plan["frames"], [STX.fword(f) for f in plan["tones_hz"]])]


# ================================================================ decoder
def tone_power(x, t, f, half):
    """|centred boxcar (half samples each side) of x exp(-j 2 pi f t)|^2 per sample"""
    z = x * np.exp(-2j * np.pi * f * t)
    c = np.concatenate([[0], np.cumsum(z)])
    i = np.arange(len(x))
    lo, hi = np.clip(i - half, 0, len(x)), np.clip(i + half + 1, 0, len(x))
    return np.abs((c[hi] - c[lo]) / (hi - lo)) ** 2


def clock_delta(x, t, lines):
    """bit-free sample-clock error of the watch from the scheduled tone lines (rule 1): returns (delta, line SNR dB)"""
    w = np.hanning(len(x)); xw = (x - x.mean()) * w
    grid = np.arange(-CLOCK_PPM, CLOCK_PPM + 1, CLOCK_STEP_PPM) * 1e-6

    def power(dl):
        return sum(abs(np.sum(xw * np.exp(-2j * np.pi * f / (1 + dl) * t))) ** 2 for f in lines)
    P = np.array([power(d) for d in grid])
    k = int(np.argmax(P))
    if 0 < k < len(grid) - 1:                   # parabolic refinement
        y0, y1, y2 = np.log(P[k - 1:k + 2] + 1e-300)
        den = y0 - 2 * y1 + y2
        d = grid[k] + (0.5 * (y0 - y2) / den if den else 0) * (grid[1] - grid[0])
    else:
        d = grid[k]
    snr = 10 * np.log10(power(d) / (sum(abs(np.sum(xw * np.exp(-2j * np.pi * (f / (1 + d) + 2.5) * t))) ** 2
                                        for f in lines) + 1e-300))
    return float(d), float(snr)


def bit_index(t, tau, Tb, eps, nb):
    u = (t - tau) / (Tb * (1 + eps))
    m = np.floor(u / nb).astype(int)
    jb = u - m * nb
    j = np.floor(jb).astype(int)
    return m, j, jb - j


def header_score(f, t, tau, Tb, eps, nb, frames, hdr, reduce="mean"):
    """decode_v4.header_score with the header as an argument; reduce='sum' is v4's score"""
    m, j, _ = bit_index(t, tau, Tb, eps, nb)
    sel = (j < len(hdr)) & np.isin(m, frames)
    if not sel.any():
        return -np.inf
    idx = np.searchsorted(frames, m[sel]) * len(hdr) + j[sel]
    n = len(frames) * len(hdr)
    s = np.bincount(idx, f[sel], n); c = np.bincount(idx, minlength=n)
    H = np.where(c > 0, s / np.maximum(c, 1), np.nan).reshape(len(frames), len(hdr))
    ok = ~np.isnan(H).any(axis=1)
    if not ok.any():
        return -np.inf
    Hc = H[ok] - H[ok].mean(axis=1, keepdims=True); tc = hdr - hdr.mean()
    den = np.linalg.norm(Hc, axis=1) * np.linalg.norm(tc)
    r = np.where(den > 0, Hc @ tc / np.where(den > 0, den, 1), 0)
    return float(np.sum(r) if reduce == "sum" else np.mean(r))


def search(f, t, Tb, nb, tau_lo, tau_hi, frames_of, hdr, reduce="mean"):
    """decode_v4.search with the header (and the score's reduction) as arguments"""
    hdr = np.asarray(hdr, float)
    best = (-np.inf, tau_lo, 0.0)
    for tau in np.arange(tau_lo, tau_hi, TAU_STEP):
        s = header_score(f, t, tau, Tb, 0.0, nb, frames_of(tau, 0.0), hdr, reduce)
        if s > best[0]:
            best = (s, tau, 0.0)
    tau_c = best[1]
    for eps in EPS_PPM * 1e-6:
        for tau in np.arange(tau_c - FINE_HALF, tau_c + FINE_HALF + 1e-9, FINE_STEP):
            s = header_score(f, t, tau, Tb, eps, nb, frames_of(tau, eps), hdr, reduce)
            if s > best[0]:
                best = (s, tau, eps)
    return best[1], best[2], best[0]


def bit_energies(x, t, tau, Tb, eps, nb, frames, freqs, guard_s):
    """per (frame, bit): |mean of x exp(-j 2 pi f t)|^2 over the guarded samples, for each f; NaN where no sample"""
    m, j, pos = bit_index(t, tau, Tb, eps, nb)
    g = guard_s / (Tb * (1 + eps))
    inner = (pos > g) & (pos < 1 - g)
    E = np.full((len(freqs), len(frames), nb), np.nan)
    cnt = np.zeros((len(frames), nb), int)
    for k, mm in enumerate(frames):
        use = inner & (m == mm)
        jj = j[use]
        if not len(jj):
            continue
        c = np.bincount(jj, minlength=nb)
        cnt[k] = c
        for fi, f in enumerate(freqs):
            z = x[use] * np.exp(-2j * np.pi * f * t[use])
            s = np.bincount(jj, z.real, nb) + 1j * np.bincount(jj, z.imag, nb)
            E[fi, k] = np.where(c > 0, np.abs(s / np.maximum(c, 1)) ** 2, np.nan)
    return E, cnt


def decode(x, t, plan, truth=None):
    """x [mV] at times t [s] (watch time base, nominal fs). Returns (report, trace). truth = payload bits, used only
    to count errors."""
    mod, bps = plan["modulation"], plan["bps"]
    bits, pl, _ = frame(bps)
    truth = pl if truth is None else truth
    nb, Tb = len(bits), plan["bit_us"] / 1e6
    P = nb * Tb
    hdr = np.array(HDR, float)
    x = np.asarray(x, float) - np.mean(x)
    f1, f0 = plan["f1_line_hz"], plan["f0_line_hz"]
    lines = [f for f in (f1, f0) if f]
    delta, line_snr = clock_delta(x, t, lines)
    tc = t / (1 + delta)                                    # rule 1
    fs = (len(t) - 1) / (tc[-1] - tc[0])
    half = max(1, int(round(Tb * fs / 4)))                  # half-bit boxcar = +-Tb/4
    p1 = tone_power(x, tc, f1, half)
    s1 = float(np.percentile(p1, NOISE_Q))
    if mod == "fsk":
        p0 = tone_power(x, tc, f0, half)
        s0 = float(np.percentile(p0, NOISE_Q))
        feat = np.log10((p1 + s1) / (p0 + s0))
    else:
        feat = np.log10(1 + p1 / max(s1, 1e-12))

    def all_frames(tau, eps):                               # as decode_v4: frames whose header lies inside
        Pe = P * (1 + eps)
        return np.arange(max(0, int(np.ceil((tc[0] - tau) / Pe))), int((tc[-1] - tau) / Pe) + 1)

    tau, eps, score = search(feat, tc, Tb, nb, tc[0], tc[0] + P, all_frames, hdr)   # rule 3
    full = [int(mm) for mm in all_frames(tau, eps) if tau + (mm + 1) * P * (1 + eps) <= tc[-1] + 1e-9]
    guard = max(GUARD_MS / 1e3, GUARD_FRAC * Tb)
    freqs = [f1] + ([f0] if mod == "fsk" else [])
    E, cnt = bit_energies(x, tc, tau, Tb, eps, nb, full, freqs, guard)
    pq = lambda a: float(np.nanpercentile(a, NOISE_Q)) if np.isfinite(a).any() else 1e-12
    n1 = pq(E[0])
    if mod == "fsk":
        nz = max((n1 + pq(E[1])) / 2, 1e-12)
        V = np.log10((E[0] + nz) / (E[1] + nz))
    else:
        V = np.log10(1 + E[0] / max(n1, 1e-12))
    fb = np.array(bits); kn = np.zeros(nb, bool); kn[:len(HDR)] = True; kn[len(HDR) + len(pl):] = True
    psl = slice(len(HDR), len(HDR) + len(pl))
    tr = np.array(truth, int)
    out = []
    for k, mm in enumerate(full):
        v = V[k]
        if np.isnan(v[:len(HDR)]).any() or np.isnan(v[psl]).any():
            continue
        hi, lo = np.nanmean(v[kn & (fb == 1)]), np.nanmean(v[kn & (fb == 0)])
        thr = (hi + lo) / 2                                  # rule 5
        got = (v[psl] > thr).astype(int)
        err = int(np.sum(got != tr))
        hc = v[:len(HDR)] - v[:len(HDR)].mean(); tch = hdr - hdr.mean()
        r = float(hc @ tch / (np.linalg.norm(hc) * np.linalg.norm(tch) + 1e-12))
        ber = err / len(tr)
        out.append(dict(m=int(mm), s0=float(tau + mm * P * (1 + eps)), r=r, bits=len(tr), errors=err, ber=ber,
                        thr=float(thr), margin=float(hi - lo), got=got.tolist(),
                        norm=((v[psl] - thr) / max(hi - lo, 1e-9)).tolist(),
                        valid=bool(r >= R_LOCAL),                        # rule 6, truth-free
                        valid_v4=bool(r >= R_LOCAL and ber < BER_INVALID)))
    on, offe = [], []                                       # evaluation only: tone energy where sent / not sent
    for k in range(len(full)):
        on.append(E[0, k][psl][tr == 1]); offe.append(E[0, k][psl][tr == 0])
        if mod == "fsk":
            on.append(E[1, k][psl][tr == 0]); offe.append(E[1, k][psl][tr == 1])
    es = None
    if on:
        S, N = np.nanmean(np.concatenate(on)), np.nanmean(np.concatenate(offe))
        es = float(10 * np.log10((S - N) / N)) if S > N > 0 else None      # None: no tone above the off level
    return dict(decoder=VERSION, rules_sha=RULES_SHA, frozen_at=FROZEN_AT, modulation=mod, bps=bps,
                clock_ppm=delta * 1e6, residual_ppm=eps * 1e6, line_snr_db=line_snr, header_score=score,
                tau_s=tau, frames_complete=len(full), frames_expected=len(full), guard_ms=guard * 1e3,
                es_n0_db=es, frames=out), \
        dict(t=tc, feat=feat, x=x, tau=tau, eps=eps, Tb=Tb, nb=nb, V=V, frames_full=full, cnt=cnt)


# ================================================================ accounting (decode_v2 / decode_v3, unchanged)
def frames_summary(frames, n_expected, payload_bits, key="valid"):
    """decode_v2.frames_summary with the validity flag chosen (truth-free 'valid' or v4's 'valid_v4')"""
    import decode_v2 as V2
    return V2.frames_summary([dict(fr, valid=fr[key]) for fr in frames], n_expected, payload_bits)


def error_types(frames, truth, key="valid"):
    import decode_v3 as V3
    return V3.error_types([dict(fr, valid=fr[key]) for fr in frames], truth)


# ================================================================ signal quality (bit-free unless stated)
def quality(x, fs, lines, mains=(50.0, 60.0)):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / fs)
    S = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    band = (f >= 2) & (f <= 40)
    near = np.zeros(len(f), bool)
    for L in list(lines) + list(mains):
        near |= np.abs(f - L) < 1.0
    floor = float(np.median(S[band & ~near])) if (band & ~near).any() else float("nan")
    q = dict(rms_mv=float(np.std(x)), ptp_mv=float(np.ptp(x)))
    for L in lines:
        k = np.abs(f - L) < 0.3
        q[f"line_{L:g}hz_db_over_floor"] = float(10 * np.log10(S[k].max() / floor)) if k.any() else None
    for L in mains:
        k = np.abs(f - L) < 0.5
        q[f"mains_{L:g}hz_db_over_floor"] = float(10 * np.log10(S[k].max() / floor)) if k.any() else None
    d = np.diff(x)                               # flat runs = saturation / dropped data (>= 0.1 s identical values)
    flat, run = 0, 0
    for v in d:
        run = run + 1 if v == 0 else 0
        if run == int(0.1 * fs):
            flat += 1
    q["flat_runs_0p1s"] = flat
    return q

