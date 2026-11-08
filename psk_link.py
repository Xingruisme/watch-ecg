"""psk-x2 (frozen; psk-x1 = development): EXPERIMENTAL phase/amplitude modulation for the Pixel Watch 2 ECG link (added 2026-11-06).

Separate from the frozen ecg-v1 (ecg_link.py is only imported for its frame definition, never changed). Same frame
bits as ecg-v1 - header 1010 + Barker-7, PRBS payload, gap 00, frame <= 14 s (ecg_link.frame) - so BER statistics stay
comparable; only the way bits become a waveform, and the receiver, are new.

Modulations (all on one carrier fc, default 16 Hz, transmitted as Re{b(t) e^{j 2 pi fc t}}, b = sum_k a_k p(t/T - k - 1/2)):
  dbpsk  1 bit/symbol, bit 1 = phase flip (pi), differential detection (no carrier phase needed)
  bpsk   1 bit/symbol, coherent: phase from the frame header, decision-directed tracking
  dqpsk  2 bits/symbol, Gray phase steps 0, +pi/2, pi, -pi/2, differential detection
  ask4   2 bits/symbol, 4 amplitude levels (-3,-1,+1,+3)/3 on one phase, coherent (gain/phase from the header)
Pulses p: 'rect' (constant envelope for the PSKs, abrupt phase steps), 'rc' (raised-cosine transitions of width beta
symbols: smooth, the amplitude dips through zero at a flip), 'rrc' (root raised cosine, roll-off alpha, span 6).
The waveform is normalised so its PEAK is 1 (= the configured mVpp / 2 at the watch): every scheme is compared at the
same peak voltage, the constraint that matters for safety and for the watch's input window.

Receiver (decode): (1) clock = carrier offset from the power-law line (x^2 at 2 fc, x^4 at 4 fc for dqpsk) - the
carrier and the symbol clock come from the same crystals, so one ppm describes both; (2) complex down-conversion and a
filter matched to p; (3) data-aided symbol timing + frame sync: grid over fine clock x sampling phase, header
correlation folded over the frame period; (4) detection per frame; (5) optional decision-directed least-squares
fractionally spaced equaliser (eq=True) for the watch's ~25 Hz low-pass; (6) per-frame validity from the header alone
(truth-free), BER only on valid frames, as ecg-v1.

Thresholds R_MIN / G_MIN were set from noise-only runs on REAL residuals (wearer ECG + noise, no PSK) before any PSK
measurement: see psk_sim.py falsealarm.
"""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np

import ecg_link as L

# psk-x1 = development (simulation of 2026-11-06 16:33). psk-x2 = clock search narrowed to +-200 ppm around the power
# line (never used on real data). psk-x3 = x2's receiver kept as decode() ("x2-diff") + the new decode_wl() ("x3-wl",
# primary for the real DBPSK experiment), FROZEN before any DBPSK measurement.
# Any later change to this file changes RULES_SHA; psk_ber.py refuses to run or analyse unless a new VERSION is set.
VERSION = "psk-x3"
FROZEN_AT = "2026-11-06 17:40"
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
FS = 250.0
LEAD_SYMS, TAIL_SYMS = 8, 2          # plain carrier before the first frame (reference + settling), after the last
MODS = dict(dbpsk=dict(bits=1, diff=True), bpsk=dict(bits=1, diff=False),
            dqpsk=dict(bits=2, diff=True), ask4=dict(bits=2, diff=False))
DQPSK_STEP = {(0, 0): 0.0, (0, 1): np.pi / 2, (1, 1): np.pi, (1, 0): -np.pi / 2}       # Gray
ASK4_LEVEL = {(0, 0): -1.0, (0, 1): -1 / 3, (1, 1): 1 / 3, (1, 0): 1.0}                # Gray
R_MIN = 0.70                         # per-frame header correlation for a valid frame
KNOWN_ERR_MAX = 1                    # and at most 1 of the 13 known bits (header 11 + gap 2) wrong after detection.
                                     # Added after the noise-only check on real residuals (4-ASK: 15/21 false frames
                                     # with the 5-symbol correlation alone); set before any PSK measurement.
G_MIN = 0.70                         # mean header correlation over the recording's frames for a sync lock
CLOCK_PPM = 6000
EQ_SPAN_S = 0.08                     # equaliser reach either side (watch low-pass impulse ~ +-40-60 ms)
CLOCK_WINDOW_PPM = 200               # fine search around the power-line clock
EQ_ITER = 3


def params():
    """every receiver parameter, recorded with each result"""
    return dict(version=VERSION, rules_sha=RULES_SHA, frozen_at=FROZEN_AT, R_MIN=R_MIN, G_MIN=G_MIN,
                KNOWN_ERR_MAX=KNOWN_ERR_MAX, CLOCK_PPM=CLOCK_PPM, CLOCK_WINDOW_PPM=CLOCK_WINDOW_PPM,
                EQ_SPAN_S=EQ_SPAN_S, EQ_ITER=EQ_ITER, LEAD_SYMS=LEAD_SYMS, TAIL_SYMS=TAIL_SYMS, PTS=PTS,
                WL_CHANNEL=list(WL_CHANNEL), WL_ITER=WL_ITER, WL_RIDGE=WL_RIDGE, WL_CLOCK_REFINE_PPM=WL_CLOCK_REFINE_PPM, WL_HYPOTHESES=WL_HYPOTHESES,
                receivers=["x3-wl (decode_wl, primary)", "x2-diff (decode)"])


# ================================================================ transmitter side
def plan(mod, bps, fc=16.0, shape="rect", beta=0.5, alpha=0.35, seconds=60.0):
    m = MODS[mod]
    baud = bps / m["bits"]
    sym_us = int(round(1e6 / baud))
    bits, pl, order = L.frame(bps)
    if len(bits) % m["bits"]:
        raise ValueError("frame length not a multiple of the bits per symbol")
    nf = len(bits) // m["bits"]
    T = sym_us / 1e6
    frames = max(1, math.ceil((seconds - (LEAD_SYMS + TAIL_SYMS) * T) / (nf * T)))
    return dict(kind="psk", version=VERSION, modulation=mod, bps=m["bits"] / T, baud=1 / T, sym_us=sym_us,
                bits_per_symbol=m["bits"], fc_hz=float(fc), shape=shape, beta=beta, alpha=alpha,
                header=L.HDR, gap=L.GAP, prbs_order=order, frame_bits=len(bits), payload_bits=len(pl),
                frame_symbols=nf, frame_s=nf * T, frames=frames, lead_symbols=LEAD_SYMS, tail_symbols=TAIL_SYMS,
                n_symbols=LEAD_SYMS + frames * nf + TAIL_SYMS, total_s=(LEAD_SYMS + frames * nf + TAIL_SYMS) * T,
                cycles_per_symbol=fc * T, payload_bps=m["bits"] / T * len(pl) / len(bits))


def frame_bits(p):
    return L.frame(p["bps"])[0]


def symbols(p):
    """complex symbols of the whole transmission (lead-in, frames, tail)"""
    bits = frame_bits(p)
    mod = p["modulation"]
    out = [1.0 + 0j] * LEAD_SYMS
    ph = 0.0
    for _ in range(p["frames"]):
        if mod == "dbpsk":
            for b in bits:
                ph += np.pi * b
                out.append(np.exp(1j * ph))
        elif mod == "bpsk":
            out += [complex(1 - 2 * b) for b in bits]
        elif mod == "dqpsk":
            for i in range(0, len(bits), 2):
                ph += DQPSK_STEP[(bits[i], bits[i + 1])]
                out.append(np.exp(1j * ph))
        else:
            out += [complex(ASK4_LEVEL[(bits[i], bits[i + 1])]) for i in range(0, len(bits), 2)]
    last = out[-1] if mod in ("dbpsk", "dqpsk") else 1.0 + 0j
    out += [last] * TAIL_SYMS
    a = np.array(out)
    if mod in ("dbpsk", "dqpsk"):
        a = np.round(a.real) + 1j * np.round(a.imag)          # exact +-1 / +-j
    return a


PTS = 32                              # pulse table points per symbol (firmware uses the same table)


def pulse_table(p):
    """(span in symbols, table of span*PTS + 1 values of p(tau), tau from -span/2 to +span/2)"""
    shape = p["shape"]
    if shape == "rect":
        return 1, np.ones(PTS + 1)
    if shape == "rc":
        b = min(max(p["beta"], 1e-3), 1.0)
        span = 2
        tau = np.linspace(-1, 1, span * PTS + 1)
        a = np.abs(tau)
        v = np.where(a <= (1 - b) / 2, 1.0, np.where(a >= (1 + b) / 2, 0.0,
                                                   0.5 * (1 + np.cos(np.pi * (a - (1 - b) / 2) / b))))
        return span, v
    if shape == "rrc":
        al, span = p["alpha"], 6
        tau = np.linspace(-span / 2, span / 2, span * PTS + 1)
        v = np.empty_like(tau)
        for i, t in enumerate(tau):
            if abs(t) < 1e-9:
                v[i] = 1 - al + 4 * al / np.pi
            elif abs(abs(t) - 1 / (4 * al)) < 1e-9:
                v[i] = al / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * al)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * al)))
            else:
                v[i] = (np.sin(np.pi * t * (1 - al)) + 4 * al * t * np.cos(np.pi * t * (1 + al))) / \
                       (np.pi * t * (1 - (4 * al * t) ** 2))
        return span, v * np.hanning(len(v)) ** 0.25 / (1 - al + 4 * al / np.pi)
    raise ValueError(shape)


def pulse_at(tau, span, table):
    """the firmware's table interpolation; zero outside [-span/2, span/2)"""
    u = (tau + span / 2) * PTS
    i = np.floor(u).astype(int)
    f = u - i
    ok = (u >= 0) & (i < span * PTS)
    i = np.clip(i, 0, span * PTS - 1)
    return np.where(ok, table[i] * (1 - f) + table[i + 1] * f, 0.0)


def baseband(p, a, span, table, t_tx):
    """b(t) at transmitter times t (s from the first symbol), exactly as the firmware computes it"""
    T = p["sym_us"] / 1e6
    u = t_tx / T
    k0 = np.floor(u).astype(int)
    b = np.zeros(len(t_tx), complex)
    h = span // 2 + 1
    for dk in range(-h, h + 1):
        k = k0 + dk
        ok = (k >= 0) & (k < len(a))
        tau = u - k - 0.5
        b += np.where(ok, a[np.clip(k, 0, len(a) - 1)], 0) * pulse_at(tau, span, table)
    return b


def peak_norm(p, a=None):
    """peak |Re{b e^{jwt}}| of the whole transmission (the waveform is divided by it: peak = 1)"""
    a = symbols(p) if a is None else a
    span, table = pulse_table(p)
    t = np.arange(0, p["total_s"], 1 / 4000.0)
    return float(np.max(np.abs(baseband(p, a, span, table, t))))


def waveform(p, t_tx, a=None, norm=None):
    """the transmitter output (relative, peak 1) at TX times t (s after go)"""
    a = symbols(p) if a is None else a
    span, table = pulse_table(p)
    norm = norm or peak_norm(p, a)
    b = baseband(p, a, span, table, t_tx)
    return np.real(b * np.exp(2j * np.pi * p["fc_hz"] * t_tx)) / norm


# ================================================================ receiver
def _power_line(x, fs, f, ppm):
    n = len(x); N = 16 * n
    sp = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n), N)); fq = np.fft.rfftfreq(N, 1 / fs)
    k = np.flatnonzero(np.abs(fq - f) <= f * ppm * 1e-6 * 1.2)
    i = k[np.argmax(sp[k])]
    if 0 < i < len(sp) - 1:                                     # parabolic interpolation
        a, b, c = np.log(sp[i - 1:i + 2] + 1e-30)
        d = 0.5 * (a - c) / (a - 2 * b + c)
    else:
        d = 0.0
    fpk = fq[i] + d * (fq[1] - fq[0])
    floor = np.median(sp[(np.abs(fq - f) > 1) & (np.abs(fq - f) < 3)])
    return fpk, float(20 * np.log10(sp[i] / floor))


def coarse_clock(x, fs, p):
    power = 4 if p["modulation"] == "dqpsk" else 2
    xn = (x - x.mean()) / (np.std(x) + 1e-12)
    f = power * p["fc_hz"]
    fpk, snr = _power_line(xn ** power, fs, f, CLOCK_PPM)
    return f / fpk, snr                                          # scale = 1 + ppm (tone seen at f / scale)


def matched(x, t, p, scale):
    """complex down-conversion at fc/scale and the filter matched to the pulse"""
    z = 2 * x * np.exp(-2j * np.pi * p["fc_hz"] / scale * t)
    span, table = pulse_table(p)
    T = p["sym_us"] / 1e6 * scale
    fs = 1 / (t[1] - t[0])
    n = int(np.ceil(span / 2 * T * fs))
    tau = np.arange(-n, n + 1) / fs / T
    h = pulse_at(tau, span, table)
    h = h / h.sum()
    return np.convolve(z, h, mode="same")


def sample(y, t, times):
    return np.interp(times, t, y.real, left=np.nan, right=np.nan) + 1j * np.interp(times, t, y.imag, left=np.nan, right=np.nan)


def _hdr_ref(p):
    bits = frame_bits(p)
    mod = p["modulation"]
    if mod == "dbpsk":
        return np.array([2.0 * b - 1 for b in L.HDR])
    if mod == "dqpsk":
        return np.array([np.exp(1j * DQPSK_STEP[(bits[i], bits[i + 1])]) for i in range(0, 10, 2)])
    if mod == "bpsk":
        return np.array([1.0 - 2 * b for b in L.HDR], complex)
    return np.array([ASK4_LEVEL[(bits[i], bits[i + 1])] for i in range(0, 10, 2)], complex)


def soft(y, p):
    """per-symbol statistic used for sync: differential products (normalised) or the raw symbols"""
    if p["modulation"] in ("dbpsk", "dqpsk"):
        d = y[1:] * np.conj(y[:-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            d = d / (np.abs(d) + 1e-30)
        d = np.r_[0, d]
        return -d.real if p["modulation"] == "dbpsk" else d
    return y


def header_corr(s, p, ref, pos):
    """header correlation at symbol positions pos (vectorised); ref from _hdr_ref"""
    H = len(ref)
    idx = pos[:, None] + np.arange(H)[None, :]
    ok = (idx >= 1) & (idx < len(s))
    idx = np.clip(idx, 0, len(s) - 1)
    w = np.where(ok, s[idx], np.nan)
    mod = p["modulation"]
    if mod == "dbpsk":
        return np.nanmean(w * ref, 1)
    if mod == "dqpsk":
        return np.real(np.nanmean(w * np.conj(ref), 1))
    if mod == "bpsk":
        return np.abs(np.nansum(w * ref, 1)) / np.nansum(np.abs(w), 1)
    return np.abs(np.nansum(w * ref, 1)) / np.sqrt(np.nansum(np.abs(w) ** 2, 1) * np.sum(np.abs(ref) ** 2))


def sync(y_of, p, scales, phases, sc0=None):
    """best (score, scale, phase, frame position) over the grid; y_of(scale, phase) -> symbol samples"""
    ref = _hdr_ref(p)
    nf = p["frame_symbols"]
    best = (-9, None, None, None, None)
    for sc in scales:
        for ph in phases:
            y = y_of(sc, ph)
            s = soft(y, p)
            n = len(s)
            pos = np.arange(0, n - len(ref))
            r = header_corr(s, p, ref, pos)
            r = np.where(np.isfinite(r), r, -9)
            # fold over frames: mean over all complete headers at the same frame phase
            q = np.full(nf, -9.0)
            for f0 in range(min(nf, len(r))):
                v = r[f0::nf]
                v = v[v > -9]
                if len(v):
                    q[f0] = v.mean()
            j = int(np.argmax(q))
            q[j] -= 1e-4 * abs(sc - sc0) * 1e4 if sc0 else 0      # tie-break toward the power-line clock
            if q[j] > best[0]:
                best = (float(q[j]), sc, ph, j, y)
    return best


def detect(y, p, start):
    """bits of one frame starting at symbol index start (y: symbol samples, possibly equalised)"""
    mod = p["modulation"]
    nf = p["frame_symbols"]
    seg = y[start:start + nf]
    if mod == "dbpsk":
        d = seg * np.conj(y[start - 1:start + nf - 1])
        bits = (d.real < 0).astype(int)
        ref = np.cumprod(np.where(bits == 1, -1.0, 1.0)) + 0j
    elif mod == "dqpsk":
        d = seg * np.conj(y[start - 1:start + nf - 1])
        q = np.mod(np.round(np.nan_to_num(np.angle(d)) / (np.pi / 2)), 4).astype(int)   # 0, 1(+pi/2), 2(pi), 3(-pi/2)
        inv = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
        bits = np.array([b for k in q for b in inv[k]])
        ref = np.exp(1j * np.cumsum(q * np.pi / 2))
    elif mod == "bpsk":
        a_h = np.array([1.0 - 2 * b for b in L.HDR])
        th = np.angle(np.sum(seg[:len(a_h)] * a_h))
        bits, ref = [], []
        for v in seg:
            w = v * np.exp(-1j * th)
            a_ = 1.0 if w.real >= 0 else -1.0
            th += 0.1 * np.angle(w * a_)
            bits.append(int(a_ < 0)); ref.append(a_ * np.exp(1j * th))
        bits, ref = np.array(bits), np.array(ref)
    else:
        allb = frame_bits(p)
        a_h = np.array([ASK4_LEVEL[(allb[i], allb[i + 1])] for i in range(0, 10, 2)])
        g = np.sum(seg[:5] * a_h) / np.sum(a_h ** 2)
        levels = np.array([-1, -1 / 3, 1 / 3, 1])
        inv = {-1.0: (0, 0), -1 / 3: (0, 1), 1 / 3: (1, 1), 1.0: (1, 0)}
        bits, ref = [], []
        for v in seg:
            w = (v / g).real
            lv = levels[np.argmin(np.abs(levels - w))]
            bits += list(inv[lv]); ref.append(lv * g)
            g = g + 0.05 * (v - lv * g) * lv / max(lv * lv, 1e-9)
        bits, ref = np.array(bits), np.array(ref)
    return np.asarray(bits)[:p["frame_bits"]], np.asarray(ref)


def phase_align(ref, seg):
    """rotate/scale a coherent reference chain onto the received symbols"""
    g = np.sum(seg * np.conj(ref)) / max(np.sum(np.abs(ref) ** 2), 1e-30)
    return ref * g


def equalise(Y, targets, L_taps):
    """LS fractionally spaced FFE: Y (N, taps) regressors, targets (N,) -> weights"""
    ok = np.isfinite(targets) & np.all(np.isfinite(Y), 1)
    A, b = Y[ok], targets[ok]
    G = A.conj().T @ A
    lam = 1e-3 * np.real(np.trace(G)) / len(G)
    return np.linalg.solve(G + lam * np.eye(len(G)), A.conj().T @ b)


def decode(x, t, p, eq=False, truth=None):
    """x [mV] at watch times t (nominal fs). Returns a report dict (bits, frames, errors vs the PRBS, quality)."""
    x = np.asarray(x, float) - np.mean(x)
    t = np.asarray(t, float)
    fs = 1 / (t[1] - t[0])
    _, pl, _ = L.frame(p["bps"])
    truth = pl if truth is None else truth
    T = p["sym_us"] / 1e6
    sc0, line_db = coarse_clock(x, fs, p)
    dur = t[-1] - t[0]
    step_ppm = max(T / 8 / dur * 1e6, 10.0)
    scales = sc0 + np.arange(-CLOCK_WINDOW_PPM, CLOCK_WINDOW_PPM + 1e-9, step_ppm) * 1e-6   # power line ~+-150 ppm
    yc = matched(x, t, p, sc0)
    nph = max(8, int(round(T * fs * 2)))
    phases = np.arange(nph) / nph * T

    def y_of(sc, ph):
        times = t[0] + ph * sc + np.arange(int(dur / (T * sc)) + 1) * T * sc
        return sample(yc, t, times)

    score, sc, ph, f0, y = sync(y_of, p, scales, phases, sc0)
    if sc is not None:                                           # fine pass around the best grid point
        fine_s = sc + np.arange(-1, 1.01, 0.125) * step_ppm * 1e-6
        fine_p = ph + np.arange(-1, 1.01, 0.25) * (T / nph)
        score, sc, ph, f0, y = sync(y_of, p, fine_s, fine_p, sc0)
    nf = p["frame_symbols"]
    ref_h = _hdr_ref(p)
    rep = dict(decoder=VERSION, rules_sha=RULES_SHA, modulation=p["modulation"], bps=p["bps"], baud=p["baud"], shape=p["shape"],
               eq=bool(eq), clock_ppm=(sc - 1) * 1e6 if sc else None, coarse_clock_ppm=(sc0 - 1) * 1e6,
               power_line_db=line_db, sync_score=score, frames=[])
    if sc is None:
        rep.update(reason="no sync")
        return rep
    starts = [s for s in range(f0, len(y) - nf + 1, nf) if s >= 1 and np.all(np.isfinite(y[s - 1:s + nf]))]

    def run(yv):
        frames, chain = [], np.full(len(yv), np.nan + 0j)
        s_ = soft(yv, p)
        for m, s in enumerate(starts):
            if not np.all(np.isfinite(yv[s - 1:s + nf])):
                continue
            r = float(header_corr(s_, p, ref_h, np.array([s]))[0])
            bits, ref = detect(yv, p, s)
            pay = bits[len(L.HDR):len(L.HDR) + len(truth)]
            err = int(np.sum(pay != np.array(truth)))
            known = np.r_[bits[:len(L.HDR)], bits[-len(L.GAP):]]          # header + gap: frame structure, not payload
            known_err = int(np.sum(known != np.array(L.HDR + L.GAP)))
            valid = bool(r >= R_MIN and score >= G_MIN and known_err <= KNOWN_ERR_MAX)
            frames.append(dict(m=m, start_symbol=int(s), r=r, known_errors=known_err, valid=valid, bits=len(truth), errors=err,
                               decided=[int(v_) for v_ in bits],
                               err_pos=[int(i) for i in np.flatnonzero(pay != np.array(truth))],
                               t_start_s=float(t[0] + (ph + s * T) * sc)))
            if valid:
                chain[s:s + nf] = phase_align(ref, yv[s:s + nf])
        return frames, chain

    frames, chain = run(y)
    yeq = y
    if eq and any(f["valid"] for f in frames):
        Lt = max(2, int(np.ceil(EQ_SPAN_S / (T / 2))))
        offs = np.arange(-Lt, Lt + 1) * T / 2 * sc
        base = t[0] + ph * sc + np.arange(len(y)) * T * sc
        Y = np.vstack([sample(yc, t, base + o) for o in offs]).T
        for _ in range(EQ_ITER):
            tgt = chain / np.where(np.isfinite(chain), np.abs(chain), 1)       # unit targets, phase from the chain
            w = equalise(Y, tgt, Lt)
            yeq = Y @ w
            frames, chain = run(yeq)
            if not any(f["valid"] for f in frames):
                break
    # quality: symbol SNR from the coherent chains of valid frames
    ok = np.isfinite(chain)
    if ok.any():
        e = yeq[ok] - chain[ok]
        snr = 10 * np.log10(np.mean(np.abs(chain[ok]) ** 2) / np.mean(np.abs(e) ** 2))
    else:
        snr = None
    v = [f for f in frames if f["valid"]]
    rep.update(frames=frames, frames_complete=len(frames), frames_valid=len(v),
               bits=sum(f["bits"] for f in v), errors=sum(f["errors"] for f in v), symbol_snr_db=snr,
               reason="" if v else ("no valid frame" if frames else "no complete frame"))
    return rep


# ================================================================ psk-x3 receiver "x3-wl" (DBPSK), added 2026-11-06
# (equaliser inputs: wl_regressors - the signal and its copy counter-rotated by the image rotation)
# Why: the x2 receiver (complex matched filter + differential detection) cannot separate the carrier's folded image
# when the carrier has few cycles per symbol (bench test on a PERFECT synthetic capture: 0 frames at 64 and 100 bps on
# 16 Hz, 0.25 and 0.16 cycles/symbol). x3-wl works on the real signal (widely linear), so the image is not lost:
#   1. clock: power line, as x2
#   2. acquisition: correlation with the KNOWN header waveform (gap 00 + header = 13 known bits -> 14 symbols),
#      transmitter pulse through the FIXED channel model below, both carrier phases (magnitude), folded over the frame
#      period; grid over clock (+-CLOCK_WINDOW_PPM) and every watch sample
#   3. cyclic widely-linear least-squares equaliser (wl_regressors) at T/2 taps; start: the known symbols (gap +
#      header, all frames share one phase because every DBPSK frame has an even number of flips) and the decisions of
#      the x2-diff receiver's valid frames, if any (never the payload truth); then WL_ITER decision-directed refits
#      while widening to +-EQ_SPAN_S; clock refined by that error; 5 alignment hypotheses (+-2 symbols) chosen by the
#      header fit; finally each frame decoded by a filter fitted without that frame
#   4. differential decisions, the same validity rules as x2 (R_MIN, G_MIN, KNOWN_ERR_MAX)
WL_CHANNEL = ("linphase", 4, 1.0, 27.5)     # channel_est parametric fit of the 10 real OOK/FSK recordings (fixed)
WL_ITER = 3
WL_RIDGE = 1e-2
WL_CLOCK_REFINE_PPM = 150
WL_HYPOTHESES = 5                    # acquisition phases tried (each with +-1, +-2 symbol shifts)


def _known_symbols(p):
    """relative symbol values of the known span: the symbol before the gap, the gap (00) and the header (13 bits)"""
    bits = list(L.GAP) + list(L.HDR)
    a = [1.0]
    for b in bits:
        a.append(a[-1] * (-1.0 if b else 1.0))
    return np.array(a)                      # 14 values: symbols s-3 .. s+10 for a frame starting at symbol s


def _wl_template(p, sc, fs):
    """complex template (Re and Im carrier phases) of the known span through the fixed channel, at watch spacing"""
    import channel_est as C
    T = p["sym_us"] / 1e6
    span, table = pulse_table(p)
    a = _known_symbols(p)
    k0 = -3                                  # first known symbol relative to the frame start
    pre, post = (span / 2 + 0.5) * T + 0.15, (span / 2 + 0.5) * T + 0.15
    tt = np.arange(-pre + k0 * T, (k0 + len(a)) * T + post, 1 / fs / sc)    # TX time relative to the frame start
    b = baseband(p, a, span, table, tt - k0 * T)                           # symbols placed from k0
    out = []
    for car in (np.cos, np.sin):
        w = np.real(b) * car(2 * np.pi * p["fc_hz"] * tt)
        out.append(C.model_filter(np.r_[np.zeros(200), w], WL_CHANNEL[2], WL_CHANNEL[3], WL_CHANNEL[1], WL_CHANNEL[0])[200:])
    return out[0] + 1j * out[1], tt[0]       # template, TX time (rel. frame start) of its first sample


def _wl_acquire(x, fs, p, scales, top=1):
    """best (score, scale, start sample, t_first); with top > 1 a list of the best distinct frame phases"""
    nf, T = p["frame_symbols"], p["sym_us"] / 1e6
    best = (-1.0, None, None, None)
    allc = []
    xe = np.r_[0, np.cumsum(x ** 2)]
    for sc in scales:
        h, t_first = _wl_template(p, sc, fs)
        L_ = len(h)
        if L_ >= len(x):
            continue
        c = np.abs(np.correlate(x, h.real, "valid") + 1j * np.correlate(x, h.imag, "valid"))
        en = np.sqrt(np.maximum(xe[L_:] - xe[:-L_], 1e-30))[:len(c)]
        rho = c / (en * np.sqrt(np.sum(np.abs(h) ** 2) / 2))
        Pw = nf * T * sc * fs                                   # frame period in watch samples
        n_fr = int((len(rho) - 1) // Pw) + 1
        for t0 in range(int(np.ceil(Pw))):
            idx = np.round(t0 + np.arange(n_fr) * Pw).astype(int)
            idx = idx[idx < len(rho)]
            if not len(idx):
                continue
            q = rho[idx].mean()
            allc.append((float(q), sc, int(t0), t_first))
            if q > best[0]:
                best = (float(q), sc, int(t0), t_first)
    if top <= 1:
        return best
    allc.sort(key=lambda c: -c[0])
    out = []
    for c in allc:                                      # distinct phases (>= half a symbol apart)
        if all(abs(c[2] - o[2]) >= 0.5 * T * fs for o in out):
            out.append(c)
        if len(out) >= top:
            break
    return out


def wl_regressors(zw, ns, p):
    """[Re z, Im z, Re z r_n, Im z r_n, 1] with r_n = exp(j 4 pi fc T n): the folded image's phase turns by 4 pi fc T
    per symbol, so a fixed widely-linear filter alone cannot cancel it unless 2 fc T is a whole number; the rotated
    copy makes the filter periodic in the symbol index (a frequency-shift filter for the cyclostationary signal)"""
    c = p["fc_hz"] * p["sym_us"] / 1e6
    rot = np.exp(1j * 4 * np.pi * c * np.asarray(ns, float))[:, None]
    zr = zw * rot
    return np.hstack([zw.real, zw.imag, zr.real, zr.imag, np.ones((len(ns), 1))])


def decode_wl(x, t, p, truth=None):
    """psk-x3 'x3-wl' receiver for DBPSK. Same report fields as decode()."""
    if p["modulation"] != "dbpsk":
        raise ValueError("x3-wl is for DBPSK")
    x = np.asarray(x, float) - np.mean(x)
    t = np.asarray(t, float) - t[0]
    fs = 1 / (t[1] - t[0])
    _, pl, _ = L.frame(p["bps"])
    truth = pl if truth is None else truth
    T = p["sym_us"] / 1e6
    nf = p["frame_symbols"]
    sc0, line_db = coarse_clock(x, fs, p)
    dur = t[-1]
    step = max(T / 8 / dur * 1e6, 10.0)
    rep = dict(decoder=VERSION, receiver="x3-wl", rules_sha=RULES_SHA, modulation="dbpsk", bps=p["bps"], baud=p["baud"],
               shape=p["shape"], eq=True, coarse_clock_ppm=(sc0 - 1) * 1e6, power_line_db=line_db, frames=[])
    scales = sc0 + np.arange(-CLOCK_WINDOW_PPM, CLOCK_WINDOW_PPM + 1e-9, step) * 1e-6
    q, sc, i0, t_first = _wl_acquire(x, fs, p, scales)
    if sc is not None:                                           # finer clock around the best
        q2, sc2, i02, tf2 = _wl_acquire(x, fs, p, sc + np.arange(-1, 1.01, 0.25) * step * 1e-6)
        if q2 >= q:
            q, sc, i0, t_first = q2, sc2, i02, tf2
    if sc is None:
        rep.update(reason="no acquisition", sync_score=None, acquisition_score=q)
        return rep
    known = _known_symbols(p)
    z_cache = {}
    tf0 = [i0 / fs - t_first * sc]                    # watch time of frame 0's start-symbol boundary (mutable)
    t_frame0_of = lambda: tf0[0]
    x2 = decode(x, t, p, eq=True)                     # seed: the x2-diff receiver's valid frames
    x2_frames = [f for f in x2.get("frames", []) if f["valid"]]

    def tgt_of(ns):
        rel = np.mod(ns, nf)
        tg = np.full(len(ns), np.nan)
        for jj, v in enumerate(known):
            tg[np.mod(rel - (jj - 3), nf) == 0] = v
        return tg

    def build(scx, Lh):
        """symbol grid, regressors with half-span Lh (T/2 taps), targets of the known span"""
        t_frame0 = tf0[0]
        n_lo = int(np.ceil((t[0] - t_frame0) / (T * scx) - 0.5)) + 1
        n_hi = int(np.floor((t[-1] - t_frame0) / (T * scx) - 0.5)) - 1
        ns = np.arange(n_lo, n_hi + 1)
        tc = t_frame0 + (ns + 0.5) * T * scx
        if scx not in z_cache:
            z_cache[scx] = 2 * x * np.exp(-2j * np.pi * p["fc_hz"] * t / scx)
        z = z_cache[scx]
        offs = np.arange(-Lh, Lh + 1) * (T / 2) * scx
        tw = tc[:, None] + offs[None, :]
        inside = (tw[:, 0] >= t[0]) & (tw[:, -1] <= t[-1])
        zw = np.interp(tw, t, z.real) + 1j * np.interp(tw, t, z.imag)
        X = wl_regressors(zw, ns, p)
        rel = np.mod(ns, nf)
        tgt = np.full(len(ns), np.nan)
        for jj, v in enumerate(known):                  # known span: positions nf-3, nf-2, nf-1, 0 .. 10
            tgt[np.mod(rel - (jj - 3), nf) == 0] = v
        return ns, X, tgt, inside

    def fit(X, mask, target):
        A = X[mask]; G = A.T @ A
        return np.linalg.solve(G + WL_RIDGE * np.trace(G) / len(G) * np.eye(len(G)), A.T @ target[mask])

    Lfull = max(1, int(np.ceil(EQ_SPAN_S / (T / 2))))

    def seed(dec, ns, scx):
        """decisions of the x2-diff receiver's valid frames (if any) as the starting point - never the payload truth"""
        if not x2_frames:
            return dec, 0
        tc = t_frame0_of() + (ns + 0.5) * T * scx
        n_seeded = 0
        a_prev = known[2]                                    # symbol s-1 (last gap symbol) of the known span
        for f in x2_frames:
            k = int(np.argmin(np.abs(tc - f["t_start_s"])))
            if abs(tc[k] - f["t_start_s"]) > T * scx / 2 or k < 1 or k + nf > len(ns):
                continue
            a = a_prev
            for jj, b_ in enumerate(f["decided"]):
                a = -a if b_ else a
                if not np.isfinite(tgt_of(ns)[k + jj]):
                    dec[k + jj] = a
            n_seeded += 1
        return dec, n_seeded

    def train(scx):
        """stage A: short equaliser on the known span (+ x2 decisions as seed); stage B: decision-directed,
        widening to the full span"""
        ns, Xs, tgt, ins = build(scx, 1)
        tr = np.isfinite(tgt) & ins
        if tr.sum() < 2 * Xs.shape[1]:
            return None
        dec = np.where(np.isfinite(tgt), tgt, np.sign(Xs @ fit(Xs, tr, tgt)))
        dec, n_seed = seed(dec, ns, scx)
        for Lh in sorted({max(1, Lfull // 2), Lfull}):
            ns, X, tgt, ins = build(scx, Lh)
            for _ in range(WL_ITER):
                dec = np.where(np.isfinite(tgt), tgt, np.sign(X @ fit(X, ins, dec)))
        w = fit(X, ins, dec)
        est = X @ w
        mse = float(np.mean((est[ins] - dec[ins]) ** 2))
        return dict(sc=scx, ns=ns, X=X, tgt=tgt, inside=ins, train=ins, dec=dec, w=w, mse=mse, Lh=Lfull, seeded=n_seed)

    def header_fit(c):
        """mean header correlation of the (in-sample) equalised output: picks the frame alignment"""
        if c is None:
            return -9.0
        est = c["X"] @ c["w"]
        hdr_ = np.array([2.0 * b_ - 1 for b_ in L.HDR]); rs = []
        for m in range(int(np.ceil((c["ns"][0] + 1) / nf)), int(np.floor((c["ns"][-1] - nf + 1) / nf)) + 1):
            i = m * nf - c["ns"][0]
            if i < 1 or i + len(hdr_) > len(est):
                continue
            e_ = est[i - 1:i + len(hdr_)]
            rs.append(np.mean(-(e_[1:] * e_[:-1]) / (np.abs(e_[1:] * e_[:-1]) + 1e-30) * hdr_))
        return float(np.mean(rs)) if rs else -9.0
    # frame alignment: the top acquisition phases, each also shifted by +-1, +-2 symbols (at few carrier cycles per
    # symbol a shifted position can match the known span almost as well)
    hyps = []
    for qq, scq, i0q, tfq in _wl_acquire(x, fs, p, [sc], top=WL_HYPOTHESES):
        for k in (-2, -1, 0, 1, 2):
            hyps.append((qq, i0q / fs - tfq * sc + k * T * sc))
    scored = []
    for qq, t0h in hyps:
        tf0[0] = t0h
        scored.append((header_fit(train(sc)), t0h, qq))
    hf, tf0[0], q = max(scored, key=lambda h: h[0])
    rep["alignment_hypotheses"] = len(hyps)
    t_frame0 = tf0[0]
    # clock refinement by the decision-directed error (the carrier phase drifts with any clock error)
    cands = [train(sc + d * 1e-6) for d in np.arange(-WL_CLOCK_REFINE_PPM, WL_CLOCK_REFINE_PPM + 0.1, 10)]
    cands = [c for c in cands if c]
    if not cands:
        rep.update(reason="too few known symbols for the equaliser", sync_score=q, acquisition_score=q)
        return rep
    B_ = min(cands, key=lambda c: c["mse"])
    sc = B_["sc"]
    ns, X, tgt, inside, dec = B_["ns"], B_["X"], B_["tgt"], B_["inside"], B_["dec"]
    rep.update(acquisition_score=q, clock_ppm=(sc - 1) * 1e6, eq_taps=int(X.shape[1]), seeded_frames=B_["seeded"],
               x2_valid_frames=len(x2_frames))
    # stage C: leave-one-frame-out - each frame is equalised (and judged) by a filter that never saw it
    hdr = np.array([2.0 * b - 1 for b in L.HDR])
    frames, err2, pow2 = [], [], []
    m_lo = int(np.ceil((ns[0] + 1) / nf)); m_hi = int(np.floor((ns[-1] - nf + 1) / nf))
    for m in range(m_lo, m_hi + 1):
        s0 = m * nf
        i = s0 - ns[0]
        if i < 1 or i + nf > len(ns) or not inside[i - 1:i + nf].all():
            continue
        own = np.zeros(len(ns), bool); own[i - 1:i + nf] = True
        w = fit(X, inside & ~own, dec)
        e_ = X[i - 1:i + nf] @ w
        soft = -(e_[1:] * e_[:-1]) / (np.abs(e_[1:] * e_[:-1]) + 1e-30)
        bits = (soft > 0).astype(int)
        r = float(np.mean(soft[:len(hdr)] * hdr))
        pay = bits[len(L.HDR):len(L.HDR) + len(truth)]
        err = int(np.sum(pay != np.array(truth)))
        kn = np.r_[bits[:len(L.HDR)], bits[-len(L.GAP):]]
        kerr = int(np.sum(kn != np.array(L.HDR + L.GAP)))
        frames.append(dict(m=len(frames), start_symbol=int(s0), r=r, known_errors=kerr, bits=len(truth), errors=err,
                           err_pos=[int(k) for k in np.flatnonzero(pay != np.array(truth))],
                           t_start_s=float(t_frame0 + (s0 + 0.5) * T * sc)))
        err2.append(np.mean((np.abs(e_) - 1) ** 2)); pow2.append(1.0)
    score = float(np.mean([f["r"] for f in frames])) if frames else None
    for f in frames:
        f["valid"] = bool(f["r"] >= R_MIN and score is not None and score >= G_MIN and f["known_errors"] <= KNOWN_ERR_MAX)
    v = [f for f in frames if f["valid"]]
    snr = 10 * np.log10(1.0 / max(np.mean(err2), 1e-30)) if frames else None
    rep.update(frames=frames, frames_complete=len(frames), frames_valid=len(v), sync_score=score,
               bits=sum(f["bits"] for f in v), errors=sum(f["errors"] for f in v), symbol_snr_db=snr,
               reason="" if v else ("no valid frame" if frames else "no complete frame"))
    return rep
    # watch time of frame m's start symbol boundary: the template's first sample is at TX time t_first (rel. frame start)
    t_frame0 = i0 / fs - t_first * sc
    Pw_s = nf * T * sc
    # symbols on the grid: symbol n (global, relative to frame 0's start symbol) centre at t_frame0 + (n + 0.5) T sc
    n_lo = int(np.ceil((t[0] - t_frame0) / (T * sc) - 0.5)) + 1
    n_hi = int(np.floor((t[-1] - t_frame0) / (T * sc) - 0.5)) - 1
    ns = np.arange(n_lo, n_hi + 1)
    tc = t_frame0 + (ns + 0.5) * T * sc
    z = 2 * x * np.exp(-2j * np.pi * p["fc_hz"] * t / sc)
    Lh = max(1, int(np.ceil(EQ_SPAN_S / (T / 2))))
    offs = np.arange(-Lh, Lh + 1) * (T / 2) * sc
    tw = tc[:, None] + offs[None, :]
    inside = (tw[:, 0] >= t[0]) & (tw[:, -1] <= t[-1])
    zw = np.interp(tw, t, z.real) + 1j * np.interp(tw, t, z.imag)
    X = wl_regressors(zw, ns, p)
    known = _known_symbols(p)
    rel = np.mod(ns, nf)                                         # position within its frame (0 = start symbol)
    tgt = np.full(len(ns), np.nan)
    for j, v in enumerate(known):                                # known span = positions nf-3 .. nf-1 and 0 .. 10
        tgt[np.mod(rel - (j - 3), nf) == 0] = v
    train = np.isfinite(tgt) & inside

    def fit(mask, target):
        A = X[mask]; G = A.T @ A
        return np.linalg.solve(G + WL_RIDGE * np.trace(G) / len(G) * np.eye(len(G)), A.T @ target[mask])
    if train.sum() < X.shape[1] // 2:
        rep.update(reason=f"too few training symbols ({int(train.sum())})", sync_score=q)
        return rep
    w = fit(train, tgt)
    for _ in range(WL_ITER):
        est = X @ w
        dec = np.where(np.isfinite(tgt), tgt, np.sign(est))
        w = fit(inside, dec)
    est = X @ w
    hdr = np.array([2.0 * b - 1 for b in L.HDR])
    frames, snr_e, snr_s = [], [], []
    m_lo = int(np.ceil((ns[0] + 1) / nf)); m_hi = int(np.floor((ns[-1] - nf + 1) / nf))
    for m in range(m_lo, m_hi + 1):
        s = m * nf
        i = s - ns[0]
        if i < 1 or i + nf > len(ns) or not inside[i - 1:i + nf].all():
            continue
        e_ = est[i - 1:i + nf]
        soft = -(e_[1:] * e_[:-1]) / (np.abs(e_[1:] * e_[:-1]) + 1e-30)
        bits = (soft > 0).astype(int)
        r = float(np.mean(soft[:len(hdr)] * hdr))
        pay = bits[len(L.HDR):len(L.HDR) + len(truth)]
        err = int(np.sum(pay != np.array(truth)))
        kn = np.r_[bits[:len(L.HDR)], bits[-len(L.GAP):]]
        kerr = int(np.sum(kn != np.array(L.HDR + L.GAP)))
        frames.append(dict(m=len(frames), start_symbol=int(s), r=r, known_errors=kerr, bits=len(truth), errors=err,
                           err_pos=[int(k) for k in np.flatnonzero(pay != np.array(truth))],
                           t_start_s=float(t_frame0 + (s + 0.5) * T * sc)))
        snr_e.append(np.mean((np.abs(e_) - 1) ** 2)); snr_s.append(np.mean(np.abs(e_) ** 2))
    score = float(np.mean([f["r"] for f in frames])) if frames else None
    for f in frames:
        f["valid"] = bool(f["r"] >= R_MIN and score is not None and score >= G_MIN and f["known_errors"] <= KNOWN_ERR_MAX)
    v = [f for f in frames if f["valid"]]
    snr = 10 * np.log10(np.mean(snr_s) / max(np.mean(snr_e), 1e-30)) if frames else None
    rep.update(frames=frames, frames_complete=len(frames), frames_valid=len(v), sync_score=score,
               bits=sum(f["bits"] for f in v), errors=sum(f["errors"] for f in v), symbol_snr_db=snr,
               reason="" if v else ("no valid frame" if frames else "no complete frame"))
    return rep
