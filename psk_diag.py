"""Known-transmission ("genie") diagnosis of a psk recording (added 2026-11-06). DIAGNOSTIC, NOT A RECEIVER.

It uses what the receiver may not: the exact transmitted symbols and waveform. That separates the causes of a failure:
  1. locate: the known waveform, passed through the watch channel fitted from the real OOK/FSK recordings, is
     cross-correlated with the recording over every clock and start allowed by the 'Recorded on' minute
     -> genie start/clock, normalised correlation rho, and exactly how many complete frames were transmitted while
        the watch recorded (the denominator for frame-sync success, also when the receiver finds nothing)
  2. ideal linear receiver: a widely-linear least-squares filter (real and imaginary parts of the recording, carrier
     removed with the KNOWN phase, +-WIN_S around each symbol) trained on the first half of the symbols with their
     true values, TESTED on the second half -> genie symbol SNR and genie BER on unseen symbols
Classification of a recording (thresholds fixed here, before any DBPSK measurement):
  - 'not usable'                 recording too short / flat / clipped (ecg_ber.usability)
  - 'transmission not found'     rho < RHO_MIN: the signal is not in the recording (contact, level, timing)
  - 'channel-limited'            genie BER >= GENIE_BER_LIMIT: even the ideal linear receiver with perfect knowledge
                                 fails - a limit of the watch channel FOR THIS WAVEFORM (rate, carrier, pulse)
  - 'receiver-limited: sync'     genie fine, but the receiver recovers < 95 % of the transmitted frames
  - 'receiver-limited: false lock'  genie fine, the receiver's frames pass the header checks but carry >= 20 % errors
                                 (identical repeated frames let decision-directed training converge on a wrong but
                                 self-consistent pattern; the BER still counts every error)
  - 'receiver-limited: errors'   genie fine, but receiver BER >= 10 x genie BER and >= 1 %
  - 'ok'
"""
from __future__ import annotations

import numpy as np
from scipy import signal

import channel_est as C
import psk_link as P
import psk_tx as PT

CHANNEL = ("linphase", 4, 1.0, 27.5)       # channel_est parametric best fit of the 10 real recordings
RHO_MIN = 0.30
GENIE_BER_LIMIT = 0.01
WIN_S = 0.08
RIDGE = 1e-2


def template(p, fw, t_tx):
    kind, n, fh, fl = CHANNEL
    pre = 750
    tt = np.r_[t_tx[0] - np.arange(pre, 0, -1) / 250.0, t_tx]
    return C.model_filter(PT.model(p, tt, fw), fh, fl, n, kind)[pre:]


def locate(x, fs, p, fw, lag_lo, lag_hi, ppm_hint=None):
    """best (rho, lag, ppm): recording sample 0 is at transmitter time lag; watch time = (tx time - lag) * (1 + ppm)"""
    x = x - x.mean()
    N = len(x)
    T = p["sym_us"] / 1e6
    dur = N / fs
    step = max(20.0, min(500.0, T / 4 / dur * 1e6))
    if ppm_hint is not None:
        grid = ppm_hint + np.arange(-400, 401, step)
    else:
        grid = np.arange(-6000, 6001, step)
    best = (-1.0, None, None)
    lo, hi = max(lag_lo, -dur), min(lag_hi, p["total_s"])
    for ppm in grid:
        sc = 1 + ppm * 1e-6
        t_tx = np.arange(int((hi - lo + dur) * fs) + 2) / fs / sc + lo
        m = template(p, fw, t_tx)
        # correlate the recording (sampled at watch times) against the template at the same watch-time spacing
        xc = signal.fftconvolve(m, x[::-1], mode="valid")
        e_m = np.sqrt(signal.fftconvolve(m ** 2, np.ones(N), mode="valid"))
        r = xc / (e_m * np.linalg.norm(x) + 1e-30)
        k = int(np.argmax(np.abs(r)))
        if abs(r[k]) > best[0]:
            best = (float(abs(r[k])), float(lo + k / fs / sc), float(ppm))
    return best


def ideal_receiver(x, t, p, fw, lag, ppm):
    """widely-linear LS symbol estimator trained on the first half, tested on the second (true symbols)"""
    T = p["sym_us"] / 1e6
    sc = 1 + ppm * 1e-6
    fs = 1 / (t[1] - t[0])
    a = fw["aq"]
    t_tx = t / sc + lag
    fc = int(round(p["fc_hz"] * 1000)) / 1000.0
    z = 2 * x * np.exp(-2j * np.pi * fc * t_tx)                      # carrier removed with the KNOWN phase
    k_lo = int(np.ceil((t_tx[0] + WIN_S) / T - 0.5)); k_hi = int(np.floor((t_tx[-1] - WIN_S) / T - 0.5))
    ks = np.arange(max(k_lo, 1), min(k_hi, len(a) - 1) + 1)
    if len(ks) < 40:
        return None
    J = int(np.ceil(WIN_S * fs))
    offs = np.arange(-J, J + 1) / fs
    tc = ((ks + 0.5) * T - lag) * sc
    tt = tc[:, None] + offs[None, :] * sc
    zw = np.interp(tt, t, z.real) + 1j * np.interp(tt, t, z.imag)
    X = P.wl_regressors(zw, ks, p)                                  # same structure as the x3-wl receiver
    y = a[ks]
    half = len(ks) // 2
    out = {}
    for name, comp in (("re", y.real), ("im", y.imag)):
        if np.std(comp) < 1e-9:
            out[name] = np.zeros(len(ks)); continue
        A = X[:half]; G = A.T @ A
        w = np.linalg.solve(G + RIDGE * np.trace(G) / len(G) * np.eye(len(G)), A.T @ comp[:half])
        out[name] = X @ w
    est = out["re"] + 1j * out["im"]
    te = slice(half, None)
    mse = np.mean(np.abs(est[te] - y[te]) ** 2)
    snr = 10 * np.log10(np.mean(np.abs(y[te]) ** 2) / max(mse, 1e-30))
    mod = p["modulation"]
    if mod == "dbpsk":
        dh = np.sign((est[1:] * np.conj(est[:-1])).real); dt = np.sign((y[1:] * np.conj(y[:-1])).real)
        bits_ok = (dh == dt)[half:]
        nb, ne = len(bits_ok), int(np.sum(~bits_ok))
    else:
        nb, ne = None, None
    return dict(symbols_train=int(half), symbols_test=int(len(ks) - half), snr_db=float(snr), bits=nb, errors=ne,
                ber=(ne / nb) if nb else None)


def complete_frames(p, lag, ppm, dur):
    """transmitted frames lying wholly inside the recording (from the genie alignment)"""
    sc = 1 + ppm * 1e-6
    T = p["sym_us"] / 1e6
    t0, t1 = lag, lag + dur / sc                                 # transmitter-time span of the recording
    n = 0
    for m in range(p["frames"]):
        s = (p["lead_symbols"] + m * p["frame_symbols"]) * T
        if s - T >= t0 and s + p["frame_s"] <= t1:
            n += 1
    return n


def diagnose(e, p, lag_lo, lag_hi, rep, usable):
    """e: ecg (t, mv, fs); rep: the receiver's report. Returns the diagnosis dict."""
    fw = PT.fw_payload(p)
    x = np.asarray(e.mv, float); x = x - x.mean()
    t = np.asarray(e.t, float) - e.t[0]
    fs = float(e.fs)
    hint = rep.get("coarse_clock_ppm") if rep.get("power_line_db", 0) >= 10 else None
    rho, lag, ppm = locate(x, fs, p, fw, lag_lo, lag_hi, hint)
    if hint is not None and rho < RHO_MIN:                         # the line may be a false peak: full grid
        rho2, lag2, ppm2 = locate(x, fs, p, fw, lag_lo, lag_hi, None)
        if rho2 > rho:
            rho, lag, ppm = rho2, lag2, ppm2
    out = dict(method="genie (known transmission) - diagnostic", channel_model=CHANNEL, rho=rho, lag_s=lag,
               clock_ppm=ppm, frames_transmitted=None, ideal=None)
    if lag is None:
        out["diagnosis"] = "transmission not found"; return out
    out["frames_transmitted"] = complete_frames(p, lag, ppm, len(x) / fs)
    ideal = ideal_receiver(x, t, p, fw, lag, ppm) if rho >= RHO_MIN else None
    out["ideal"] = ideal
    v = [f for f in rep.get("frames", []) if f["valid"]]
    nb = sum(f["bits"] for f in v); ne = sum(f["errors"] for f in v)
    ft = out["frames_transmitted"] or 0
    if rep.get("clock_ppm") is not None:
        out["clock_error_ppm"] = float(rep["clock_ppm"] - ppm)
    if v:
        T = p["sym_us"] / 1e6
        f0 = v[0]
        frac = ((f0["t_start_s"] / (1 + ppm * 1e-6) + lag) / T - 0.5 - p["lead_symbols"]) % p["frame_symbols"]
        out["timing_error_symbols"] = float(min(frac, p["frame_symbols"] - frac))
    if not usable:
        out["diagnosis"] = "not usable"
    elif rho < RHO_MIN:
        out["diagnosis"] = "transmission not found"
    elif ideal is None or ideal["ber"] is None:
        out["diagnosis"] = "ideal receiver not evaluable"
    elif ideal["ber"] >= GENIE_BER_LIMIT:
        out["diagnosis"] = "channel-limited"
    elif ft and len(v) < 0.95 * ft:
        out["diagnosis"] = "receiver-limited: sync"
    elif nb and ne / nb >= 0.2:
        out["diagnosis"] = "receiver-limited: false lock (headers valid, payload near random)"
    elif nb and ne / nb >= max(0.01, 10 * ideal["ber"]):
        out["diagnosis"] = "receiver-limited: errors"
    else:
        out["diagnosis"] = "ok"
    return out
