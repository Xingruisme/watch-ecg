"""psk-x4 'x4-mb': model-based DBPSK receiver (EXPERIMENTAL, separately versioned; added 2026-11-07).

psk_link.py (psk-x3, x3-wl, frozen at sha 04dbb95193d8) is not modified; this file only imports its transmitter model.
Developed on the 40 bps recordings d40 r1 and r2 and on simulations; d40 r3 and every 16/24/32 bps recording are
held out (see psk_protocol.json revisions and README section 15).

Why (diagnosis of x3-wl on the 40 bps recordings, 2026-11-07):
  r1  identical 2-error patterns in 4 of 7 frames  - decision-directed training of a ~165-parameter filter biased
  r2  57 of 127 payload bits wrong in every frame  - false lock: training converged on a wrong, self-consistent pattern
  r3  clock 208 ppm off the true clock             - 0.25 symbol of timing drift over the recording; errors grow
x4 replaces the data-trained equaliser by the physical model, which has very few unknowns:
  watch sample at time w:  x(w) = sum_n a_n Re{ e^{j 2 pi fc (n+1/2) T} v_b(w/sc - tau0 - (n+1/2) T) } + ECG + noise
    a_n = +-1                the DBPSK symbols (all unknown to the receiver)
    v(tau)                   the transmitter pulse on its carrier, through the FIXED channel model (psk-x3 WL_CHANNEL)
    v_b = sum_j b_j v(tau - j/fs), j = -CORR_TAPS..CORR_TAPS, b complex: amplitude, carrier phase and a short
                             correction of the channel model (2 x 5 real parameters)
    sc, tau0                 clock scale and timing
  1. clock + timing: blind - no known bits. Grid around the power-line clock, then local refinement, minimising the
     model residual with the decided symbols. The residual does not depend on which symbol starts a frame, so clock
     and timing are found before (and independently of) frame sync.
  2. symbols: ridge (MMSE) least squares through the model (banded, all symbols jointly: the ISI is inverted by the
     model, not learned); b refitted from the decisions (10 parameters only); a few alternations.
  3. frame sync: the 13 known bits (header 11 + gap 2) searched in the DECIDED bit stream over every offset. Each
     frame's known span has its own sign (a real payload would not keep every frame's phase), so nothing assumes the
     frames repeat. The known bits are never imposed on the decisions, so the header check stays a genuine check.
  4. validity: the psk-x3 rules unchanged (R_MIN, G_MIN, KNOWN_ERR_MAX). The payload truth is only used to COUNT
     errors after the decisions (exactly like psk-x3).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spl

import channel_est as C
import ecg_link as L
import psk_link as P

VERSION = "psk-x4"
FROZEN_AT = "2026-11-07 19:20"           # after development on d40 r1/r2 + simulations; before any held-out decode
RECEIVER = "x4-mb"
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
CHANNEL = P.WL_CHANNEL                 # the same fixed channel model as psk-x3
OS = 16                                # fine grid of v: 16 points per watch sample
V_ENERGY = 0.9995                      # v truncated to the span holding this fraction of its energy
CORR_TAPS = 2                          # correction filter taps -2..2 (watch sample spacing), complex
CLOCK_WINDOW_PPM = 400                 # search around the power-line clock
ITER = 3                               # symbol / correction alternations per evaluation
WHITEN_ORDER = 12                      # >0: AR noise model of this order fitted to the residual, then GLS (whitened)
R_MIN, G_MIN, KNOWN_ERR_MAX = P.R_MIN, P.G_MIN, P.KNOWN_ERR_MAX


def params():
    return dict(version=VERSION, receiver=RECEIVER, rules_sha=RULES_SHA, frozen_at=FROZEN_AT, channel=list(CHANNEL), OS=OS,
                V_ENERGY=V_ENERGY, CORR_TAPS=CORR_TAPS, CLOCK_WINDOW_PPM=CLOCK_WINDOW_PPM, ITER=ITER,
                WHITEN_ORDER=WHITEN_ORDER,
                R_MIN=R_MIN, G_MIN=G_MIN, KNOWN_ERR_MAX=KNOWN_ERR_MAX, transmitter_model=f"{P.VERSION} {P.RULES_SHA}")


# ---------------------------------------------------------------- the model
def kernel(p, fs):
    """(tau grid start, step, v): the pulse on its carrier through the fixed channel, on a grid of 1/(fs OS)"""
    kind, n, fh, fl = CHANNEL
    T = p["sym_us"] / 1e6
    span, table = P.pulse_table(p)
    d = 1.0 / (fs * OS)
    tail = 0.8
    m0 = int(np.ceil(((span / 2) * T + tail) * fs))            # watch samples either side
    tau = (np.arange(-m0 * OS, m0 * OS + 1)) * d
    pul = P.pulse_at(tau / T, span, table) * np.exp(2j * np.pi * p["fc_hz"] * tau)
    v = np.zeros(len(tau), complex)
    for j in range(OS):                                        # polyphase: the channel runs at the watch rate
        seg = pul[j::OS]
        v[j::OS] = (C.model_filter(seg.real, fh, fl, n, kind) + 1j * C.model_filter(seg.imag, fh, fl, n, kind))
    e = np.cumsum(np.abs(v) ** 2); e /= e[-1]
    lo = int(np.searchsorted(e, (1 - V_ENERGY) / 2)); hi = int(np.searchsorted(e, 1 - (1 - V_ENERGY) / 2))
    pad = CORR_TAPS * OS
    lo, hi = max(lo - pad, 0), min(hi + pad, len(v) - 1)
    return tau[lo], d, v[lo:hi + 1]


class Model:
    def __init__(self, x, fs, p):
        self.x, self.fs, self.p = np.asarray(x, float), float(fs), p
        self.N = len(self.x)
        self.T = p["sym_us"] / 1e6
        self.tau_a, self.d, self.v = kernel(p, fs)
        self.tau_b = self.tau_a + (len(self.v) - 1) * self.d
        self.x_raw = self.x
        self.w = None                                              # whitening FIR (None = white noise assumed)

    def whiten(self, residual, order):
        """fit an AR(order) model to the residual; afterwards x and every model column are filtered by it"""
        from scipy.linalg import solve_toeplitz
        r = residual - residual.mean()
        ac = np.array([r[:len(r) - k] @ r[k:] for k in range(order + 1)]) / len(r)
        a = solve_toeplitz(ac[:-1], ac[1:])
        self.w = np.r_[1.0, -a]
        self.x = self._filt(self.x_raw)
        diags = [np.full(self.N - k, c) for k, c in enumerate(self.w)]
        self.W = sps.diags(diags, [-k for k in range(len(self.w))], shape=(self.N, self.N), format="csc")

    def _filt(self, y):
        from scipy.signal import lfilter
        return lfilter(self.w, [1.0], y) if self.w is not None else y

    def _v_at(self, tau, shift):
        """v(tau - shift/fs) by linear interpolation on the fine grid (0 outside)"""
        f = (tau - shift / self.fs - self.tau_a) / self.d
        i = np.floor(f).astype(int); r = f - i
        ok = (i >= 0) & (i < len(self.v) - 1)
        i = np.clip(i, 0, len(self.v) - 2)
        return np.where(ok, self.v[i] * (1 - r) + self.v[i + 1] * r, 0)

    def grid(self, sc, tau0):
        """symbol indices whose waveform touches the recording, and which lie wholly inside it"""
        T = self.T
        t_end = (self.N - 1) / self.fs
        n_lo = int(np.floor((0 / sc - tau0 - self.tau_b - CORR_TAPS / self.fs) / T - 0.5))
        n_hi = int(np.ceil((t_end / sc - tau0 - self.tau_a + CORR_TAPS / self.fs) / T - 0.5))
        ns = np.arange(n_lo, n_hi + 1)
        c = tau0 + (ns + 0.5) * T                                  # TX-time centre (relative)
        inside = ((c + self.tau_a) * sc >= 0) & ((c + self.tau_b) * sc <= t_end)
        return ns, inside

    def U(self, sc, tau0, ns, shift=0):
        """sparse N x K complex matrix: column n = e^{j phi_n} v(w/sc - tau0 - (n+1/2)T - shift/fs)"""
        T, fs = self.T, self.fs
        c = tau0 + (ns + 0.5) * T
        M = int(np.ceil((self.tau_b - self.tau_a) * sc * fs)) + 2
        i0 = np.ceil((c + self.tau_a + shift / fs) * sc * fs).astype(int)
        rows = i0[:, None] + np.arange(M)[None, :]
        tau = rows / fs / sc - c[:, None]
        val = self._v_at(tau, shift) * np.exp(2j * np.pi * self.p["fc_hz"] * c)[:, None]
        ok = (rows >= 0) & (rows < self.N) & (val != 0)
        cols = np.broadcast_to(np.arange(len(ns))[:, None], rows.shape)
        return sps.csc_matrix((val[ok], (rows[ok], cols[ok])), shape=(self.N, len(ns)))

    def Us(self, sc, tau0, ns):
        return [self.U(sc, tau0, ns, j) for j in range(-CORR_TAPS, CORR_TAPS + 1)]

    @staticmethod
    def G(Us, b):
        """real design matrix for correction b (complex, one per shift)"""
        acc = None
        for Uj, bj in zip(Us, b):
            g = (Uj * bj).real if bj != 0 else None
            if g is not None:
                acc = g if acc is None else acc + g
        return acc.tocsc()

    def Gw(self, Us, b):
        G = self.G(Us, b)
        return (self.W @ G).tocsc() if self.w is not None else G

    def solve(self, G, lam):
        A = (G.T @ G).tocsc() + lam * sps.identity(G.shape[1], format="csc")
        return spl.spsolve(A, G.T @ self.x)

    def fit_b(self, Us, a):
        """least-squares correction b for fixed symbols a"""
        cols = []
        for Uj in Us:
            s = Uj @ a
            cols += [self._filt(s.real), -self._filt(s.imag)]
        X = np.array(cols).T
        g, *_ = np.linalg.lstsq(X, self.x, rcond=None)
        return g[0::2] + 1j * g[1::2]

    def blind_start(self, Us, lam):
        """closed-form start without any known symbol: x ~ Re(U) cr a - Im(U) ci a, solved for (cr a, ci a),
        then the rank-1 direction gives c = cr + j ci"""
        U0 = Us[CORR_TAPS]
        X = sps.hstack([U0.real, -U0.imag]).tocsc()
        A = (X.T @ X).tocsc() + lam * sps.identity(X.shape[1], format="csc")
        z = spl.spsolve(A, X.T @ self.x)
        K = U0.shape[1]
        al, be = z[:K], z[K:]
        Mx = np.array([[al @ al, al @ be], [al @ be, be @ be]])
        w, V = np.linalg.eigh(Mx)
        cr, ci = V[:, -1]
        scale = np.sqrt(max(w[-1], 1e-30) / K)
        b = np.zeros(len(Us), complex); b[CORR_TAPS] = (cr + 1j * ci) * scale
        return b

    def evaluate(self, sc, tau0, iters=ITER, b0=None):
        """fit at (sc, tau0): returns dict with residual (per sample) of the DECIDED symbols"""
        ns, inside = self.grid(sc, tau0)
        Us = self.Us(sc, tau0, ns)
        b = self.blind_start(Us, 1e-6) if b0 is None else b0
        if not np.any(b):
            return None
        res = None
        for _ in range(iters):
            G = self.Gw(Us, b)
            # MMSE regularisation: noise variance relative to the symbol power (1)
            sig2 = np.mean(self.x ** 2) if res is None else res
            a = self.solve(G, sig2)
            dec = np.sign(a); dec[dec == 0] = 1
            b = self.fit_b(Us, dec)
            G = self.Gw(Us, b)
            res = float(np.mean((self.x - G @ dec) ** 2))
        return dict(sc=sc, tau0=tau0, ns=ns, inside=inside, a=a, dec=dec, b=b, res=res, Us=Us)


# ---------------------------------------------------------------- decoding
def _frames(m, ev, p, truth):
    """frame sync on the decided bits, then per-frame checks (psk-x3 validity rules)"""
    nf = p["frame_symbols"]
    a, ns, inside = ev["a"], ev["ns"], ev["inside"]
    soft = -(a[1:] * a[:-1]) / (np.abs(a[1:] * a[:-1]) + 1e-30)          # soft bit of symbol k+1 (1 = flip)
    bit = (soft > 0).astype(int)
    known_pos = list(range(len(L.HDR))) + [nf - 2, nf - 1]
    known_val = np.array(L.HDR + L.GAP)
    hdr = np.array([2.0 * b - 1 for b in L.HDR])
    best = None
    for o in range(nf):                                  # frame start offset (in symbols, from the grid's first)
        sc_, n_ = 0, 0
        for s0 in range(o + 1, len(a) - nf + 1, nf):     # s0 = index into a of the frame's first symbol
            if not inside[s0 - 1:s0 + nf].all():
                continue
            bits_f = bit[s0 - 1:s0 - 1 + nf]
            sc_ += int(np.sum(bits_f[known_pos] == known_val)); n_ += 1
        if n_ and (best is None or sc_ / n_ > best[0]):
            best = (sc_ / n_, o)
    frames = []
    if best is None:
        return frames
    o = best[1]
    T, sc = p["sym_us"] / 1e6, ev["sc"]
    for s0 in range(o + 1, len(a) - nf + 1, nf):
        if not inside[s0 - 1:s0 + nf].all():
            continue
        sf = soft[s0 - 1:s0 - 1 + nf]
        bits_f = bit[s0 - 1:s0 - 1 + nf]
        r = float(np.mean(sf[:len(hdr)] * hdr))
        kerr = int(np.sum(bits_f[known_pos] != known_val))
        pay = bits_f[len(L.HDR):len(L.HDR) + len(truth)]
        err = int(np.sum(pay != np.array(truth)))
        t_start = float((ev["tau0"] + (ns[s0] + 0.5) * T) * sc)       # watch time of the start symbol's centre
        frames.append(dict(m=len(frames), start_symbol=int(ns[s0]), r=r, known_errors=kerr, bits=len(truth),
                           errors=err, err_pos=[int(k) for k in np.flatnonzero(pay != np.array(truth))],
                           t_start_s=t_start, decided=[int(b_) for b_ in bits_f]))
    return frames


def search(m, sc0, window_ppm=CLOCK_WINDOW_PPM):
    """blind clock + timing: coarse grid (drift <= T/8 per step, T/8 timing), then local refinement"""
    T = m.T
    dur = m.N / m.fs
    step_ppm = max(T / 8 / dur * 1e6, 5.0)
    best = None
    for ppm in np.arange(-window_ppm, window_ppm + 1e-9, step_ppm):
        sc = sc0 * (1 + ppm * 1e-6)
        for k in range(8):
            ev = m.evaluate(sc, k * T / 8, iters=2)
            if ev and (best is None or ev["res"] < best["res"]):
                best = ev
    # local refinement: shrinking steps in clock and timing, keeping the fitted correction
    d_ppm, d_tau = step_ppm / 2, T / 16
    for _ in range(6):
        improved = False
        for dp, dt in ((d_ppm, 0), (-d_ppm, 0), (0, d_tau), (0, -d_tau)):
            ev = m.evaluate(best["sc"] * (1 + dp * 1e-6), best["tau0"] + dt, iters=2, b0=best["b"])
            if ev and ev["res"] < best["res"]:
                best, improved = ev, True
        if not improved:
            d_ppm /= 2; d_tau /= 2
    return m.evaluate(best["sc"], best["tau0"], iters=ITER + 1, b0=best["b"])


def decode(x, t, p, truth=None):
    """psk-x4 'x4-mb' receiver for DBPSK. Same report fields as psk_link.decode_wl()."""
    if p["modulation"] != "dbpsk":
        raise ValueError("x4-mb is for DBPSK")
    x = np.asarray(x, float) - np.mean(x)
    t = np.asarray(t, float) - t[0]
    fs = 1 / (t[1] - t[0])
    _, pl, _ = L.frame(p["bps"])
    truth = pl if truth is None else truth
    sc0, line_db = P.coarse_clock(x, fs, p)
    rep = dict(decoder=VERSION, receiver=RECEIVER, rules_sha=RULES_SHA, modulation="dbpsk", bps=p["bps"],
               baud=p["baud"], shape=p["shape"], eq=True, coarse_clock_ppm=(sc0 - 1) * 1e6, power_line_db=line_db,
               frames=[])
    m = Model(x, fs, p)
    ev = search(m, sc0)
    if ev is None:
        rep.update(reason="no fit", sync_score=None)
        return rep
    if WHITEN_ORDER:
        G = m.G(ev["Us"], ev["b"])
        m.whiten(m.x_raw - G @ ev["dec"], WHITEN_ORDER)
        ev = m.evaluate(ev["sc"], ev["tau0"], iters=ITER, b0=ev["b"])
    frames = _frames(m, ev, p, truth)
    score = float(np.mean([f["r"] for f in frames])) if frames else None
    for f in frames:
        f["valid"] = bool(f["r"] >= R_MIN and score is not None and score >= G_MIN and f["known_errors"] <= KNOWN_ERR_MAX)
    v = [f for f in frames if f["valid"]]
    G = m.Gw(ev["Us"], ev["b"])
    sig = float(np.mean((G @ ev["dec"]) ** 2))
    snr = 10 * np.log10(sig / max(ev["res"], 1e-30))
    rep.update(frames=frames, frames_complete=len(frames), frames_valid=len(v), sync_score=score,
               bits=sum(f["bits"] for f in v), errors=sum(f["errors"] for f in v), symbol_snr_db=snr,
               clock_ppm=(ev["sc"] - 1) * 1e6, model_snr_db=snr, correction=[complex(b) for b in ev["b"]],
               reason="" if v else ("no valid frame" if frames else "no complete frame"))
    return rep
