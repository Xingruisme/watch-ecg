"""Tone-set simulation for the Arduino BER design, through the FROZEN ecg-v1 decoder (ecg_link.decode, unchanged).
Transmitter: Arduino DAC sine at the exact nominal frequency, phase restarted each bit, '0' = mid level (OOK),
clock error tx_ppm. Channel: user's measured Pixel response (fit: HP 1 Hz 1st order + Butterworth LP 3rd order
21 Hz: 20 Hz 0.75, 30 Hz 0.32, 50 Hz 0.07 of 10 Hz) or the WatchLink fit in ecg_synth (LP 5th order 31.5 Hz)."""
import sys, os, json, time, itertools
sys.path.insert(0, os.path.expanduser("~/xingru/watch-ecg"))
import numpy as np
from scipy import signal
import ecg_link as L
import ecg_synth as Y

FS_SIM, FS_NOM = 2000.0, 250.0


def chan(kind, fs):
    hp = signal.butter(1, 1.0, "highpass", fs=fs, output="sos")
    lp = signal.butter(3, 21.0, "lowpass", fs=fs, output="sos") if kind == "measured" else \
        signal.butter(5, 31.5, "lowpass", fs=fs, output="sos")
    return np.vstack([hp, lp])


def txwave(plan, t, tx_ppm):
    bits, _, _ = L.frame(plan["bps"])
    Tb = plan["bit_us"] / 1e6
    tt = t * (1 + tx_ppm * 1e-6)                    # transmitter's own clock
    k = np.floor(tt / Tb).astype(int)
    inside = (k >= 0) & (k < plan["frames"] * len(bits))
    b = np.array(bits)[np.clip(k, 0, None) % len(bits)]
    f = np.where(b == 1, plan["f1_hz"], plan["f0_hz"] or 0.0)
    return np.where(inside, np.sin(2 * np.pi * f * (tt - k * Tb)), 0.0)


def record(plan, rng, amp, ch, tx_ppm=-2400.0, watch_ppm=1227.0, white_uv=10.0, wander_uv=30.0, heart=None):
    start = rng.uniform(3.0, plan["total_s"] - 33.0)
    ts = np.arange(start - 3.0, start + 30.2, 1 / FS_SIM)
    v = 0.5 * amp * txwave(plan, ts, tx_ppm)
    if heart:
        v = v + Y.heartbeat(ts, heart, rng)
    v = signal.sosfilt(chan(ch, FS_SIM), v)
    n = int(30 * FS_NOM)
    tk = start + np.arange(n) / (FS_NOM * (1 + watch_ppm * 1e-6))
    x = np.interp(tk, ts, v) + rng.normal(0, white_uv * 1e-3, n)
    x += 5e-3 * np.sin(2 * np.pi * 50 * tk + rng.uniform(0, 6.3))
    w = np.cumsum(rng.normal(0, 1, n)); w = signal.sosfiltfilt(signal.butter(2, 0.3, fs=FS_NOM, output="sos"), w)
    x += wander_uv * 1e-3 * w / (w.std() + 1e-12)
    x = x[125:]
    return np.arange(len(x)) / FS_NOM, x


def run(mod, bps, f1, f0, amps, ch, n, seed, **kw):
    plan = L.ber_plan(mod, bps, f1=f1, f0=f0, seconds=90.0)
    _, truth, _ = L.frame(bps)
    rows = []
    for amp in amps:
        rng = np.random.default_rng(seed)
        E = B = lost = exp = 0
        es = []
        for i in range(n):
            t, x = record(plan, rng, amp, ch, **kw)
            rep, _ = L.decode(x, t, plan)
            fr = rep["frames"]
            s = L.frames_summary(fr, rep["frames_expected"], len(truth))
            v = [f for f in fr if f["valid"]]
            E += sum(f["errors"] for f in v); B += sum(f["bits"] for f in v)
            exp += rep["frames_expected"]; lost += rep["frames_expected"] - len(v)
            if rep["es_n0_db"] is not None:
                es.append(rep["es_n0_db"])
        rows.append(dict(amp=amp, ber=E / B if B else None, bits=B, err=E, lost=lost, frames=exp,
                         es=float(np.median(es)) if es else None))
    return rows


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    out = run(**cfg)
    print(json.dumps(dict(cfg=cfg, rows=out)))
