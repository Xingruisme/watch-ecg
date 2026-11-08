"""SIMULATED: x4-mb vs x3-wl, random payload in every frame, real ECG residuals, matched or mismatched channel"""
import sys, os, json, time, itertools; sys.path.insert(0, os.path.expanduser("~/xingru/watch-ecg"))
import numpy as np, multiprocessing as mp
import psk_link as P, psk_sim as S, channel_est as C, ecg_link as L, psk_rx4 as X
RES = None

def symbols_random(p, rng):
    bits_hdr, nf = L.HDR, p["frame_symbols"]
    _, pl, _ = L.frame(p["bps"])
    out, ph, pays = [1.0] * p["lead_symbols"], 0.0, []
    for _ in range(p["frames"]):
        pay = list(rng.integers(0, 2, len(pl)))
        pays.append(pay)
        for b in L.HDR + pay + L.GAP:
            ph += np.pi * b; out.append(np.cos(ph))
    out += [out[-1]] * p["tail_symbols"]
    return np.round(np.array(out)) + 0j, pays

def errors(rep, p, ppm, lag, pays):
    T, nf = p["sym_us"] / 1e6, p["frame_symbols"]; sc = 1 + ppm * 1e-6
    v = [f for f in rep.get("frames", []) if f["valid"]]
    nb = ne = 0; bad = 0
    for f in v:
        k = (f["t_start_s"] / sc + lag) / T - 0.5
        m = (k - p["lead_symbols"]) / nf
        mi = int(round(m))
        if abs(m - mi) > 0.25 or not 0 <= mi < len(pays) or "decided" not in f:
            bad += 1; nb += len(pays[0]); ne += len(pays[0]) // 2; continue   # misaligned frame: count as half wrong
        dec = f["decided"][len(L.HDR):len(L.HDR) + len(pays[mi])]
        nb += len(dec); ne += int(np.sum(np.array(dec) != np.array(pays[mi])))
    return nb, ne, bad

def x3_decided(rep):
    # x3 frames have no 'decided' bits: rebuild payload decisions from err_pos vs the fixed payload is impossible for
    # random payloads, so x3 is run on the standard (repeating) payload instead - see cell()
    return rep

def cell(args):
    bps, fc, shape, ch, i, mode = args
    name, gain, res = RES[i]
    rng = np.random.default_rng(1000 * i + int(bps) + (7 if ch == "min22" else 0))
    p = P.plan("dbpsk", bps, fc=fc, shape=shape, seconds=55)
    kind, n, fh, fl = S.CHANNELS[ch]
    N = len(res); t = np.arange(N) / C.FS
    ppm = 3400 + rng.uniform(-300, 300); sc = 1 + ppm * 1e-6
    lag = rng.uniform(1.0, p["total_s"] - N / C.FS - 1.0)
    pre = 750; tt = np.arange(-pre, N) / C.FS
    if mode == "noise":
        x = res
        rep = X.decode(x, t, p)
        return dict(bps=bps, fc=fc, shape=shape, ch=ch, i=i, mode=mode, valid=rep.get("frames_valid", 0),
                    frames=rep.get("frames_complete", 0))
    a, pays = symbols_random(p, rng)
    y = C.model_filter(P.waveform(p, tt / sc + lag, a=a), fh, fl, n, kind)[pre:]
    x = gain * y + res
    t0 = time.time(); rep = X.decode(x, t, p); dt = time.time() - t0
    nb, ne, bad = errors(rep, p, ppm, lag, pays)
    # frames transmitted wholly inside the recording
    T, nf = p["sym_us"] / 1e6, p["frame_symbols"]
    ft = sum(1 for m in range(p["frames"]) if (p["lead_symbols"] + m * nf - 1) * T >= lag and
             (p["lead_symbols"] + (m + 1) * nf) * T <= lag + N / C.FS / sc)
    return dict(bps=bps, fc=fc, shape=shape, ch=ch, i=i, mode=mode, ft=ft, valid=rep.get("frames_valid", 0), bits=nb,
                errors=ne, misaligned=bad, clock_err=rep.get("clock_ppm", 0) - ppm, s=dt)

if __name__ == "__main__":
    globals()["RES"] = S.real_residuals("lin27")
    X.WHITEN_ORDER = int(os.environ.get("X4_WHITEN", "0"))
    rates = [float(r) for r in sys.argv[1].split(",")]
    chs = sys.argv[2].split(",")
    shapes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["rrc"]
    mode = sys.argv[4] if len(sys.argv) > 4 else "signal"
    fcs = [float(f) for f in sys.argv[5].split(",")] if len(sys.argv) > 5 else [16.0]
    jobs = [(b, fc, s, c, i, mode) for b in rates for fc in fcs for s in shapes for c in chs for i in range(len(RES))]
    with mp.get_context("fork").Pool(min(20, os.cpu_count())) as pool:
        out = pool.map(cell, jobs, chunksize=1)
    tag = sys.argv[6] if len(sys.argv) > 6 else "sim"
    json.dump(out, open(os.path.join(os.path.dirname(__file__), f"{tag}.json"), "w"), default=float)
    key = lambda r: (r["bps"], r["fc"], r["shape"], r["ch"])
    for k in sorted({key(r) for r in out}):
        rr = [r for r in out if key(r) == k]
        if mode == "noise":
            print(k, "false valid frames", sum(r["valid"] for r in rr), "of", sum(r["frames"] for r in rr)); continue
        nb = sum(r["bits"] for r in rr); ne = sum(r["errors"] for r in rr)
        print(f"{k}  sync {sum(r['valid'] for r in rr)}/{sum(r['ft'] for r in rr)}  errors {ne}/{nb}  "
              f"misaligned {sum(r['misaligned'] for r in rr)}  clock err med {np.median([abs(r['clock_err']) for r in rr]):.0f} ppm  "
              f"{np.mean([r['s'] for r in rr]):.0f} s/rec")
