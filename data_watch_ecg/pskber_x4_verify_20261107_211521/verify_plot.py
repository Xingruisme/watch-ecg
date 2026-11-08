"""Verification plot for psk-x4 (x4-mb) on the real worn-watch DBPSK recordings.

The frozen receivers run UNCHANGED (X4.decode / P.decode_wl). A wrapper around X4._frames only *records* the
fitted symbol estimates a_k that x4 makes its decisions from. The transmitted payload is used afterwards, for
scoring only: each symbol estimate is multiplied by its TRUE transmitted sign, so a value > 0 is a correct
symbol and a value < 0 is a wrong one. The distance from 0 is the decision margin.
"""
import csv, datetime, glob, json, os, shutil, sys
from multiprocessing import get_context

HERE = os.path.expanduser("~/xingru/watch-ecg")
sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import norm, beta
import ecg_io as IO, ecg_link as L, psk_link as P, psk_rx4 as X4

LABELS = ["d16", "d24", "d32", "d40", "d40b", "d50"]
DEV = {"psk_d40_r1_20261107_174709", "psk_d40_r2_20261107_174843"}


def runs():
    out = []
    for lab in LABELS:
        for f in sorted(glob.glob(f"{HERE}/data_watch_ecg/psk_session_{lab}_*/psk_*/tx.json")):
            d = os.path.dirname(f)
            if os.path.exists(d + "/ecg.npz"):
                out.append((lab, d))
    return out


def one(job):
    lab, d = job
    p = json.load(open(d + "/tx.json"))["plan"]
    e = IO.load(d + "/ecg.npz")
    cap = {}
    orig = X4._frames

    def rec(m, ev, p_, truth):
        cap["ev"] = ev
        return orig(m, ev, p_, truth)
    X4._frames = rec
    try:
        r4 = X4.decode(e.mv, e.t - e.t[0], p)
    finally:
        X4._frames = orig
    r3 = P.decode_wl(e.mv, e.t - e.t[0], p)
    full, pl, _ = L.frame(p["bps"])
    nf = p["frame_symbols"]
    assert len(full) == nf
    ev = cap["ev"]
    a, ns = ev["a"], ev["ns"]
    ys, frames = [], []
    for f in r4["frames"]:
        if not f["valid"]:
            continue
        s0 = int(np.searchsorted(ns, f["start_symbol"]))
        seg = a[s0 - 1:s0 + nf]
        s = np.empty(nf + 1); s[0] = 1.0
        for j, b in enumerate(full):                          # true symbol signs from the true frame bits
            s[j + 1] = -s[j] if b else s[j]
        y = seg * s
        y *= np.sign(np.sum(y))                                # per-frame sign is free in DBPSK
        # bits decided from y must reproduce x4's own error count (cross-check of the alignment)
        wrong = (np.sign(y[1:]) != np.sign(y[:-1]))[len(L.HDR):len(L.HDR) + len(pl)]
        assert int(wrong.sum()) == f["errors"], (d, f["m"], int(wrong.sum()), f["errors"])
        ys.append(y)
        frames.append(dict(m=f["m"], errors=f["errors"], min_margin=float(y.min() / np.mean(y))))
    y = np.concatenate(ys) if ys else np.zeros(0)
    y = y / np.mean(y) if len(y) else y
    v3 = [f for f in r3.get("frames", []) if f["valid"]]
    return dict(label=lab, run=os.path.basename(d), bps=float(p["bps"]), dev=os.path.basename(d) in DEV, y=y,
                x4_bits=r4.get("bits", 0), x4_err=r4.get("errors", 0), x4_valid=r4.get("frames_valid", 0),
                x4_frames=frames, x3_bits=r3.get("bits", 0), x3_err=r3.get("errors", 0),
                x3_frame_err=[f["errors"] for f in v3], x4_sha=X4.RULES_SHA, x3_sha=P.RULES_SHA)


def tail_est(y):
    """DBPSK bit-error estimate from the side of the cluster that faces the threshold only: Gaussian with the
    lower-half spread (values > mean cannot cause errors and only inflate a two-sided sd). An extrapolation."""
    lo = y[y < 1.0]
    sd = np.sqrt(np.mean((1.0 - lo) ** 2)) if len(lo) else np.nan
    return 2 * norm.sf(1.0 / sd), 1.0 / sd


def tag(r):
    return f"{r['run'][4:-16]} {r['run'][-6:-4]}:{r['run'][-4:-2]}"


def cp_upper(k, n):
    return 1.0 if n == 0 else float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0


def main():
    jobs = runs()
    with get_context("fork").Pool(min(8, len(jobs))) as pool:
        res = pool.map(one, jobs)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{HERE}/data_watch_ecg/pskber_x4_verify_{ts}"
    os.makedirs(out)
    shutil.copy(__file__, out)

    with open(out + "/per_recording.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "label", "bps", "role", "x4_frames_valid", "x4_bits", "x4_errors", "x4_min_margin",
                    "x4_margin_mean_over_lower_sd", "x4_tail_est_ber_NOT_MEASURED", "x3_bits", "x3_errors"])
        for r in res:
            y = r["y"]
            r["est"], q = tail_est(y) if len(y) else (np.nan, np.nan)
            r["q"] = q
            r["min"] = float(y.min()) if len(y) else np.nan
            r["n_below_025"] = int(np.sum(y < 0.25))
            w.writerow([r["run"], r["label"], r["bps"], "development" if r["dev"] else "held-out", r["x4_valid"],
                        r["x4_bits"], r["x4_err"], f"{r['min']:.3f}", f"{q:.2f}", f"{r['est']:.2e}",
                        r["x3_bits"], r["x3_err"]])
    with open(out + "/per_frame_x4.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["run", "frame", "errors", "min_margin"])
        for r in res:
            for f in r["x4_frames"]:
                w.writerow([r["run"], f["m"], f["errors"], f"{f['min_margin']:.3f}"])
    np.savez_compressed(out + "/x4_symbol_margins.npz", **{r["run"]: r["y"] for r in res})

    plot(res, out)
    print_table(res)
    print("->", out)


def print_table(res):
    print(f"{'run':30s} {'bps':>5s} {'role':10s} {'x4 err/bits':>12s} {'min':>6s} {'<0.25':>6s} {'1/sdL':>6s} "
          f"{'tail est':>9s} {'x3 err/bits':>12s}")
    for r in res:
        print(f"{r['run']:30s} {r['bps']:5.0f} {'dev' if r['dev'] else 'held-out':10s} "
              f"{r['x4_err']:>5d}/{r['x4_bits']:<6d} {r['min']:6.2f} {r['n_below_025']:6d} {r['q']:6.2f} {r['est']:9.1e} "
              f"{r['x3_err']:>5d}/{r['x3_bits']:<6d}")


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#9a9890", "#0b0b0b", "#52514e", "#e6e5e0"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 11), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=0.32, wspace=0.28)

    # ---- A: every decoded symbol, aligned to its true sign
    ax = fig.add_subplot(gs[0, :]); ax.set_facecolor("#fcfcfb")
    rng = np.random.default_rng(0)
    n = len(res)
    for i, r in enumerate(res):
        row = n - 1 - i
        y = r["y"]
        c = GRAY if r["dev"] else BLUE
        ax.scatter(y, row + rng.uniform(-0.28, 0.28, len(y)), s=2.5, color=c, alpha=0.35, linewidths=0)
        if len(y):
            ax.plot([y.min()] * 2, [row - 0.38, row + 0.38], color=INK, lw=1.2)
        role = "development" if r["dev"] else "held-out"
        ax.text(2.42, row, f"{r['x4_err']}/{r['x4_bits']} errors · min margin {r["min"]:.2f} · {role}",
                va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=ORANGE, lw=1.5)
    ax.text(0.03, n - 0.35, "decision threshold\n(left of this line = wrong symbol)", color=INK2, fontsize=8, va="top")
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{r['bps']:.0f} bps  {tag(r)}" for r in res][::-1])
    ax.set_xlim(-0.6, 2.4); ax.set_ylim(-0.7, n - 0.3)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("x4 symbol estimate × true transmitted sign  (normalised: 1 = mean)")
    ax.set_title("A.  Every symbol x4 decided, scored against the true transmission (black tick = worst symbol)",
                 loc="left", fontsize=10.5, color=INK)

    # ---- B: per-recording errors, x3-wl vs x4-mb, 40 and 50 bps
    ax = fig.add_subplot(gs[1, 0]); ax.set_facecolor("#fcfcfb")
    sel = [r for r in res if r["bps"] >= 40]
    xs = np.arange(len(sel))
    b3 = [100 * r["x3_err"] / r["x3_bits"] if r["x3_bits"] else np.nan for r in sel]
    b4 = [100 * r["x4_err"] / r["x4_bits"] if r["x4_bits"] else np.nan for r in sel]
    ax.bar(xs - 0.2, b3, 0.36, color=ORANGE, label="x3-wl (frozen primary)")
    ax.bar(xs + 0.2, [max(v, 0) for v in b4], 0.36, color=BLUE, label="x4-mb")
    for xx, v, r in zip(xs, b4, sel):
        ax.text(xx + 0.2, 0.6, f"{r['x4_err']}", ha="center", color=BLUE, fontsize=8, fontweight="bold")
    for xx, v in zip(xs, b3):
        if not np.isnan(v):
            ax.text(xx - 0.2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels([f"{r['bps']:.0f}\n{r['run'][4:-16]}" + ("\n(dev)" if r["dev"] else "")
                                           for r in sel], fontsize=8)
    ax.set_ylabel("bit error rate on valid frames (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("B.  Same recordings, two receivers (numbers on blue = x4 bit errors)", loc="left",
                 fontsize=10.5, color=INK)

    # ---- C: what 0 errors actually bounds, per rate (held-out only)
    ax = fig.add_subplot(gs[1, 1]); ax.set_facecolor("#fcfcfb")
    rates = sorted({r["bps"] for r in res})
    up, est, lab = [], [], []
    for b in rates:
        h = [r for r in res if r["bps"] == b and not r["dev"]]
        k, nb = sum(r["x4_err"] for r in h), sum(r["x4_bits"] for r in h)
        up.append(cp_upper(k, nb))
        est.append(tail_est(np.concatenate([r["y"] for r in h]))[0])
        lab.append(f"{b:.0f}\n{k}/{nb}\n({len(h)} rec)" + ("\nest. < 1e-12" if est[-1] < 1e-12 else ""))
    xs = np.arange(len(rates))
    est = np.array(est)
    ax.axhline(1e-2, color=ORANGE, lw=1.2, ls="--")
    ax.text(len(rates) - 0.5, 1.25e-2, "1 % reliability limit", ha="right", fontsize=8, color=INK2)
    ax.bar(xs, up, 0.5, color=BLUE, alpha=0.85, label="measured: 95 % upper bound on BER (0 errors seen)")
    shown = est >= 1e-12
    ax.scatter(xs[shown], est[shown], marker="D", s=40, color=INK, zorder=3,
               label="extrapolated from the margin (lower-tail Gaussian, NOT measured)")
    for xx, u in zip(xs, up):
        ax.text(xx + 0.3, u * 1.3, f"< {100 * u:.2f}%", ha="center", fontsize=7.5, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(1e-12, 0.1)
    ax.set_xticks(xs); ax.set_xticklabels(lab, fontsize=8)
    ax.set_xlabel("bps  ·  x4 errors / bits  (held-out recordings only)")
    ax.set_ylabel("bit error rate")
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor="#fcfcfb", edgecolor=GRID, framealpha=1, fontsize=7.5, loc="lower left")
    ax.set_title("C.  What \"0 errors\" can and cannot prove", loc="left", fontsize=10.5, color=INK)

    fig.suptitle(f"psk-x4 (x4-mb, sha {X4.RULES_SHA}) on real worn-watch DBPSK recordings, 16 Hz rrc, 2 mVpp "
                 f"calculated · x3-wl sha {P.RULES_SHA} unchanged", x=0.06, ha="left", fontsize=11, color=INK)
    fig.savefig(out + "/x4_verify.png", dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
