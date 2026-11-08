"""From PDF waveform to bits, on one real frame: what the watch gives, what x4 decides, what was sent.
x4 runs unchanged; a wrapper only records its fitted model. The sent bits are used only for the comparison row."""
import glob, json, os, sys
HERE = os.path.expanduser("~/xingru/watch-ecg")
sys.path.insert(0, HERE)
import numpy as np
import ecg_io as IO, ecg_link as L, psk_rx4 as X4

run = sys.argv[1] if len(sys.argv) > 1 else "psk_d50_r3_20261107_204218"
out = sys.argv[2]
d = glob.glob(f"{HERE}/data_watch_ecg/psk_session_*/{run}")[0]
p = json.load(open(d + "/tx.json"))["plan"]
e = IO.load(d + "/ecg.npz")
x, t = np.asarray(e.mv, float), np.asarray(e.t, float) - e.t[0]

# amplitude resolution of the PDF trace
steps = np.abs(np.diff(x)); steps = steps[steps > 0]
q = np.min(steps)
print(f"PDF trace: {len(x)} samples at {e.fs:.0f} Hz, {t[-1]:.2f} s; smallest amplitude step {q * 1000:.2f} uV")

cap = {}
orig = X4._frames
def rec(m, ev, p_, truth):
    cap.update(m=m, ev=ev); return orig(m, ev, p_, truth)
X4._frames = rec
rep = X4.decode(x, t, p)
X4._frames = orig
m, ev = cap["m"], cap["ev"]
full, pl, order = L.frame(p["bps"])
nf, nh = p["frame_symbols"], len(L.HDR)
a, ns, sc, tau0, T = ev["a"], ev["ns"], ev["sc"], ev["tau0"], p["sym_us"] / 1e6
tsym = (tau0 + (ns + 0.5) * T) * sc                                # watch time of each symbol centre
model = np.asarray(m.G(ev["Us"], ev["b"]) @ ev["dec"]).ravel()     # x4's reconstruction of the transmitted signal

# pick the frame with the smallest margin (the near-miss)
best = None
for f in rep["frames"]:
    if not f["valid"]:
        continue
    s0 = int(np.searchsorted(ns, f["start_symbol"]))
    s = np.empty(nf + 1); s[0] = 1
    for j, b in enumerate(full):
        s[j + 1] = -s[j] if b else s[j]
    y = a[s0 - 1:s0 + nf] * s; y = y * np.sign(y.sum())
    if best is None or y.min() / y.mean() < best[2]:
        best = (f, s0, y.min() / y.mean(), y / y.mean())
f, s0, mn, yn = best
dec_bits = f["decided"]
print(f"frame {f['m']} (of {rep['frames_valid']} valid), starts t = {tsym[s0]:.2f} s, worst symbol margin {mn:.3f}")
print(f"frame = {nh} header bits + {len(pl)} payload bits (PRBS{order}) + 2 gap bits = {nf}")
print("sent    header :", "".join(map(str, L.HDR)))
print("decided header :", "".join(map(str, dec_bits[:nh])))
sent_pl, dec_pl = "".join(map(str, pl)), "".join(map(str, dec_bits[nh:nh + len(pl)]))
for i in range(0, len(pl), 64):
    print(f"sent    payload {i:3d}: {sent_pl[i:i + 64]}")
    print(f"decided payload {i:3d}: {dec_pl[i:i + 64]}")
print("payload bit errors in this frame:", sum(a_ != b_ for a_, b_ in zip(sent_pl, dec_pl)))
print("all frames:", [(fr["m"], fr["errors"]) for fr in rep["frames"] if fr["valid"]])

# ------------------------------------------------------------------ figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, ORANGE, GRAY, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#9a9890", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(13, 12), facecolor=SURF)
gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1.2, 1.0, 0.75], hspace=0.75)
t0, t1 = tsym[s0 - 1] - T / 2, tsym[s0 + nf - 1] + T / 2
w = (t >= t0 - 0.3) & (t <= t1 + 0.3)

ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURF)
ax.plot(t[w], x[w], color=GRAY, lw=0.9, label="ECG trace read from the PDF (heartbeat + DBPSK signal + noise)")
ax.plot(t[w], model[w], color=BLUE, lw=0.9, label="x4's fitted DBPSK signal")
ax.axvspan(t0, t1, color=BLUE, alpha=0.05, lw=0)
ax.set_xlim(t0 - 0.3, t1 + 0.3); ax.set_xlabel("time in the recording (s)"); ax.set_ylabel("mV")
ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
ax.set_title(f"1.  The raw data: the watch's ECG waveform for one frame ({nf} symbols, {nf / p['baud']:.2f} s) "
             f"- {run}", loc="left", fontsize=10.5, color=INK)

# zoom: 20 symbols around the near-miss
k_worst = s0 - 1 + int(np.argmin(yn))
z0, z1 = max(s0, k_worst - 10), min(s0 + nf - 1, k_worst + 10)
ax = fig.add_subplot(gs[1]); ax.set_facecolor(SURF)
wz = (t >= tsym[z0] - T / 2) & (t <= tsym[z1] + T / 2)
ax.plot(t[wz], x[wz] - np.mean(x[wz]), color=GRAY, lw=1.0, marker="o", ms=2, label="PDF samples (250 Hz)")
ax.plot(t[wz], model[wz] - np.mean(model[wz]), color=BLUE, lw=1.4, label="x4's fitted DBPSK signal")
import matplotlib.transforms as mtr
bt = mtr.blended_transform_factory(ax.transData, ax.transAxes)
for k in range(z0, z1 + 1):
    ax.axvline(tsym[k] - T / 2, color=GRID, lw=0.8, zorder=0)
    j = k - s0                                          # bit j of the frame is carried by symbol s0 + j
    if 0 <= j < nf:
        ax.text(tsym[k], 1.10, f"{dec_bits[j]}", ha="center", va="bottom", fontsize=9, color=INK,
                fontweight="bold", transform=bt)
        ax.text(tsym[k], 1.02, f"{full[j]}", ha="center", va="bottom", fontsize=9, color=INK2, transform=bt)
ax.text(1.005, 1.10, "decided (from the waveform)", fontsize=8, color=INK, transform=ax.transAxes, va="bottom")
ax.text(1.005, 1.02, "sent (from tx.json)", fontsize=8, color=INK2, transform=ax.transAxes, va="bottom")
ax.axvspan(tsym[k_worst] - T / 2, tsym[k_worst] + T / 2, color=ORANGE, alpha=0.10, lw=0)
ax.text(tsym[k_worst], 0.04, "near-miss\nsymbol", ha="center", fontsize=8, color=ORANGE, transform=bt)
ax.set_xlim(tsym[z0] - T / 2, tsym[z1] + T / 2)
ax.set_ylabel("mV (offset removed)"); ax.set_xlabel("time (s)  ·  grey lines = symbol boundaries (20 ms)")
ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, -0.42), ncol=2)
ax.set_title("2.  Zoom: a 16 Hz carrier whose phase flips (bit 1) or does not flip (bit 0) at each symbol boundary",
             loc="left", fontsize=10.5, color=INK, pad=34)

ax = fig.add_subplot(gs[2]); ax.set_facecolor(SURF)
kk = np.arange(s0 - 1, s0 + nf)
ax.vlines(tsym[kk], 0, yn, color=[BLUE if v > 0 else ORANGE for v in yn], lw=1.4)
ax.axhline(0, color=ORANGE, lw=1.2)
ax.plot(tsym[k_worst], yn[np.argmin(yn)], "o", color=ORANGE, ms=6)
ax.text(tsym[k_worst], yn[np.argmin(yn)] - 0.25, f"margin {yn.min():.3f}", ha="center", va="top", fontsize=8,
        color=ORANGE)
ax.set_ylim(-0.6, max(2.2, yn.max() * 1.05)); ax.set_xlim(t0 - 0.3, t1 + 0.3)
ax.set_ylabel("symbol estimate\n× true sign"); ax.set_xlabel("time (s)")
ax.axvspan(tsym[s0 - 1], tsym[s0 + nh - 1], color=GRAY, alpha=0.12, lw=0)
ax.text(tsym[s0 - 1], 2.0, " header", fontsize=8, color=INK2)
ax.set_title("3.  What x4 measured for each symbol in this frame, scored against what was sent (above 0 = correct)",
             loc="left", fontsize=10.5, color=INK)

ax = fig.add_subplot(gs[3]); ax.axis("off")
lines = [("sent header", "".join(map(str, L.HDR))), ("decided header", "".join(map(str, dec_bits[:nh])))]
for i in range(0, len(pl), 64):
    lines += [(f"sent payload {i}-{min(i + 63, len(pl) - 1)}", sent_pl[i:i + 64]),
              (f"decided payload {i}-{min(i + 63, len(pl) - 1)}", dec_pl[i:i + 64])]
for r, (lab, s) in enumerate(lines):
    yy = 1 - r * 0.15
    ax.text(0.0, yy, lab, family="monospace", fontsize=8.5, color=INK2, transform=ax.transAxes, va="top")
    ax.text(0.2, yy, s, family="monospace", fontsize=8.5, color=INK, transform=ax.transAxes, va="top")
ax.set_title(f"4.  The bits: decided from the waveform vs sent - payload errors in this frame: {f['errors']}",
             loc="left", fontsize=10.5, color=INK)
fig.savefig(out, dpi=125, bbox_inches="tight", facecolor=SURF)
print("->", out)
