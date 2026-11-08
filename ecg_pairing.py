"""Pair exported ECG recordings with the transmitted runs of a session - by time AND by signal (added 2026-11-06).

Why: a PDF's 'Recorded on' time has minute resolution. Two runs can both fit one minute (e.g. OOK ends 14:13:00,
FSK starts 14:12:48), and ingest used to assign such PDFs in file-name order - which swapped the 4 bps session of
2026-11-06 (session_ard_b4_4bps_ard-baseline_20261106_141140). BER does not catch it: the OOK decoder reads FSK's
16 Hz '1's fine, and the FSK decoder reads OOK's tone/no-tone fine, so a swapped pair still decodes with 0 errors.

The check here does NOT use the decoder (ecg_link stays frozen; only its frame() definition of the sent bits is used):
  1. tones:   every tone the run sends must show a spectral line >= PRESENT_DB over the local floor, and every tone
              that only the OTHER candidate runs send must be absent (< ABSENT_DB) - e.g. FSK's 8 Hz in OOK.
  2. pattern: the per-bit tone envelopes must follow the run's transmitted bit/tone sequence (correlation >= MATCH_R)
              at a start time allowed by the PDF minute: recording start in [Recorded-on, +60 s) +- CLOCK_SLACK_S.
  3. timing:  that start time, on the transmitter's time line (T+x s after its 'go').
A pairing is used only if exactly one one-to-one assignment makes every pair consistent; otherwise the readings
are reported as AMBIGUOUS and not imported.

CLI:  python ecg_pairing.py <session folder> [--apply]
      checks the recordings already imported in a session; --apply moves them to the runs the signal check finds
      (old files kept in <run>/superseded_<time>/, a pairing_correction_<time>.json in the session) and re-analyses.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import itertools
import json
import os
import shutil

import numpy as np

import ecg_io as IO
import ecg_link as L

VERSION = "pairing-v2"   # v2 (2026-11-06): watch -> transmitter time mapping fixed (v1: t * scale, now t / scale).
                         # v1 drifted 2 x 3400 ppm = ~200 ms over a recording: harmless at 2 bps, r 0.41/0.30 at 16 bps.
                         # Thresholds unchanged.
PRESENT_DB = 6.0        # a tone the run sends: line >= this over the floor 1-3 Hz away (as the usability rule)
ABSENT_DB = 6.0         # a tone only the other runs send: line below this
MATCH_R = 0.5           # tone-envelope vs sent-sequence correlation
CLOCK_SLACK_S = 5.0     # laptop clock vs phone clock, either way


def tones(plan):
    return [f for f in (plan.get("f1_line_hz"), plan.get("f0_line_hz")) if f]


def sent(plan):
    """(bit duration s, tone per transmitted bit; 0 = no tone), from the frozen frame definition"""
    bits = L.frame(plan["bps"])[0] * int(plan["frames"])
    return plan["bit_us"] * 1e-6, np.array([plan["f1_line_hz"] if b else (plan["f0_line_hz"] or 0.0) for b in bits])


def line_db(x, fs, f, exclude=()):
    """spectral line near f (+-0.6 %, covers +-6000 ppm) over the median floor 1-3 Hz away; (dB, peak Hz)"""
    n = len(x); N = 8 * n
    sp = np.abs(np.fft.rfft(x * np.hanning(n), N)); fq = np.fft.rfftfreq(N, 1 / fs)
    k = np.flatnonzero(np.abs(fq - f) <= max(0.006 * f, 0.05))
    noise = (np.abs(fq - f) >= 1) & (np.abs(fq - f) <= 3)
    for g in exclude:
        noise &= np.abs(fq - g) > 0.5
    pk = k[np.argmax(sp[k])]
    return float(20 * np.log10(sp[pk] / np.median(sp[noise]))), float(fq[pk])


def envelope(x, fs, f, win_s):
    t = np.arange(len(x)) / fs
    k = max(int(round(win_s * fs)), 1)
    c = np.cumsum(np.r_[0, x * np.exp(-2j * np.pi * f * t)])
    return (np.arange(len(c) - k) + k / 2) / fs, np.abs(c[k:] - c[:-k]) / k


def pattern(x, fs, plan, scale, lag_lo, lag_hi):
    """best correlation of the tone envelopes with the sent sequence over recording starts lag in [lag_lo, lag_hi]
    (seconds after the run's go, transmitter time). scale = f_sent / f_seen = recording seconds per transmitter
    second (1 + clock ppm), so transmitter time = recording time / scale (as the decoder's tc = t / (1 + delta))"""
    tb, seq = sent(plan)
    m = max(1, int(fs * tb / 8))
    feats = []
    for f in tones(plan):
        tc, e = envelope(x, fs, f / scale, tb)
        tc, e = tc[::m], e[::m]
        feats.append((f, e - e.mean()))
    tx0 = tc / scale             # pairing-v2 fix: v1 multiplied by scale, a 2 x clock-ppm timing error (2026-11-06)
    lags = np.arange(lag_lo, lag_hi, m / fs)
    R, per = [], {f: [] for f, _ in feats}
    for i in range(0, len(lags), 256):
        lg = lags[i:i + 256]
        idx = np.floor((tx0[None, :] + lg[:, None]) / tb).astype(int)
        ins = (idx >= 0) & (idx < len(seq))
        tone = np.where(ins, seq[np.clip(idx, 0, len(seq) - 1)], 0.0)
        for f, ez in feats:
            p = (tone == f).astype(float)
            pc = p - p.mean(1, keepdims=True)
            den = np.sqrt((pc ** 2).sum(1)) * np.sqrt((ez ** 2).sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                per[f].append(np.where(den > 0, (pc @ ez) / den, np.nan))
    per = {f: np.concatenate(v) for f, v in per.items()}
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sw:
        sw.filter(RuntimeWarning)
        R = np.nanmean(np.vstack(list(per.values())), 0)
    if not np.any(np.isfinite(R)):
        return float("nan"), None, [], {}
    R = np.where(np.isfinite(R), R, -2.0)
    j = int(np.argmax(R))
    # the frames repeat, so the start is only known modulo the frame length: list every start that fits as well
    near = [k for k in range(len(R)) if R[k] >= R[j] - 0.03 and R[k] == R[max(k - 1, 0):k + 2].max()]
    starts = []
    for k in near:
        if not starts or lags[k] - starts[-1][0] > tb:
            starts.append((float(lags[k]), float(R[k])))
        elif R[k] > starts[-1][1]:
            starts[-1] = (float(lags[k]), float(R[k]))
    return float(R[j]), float(lags[j]), starts, {f: float(v[j]) for f, v in per.items()}


def check(e, w, rec, foreign=()):
    """is recording e (Recorded-on minute w, epoch s) consistent with run rec? foreign = tones only other runs send"""
    plan = rec["plan"]
    x = np.asarray(e.mv, float); x = x - x.mean(); fs = float(e.fs)
    fl = tones(plan)
    excl = set(fl) | set(foreign)
    present = {f: line_db(x, fs, f, excl - {f}) for f in fl}
    absent = {f: line_db(x, fs, f, excl - {f}) for f in foreign}
    f1 = plan["f1_line_hz"]
    scale = f1 / present[f1][1]                       # tone seen at f1 / scale on the watch time base
    dur = len(x) / fs
    lo = max(w - rec["t_go"] - CLOCK_SLACK_S, -dur)
    hi = min(w + 60 - rec["t_go"] + CLOCK_SLACK_S, sent(plan)[0] * len(sent(plan)[1]))
    r, lag, starts, per_tone = pattern(x, fs, plan, scale, lo, hi) if hi > lo else (float("nan"), None, [], {})
    why = [f"{f:g} Hz missing ({present[f][0]:.1f} dB)" for f in fl if present[f][0] < PRESENT_DB]
    why += [f"{f:g} Hz present but this run does not send it ({absent[f][0]:.1f} dB)" for f in foreign
            if absent[f][0] >= ABSENT_DB]
    if not (r >= MATCH_R):
        why.append(f"sent pattern not found in the allowed start window (r {r:.2f})")
    return dict(version=VERSION, run=os.path.basename(rec.get("_dir", "")), modulation=plan["modulation"],
                tones_db={f"{f:g}": round(present[f][0], 1) for f in fl},
                foreign_db={f"{f:g}": round(absent[f][0], 1) for f in foreign},
                pattern_r=None if r != r else round(r, 3), start_tx_s=None if lag is None else round(lag, 2),
                start_local=None if lag is None else
                datetime.datetime.fromtimestamp(rec["t_go"] + lag).isoformat(timespec="seconds"),
                pattern_r_per_tone={f"{f:g}": round(v, 3) for f, v in per_tone.items()},
                starts_tx_s=[round(t, 2) for t, _ in starts],
                frame_s=plan["frame_s"],
                clock_ppm=round((scale - 1) * 1e6), ok=not why, why="; ".join(why))


def resolve(readings, runs, latest_s):
    """readings: [(w, path, ecg)], runs: [(t_go, dir, rec)]. Returns [(reading index, run dir or None, method, evidence)]
    method: 'time' (one candidate), 'signal' (overlap resolved), 'ambiguous', 'none' (no run fits the time)"""
    cand = {i: [d for t0, d, _ in runs if w <= t0 + latest_s and w + 60 >= t0] for i, (w, _, _) in enumerate(readings)}
    rec_of = {d: rec for _, d, rec in runs}
    out, done = [], set()
    for i in range(len(readings)):
        if i in done:
            continue
        if not cand[i]:
            out.append((i, None, "none", [])); done.add(i); continue
        comp_r, comp_d, grow = {i}, set(cand[i]), True        # readings and runs linked through shared candidates
        while grow:
            new_r = {j for j in cand if j not in comp_r and set(cand[j]) & comp_d}
            grow = bool(new_r); comp_r |= new_r
            for j in new_r:
                comp_d |= set(cand[j])
        comp_r = sorted(comp_r); done |= set(comp_r)
        allt = {f for d in comp_d for f in tones(rec_of[d]["plan"])}
        ev = {(j, d): check(readings[j][2], readings[j][0], rec_of[d],
                            foreign=sorted(allt - set(tones(rec_of[d]["plan"]))))
              for j in comp_r for d in cand[j]}
        if len(comp_r) == 1 and len(comp_d) == 1:
            d = cand[i][0]
            out.append((i, d, "time", [ev[(i, d)]])); continue
        valid = [a for a in itertools.product(*[cand[j] for j in comp_r])
                 if len(set(a)) == len(a) and all(ev[(j, d)]["ok"] for j, d in zip(comp_r, a))]
        evid = lambda j: [ev[(j, d)] for d in cand[j]]
        if len(valid) == 1:
            out += [(j, d, "signal", evid(j)) for j, d in zip(comp_r, valid[0])]
        else:
            out += [(j, None, "ambiguous", evid(j)) for j in comp_r]
    return sorted(out)


def print_evidence(ev, indent="      "):
    for x in ev:
        tn = ", ".join(f"{k} Hz {v:+.1f} dB" for k, v in x["tones_db"].items())
        fo = ", ".join(f"{k} Hz {v:+.1f} dB" for k, v in x["foreign_db"].items())
        print(f"{indent}vs {x['run']}: sends {tn}" + (f"; not sent {fo}" if fo else "")
              + f"; pattern r {x['pattern_r']} (" + ", ".join(f"{k} Hz {v:.2f}" for k, v in x.get("pattern_r_per_tone", {}).items())
              + (f"), start T+" + " / ".join(f"{t:g}" for t in x["starts_tx_s"])
                 + f" s (frames repeat every {x.get('frame_s', 0):g} s) -> " if x.get("starts_tx_s")
                 else "), no start time possible in the allowed window -> ")
              + ("CONSISTENT" if x["ok"] else "inconsistent: " + x["why"]))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def session_runs(sess):
    runs = []
    for p in glob.glob(os.path.join(sess, "ecg_*", "tx.json")):
        rec = json.load(open(p))
        if "t_go" in rec:
            rec["_dir"] = os.path.dirname(p)
            runs.append((rec["t_go"], os.path.dirname(p), rec))
    return sorted(runs, key=lambda r: r[0])


IMPORTED = ("ecg.npz", "ecg_source.pdf", "ecg_import.json")


def main():
    import ecg_ber as B
    ap = argparse.ArgumentParser(description="check (and correct) which recording is imported into which run")
    ap.add_argument("session"); ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    sess = os.path.normpath(a.session)
    runs = session_runs(sess)
    readings, current = [], []
    for _, d, _ in runs:
        if os.path.exists(os.path.join(d, "ecg.npz")):
            meta = json.load(open(os.path.join(d, "ecg_import.json")))
            w = B.parse_recorded(meta.get("recorded_on", ""))
            readings.append((w.timestamp(), d, IO.load(os.path.join(d, "ecg.npz"))))
            current.append(d)
    res = resolve(readings, runs, B.ECG_LATEST_S)
    moves = {}
    for i, d, method, ev in res:
        src = os.path.basename(json.load(open(os.path.join(current[i], "ecg_import.json")))["original"])
        print(f"\nrecording {src} (Recorded on {datetime.datetime.fromtimestamp(readings[i][0]):%H:%M}), "
              f"now in {os.path.basename(current[i])}")
        print_evidence(ev)
        if method == "time":
            print(f"   -> only one run fits the time: {os.path.basename(d)}"
                  + ("" if ev[0]["ok"] else "   ! but the signal check is NOT consistent - inspect it"))
        elif method == "signal":
            print(f"   -> signal check: belongs to {os.path.basename(d)}"
                  + ("   (= current)" if d == current[i] else "   ** DIFFERS from the current pairing **"))
        else:
            print(f"   -> {method.upper()}: no unique consistent pairing")
        if d and d != current[i]:
            moves[current[i]] = d
    if not moves:
        print("\npairing verified: every imported recording is in the run the checks find")
        return
    print("\nproposed moves: " + "; ".join(f"{os.path.basename(s)} -> {os.path.basename(t)}" for s, t in moves.items()))
    if any(m == "ambiguous" for _, _, m, _ in res):
        print("some recordings are ambiguous - nothing moved"); return
    if not a.apply:
        print("run again with --apply to move them (old files kept)"); return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    held = {s: {f: open(os.path.join(s, f), "rb").read() for f in IMPORTED} for s in moves}
    for s in set(moves) | set(moves.values()):                 # keep everything that is replaced
        old = os.path.join(s, f"superseded_{stamp}")
        os.makedirs(old, exist_ok=True)
        for f in os.listdir(s):
            if f in IMPORTED or f.startswith("result_"):
                shutil.move(os.path.join(s, f), os.path.join(old, f))
    ev_of = {current[i]: ev for i, _, _, ev in res}
    for s, t in moves.items():
        for f in ("ecg.npz", "ecg_source.pdf"):
            open(os.path.join(t, f), "wb").write(held[s][f])
        meta = json.loads(held[s]["ecg_import.json"])
        meta["pairing"] = dict(method="signal (ecg_pairing.py)", corrected=stamp, previously_in=os.path.basename(s),
                               evidence=ev_of[s])
        json.dump(meta, open(os.path.join(t, "ecg_import.json"), "w"), indent=1, default=str)
    json.dump(dict(session=os.path.basename(sess), corrected=stamp, moves={os.path.basename(s): os.path.basename(t)
                                                                           for s, t in moves.items()},
                   reason="ingest assigned two PDFs with the same 'Recorded on' minute in file-name order",
                   evidence={os.path.basename(s): ev_of[s] for s in moves},
                   old_files=f"<run>/superseded_{stamp}/"),
              open(os.path.join(sess, f"pairing_correction_{stamp}.json"), "w"), indent=1)
    for old in glob.glob(os.path.join(sess, "ecgber_*")):
        open(os.path.join(old, "SUPERSEDED.txt"), "w").write(
            f"Analysed before the pairing correction of {stamp} (recordings were in the wrong runs). "
            f"See ../pairing_correction_{stamp}.json; use the newer ecgber_* folder.\n")
    print(f"moved; correction record pairing_correction_{stamp}.json; re-analysing")
    B.analyse([d for _, d, _ in runs], os.path.basename(sess), outroot=sess)


if __name__ == "__main__":
    main()
