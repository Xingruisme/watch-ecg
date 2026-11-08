#!/usr/bin/env python3
"""
How often the frozen decoder (ecg-v1) and the usability rules accept a recording that contains NO tone.

Exploratory (not a benchmark): noise-only synthetic recordings with the dry-run noise model (white 10 uV, mains 5 uV,
wander 30 uV, ecg_synth.record(no_signal=True)), decoded exactly as analyse_ber does. Writes
data_watch_ecg/exploratory_falsealarm_<time>/ (scan.json, scan.csv). Nothing else is read or changed.

   ~/.venvs/galvanic/bin/python tests/false_alarm_scan.py --n 100
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import ecg_ber as B                                         # noqa: E402
import ecg_link as L                                        # noqa: E402
import ecg_synth as Y                                       # noqa: E402


def one(plan, rng):
    t, x, info = Y.record(plan, rng, no_signal=True)
    rep, _ = L.decode(x, t, plan)
    e = types.SimpleNamespace(t=t, mv=x, fs=Y.FS_NOM)
    u = B.usability(e, rep["line_snr_db"])
    fr = rep["frames"]
    return dict(line_snr_db=rep["line_snr_db"], clock_ppm=rep["clock_ppm"], usable=u["usable"],
                frames_valid=sum(f["valid"] for f in fr), frames=len(fr),
                errors_valid=sum(f["errors"] for f in fr if f["valid"]), bits_valid=sum(f["bits"] for f in fr if f["valid"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="noise-only recordings per modulation x rate")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--mods", default="ook,fsk"); ap.add_argument("--rates", default="2,4,8,16")
    a = ap.parse_args()
    if L.RULES_SHA != B.FROZEN_SHA:
        sys.exit("ecg_link.py is not the frozen decoder")
    rng = np.random.default_rng(a.seed)
    out = os.path.join(B.DATADIR, f"exploratory_falsealarm_{B.stamp()}")
    os.makedirs(out)
    rows, cells = [], []
    for mod in a.mods.split(","):
        for bps in (float(r) for r in a.rates.split(",")):
            plan = L.ber_plan(mod, bps)
            rr = [dict(one(plan, rng), modulation=mod, bps=bps, i=i) for i in range(a.n)]
            rows += rr
            us = [r for r in rr if r["usable"]]
            c = dict(modulation=mod, bps=bps, n=a.n,
                     line_snr_ge_6db=sum(r["line_snr_db"] >= B.MIN_LINE_SNR_DB for r in rr),
                     usable=len(us), usable_with_valid_frame=sum(r["frames_valid"] > 0 for r in us),
                     valid_frames=sum(r["frames_valid"] for r in us), frames=sum(r["frames"] for r in us),
                     ber_of_false_frames=(sum(r["errors_valid"] for r in us) / max(1, sum(r["bits_valid"] for r in us))),
                     line_snr_db_median=float(np.median([r["line_snr_db"] for r in rr])))
            cells.append(c)
            print(f"{L.MOD_NAMES[mod]} {bps:4g} bps: line >= 6 dB in {c['line_snr_ge_6db']}/{a.n}, usable {c['usable']}/{a.n}, "
                  f"usable with >= 1 valid frame {c['usable_with_valid_frame']}/{a.n}, valid frames {c['valid_frames']}/"
                  f"{c['frames']}, BER of those frames {c['ber_of_false_frames']:.2f}", flush=True)
    with open(os.path.join(out, "scan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(created=datetime.datetime.now().isoformat(timespec="seconds"), decoder=L.VERSION, rules_sha=L.RULES_SHA,
                   seed=a.seed, n=a.n, noise=dict(white_uv=10.0, mains_uv=5.0, wander_uv=30.0), cells=cells),
              open(os.path.join(out, "scan.json"), "w"), indent=1, default=float)
    print(f"-> {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
