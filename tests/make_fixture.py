#!/usr/bin/env python3
"""
Rebuild tests/fixtures/nosignal_ook4.{npz,json}: the noise-only synthetic recording on which the frozen decoder
ecg-v1 showed both low-SNR failures (2026-11-03, `ecg_ber.py synth --n 2 --amps 1,0.05`, default seed 20261102,
recording ecg_synth_impaired_ber_ook_4bps_00).

It replays the synth command's random draws in the same order (amps x mods x rates x n, then the impaired set) and
takes the same path as `synth` (Fitbit-style PDF -> PDF reader -> ecg.npz). Writes only into tests/fixtures/.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import ecg_io as IO                                         # noqa: E402
import ecg_link as L                                        # noqa: E402
import ecg_synth as Y                                       # noqa: E402

SEED, AMPS, MODS, RATES, N, WHITE_UV = 20261102, [1.0, 0.05], ["ook", "fsk"], [2.0, 4.0, 8.0, 16.0], 2, 10.0
FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def regenerate():
    rng = np.random.default_rng(SEED)
    for amp in AMPS:
        for mod in MODS:
            for bps in RATES:
                plan = L.ber_plan(mod, bps)
                for _ in range(N):
                    Y.record(plan, rng, amp_mvpp=amp, white_uv=WHITE_UV)
    plan = L.ber_plan("ook", 4)
    t, x, info = Y.record(plan, rng, amp_mvpp=1.0, white_uv=WHITE_UV, no_signal=True)     # impaired #0
    with tempfile.TemporaryDirectory() as d:
        Y.write_pdf(os.path.join(d, "x.pdf"), x)
        e = IO.load(os.path.join(d, "x.pdf"))
    if len(e.mv) != len(x) or np.max(np.abs(e.mv - x)) > 1e-9:
        raise RuntimeError("PDF round trip changed the samples")
    return e, plan, info


if __name__ == "__main__":
    e, plan, info = regenerate()
    os.makedirs(FIX, exist_ok=True)
    IO.save_npz(e, os.path.join(FIX, "nosignal_ook4.npz"))
    json.dump(dict(kind="ber", label="synth", condition="impaired", plan=plan, amp_mvpp=1.0, synthetic=True, synth=info,
                   provenance=dict(command="ecg_ber.py synth --n 2 --amps 1,0.05", seed=SEED,
                                   recording="ecg_synth_impaired_ber_ook_4bps_00", found="2026-11-03",
                                   decoder=L.VERSION, rules_sha=L.RULES_SHA)),
              open(os.path.join(FIX, "nosignal_ook4.json"), "w"), indent=1, default=float)
    print(f"wrote {FIX}/nosignal_ook4.npz/.json ({len(e.mv)} samples)")
