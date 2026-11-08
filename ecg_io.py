#!/usr/bin/env python3
"""
ECG recordings from the Pixel Watch 2 (Fitbit ECG app) -> timestamped numeric samples.

   python ecg_io.py <file.pdf|.json|.csv> [--plot out.png]       # parse, print what was found, optional figure

Sources (what each gives, and how sure we are):
  pdf    Fitbit app 'Export a PDF for your doctor' (Heart Rhythm Assessment). VERIFIED on the only real Pixel Watch 2
         export at hand (WatchLink repo, pdf_parser/pixel_1_Hz.pdf, Wear OS 4.0.0, feature 6-2.14.2-2.13-2.16): the
         trace is a VECTOR polyline, one point per 250 Hz sample (x step 0.5 units = 4 ms at 25 mm/s), y to 0.01 unit
         (0.2 uV at 10 mm/mV), in four form XObjects (rows of 8 s, 1000 units). So the PDF holds the full-rate
         waveform - no image processing (WatchLink rasterised it at 72 dpi). Scale is read from the page text
         ('[ 25mm/s, 10mm/mV, 250 Hz ]') and checked against the grid (5 units = 1 mm) and the 1 mV calibration pulse.
         The drawing clip (+-1.5 mV) only hides ink; the coordinates beyond it are kept. Any other app version must
         pass the same checks or the parser stops - it never guesses a scale.
  json   Google Health API v4 'electrocardiogram' dataPoints (waveformSamples / millivoltsScalingFactor /
         samplingFrequencyHertz) or the legacy Fitbit Web API ecgReadings (waveformSamples / scalingFactor /
         samplingFrequencyHz; API switched off 2026-10-30). Field names from the public docs; NOT yet seen for this
         watch.
  csv    Google Takeout (Fitbit 'Afib ECG Readings.csv' with a waveform_samples column - community report only,
         UNVERIFIED) or a plain two-column csv (time_s, mV) written by this project.
Every loader returns Ecg(t, mv, fs, meta): t in seconds from the first sample, mv in millivolts.
Needs: numpy (+ poppler 'pdftotext' for the PDF header text; without it the scale must be given with --scale).
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import zlib
from dataclasses import dataclass, field

import numpy as np

UNITS_PER_MM = 5.0                              # Fitbit PDF row forms: grid lines every 5 units, 1 mm on paper
CAL_MV = 1.0                                    # the calibration pulse at the start of row 1


@dataclass
class Ecg:
    t: np.ndarray
    mv: np.ndarray
    fs: float
    meta: dict = field(default_factory=dict)


# ================================================================ PDF (minimal reader: zlib streams + path operators)
def _objects(b):
    """{obj number: body} with incremental updates applied (last definition wins) and object streams unpacked"""
    out = {}
    for m in re.finditer(rb"(\d+)\s+(\d+)\s+obj(.*?)endobj", b, re.S):
        out[int(m.group(1))] = m.group(3)
    for n, body in list(out.items()):           # PDF 1.5 object streams: non-stream objects packed together
        if b"/ObjStm" not in body.split(b"stream", 1)[0]:
            continue
        data = _stream(body)
        nobj = int(re.search(rb"/N\s+(\d+)", body).group(1)); first = int(re.search(rb"/First\s+(\d+)", body).group(1))
        nums = list(map(int, data[:first].split()))
        offs = [(nums[2 * i], nums[2 * i + 1]) for i in range(nobj)]
        for i, (on, off) in enumerate(offs):
            end = offs[i + 1][1] if i + 1 < nobj else len(data) - first
            out.setdefault(on, data[first + off:first + end])
    return out


def _stream(body):
    i = body.find(b"stream")
    if i < 0:
        return None
    head, raw = body[:i], body[i + 6:]
    raw = raw[2:] if raw[:2] == b"\r\n" else raw[1:] if raw[:1] in b"\r\n" else raw
    raw = raw[:raw.rfind(b"endstream")]
    if b"/FlateDecode" in head:
        return zlib.decompressobj().decompress(raw)
    if b"/Filter" in head:
        raise ValueError(f"unsupported PDF stream filter: {head[:120]!r}")
    return raw


def _paths(content):
    """stroked paths of a content stream: (points as (x, y) after translation cm, stroke rgb, line width).
    Only translations are allowed inside q/Q for a trace; a scaled or rotated trace raises (never mis-scaled)."""
    toks = re.findall(rb"\[[^\]]*\]|\([^)]*\)|<[^>]*>|/[^\s/\[\]()<>]+|[-+]?\d*\.?\d+|[A-Za-z'\"*]+", content)
    stack, ctm, rgb, lw = [], np.eye(3), None, 1.0
    num, cur, path, out = [], [], [], []
    for tk in toks:
        try:
            num.append(float(tk)); continue
        except ValueError:
            pass
        op = tk.decode("latin1")
        if op == "q":
            stack.append((ctm.copy(), rgb, lw))
        elif op == "Q" and stack:
            ctm, rgb, lw = stack.pop()
        elif op == "cm" and len(num) >= 6:
            a, b_, c, d, e, f = num[-6:]
            ctm = np.array([[a, b_, 0], [c, d, 0], [e, f, 1]]) @ ctm
        elif op == "RG" and len(num) >= 3:
            rgb = tuple(num[-3:])
        elif op == "w" and num:
            lw = num[-1]
        elif op == "m" and len(num) >= 2:
            if len(cur) > 1:
                path.append(cur)
            cur = [(num[-2], num[-1])]
        elif op == "l" and len(num) >= 2:
            cur.append((num[-2], num[-1]))
        elif op in ("S", "s"):
            if len(cur) > 1:
                path.append(cur)
            for p in path:
                P = np.c_[np.array(p), np.ones(len(p))] @ ctm
                out.append(dict(xy=P[:, :2], raw=np.array(p, float), ctm=ctm.copy(), rgb=rgb, w=lw,
                                shear=bool(abs(ctm[0, 1]) + abs(ctm[1, 0]) > 1e-9),
                                scale=(float(ctm[0, 0]), float(ctm[1, 1]))))
            cur, path = [], []
        elif op in ("f", "F", "f*", "B", "B*", "b", "b*", "n"):
            cur, path = [], []
        num = []
    return out


def pdf_text(path):
    exe = shutil.which("pdftotext")
    if not exe:
        return None
    return subprocess.run([exe, "-layout", path, "-"], capture_output=True, text=True, check=True).stdout


def _header(text):
    h = {}
    if not text:
        return h
    m = re.search(r"\[\s*([\d.]+)\s*mm/s,\s*([\d.]+)\s*mm/mV,\s*([\d.]+)\s*Hz\s*\]", text)
    if m:
        h.update(mm_per_s=float(m.group(1)), mm_per_mv=float(m.group(2)), fs_hz=float(m.group(3)))
    m = re.search(r"Recorded on ([A-Z][a-z]+ \d{1,2}, \d{4},? \d{1,2}:\d{2}\s*[AP]M)", text)
    if m:
        h["recorded_on"] = m.group(1)
    for key, rx in (("recorded_on", r"Recorded on (.+?)\s*\n"), ("feature_version", r"Feature Version:\s*\[([^\]]*)\]"),
                    ("hardware", r"Hardware:\s*\[([^\]]*)\]"), ("os_version", r"OS Version:\s*\[([^\]]*)\]"),
                    ("average_bpm", r"(\d+)\s*bpm")):
        m = re.search(rx, text)
        if m and key not in h:
            h[key] = m.group(1).strip()
    for cls in ("Inconclusive", "Normal sinus rhythm", "Sinus rhythm", "Atrial fibrillation", "AFib", "Poor recording"):
        if re.search(r"^\s*" + cls, text, re.M | re.I):
            h["classification"] = cls
            break
    return h


def load_pdf(path, scale=None):
    """scale = (mm_per_s, mm_per_mv, fs_hz) when pdftotext is missing; otherwise read from the page and checked"""
    b = open(path, "rb").read()
    objs = _objects(b)
    text = pdf_text(path)
    hdr = _header(text)
    if scale:
        hdr.update(mm_per_s=scale[0], mm_per_mv=scale[1], fs_hz=scale[2])
    if not {"mm_per_s", "mm_per_mv", "fs_hz"} <= set(hdr):
        raise ValueError("scale line '[ 25mm/s, 10mm/mV, 250 Hz ]' not found (pdftotext missing or another layout) - "
                         "give --scale 25,10,250 only if the page really says so")
    u_s, u_mv = hdr["mm_per_s"] * UNITS_PER_MM, hdr["mm_per_mv"] * UNITS_PER_MM
    dx = u_s / hdr["fs_hz"]
    # row forms = Form XObjects whose stroked paths include a long polyline at the sample step; order = page placement
    rows = {}
    for n, body in objs.items():
        if b"/Form" not in body.split(b"stream", 1)[0]:
            continue
        data = _stream(body)
        if not data:
            continue
        P = _paths(data)
        long = [p for p in P if len(p["xy"]) >= 50]
        if not long:
            continue
        tr = max(long, key=lambda p: len(p["xy"]))
        traces = [p for p in long if p["rgb"] == tr["rgb"] and p["w"] == tr["w"]]
        vert = {}                               # vertical grid lines by colour; the finest family is the 1 mm grid
        for p in P:
            if len(p["xy"]) == 2 and abs(p["xy"][0, 0] - p["xy"][1, 0]) < 1e-6:
                vert.setdefault(p["rgb"], set()).add(round(p["xy"][0, 0], 2))
        grid = sorted(max(vert.values(), key=len)) if vert else []
        cal = [p for p in P if len(p["xy"]) == 6 and p["rgb"] != tr["rgb"]]
        bbox = re.search(rb"/BBox\s*\[\s*([-\d.\s]+)\]", body)
        rows[n] = dict(traces=traces, grid=grid, cal=cal, bbox=[float(v) for v in bbox.group(1).split()] if bbox else None)
    if not rows:
        raise ValueError("no ECG trace found (no form XObject with a long stroked polyline)")
    if any(p["scale"] != (1.0, 1.0) for R in rows.values() for p in R["traces"]):
        return _load_scaled(path, rows, hdr)
    order = _row_order(objs, list(rows))
    segs, checks = [], dict(rows=len(order), dx_units=dx, units_per_s=u_s, units_per_mv=u_mv)
    y0 = None
    for r, n in enumerate(order):
        R = rows[n]
        steps = np.diff(R["grid"])
        if len(steps) and abs(np.median(steps) - UNITS_PER_MM) > 0.01:
            raise ValueError(f"row {r}: grid step {np.median(steps):.3f} units, expected {UNITS_PER_MM} (1 mm)")
        for c in R["cal"]:                      # 1 mV pulse: rise of mm_per_mv mm
            rise = float(np.ptp(c["xy"][:, 1]))
            if abs(rise - u_mv * CAL_MV) < 0.5:
                y0 = float(c["xy"][0, 1]); checks["cal_pulse_units"] = rise
        x_left = R["bbox"][0] + 2.0 if R["bbox"] else 2.0     # clip box starts 2 units in (sample form)
        for p in R["traces"]:
            xy = p["xy"]
            if p["shear"] or abs(p["scale"][0] - 1) > 1e-9 or abs(p["scale"][1] - 1) > 1e-9:
                raise ValueError(f"row {r}: trace drawn with a scaled/rotated matrix {p['scale']}")
            d = np.diff(xy[:, 0])
            if np.max(np.abs(d - dx)) > 0.011:
                raise ValueError(f"row {r}: x steps {np.min(d):.3f}..{np.max(d):.3f} units, expected {dx:g} "
                                 "(one point per sample) - not the known layout")
            segs.append((r, x_left, xy))
    if y0 is None:
        y0 = float(np.mean([np.median(xy[:, 1]) for _, _, xy in segs]))
        checks["cal_pulse_units"] = None
    row_s = (rows[order[0]]["bbox"][2] - 4.0) / u_s if rows[order[0]]["bbox"] else 8.0
    t, v = [], []
    for r, xl, xy in segs:
        t.append(r * row_s + (xy[:, 0] - xl) / u_s)
        v.append((xy[:, 1] - y0) / u_mv)
    t, v = np.concatenate(t), np.concatenate(v)
    k = np.argsort(t, kind="stable"); t, v = t[k], v[k]
    n_idx = np.round(t * hdr["fs_hz"]).astype(int)              # sample numbers: every point on the 4 ms lattice
    off = float(np.max(np.abs(t * hdr["fs_hz"] - n_idx)))
    dup = int(np.sum(np.diff(n_idx) == 0))
    gaps = np.diff(n_idx)
    checks.update(row_seconds=row_s, lattice_offset_samples=off, duplicate_points=dup,
                  gaps=[(float(t[i]), int(g - 1)) for i, g in enumerate(gaps) if g > 1],
                  y_resolution_uv=0.01 / u_mv * 1e3, cal_baseline_units=y0)
    if dup:                                     # row ends repeat the boundary sample: keep the first
        keep = np.r_[True, np.diff(n_idx) != 0]
        n_idx, v = n_idx[keep], v[keep]
    t0 = n_idx[0]
    meta = dict(source="pdf", file=os.path.abspath(path), checks=checks, **hdr)
    return Ecg(t=(n_idx - t0) / hdr["fs_hz"], mv=v, fs=hdr["fs_hz"],
               meta=dict(meta, first_sample_s_on_page=float(t0 / hdr["fs_hz"])))


def _load_scaled(path, rows, hdr):
    """Pixel Watch / Fitbit app export as seen 2026-11-06 (semanticsignals_ECGReport_06 Nov 2026_11-40): each row form
    draws the trace in raw units under a uniform scale matrix, y axis pointing down, no stroked 1 mm grid. Everything
    is derived from the page and cross-checked; anything inconsistent raises instead of guessing:
      raw units per mm  from the sample step (mm_per_s / fs mm per sample)  ==  from the 1 mV pulse (mm_per_mv mm), +-1 %
      row order         by the boundary sample each row repeats from the previous one (not by page placement)
      time              row start = row index x row length (full rows equal), points on the 4 ms lattice"""
    mm_per_sample = hdr["mm_per_s"] / hdr["fs_hz"]
    R = {}
    for n, r in rows.items():
        if len(r["traces"]) != 1:
            raise ValueError(f"form {n}: {len(r['traces'])} trace polylines, expected 1 (unknown layout)")
        p = r["traces"][0]
        if p["shear"] or abs(p["scale"][0] - p["scale"][1]) > 1e-9:
            raise ValueError(f"form {n}: trace drawn with a sheared or non-uniform matrix {p['scale']}")
        raw = p["raw"]
        d = np.diff(raw[:, 0])
        if np.max(np.abs(d - np.median(d))) > 1e-3 * abs(np.median(d)) + 1e-4:      # float32 coordinates
            raise ValueError(f"form {n}: uneven x steps {d.min():.6f}..{d.max():.6f} raw units (not one point per sample)")
        cal = []
        for q in r["cal"]:                                   # 1 mV pulse: base, base, top, top, base, base
            c = q["raw"]
            if len(c) == 6 and abs(c[0, 1] - c[1, 1]) < 1e-6 and abs(c[2, 1] - c[3, 1]) < 1e-6 and abs(c[1, 0] - c[2, 0]) < 1e-6:
                cal.append(c)
        R[n] = dict(raw=raw, step=float(np.median(d)), cal=cal)
    steps = [r["step"] for r in R.values()]
    if max(steps) - min(steps) > 1e-5 * max(steps):
        raise ValueError(f"rows have different sample steps {steps}")
    u_mm = steps[0] / mm_per_sample                          # raw units per mm, from the time axis
    cals = [(n, c) for n, r in R.items() for c in r["cal"]]
    if len(cals) != 1:
        raise ValueError(f"{len(cals)} calibration pulses found, expected 1 (row 1)")
    n0, c = cals[0]
    rise = float(c[2, 1] - c[0, 1])                          # signed: + = up, - = the y axis points down
    u_mm_cal = abs(rise) / (hdr["mm_per_mv"] * CAL_MV)
    if abs(u_mm_cal / u_mm - 1) > 0.01:
        raise ValueError(f"scale mismatch: {u_mm:.5f} raw units/mm from the sample step, {u_mm_cal:.5f} from the 1 mV pulse")
    sign = 1.0 if rise > 0 else -1.0
    y0 = float(c[0, 1])
    u_s, u_mv = u_mm * hdr["mm_per_s"], u_mm * hdr["mm_per_mv"]
    order, left = [n0], set(R) - {n0}                        # chain the rows by the repeated boundary sample
    while left:
        last = R[order[-1]]["raw"][-1, 1]
        nxt = [n for n in left if abs(R[n]["raw"][0, 1] - last) < 1e-4 * u_mv]
        if len(nxt) != 1:
            raise ValueError(f"row order: {len(nxt)} rows continue row {len(order) - 1} (expected exactly 1)")
        order.append(nxt[0]); left.discard(nxt[0])
    x_left = min(R[n]["raw"][0, 0] for n in order[1:]) if len(order) > 1 else R[n0]["raw"][0, 0]
    full = [R[n]["raw"][-1, 0] - x_left for n in order[1:-1]] or [R[n0]["raw"][-1, 0] - x_left]
    if max(full) - min(full) > 0.5 * steps[0]:
        raise ValueError(f"full rows have different lengths {full}")
    row_s = round(full[0] / u_s * hdr["fs_hz"]) / hdr["fs_hz"]
    t, v = [], []
    for r, n in enumerate(order):
        raw = R[n]["raw"]
        t.append(r * row_s + (raw[:, 0] - x_left) / u_s)
        v.append(sign * (raw[:, 1] - y0) / u_mv)
    t, v = np.concatenate(t), np.concatenate(v)
    n_idx = np.round(t * hdr["fs_hz"]).astype(int)
    off = float(np.max(np.abs(t * hdr["fs_hz"] - n_idx)))
    if off > 0.01:
        raise ValueError(f"points are off the {1e3 / hdr['fs_hz']:g} ms sample lattice by up to {off:.3f} samples")
    dup = np.flatnonzero(np.diff(n_idx) == 0)
    if np.any(np.abs(v[dup] - v[dup + 1]) > 1e-6):
        raise ValueError("repeated boundary samples differ (row order or offsets wrong)")
    keep = np.r_[True, np.diff(n_idx) != 0]
    n_idx, v = n_idx[keep], v[keep]
    gaps = np.diff(n_idx)
    checks = dict(layout="scaled rows (Pixel Watch app export)", rows=len(order), matrix_scale=rows[n0]["traces"][0]["scale"][0],
                  raw_units_per_mm_time=u_mm, raw_units_per_mm_cal=u_mm_cal, y_axis="down" if sign < 0 else "up",
                  row_order="boundary-sample chain", row_seconds=row_s, lattice_offset_samples=off,
                  duplicate_points=int(len(dup)), gaps=[(float(n_idx[i] / hdr["fs_hz"]), int(g - 1)) for i, g in enumerate(gaps) if g > 1],
                  cal_baseline_raw=y0, units_per_s=u_s, units_per_mv=u_mv)
    t0 = n_idx[0]
    meta = dict(source="pdf", file=os.path.abspath(path), checks=checks, **hdr)
    return Ecg(t=(n_idx - t0) / hdr["fs_hz"], mv=v, fs=hdr["fs_hz"],
               meta=dict(meta, first_sample_s_on_page=float(t0 / hdr["fs_hz"])))


def _row_order(objs, cand):
    """order the row forms top to bottom by where the page content stream places them (/Name Do after a cm)"""
    names = {}
    for body in objs.values():
        for m in re.finditer(rb"/XObject\s*<<(.*?)>>", body, re.S):
            for nm, ref in re.findall(rb"/([^\s/<>\[\]]+)\s+(\d+)\s+\d+\s+R", m.group(1)):
                if int(ref) in cand:
                    names[nm.decode()] = int(ref)
    ypos = {}
    for body in objs.values():
        if b"stream" not in body or b"/Form" in body.split(b"stream", 1)[0]:
            continue
        try:
            data = _stream(body)
        except Exception:
            continue
        if not data or b" Do" not in data:
            continue
        f = None                                # y translation of the last cm before each '/Name Do'
        for m in re.finditer(rb"([-\d.]+)\s+([-\d.]+)\s+cm\b|/([^\s/\[\]()<>]+)\s+Do\b", data):
            if m.group(3) is None:
                f = float(m.group(2))
            elif m.group(3).decode() in names and f is not None:
                ypos[names[m.group(3).decode()]] = f
    if len(ypos) == len(cand):
        return sorted(cand, key=lambda n: -ypos[n])
    print(f"  ! row placement not found for every row ({len(ypos)}/{len(cand)}) - using object order", file=sys.stderr)
    return sorted(cand)


# ================================================================ JSON / CSV
def load_json(path, index=0):
    d = json.load(open(path))
    if isinstance(d, dict) and "ecgReadings" in d:            # legacy Fitbit Web API
        r = d["ecgReadings"][index]
        w, sc, fs = r["waveformSamples"], r["scalingFactor"], float(r["samplingFrequencyHz"])
        meta = {k: r.get(k) for k in ("startTime", "averageHeartRate", "resultClassification", "deviceName",
                                      "firmwareVersion", "featureVersion", "leadNumber")}
        src = "fitbit_web_api"
    else:                                                     # Google Health API v4 dataPoints
        pts = d.get("dataPoints", d if isinstance(d, list) else [d])
        p = pts[index]
        r = p.get("electrocardiogram", p)
        w, sc, fs = r["waveformSamples"], r["millivoltsScalingFactor"], float(r["samplingFrequencyHertz"])
        meta = dict(interval=r.get("interval"), classification=r.get("resultClassification"),
                    average_bpm=r.get("beatsPerMinuteAvg"), device=r.get("medicalDeviceInfo"))
        src = "google_health_api"
    mv = np.asarray(w, float) / float(sc)
    return Ecg(t=np.arange(len(mv)) / fs, mv=mv, fs=fs, meta=dict(source=src, file=os.path.abspath(path),
                                                                  scaling=sc, fs_hz=fs, **meta))


def load_csv(path, index=0, fs=None, scale=None):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise ValueError("empty csv")
    cols = {c.lower(): c for c in rows[0]}
    wcol = next((cols[c] for c in cols if "waveform" in c), None)
    if wcol:                                                  # Takeout-style: one reading per row (UNVERIFIED layout)
        r = rows[index]
        w = [float(v) for v in re.findall(r"-?\d+\.?\d*", r[wcol])]
        fcol = next((cols[c] for c in cols if "frequency" in c or "sampling" in c), None)
        scol = next((cols[c] for c in cols if "scal" in c), None)
        fs = fs or (float(r[fcol]) if fcol else None)
        scale = scale or (float(r[scol]) if scol else None)
        if not fs or not scale:
            raise ValueError(f"csv has {wcol!r} but no sampling-frequency / scaling column: give --fs and --scale-div "
                             f"(columns: {list(rows[0])})")
        mv = np.asarray(w) / scale
        return Ecg(t=np.arange(len(mv)) / fs, mv=mv, fs=fs, meta=dict(source="csv_waveform", file=os.path.abspath(path),
                                                                      columns=list(rows[0]), row=r.get(cols.get("reading_time", ""), index)))
    tcol = next(c for c in rows[0] if c.lower().startswith("t"))
    vcol = next(c for c in rows[0] if c != tcol)
    t = np.array([float(r[tcol]) for r in rows]); mv = np.array([float(r[vcol]) for r in rows])
    return Ecg(t=t - t[0], mv=mv, fs=float(1 / np.median(np.diff(t))), meta=dict(source="csv", file=os.path.abspath(path)))


def load(path, **kw):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path, kw.get("scale"))
    if ext == ".json":
        return load_json(path, kw.get("index", 0))
    if ext == ".csv":
        return load_csv(path, kw.get("index", 0), kw.get("fs"), kw.get("scale_div"))
    if ext == ".npz":
        z = np.load(path, allow_pickle=False)
        return Ecg(t=z["t"], mv=z["mv"], fs=float(z["fs"]), meta=json.loads(str(z["meta"])))
    raise ValueError(f"unknown ECG file type {ext}")


def save_npz(e, path):
    np.savez_compressed(path, t=e.t, mv=e.mv, fs=e.fs, meta=json.dumps(e.meta))


def describe(e):
    c = e.meta.get("checks", {})
    s = (f"{e.meta.get('source')}: {len(e.mv)} samples, {e.fs:g} Hz, {e.t[-1] + 1 / e.fs:.2f} s, "
         f"{np.min(e.mv):+.3f}..{np.max(e.mv):+.3f} mV")
    for k in ("recorded_on", "hardware", "os_version", "feature_version", "classification", "average_bpm"):
        if e.meta.get(k):
            s += f"\n  {k}: {e.meta[k]}"
    if c and c.get("layout", "").startswith("scaled"):
        s += (f"\n  checks: {c['layout']}, {c['rows']} rows of {c['row_seconds']:g} s, matrix scale {c['matrix_scale']:g}, "
              f"raw units/mm {c['raw_units_per_mm_time']:.4f} (time) vs {c['raw_units_per_mm_cal']:.4f} (1 mV pulse), "
              f"y axis {c['y_axis']}, row order by {c['row_order']}, lattice offset {c['lattice_offset_samples']:.3f} "
              f"samples, duplicates {c['duplicate_points']}, gaps {c['gaps'] or 'none'}")
    elif c:
        s += (f"\n  checks: {c['rows']} rows of {c['row_seconds']:g} s, x step {c['dx_units']:g} units, "
              f"cal pulse {c.get('cal_pulse_units')} units (= 1 mV), lattice offset {c['lattice_offset_samples']:.3f} "
              f"samples, duplicates {c['duplicate_points']}, gaps {c['gaps'] or 'none'}, y resolution "
              f"{c['y_resolution_uv']:.2f} uV")
    return s


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file")
    ap.add_argument("--plot")
    ap.add_argument("--csv", help="also write time_s,mV to this csv")
    a = ap.parse_args()
    e = load(a.file)
    print(describe(e))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["time_s", "mV"]); w.writerows(zip(np.round(e.t, 6), e.mv))
    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 5.5))
        ax[0].plot(e.t, e.mv, lw=0.7, color="#1f5fbf"); ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("mV")
        f = np.fft.rfftfreq(len(e.mv), 1 / e.fs); S = np.abs(np.fft.rfft((e.mv - e.mv.mean()) * np.hanning(len(e.mv))))
        ax[1].semilogy(f, S + 1e-9, lw=0.7, color="#1f5fbf"); ax[1].set_xlabel("frequency [Hz]"); ax[1].set_ylabel("|X|")
        fig.suptitle(os.path.basename(a.file)); fig.tight_layout(); fig.savefig(a.plot, dpi=130)
        print(f"figure {a.plot}")
