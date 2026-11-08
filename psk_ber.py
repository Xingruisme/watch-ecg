"""EXPERIMENTAL DBPSK BER through the worn Pixel Watch 2 ECG (psk-x3 receivers, watch_psk_tx x2). Added 2026-11-06.

Separate from ecg_ber.py / ecg-v1 (unchanged). Same procedure: start a session, the Arduino transmits, the terminal
cues the 30 s ECG, export the PDFs, ingest, BER. Protocol and thresholds: psk_protocol.json (pre-registered).

  python psk_ber.py run --bps 16 --label d16 [--fc 16] [--shape rrc] [--repeats 3] [calibration]
  python psk_ber.py ingest <session folder> <PDF folder or files>
  python psk_ber.py analyse <labels or sessions...> --label curve1        (BER-vs-bitrate table + curve + next step)
  python psk_ber.py next <labels or sessions...>                           (what to test next, per the protocol)

'run' starts directly (review gate removed 2026-11-07 at the operator's request). Kept: level <= 10 mVpp
(also clamped in firmware) and the watch_psk_tx firmware check.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

import arduino_tx as AT
import ecg_ber as B
import ecg_io as IO
import psk_diag as D
import psk_link as P
import psk_rx4 as X4
import psk_tx as PT

HERE = B.HERE
DATADIR = B.DATADIR
PROTOCOL = os.path.join(HERE, "psk_protocol.json")
FROZEN_RECEIVER = ("psk-x3", "04dbb95193d8")           # psk_link VERSION, RULES_SHA at the freeze
FROZEN_X4 = ("psk-x4", "96d8ccdb9e7c")                 # psk_rx4 VERSION, RULES_SHA at the freeze (x4 arm)
X4_DEVELOPMENT = ("psk_d40_r1_20261107_174709", "psk_d40_r2_20261107_174843")   # used to develop x4: never in its verdicts
SECONDS = 55.0
SPACING_S = 80.0
MAX_MVPP = 10.0                                         # firmware also clamps (MAX_WATCH_MVPP)
ECG_START_S, ECG_LATEST_S = 5, 20
RECEIVERS = (("x3-wl", lambda x, t, p: P.decode_wl(x, t, p)),            # primary (pre-registered)
             ("x2-diff eq", lambda x, t, p: P.decode(x, t, p, eq=True)),  # secondary, same recordings
             ("x4-mb", lambda x, t, p: X4.decode(x, t, p)))               # improved receiver (x4 arm, 2026-11-07)
X4_NAME = "x4-mb"


def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def protocol():
    raw = open(PROTOCOL, "rb").read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()[:12]


def check_frozen(allow=False):
    if (X4.VERSION, X4.RULES_SHA) != FROZEN_X4:
        msg = (f"psk_rx4.py is {X4.VERSION} sha {X4.RULES_SHA}, the frozen x4 receiver is {FROZEN_X4[0]} sha {FROZEN_X4[1]}")
        if not allow:
            sys.exit(msg + " - restore it, or give the change a new VERSION and pass --allow-modified-receiver")
        print("  ! " + msg)
    ok = (P.VERSION, P.RULES_SHA) == FROZEN_RECEIVER
    if not ok:
        msg = (f"psk_link.py is {P.VERSION} sha {P.RULES_SHA}, the frozen receiver is {FROZEN_RECEIVER[0]} sha "
               f"{FROZEN_RECEIVER[1]}: results would not be comparable")
        if not allow:
            sys.exit(msg + " - restore it, or give the change a new VERSION and pass --allow-modified-receiver")
        print("  ! " + msg)
    return ok


def voltage(cal, nominal, bench=None):
    """watch input voltage: CALCULATED from the settings; an off-body bench measurement is kept separate"""
    ratio = cal["r2"] / (cal["r1"] + cal["r2"])
    v = dict(point=B.WATCH_POINT, status="calculated", directly_measured=False, vpp_mv=cal["amp_mvpp"],
             dc_mv=cal["vfull"] * AT.MID / 4095 * ratio * 1e3,
             basis=f"DAC peak-to-peak for {cal['amp_mvpp']:g} mVpp x divider R1 {cal['r1']:g} / R2 {cal['r2']:g} Ohm; "
                   f"nominal (not measured): {', '.join(nominal) or 'none'}")
    if bench:
        v["bench_offbody"] = bench
    return v


def bench_level(path, fc, shape):
    """off-body divider-output amplitude from a psk_bench report (watch NOT connected) - never the worn voltage"""
    r = json.load(open(path))
    seg = [s for s in r["segments"] if s["plan"]["fc_hz"] == fc and s["plan"]["shape"] == shape] or r["segments"]
    return dict(report=os.path.abspath(path), mvpp=float(np.median([s["fit"]["amp_mvpp"] for s in seg])),
                note="PicoScope, divider output, watch NOT connected (off-body); not the worn electrode voltage")


# ================================================================ run (human-connected)
def parse_ohm(v):
    v = v.strip().lower().replace("ohm", "").replace("ω", "")
    mult = 1.0
    if v.endswith("k"):
        mult, v = 1e3, v[:-1]
    elif v.endswith("m"):
        mult, v = 1e6, v[:-1]
    return float(v) * mult


def run(a):
    check_frozen(a.allow_modified_receiver)
    prot, psha = protocol()
    given = [k for k in AT.DEFAULTS if getattr(a, k) is not None]
    cal = {k: (getattr(a, k) if getattr(a, k) is not None else v) for k, v in AT.DEFAULTS.items()}
    src = {k: ("given" if k in given else "nominal default") for k in AT.DEFAULTS}
    nominal = [k for k in AT.DEFAULTS if src[k] == "nominal default"]
    if cal["amp_mvpp"] > MAX_MVPP:
        sys.exit(f"level {cal['amp_mvpp']:g} mVpp above the {MAX_MVPP:g} mVpp limit - nothing transmitted")
    p = P.plan("dbpsk", a.bps, fc=a.fc, shape=a.shape, seconds=a.seconds)
    t = PT.PskTX(a.port)
    if PT.FW_ID not in (t.ident or ""):
        t.dev.close()
        sys.exit(f"board reports '{t.ident}', not '{PT.FW_ID}': upload arduino/watch_psk_tx first - nothing transmitted")
    print(f"DBPSK {p['bps']:g} bps (= {p['baud']:g} baud), fc {p['fc_hz']:g} Hz ({p['cycles_per_symbol']:.2f} cycles/symbol), "
          f"{p['shape']}; {p['frames']} frames x {p['frame_s']:.2f} s; payload {p['payload_bps']:.2f} bps; "
          f"{a.repeats} recording(s), {a.spacing:g} s apart")
    print(f"receiver {P.VERSION} sha {P.RULES_SHA}; protocol sha {psha}; level {cal['amp_mvpp']:g} mVpp CALCULATED"
          + (f" (nominal: {', '.join(nominal)})" if nominal else "") + f"; {t.ident}")
    sess = os.path.join(DATADIR, f"psk_session_{a.label}_{p['bps']:g}bps_fc{p['fc_hz']:g}_{p['shape']}_{stamp()}")
    os.makedirs(sess)
    bench = bench_level(a.bench_report, p["fc_hz"], p["shape"]) if a.bench_report else None
    safety = "none: review gate removed at the operator's request (2026-11-07)"
    S = dict(kind="psk_session", label=a.label, created=datetime.datetime.now().isoformat(timespec="seconds"),
             condition=a.condition, plan=p, receiver=P.params(), protocol_sha=psha, safety_review=safety,
             cal=cal, cal_source=src, nominal=nominal, voltage=voltage(cal, nominal, bench), runs=[])
    try:
        S["firmware"] = t.ident
        S["tx_cfg"] = t.configure(cal["vfull"], cal["r1"], cal["r2"], cal["amp_mvpp"], cal["tx_ppm"])
        json.dump(S, open(os.path.join(sess, "session.json"), "w"), indent=1, default=float)
        last = None
        for i in range(a.repeats):
            if last is not None:
                wait = last + a.spacing - time.time()
                while wait > 0:
                    print(f"\r  next recording in {wait:4.0f} s (keeps each PDF's minute unambiguous)  ", end="", flush=True)
                    time.sleep(min(1.0, wait)); wait = last + a.spacing - time.time()
                print()
            d = os.path.join(sess, f"psk_{a.label}_r{i + 1}_{stamp()}")
            os.makedirs(d)
            input(f"\n[{i + 1}/{a.repeats}] open the ECG app (do NOT start it), finger ready - press Enter to transmit ")
            rec = dict(kind="psk", run=os.path.basename(d), session=os.path.basename(sess), plan=p, condition=a.condition,
                       receiver=P.params(), protocol_sha=psha, firmware=t.ident, cal=cal, nominal=nominal,
                       voltage=S["voltage"], safety_review=safety)
            try:
                up, t_go = t.play(p)
                rec.update(upload=up, t_go=t_go, t_go_local=datetime.datetime.fromtimestamp(t_go).isoformat(timespec="seconds"))
                json.dump(rec, open(os.path.join(d, "tx.json"), "w"), indent=1, default=float)
                cues = {0: "TRANSMITTING - do not start the ECG yet",
                        ECG_START_S: ">>> NOW: start the 30 s ECG recording (by T+15 s) <<<",
                        ECG_START_S + 35: "the ECG should be finished"}
                shown, done = set(), None
                while time.time() < t_go + p["total_s"] + 5 and not done:
                    ln = t.readline(0.1)
                    if ln and ln.startswith("# done"):
                        done = ln
                    el = time.time() - t_go
                    for k, msg in cues.items():
                        if el >= k and k not in shown:
                            shown.add(k); print(f"  T+{k:3d} s  {msg}")
                rec["done_reply"] = done
                print(f"  T+{time.time() - t_go:3.0f} s  transmission " + ("complete" if done else "NOT confirmed (no '# done')"))
                last = t_go
            except KeyboardInterrupt:
                rec["stopped"] = "Ctrl+C"; print("  stopped (Ctrl+C)")
            finally:
                try:
                    t.ask("x", r"^# off")
                except Exception as ex:
                    print(f"  ! could not confirm the output is off ({ex})")
                rec["t_end"] = time.time()
                json.dump(rec, open(os.path.join(d, "tx.json"), "w"), indent=1, default=float)
            rec["operator_note"] = input("  Did the watch complete the 30 s ECG? (y / n / note) ").strip()
            json.dump(rec, open(os.path.join(d, "tx.json"), "w"), indent=1, default=float)
            S["runs"].append(rec["run"])
            json.dump(S, open(os.path.join(sess, "session.json"), "w"), indent=1, default=float)
            if rec.get("stopped"):
                break
    finally:
        try:
            t.ask("x", r"^# off")
        except Exception:
            pass
        t.dev.close()
    rel = os.path.relpath(sess, HERE)
    print(f"\nexport the {len(S['runs'])} ECG PDF(s) into ONE new folder, e.g. ~/Downloads/{a.label}/, then:\n"
          f"  python psk_ber.py ingest {rel} ~/Downloads/{a.label}/")


# ================================================================ ingest
def session_runs(sess):
    out = []
    for p in sorted(glob.glob(os.path.join(sess, "psk_*", "tx.json"))):
        r = json.load(open(p))
        if "t_go" in r:
            out.append((r["t_go"], os.path.dirname(p), r))
    return sorted(out, key=lambda x: x[0])


def ingest(a):
    check_frozen(a.allow_modified_receiver)
    sess = os.path.normpath(a.session)
    runs = session_runs(sess)
    files = []
    for f in a.files:
        if not os.path.exists(f):
            sys.exit(f"{f}: no such file or folder - nothing imported")
        files += sorted(glob.glob(os.path.join(f, "*.pdf"))) if os.path.isdir(f) else [f]
    if not files:
        sys.exit(f"no PDF files in {' '.join(a.files)} - nothing imported")
    seen = set()
    for f in files:
        try:
            e = IO.load(f)
        except Exception as ex:
            print(f"  skipped {os.path.basename(f)}: not readable as an ECG export ({ex})"); continue
        key = hash(e.mv.tobytes())
        if key in seen:
            print(f"  skipped {os.path.basename(f)}: same reading as an earlier file"); continue
        seen.add(key)
        w = B.parse_recorded(e.meta.get("recorded_on", ""))
        if w is None:
            print(f"  skipped {os.path.basename(f)}: no 'Recorded on' time"); continue
        w = w.timestamp()
        cand = [(t0, d, r) for t0, d, r in runs if w <= t0 + ECG_LATEST_S and w + 60 >= t0]
        tag = f"{os.path.basename(f)} ({datetime.datetime.fromtimestamp(w):%H:%M})"
        if not cand:
            print(f"  skipped {tag}: no run of this session was transmitting then"); continue
        method = "time"
        if len(cand) > 1:                                # spacing should prevent this; decide by signal or refuse
            sc = []
            for t0, d, r in cand:
                fw = PT.fw_payload(r["plan"])
                rho, _, _ = D.locate(e.mv - e.mv.mean(), e.fs, r["plan"], fw, w - t0 - 5, w + 60 - t0 + 5)
                sc.append(rho)
            o = np.argsort(sc)[::-1]
            if sc[o[0]] >= D.RHO_MIN and sc[o[0]] >= 1.5 * sc[o[1]]:
                cand, method = [cand[o[0]]], "signal (known-waveform correlation)"
            else:
                print(f"  NOT imported {tag}: {len(cand)} runs fit this minute and the signal does not decide "
                      f"(rho {', '.join(f'{v:.2f}' for v in sc)}); import by hand if you know which run it is")
                continue
        t0, d, r = cand[0]
        if os.path.exists(os.path.join(d, "ecg.npz")):
            same = hashlib.sha256(open(os.path.join(d, "ecg_source.pdf"), "rb").read()).digest() == \
                   hashlib.sha256(open(f, "rb").read()).digest()
            print(f"  {os.path.basename(d)} already has " + ("this ECG - kept" if same else
                  f"a DIFFERENT ECG - kept, {os.path.basename(f)} not imported"))
            continue
        shutil.copy2(f, os.path.join(d, "ecg_source.pdf"))
        IO.save_npz(e, os.path.join(d, "ecg.npz"))
        json.dump(dict(e.meta, samples=len(e.mv), imported=datetime.datetime.now().isoformat(timespec="seconds"),
                       original=os.path.abspath(f), pairing=method, recorded_minute=w),
                  open(os.path.join(d, "ecg_import.json"), "w"), indent=1, default=str)
        print(f"  {tag} -> {os.path.basename(d)}  (paired by {method})")
    return analyse([sess], os.path.basename(sess), outroot=sess)


# ================================================================ analysis
def collect(specs):
    dirs = []
    for s in specs:
        inside = glob.glob(os.path.join(s, "psk_*", "tx.json")) if os.path.isdir(s) else []
        if os.path.isdir(s) and os.path.exists(os.path.join(s, "tx.json")):
            dirs.append(s)
        elif inside:
            dirs += [os.path.dirname(p) for p in inside]
        else:                                       # a label (also when a same-named folder without runs exists)
            dirs += [os.path.dirname(p) for p in glob.glob(os.path.join(DATADIR, f"psk_session_{s}_*", "psk_*", "tx.json"))]
    return sorted(set(os.path.normpath(d) for d in dirs))


def analyse_run(d, rec):
    p = rec["plan"]
    e = IO.load(os.path.join(d, "ecg.npz"))
    imp = json.load(open(os.path.join(d, "ecg_import.json")))
    w = imp.get("recorded_minute") or B.parse_recorded(imp["recorded_on"]).timestamp()
    u = B.usability(e, None)
    dur = float(u["duration_s"])
    out = []
    diag = None
    for name, dec in RECEIVERS:
        rep = dec(e.mv, e.t - e.t[0], p)
        if diag is None:       # one known-transmission diagnosis per recording (independent of the receiver)
            diag = D.diagnose(e, p, w - rec["t_go"] - 5, w + 60 - rec["t_go"] + 5, rep, u["usable"])
        if name == X4_NAME and os.path.basename(d) in X4_DEVELOPMENT:
            name = X4_NAME + " (development)"                 # reported, never part of an x4 verdict
        dg = dict(diag)
        if name != RECEIVERS[0][0]:                          # re-classify for this receiver
            dg = D.diagnose(e, p, w - rec["t_go"] - 5, w + 60 - rec["t_go"] + 5, rep, u["usable"])
        v = [f for f in rep.get("frames", []) if f["valid"]]
        ft = dg.get("frames_transmitted")
        if ft is None:
            ft = max(int(dur // p["frame_s"]) - 1, 0)
        bits = sum(f["bits"] for f in v); errs = sum(f["errors"] for f in v)
        ideal = dg.get("ideal") or {}
        out.append(dict(run=os.path.basename(d), session=rec.get("session"), receiver=name, receiver_sha=rep.get("rules_sha"),
                        condition=rec.get("condition"), bps=p["bps"], baud=p["baud"], fc_hz=p["fc_hz"], shape=p["shape"],
                        cycles_per_symbol=p["cycles_per_symbol"], payload_bps=p["payload_bps"],
                        usable=u["usable"], usable_reason=u["reason"], duration_s=dur,
                        frames_transmitted=ft, frames_valid=len(v), frames_found=len(rep.get("frames", [])),
                        bits=bits, errors=errs, ber=(errs / bits) if bits else None,
                        correct_payload_bits=bits - errs, sync_score=rep.get("sync_score"),
                        power_line_db=rep.get("power_line_db"), clock_ppm=rep.get("clock_ppm"),
                        genie_clock_ppm=dg.get("clock_ppm"), clock_error_ppm=dg.get("clock_error_ppm"),
                        timing_error_symbols=dg.get("timing_error_symbols"), symbol_snr_db=rep.get("symbol_snr_db"),
                        genie_rho=dg.get("rho"), genie_snr_db=ideal.get("snr_db"), genie_bits=ideal.get("bits"),
                        genie_errors=ideal.get("errors"), diagnosis=dg.get("diagnosis"),
                        receiver_reason=rep.get("reason"),
                        voltage_status=rec["voltage"]["status"], voltage_vpp_mv=rec["voltage"]["vpp_mv"],
                        bench_offbody_mvpp=(rec["voltage"].get("bench_offbody") or {}).get("mvpp")))
    return out


def verdict(s, prot):
    r = prot["reliable"]
    if s["usable"] < r["min_usable_recordings"] or s["bits"] < r["min_decoded_bits"]:
        return "unreliable" if (s["usable"] >= r["min_usable_recordings"] and s["bits"] == 0) else "insufficient data"
    ok = s["sync_rate"] >= r["frame_sync_min"] and s["ci95_hi"] < r["ber_ci95_upper_max"]
    return "reliable" if ok else "unreliable"


def summarise(rows, prot):
    key = lambda r: (r["receiver"], r["condition"] or "", r["fc_hz"], r["shape"], r["voltage_status"], r["bps"])
    out = []
    for k in sorted({key(r) for r in rows}):
        rr = [r for r in rows if key(r) == k]
        ok = [r for r in rr if r["usable"]]
        n = sum(r["bits"] for r in ok); e = sum(r["errors"] for r in ok)
        ft = sum(r["frames_transmitted"] for r in ok); fv = sum(r["frames_valid"] for r in ok)
        lo, hi = B.cp(e, n)
        secs = sum(r["duration_s"] for r in ok)
        gb = sum(r["genie_bits"] or 0 for r in ok); ge = sum(r["genie_errors"] or 0 for r in ok)
        med = lambda f: float(np.median([r[f] for r in ok if r[f] is not None])) if any(r[f] is not None for r in ok) else None
        diag = {}
        for r in ok:
            diag[r["diagnosis"]] = diag.get(r["diagnosis"], 0) + 1
        s = dict(receiver=k[0], condition=k[1], fc_hz=k[2], shape=k[3], voltage_status=k[4], bps=k[5],
                 payload_bps=rr[0]["payload_bps"], recordings=len(rr), usable=len(ok), frames_transmitted=ft,
                 frames_valid=fv, sync_rate=(fv / ft) if ft else 0.0, bits=n, errors=e,
                 ber=(e / n) if n else None, ci95_lo=lo, ci95_hi=hi,
                 goodput_bps=((n - e) / secs) if secs else 0.0,
                 clock_error_ppm_median=med("clock_error_ppm"), timing_error_symbols_median=med("timing_error_symbols"),
                 symbol_snr_db_median=med("symbol_snr_db"), genie_snr_db_median=med("genie_snr_db"),
                 genie_bits=gb, genie_errors=ge, genie_ci95_hi=B.cp(ge, gb)[1] if gb else None,
                 diagnosis=";".join(f"{k_} x{v}" for k_, v in sorted(diag.items())))
        s["verdict"] = verdict(s, prot)
        out.append(s)
    return out


def next_step(summ, prot):
    """the protocol's stepping rule on the primary receiver + primary condition"""
    pc = prot["primary_condition"]
    # the rate is keyed to 0.01 bps: the symbol period is whole microseconds, so 24 bps is played as 23.9998 bps
    prim = {round(s["bps"], 2): s for s in summ if s["receiver"] == RECEIVERS[0][0] and s["fc_hz"] == pc["fc_hz"]
            and s["shape"] == pc["shape"]}
    ladder = prot["ladder_bps"]
    tested = sorted(prim)
    for b in tested:
        if prim[b]["verdict"] == "insufficient data":
            return f"collect more recordings at {b:g} bps (need >= 3 usable and >= 400 decoded bits)", None
    rel = [b for b in tested if prim[b]["verdict"] == "reliable"]
    unrel = [b for b in tested if prim[b]["verdict"] == "unreliable"]
    if not unrel:
        nxt = [b for b in ladder if b > (max(tested) if tested else 0)]
        return (f"next ladder rate: {nxt[0]} bps" if nxt else "every ladder rate is reliable"), None
    lo_un = min(unrel)
    ok_below = [b for b in rel if b < lo_un and all(prim[x]["verdict"] == "reliable" for x in tested if x <= b)]
    hi_ok = max(ok_below) if ok_below else None
    after = [b for b in ladder if b > lo_un]
    if after and after[0] not in tested:
        return f"confirm the failure: test {after[0]} bps (next ladder rate after the first unreliable {lo_un:g})", hi_ok
    if hi_ok is None:
        return f"lowest tested rate {lo_un:g} bps is unreliable: test below it", None
    if lo_un - hi_ok > 4:
        mid = int(round((hi_ok + lo_un) / 2))
        return f"bisect: test {mid} bps (between reliable {hi_ok:g} and unreliable {lo_un:g})", hi_ok
    return (f"limit found: highest reliable {hi_ok:g} bps (next tested {lo_un:g} unreliable). Then the carrier arm: "
            f"{lo_un:g} bps at fc {lo_un / 4:g} Hz and 24 Hz, and the pulse arm (rect)"), hi_ok


def next_step_x4(summ, prot):
    """the x4 arm's stepping rule (psk_protocol.json x4_arm) on the held-out x4 results"""
    arm = prot.get("x4_arm")
    if not arm:
        return "no x4 arm in the protocol", None
    res = {(round(s["bps"], 2), s["fc_hz"], s["shape"]): s for s in summ if s["receiver"] == X4_NAME}
    v = lambda b, fc, sh: res[(b, fc, sh)]["verdict"] if (b, fc, sh) in res else None
    lad = arm["ladder"]
    hi_ok = None
    for i, r in enumerate(lad):
        b, fc, sh = float(r["bps"]), float(r["fc_hz"]), r["shape"]
        vv = v(b, fc, sh)
        if vv is None:
            return f"x4: test {b:g} bps at fc {fc:g} Hz, {sh}", hi_ok
        if vv == "insufficient data":
            return f"x4: collect more recordings at {b:g} bps fc {fc:g} {sh} (>= 3 usable held-out, >= 400 bits)", hi_ok
        if vv == "reliable":
            hi_ok = (b, fc, sh); continue
        if i + 1 < len(lad) and v(float(lad[i + 1]["bps"]), float(lad[i + 1]["fc_hz"]), lad[i + 1]["shape"]) is None:
            n = lad[i + 1]
            return f"x4: confirm the failure: test {n['bps']:g} bps at fc {n['fc_hz']:g} Hz, {n['shape']}", hi_ok
        if hi_ok is None:
            return f"x4: the first rung {b:g} bps is unreliable", None
        lo_b = b
        ok_b = max([hi_ok[0]] + [k[0] for k, s in res.items() if k[1:] == hi_ok[1:] and hi_ok[0] < k[0] < lo_b
                                 and s["verdict"] == "reliable"])
        un = [k[0] for k, s in res.items() if k[1:] == hi_ok[1:] and ok_b < k[0] < lo_b and s["verdict"] == "unreliable"]
        lo_b = min(un + [lo_b])
        pend = [k for k, s in res.items() if k[1:] == hi_ok[1:] and ok_b < k[0] < lo_b and s["verdict"] == "insufficient data"]
        if pend:
            return f"x4: collect more recordings at {pend[0][0]:g} bps fc {pend[0][1]:g} {pend[0][2]}", (ok_b,) + hi_ok[1:]
        if lo_b - ok_b > 4:
            mid = int(round((ok_b + lo_b) / 2))
            return f"x4: bisect: test {mid} bps at fc {hi_ok[1]:g} Hz, {hi_ok[2]}", (ok_b,) + hi_ok[1:]
        g = res[(ok_b,) + hi_ok[1:]]["goodput_bps"]
        return (f"x4: limit found: highest reliable {ok_b:g} bps (fc {hi_ok[1]:g} {hi_ok[2]}), goodput {g:.1f} bps"), \
            (ok_b,) + hi_ok[1:]
    return "x4: every rung of the x4 ladder is reliable", hi_ok


def fmt(v, f):
    return "-" if v is None else format(v, f)


def report(summ, prot):
    print(f"\n{'receiver':10s} {'fc':>3s} {'pulse':5s} {'bps':>5s} {'payload':>7s} {'rec':>3s} {'use':>3s} {'frames':>9s} "
          f"{'sync':>6s} {'errors/bits':>13s} {'BER':>8s} {'95% CI':>17s} {'goodput':>7s} {'clk err':>7s} "
          f"{'SNR':>5s} {'genie':>11s}  verdict / diagnosis")
    for s in summ:
        ber = "no bits" if not s["bits"] else f"{s['ber']:.2%}"
        ci = "-" if not s["bits"] else f"{s['ci95_lo']:.2%}-{s['ci95_hi']:.2%}"
        g = "-" if not s["genie_bits"] else f"{s['genie_errors']}/{s['genie_bits']}"
        print(f"{s['receiver']:10s} {s['fc_hz']:3g} {s['shape']:5s} {s['bps']:5.4g} {s['payload_bps']:7.2f} {s['recordings']:3d} "
              f"{s['usable']:3d} {s['frames_valid']:>4d}/{s['frames_transmitted']:<4d} {s['sync_rate']:6.1%} "
              f"{s['errors']:>5d}/{s['bits']:<7d} {ber:>8s} {ci:>17s} {s['goodput_bps']:7.2f} "
              f"{fmt(s['clock_error_ppm_median'], '+7.0f'):>7s} {fmt(s['symbol_snr_db_median'], '5.1f'):>5s} {g:>11s}  "
              f"{s['verdict'].upper()}  [{s['diagnosis']}]")
    print("  nominal bps = symbols x bits/symbol; payload = nominal x payload/frame bits; goodput = correct payload bits per "
          "usable recording second. frames = valid / transmitted while recording (known-transmission alignment). "
          "clk err = receiver - known-transmission clock. genie = ideal linear receiver on the unseen half (diagnostic). "
          f"Reliable = sync >= {prot['reliable']['frame_sync_min']:.0%} and BER 95 % upper < {prot['reliable']['ber_ci95_upper_max']:.0%} "
          f"with >= {prot['reliable']['min_usable_recordings']} usable recordings and >= {prot['reliable']['min_decoded_bits']} bits.")
    vs = sorted({(s["voltage_status"]) for s in summ})
    print(f"  watch input voltage: {', '.join(vs)} (2 mVpp setting; an off-body bench level, if recorded, is per run in per_run.csv)")


def plot(summ, path, prot):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    groups = sorted({(s["receiver"], s["fc_hz"], s["shape"]) for s in summ})
    for gk in groups:
        ss = sorted([s for s in summ if (s["receiver"], s["fc_hz"], s["shape"]) == gk], key=lambda s: s["bps"])
        lab = f"{gk[0]}, fc {gk[1]:g} Hz, {gk[2]}"
        x = [s["bps"] for s in ss]
        ber = [s["ber"] if s["bits"] else np.nan for s in ss]
        hi = [s["ci95_hi"] if s["bits"] else np.nan for s in ss]
        ln = ax[0].plot(x, [max(b, 1e-5) if b == b else np.nan for b in ber], "o-", label=lab)[0]
        ax[0].errorbar(x, [max(b, 1e-5) if b == b else np.nan for b in ber],
                       yerr=[[0] * len(x), [h - max(b, 1e-5) if h == h else 0 for b, h in zip(ber, hi)]], fmt="none",
                       color=ln.get_color(), capsize=3)
        nb = [s["bps"] for s in ss if not s["bits"]]
        ax[0].plot(nb, [0.5] * len(nb), "x", color=ln.get_color(), ms=10)
        ax[1].plot(x, [s["sync_rate"] for s in ss], "o-", color=ln.get_color())
        ax[2].plot(x, [s["goodput_bps"] for s in ss], "o-", color=ln.get_color())
        ax[2].plot(x, [s["payload_bps"] for s in ss], ":", color=ln.get_color())
    ax[0].axhline(prot["reliable"]["ber_ci95_upper_max"], color="#c9c7c1", ls="--")
    ax[0].set_yscale("log"); ax[0].set_ylim(1e-5, 1); ax[0].set_xlabel("nominal bit rate [bps]")
    ax[0].set_ylabel("BER (bars: 95 % upper; 0 errors plotted at 1e-5; x = no bits)"); ax[0].legend(fontsize=7)
    ax[1].axhline(prot["reliable"]["frame_sync_min"], color="#c9c7c1", ls="--")
    ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("nominal bit rate [bps]"); ax[1].set_ylabel("frame sync success")
    ax[2].axvline(16, color="#c9c7c1", ls=":"); ax[2].text(16.5, 1, "OOK/FSK 16 bps (measured)", fontsize=7, color="#52514e")
    ax[2].set_xlabel("nominal bit rate [bps]"); ax[2].set_ylabel("goodput (solid) / payload rate (dotted) [bps]")
    fig.suptitle(f"DBPSK through the worn Pixel Watch 2 ECG - REAL recordings, {P.VERSION}")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def analyse(specs, label, outroot=None, quiet=False):
    prot, psha = protocol()
    dirs = collect(specs)
    if not dirs:
        sys.exit(f"no psk run folders for {specs}")
    out = os.path.join(outroot or DATADIR, f"pskber_{label}_{stamp()}")
    os.makedirs(out)
    rows, missing = [], []
    for d in dirs:
        rec = json.load(open(os.path.join(d, "tx.json")))
        if not os.path.exists(os.path.join(d, "ecg.npz")):
            missing.append(dict(run=os.path.basename(d), note=rec.get("operator_note"))); continue
        rr = analyse_run(d, rec)
        json.dump(rr, open(os.path.join(d, f"result_{P.VERSION}_{P.RULES_SHA}.json"), "w"), indent=1, default=float)
        rows += rr
    summ = summarise(rows, prot) if rows else []
    if rows:
        with open(os.path.join(out, "per_run.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
        plot(summ, os.path.join(out, "curve.png"), prot)
    nxt, hi_ok = next_step(summ, prot) if summ else ("no recordings yet", None)
    nxt4, hi4 = next_step_x4(summ, prot) if summ else ("no recordings yet", None)
    json.dump(dict(label=label, created=datetime.datetime.now().isoformat(timespec="seconds"), receiver=P.params(),
                   receiver_x4=X4.params(), x4_development_runs=list(X4_DEVELOPMENT), next_step_x4=nxt4,
                   highest_reliable_x4=hi4,
                   receivers=[r[0] for r in RECEIVERS], protocol_sha=psha, diag=dict(rho_min=D.RHO_MIN,
                   genie_ber_limit=D.GENIE_BER_LIMIT, channel_model=D.CHANNEL), runs=dirs, missing_ecg=missing,
                   rows=rows, summary=summ, next_step=nxt, highest_reliable_bps=hi_ok),
              open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    if not quiet:
        if summ:
            report(summ, prot)
        for m in missing:
            print(f"  no ECG imported: {m['run']} (operator note: {m['note']}) - counted nowhere until a PDF is imported")
        print(f"\nnext (protocol {psha}): {nxt}   [x3-wl, pre-registered primary]")
        print(f"next for the x4 arm: {nxt4}")
        print(f"-> {os.path.relpath(out, HERE)}")
    return out, summ


def main():
    ap = argparse.ArgumentParser(description="EXPERIMENTAL DBPSK BER through the worn watch (psk-x3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="worn-watch DBPSK BER session")
    p.add_argument("--bps", type=float, required=True); p.add_argument("--label", required=True)
    p.add_argument("--fc", type=float, default=16.0); p.add_argument("--shape", choices=("rect", "rrc", "rc"), default="rrc")
    p.add_argument("--repeats", type=int, default=3); p.add_argument("--condition", default="worn-still")
    p.add_argument("--safety-review", help=argparse.SUPPRESS)
    p.add_argument("--bench-report", help="psk_bench report.json: its off-body level is stored, labelled as such")
    p.add_argument("--seconds", type=float, default=SECONDS); p.add_argument("--spacing", type=float, default=SPACING_S)
    p.add_argument("--port"); p.add_argument("--allow-modified-receiver", action="store_true")
    for k in AT.DEFAULTS:
        p.add_argument("--" + k.replace("_", "-"), type=float)
    p = sub.add_parser("ingest"); p.add_argument("session"); p.add_argument("files", nargs="+")
    p.add_argument("--allow-modified-receiver", action="store_true")
    p = sub.add_parser("analyse"); p.add_argument("specs", nargs="+"); p.add_argument("--label", default="curve")
    p.add_argument("--allow-modified-receiver", action="store_true")
    p = sub.add_parser("next"); p.add_argument("specs", nargs="+")
    p.add_argument("--receiver", choices=("x3-wl", X4_NAME), default="x3-wl", help="whose stepping rule (default the pre-registered x3-wl)")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "ingest":
        ingest(a)
    elif a.cmd == "analyse":
        check_frozen(a.allow_modified_receiver)
        analyse(a.specs, a.label)
    else:
        _, summ = analyse(a.specs, "next", quiet=True)
        prot, _ = protocol()
        print((next_step_x4 if a.receiver == X4_NAME else next_step)(summ, prot)[0])


if __name__ == "__main__":
    main()
