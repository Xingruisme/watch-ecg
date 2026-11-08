#!/usr/bin/env python3
"""
Pixel Watch 2 ECG as a receiver for the TeensyTRX link: transmit, import the watch's ECG export, decode, BER.
The receiver-side counterpart of maxtouch-screen-main/stent_modcomp.py (same transmitter, same TeensyTRX schedules via
stent_tx.py, same frame accounting via decode_v2 / decode_v3, same Clopper-Pearson intervals and colours).

Worn watch:
   python ecg_ber.py run --bps 2 --label wear1                                                  # OOK + FSK
   python ecg_ber.py ingest data_watch_ecg/session_wear1_2bps_<time> ~/Downloads/*.pdf          # import + BER
   python ecg_ber.py analyse wear1                          # every session of a label together
Bench and tools:
   python ecg_ber.py plan                                   # frames, tones, durations for every rate
   python ecg_ber.py divider --vout-vpp 0.60 --r1 320e3 --r2 560          # mV at the watch, fault currents
   python ecg_ber.py selftest                               # decoder search = decode_v4 search; PRBS7 = stent_frames
   python ecg_ber.py synth --label dry1                     # dry run: synthetic recordings -> PDF -> import -> BER
   # bench (NO person touching the watch or the leads; see README.md before the first run):
   python ecg_ber.py tx --kind sweep --label bench1 --amp-mvpp 1.0 --monitor          # passband sweep, 20 tones
   python ecg_ber.py tx --kind alias --label bench1 --amp-mvpp 1.0 --monitor          # out-of-band / alias probe
   python ecg_ber.py tx --kind ber --mod ook --bps 4 --label bench1 --amp-mvpp 1.0 --monitor
   python ecg_ber.py import ~/Downloads/<exported>.pdf [--run data_watch_ecg/ecg_bench1_ber_ook_4bps_<time>]
   python ecg_ber.py analyse bench1 [--label bench1_day1]   # every run of a label (or run folders) -> tables, figures
   python ecg_ber.py compare MAXTOUCH=<modcomp folder> ECG=<ecgber folder> --label both

Run folders: data_watch_ecg/ecg_<label>_<kind>[_<mod>_<bps>bps]_<time>/ with tx.json (written by 'tx'), then
ecg_source.<ext> + ecg.npz + ecg_import.json (by 'import'), monitor.npz (the transmitter's own A0, by --monitor) and
result_<decoder>_<sha>.json (by 'analyse'; a new decoder version writes a new file, nothing is overwritten).
'analyse' writes a NEW folder data_watch_ecg/ecgber_<label>_<time>/ every time.

Usable recordings (truth-free, fixed before any real recording): >= MIN_DURATION_S long, scheduled tone line
>= MIN_LINE_SNR_DB over the same lines 2.5 Hz away, < MAX_BAD_FRAC of the samples in flat runs (>= 0.1 s identical
values) or beyond +-MAX_ABS_MV. Unusable recordings are listed with the reason and left out of the BER; frames lost
inside a usable recording count (conservative BER, v2 accounting). Runs without an ECG file are listed as dropped.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ecg_io as IO
import ecg_link as L

STX = L.STX
DATADIR = os.path.join(HERE, "data_watch_ecg")
COLORS = dict(ook="#ff7f0e", fsk="#2ca02c")                 # as stent_modcomp (slide 155)
MARKERS = dict(ook="s", fsk="^")
MIN_DURATION_S, MIN_LINE_SNR_DB, MAX_BAD_FRAC, MAX_ABS_MV = 25.0, 6.0, 0.10, 4.9
AD9833_VPP, AD9833_ROUT = 0.60, 200.0                       # datasheet typ. sine 0.038-0.65 V; measure the real one
MONITOR_SPS = 2000
FROZEN_SHA = "b422eaa49c6a"                                    # ecg_link.py sha at freezing (2026-11-02 16:34:17)
ECG_START_S = 5                                             # worn runs: finger on the crown 5 s after the start
ECG_LATEST_S = 20                                           # a recording that starts later than this misses the end
STEP_EDGE_S = 0.10                                          # sweep / alias: trimmed at both ends of every step
SAFETY = [
    "Nobody is touching the watch, its electrodes, the leads or the attenuator (the watch is NOT on a wrist).",
    "The output goes through the attenuator (R1 >= 320 kOhm in series, two resistors) - NOT the maXTouch current "
    "network, NOT straight from the board.",
    "TeensyTRX is isolated from mains earth (USB isolator, or laptop on battery with the charger unplugged).",
    "The watch is off its charger.",
]


# ================================================================ helpers
def stamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_dir(label, plan):
    tag = plan["kind"] if plan["kind"] != "ber" else f"ber_{plan['modulation']}_{round(plan['bps']):g}bps"
    return os.path.join(DATADIR, f"ecg_{label}_{tag}_{stamp()}")


def plan_from(a):
    if a.kind == "ber":
        return L.ber_plan(a.mod, a.bps, a.f1, a.f0, a.seconds)
    return L.tone_plan(a.kind, a.seconds)


def cp(e, n):
    from scipy.stats import beta
    if not n:
        return (None, None)
    return (beta.ppf(0.025, e, n - e + 1) if e else 0.0, beta.ppf(0.975, e + 1, n - e) if e < n else 1.0)


def divider(vpp, r1, r2, rout=AD9833_ROUT, rail=3.3):
    mv = vpp * r2 / (r1 + r2 + rout) * 1e3
    return dict(vout_vpp=vpp, r1_ohm=r1, r2_ohm=r2, watch_mvpp=mv, watch_mv_rms=mv / (2 * math.sqrt(2)),
                normal_current_ua_peak=vpp / 2 / (r1 + r2 + rout) * 1e6,
                fault_dc_current_ua=rail / (r1 + r2) * 1e6,                  # output stuck at the 3.3 V rail, cap shorted
                fault_one_r1_shorted_ua=rail / (r1 / 2 + r2) * 1e6)          # R1 = two equal resistors, one shorted


# ================================================================ transmit (bench)
def parse_mono(buf):
    """TeensyTRX mono stream (2 bytes/sample: 0x80 | hi7, lo7 of adc12 << 2) -> 12-bit counts, as stent_tx.a0"""
    a = np.frombuffer(buf, np.uint8)
    j = next((i for i in range(len(a) - 1) if a[i] & 0x80 and not a[i + 1] & 0x80), None)
    if j is None:
        return np.array([])
    pr = a[j:j + (len(a) - j) // 2 * 2].reshape(-1, 2)
    ok = (pr[:, 0] & 0x80 != 0) & (pr[:, 1] & 0x80 == 0)
    return ((((pr[:, 0] & 0x7F).astype(np.uint16) << 7) | pr[:, 1]) >> 2)[ok].astype(float)


def tx(a):
    plan = plan_from(a)
    sched = L.schedule(plan)
    print(f"{plan['kind']} schedule: {plan['total_s']:.0f} s"
          + (f", {L.MOD_NAMES[plan['modulation']]} {plan['bps']:g} bps, f1 {plan['f1_hz']:g} Hz"
             + (f", f0 {plan['f0_hz']:g} Hz" if plan["f0_hz"] else "") + f", frame {plan['frame_bits']} bits = "
             f"{plan['frame_s']:.2f} s x {plan['frames']}" if plan["kind"] == "ber" else
             f", {len(plan['tones_hz'])} tones x {plan['step_s']:g} s, cycle {plan['cycle_s']:g} s x {plan['frames']}"))
    dv = divider(a.vout_vpp, a.r1, a.r2) if a.r1 and a.r2 else None
    if dv:
        print(f"attenuator {a.r1:g} / {a.r2:g} Ohm from {a.vout_vpp:g} Vpp -> {dv['watch_mvpp']:.3f} mVpp at the watch"
              f" (declared --amp-mvpp {a.amp_mvpp:g})")
    print("\nBench checklist - answer y to each:")
    answers = []
    for q in SAFETY:
        r = input(f"  {q} [y/N] ").strip().lower()
        if r != "y":
            sys.exit("  stopped: checklist not confirmed - nothing was transmitted")
        answers.append(q)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    t = STX.StentTX(a.port)
    info = t.configure(wave="sine")             # sine only: square mode steps 0 <-> 1.65 V at every OOK edge
    d = run_dir(a.label, plan)
    os.makedirs(d)
    rec = dict(kind=plan["kind"], label=a.label, condition=a.condition, plan=plan, amp_mvpp=a.amp_mvpp,
               attenuator=dv, wiring=a.wiring, checklist=answers, transmitter=info, synthetic=False,
               schedule=dict(step_us=sched[0][0], reps=sched[0][1], words=sched[0][2], checksum=STX.checksum(sched)))
    try:
        rec["upload"] = t.upload_schedule(sched)
        input(f"\nWatch: open the ECG app, contacts in place. Press Enter to start the transmitter "
              f"({plan['total_s']:.0f} s), then start the ECG recording within ~{max(plan['total_s'] - 40, 5):.0f} s ")
        sps = MONITOR_SPS if a.monitor else 0
        t_cmd = time.time()
        r = t.ask(f"sgo {STX.GO_DELAY_MS}" + (f" {sps}" if sps else ""), r"^# sweep armed in")
        rec.update(t_go=t_cmd + STX.GO_DELAY_MS / 1e3 + 0.0005, go_reply=r,
                   t_go_local=datetime.datetime.fromtimestamp(t_cmd).isoformat(timespec="seconds"))
        buf = b""
        if a.monitor:
            t.send(f"mono {sps}")
        end = rec["t_go"] + plan["total_s"] + 0.5
        print(f"  >>> TRANSMITTING until {datetime.datetime.fromtimestamp(end):%H:%M:%S} - start the ECG recording now <<<")
        last = -1
        while time.time() < end:
            if a.monitor:
                buf += t.dev.read(1 << 16)
            else:
                time.sleep(0.2)
            left = int(end - time.time())
            if left != last and left % 10 == 0:
                print(f"    {left:3d} s left"); last = left
        if a.monitor:
            t.send("x"); time.sleep(0.3)
            buf += t.dev.read(1 << 20)
            raw = buf.split(b"# mono stopped")[0]
            y = parse_mono(raw)
            np.savez_compressed(os.path.join(d, "monitor.npz"), counts=y, sps=sps)
            rec["monitor"] = dict(samples=int(len(y)), sps=sps, file="monitor.npz")
            t.dev.reset_input_buffer()
        rec["tx_status"] = t.status()
        try:
            rec["voltage"] = voltage_report(rec, d, voltage_inputs(a))
        except Exception as ex:
            rec["voltage"] = dict(error=str(ex))
    except KeyboardInterrupt:
        rec["stopped"] = "Ctrl+C"
        print("stopped (Ctrl+C)")
    finally:
        t.close()
        rec["t_end"] = time.time()
        json.dump(rec, open(os.path.join(d, "tx.json"), "w"), indent=1)
    note = input("Did the watch complete a 30 s ECG recording? (y / n / note) ").strip()
    rec["operator_note"] = note
    json.dump(rec, open(os.path.join(d, "tx.json"), "w"), indent=1)
    st = rec.get("tx_status", {})
    print(f"saved {os.path.relpath(d, HERE)}   transmitter state {st.get('state')}, events {st.get('events')} "
          f"(expected {plan['n_steps']}/{plan['n_steps']})")
    if a.monitor and plan["kind"] == "ber":
        m = monitor_decode(d)
        if m:
            print(f"  monitor (transmitter A0, independent of the watch): {m['summary']}")
    if rec.get("voltage", {}).get("watch_input"):
        print_voltage(rec["voltage"])
    print(f"next: export the ECG PDF from the Fitbit app, then  python ecg_ber.py import <file.pdf> --run {os.path.relpath(d, HERE)}")


def run_session(a):
    """the one transmit command for the worn-watch experiment: OOK then FSK (x --repeats) at one rate, each run a
    separate 30 s ECG recording. --approval / --interface are recorded in tx.json when given."""
    check_frozen(a.allow_modified_decoder)
    mods = [m.strip() for m in a.mods.split(",")]
    order = [m for _ in range(a.repeats) for m in mods]
    plans = [L.ber_plan(m, a.bps, seconds=a.seconds) for m in order]
    sess = os.path.join(DATADIR, f"session_{a.label}_{round(a.bps):g}bps_{stamp()}")
    print(f"Session {os.path.relpath(sess, HERE)}: {len(order)} runs at {a.bps:g} bps ({', '.join(L.MOD_NAMES[m] for m in order)}), "
          f"{plans[0]['total_s']:.0f} s of transmission each, decoder {L.VERSION} sha {L.RULES_SHA} (frozen {L.FROZEN_AT})")
    for m, p in zip(mods, plans):
        print(f"  {L.MOD_NAMES[m]}: '1' = {p['f1_hz']:g} Hz, '0' = " + (f"{p['f0_hz']:g} Hz" if p["f0_hz"] else "off")
              + f", frame {p['frame_bits']} bits = {p['frame_s']:.2f} s")
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    t = STX.StentTX(a.port)
    info = t.configure(wave="sine")
    os.makedirs(sess)
    done = []
    vin = voltage_inputs(a)
    meta = dict(session=os.path.basename(sess), created=datetime.datetime.now().isoformat(timespec="seconds"),
                bps=a.bps, order=order, condition=a.condition, interface=a.interface, approval=a.approval,
                decoder=L.VERSION, rules_sha=L.RULES_SHA, monitor=a.monitor, voltage_inputs=vin,
                watch_input_point=WATCH_POINT, a0_point=A0_POINT, runs=[])
    json.dump(meta, open(os.path.join(sess, "session.json"), "w"), indent=1, default=float)
    try:
        for i, (m, plan) in enumerate(zip(order, plans)):
            print(f"\n[{i + 1}/{len(order)}] {L.MOD_NAMES[m]} {a.bps:g} bps")
            input(f"  ECG app ready (not yet recording). Press Enter to start transmitting ")
            d = os.path.join(sess, f"ecg_{a.label}_ber_{m}_{round(a.bps):g}bps_{stamp()}")
            os.makedirs(d)
            rec = dict(kind="ber", label=a.label, condition=a.condition, plan=plan, amp_mvpp=None, contact="worn",
                       approval=a.approval, interface=a.interface, transmitter=info, synthetic=False,
                       session=os.path.basename(sess), decoder_sha=L.RULES_SHA, voltage_inputs=vin)
            rec["voltage"] = dict(carrier=carrier_info(plan), watch_input=watch_input(rec, vin))
            print_voltage(rec["voltage"])
            transmit(t, d, rec, plan, monitor=a.monitor, cues=True)
            done.append(d)
            meta["runs"].append(dict(run=os.path.basename(d), modulation=m, voltage=rec.get("voltage")))
            json.dump(meta, open(os.path.join(sess, "session.json"), "w"), indent=1, default=float)
    except KeyboardInterrupt:
        print("stopped (Ctrl+C)")
    finally:
        t.close()
    print(f"\n{len(done)} runs saved in {os.path.relpath(sess, HERE)}")
    print("Export each reading as a PDF in the Fitbit app (Heart rhythm assessment -> reading -> Export a PDF for your "
          "doctor), copy the PDFs to this laptop, then:")
    print(f"  ~/.venvs/galvanic/bin/python ecg_ber.py ingest {os.path.relpath(sess, HERE)} <pdf files or folder>")


def transmit(t, d, rec, plan, monitor=True, cues=False):
    """upload, start, stream the A0 monitor, give the ECG cues, stop; writes tx.json (+ monitor.npz)"""
    sched = L.schedule(plan)
    rec["schedule"] = dict(step_us=sched[0][0], reps=sched[0][1], words=sched[0][2], checksum=STX.checksum(sched))
    try:
        rec["upload"] = t.upload_schedule(sched)
        sps = MONITOR_SPS if monitor else 0
        t_cmd = time.time()
        r = t.ask(f"sgo {STX.GO_DELAY_MS}" + (f" {sps}" if sps else ""), r"^# sweep armed in")
        t0 = t_cmd + STX.GO_DELAY_MS / 1e3 + 0.0005
        rec.update(t_go=t0, go_reply=r, t_go_local=datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"))
        if monitor:
            t.send(f"mono {sps}")
        end = t0 + plan["total_s"] + 0.5
        cue = {0: "TRANSMITTING - do not start the ECG yet",
               ECG_START_S: ">>> NOW: start the 30 s ECG recording <<<",
               ECG_START_S + 35: "the ECG should be finished (if it is still recording, let it finish)"} if cues else \
              {0: "TRANSMITTING - start the ECG recording now"}
        buf, shown, ticks = b"", set(), set()
        while time.time() < end:
            if monitor:
                buf += t.dev.read(1 << 16)
            else:
                time.sleep(0.05)
            el = time.time() - t0
            for k, msg in cue.items():
                if el >= k and k not in shown:
                    shown.add(k); print(f"  T+{k:3d} s  {msg}")
            left = int(end - time.time())
            if left % 10 == 0 and left not in ticks and left > 0:
                ticks.add(left); print(f"           ({left} s of transmission left)")
        if monitor:
            t.send("x"); time.sleep(0.3)
            buf += t.dev.read(1 << 20)
            y = parse_mono(buf.split(b"# mono stopped")[0])
            np.savez_compressed(os.path.join(d, "monitor.npz"), counts=y, sps=sps)
            rec["monitor"] = dict(samples=int(len(y)), sps=sps, file="monitor.npz")
            t.dev.reset_input_buffer()
        rec["tx_status"] = t.status()
        try:
            rec.setdefault("voltage", {})["a0"] = a0_levels(rec, d)
        except Exception as ex:                    # never lose a run over the voltage report
            rec.setdefault("voltage", {})["a0"] = dict(point=A0_POINT, status="unavailable", reason=f"analysis error: {ex}")
        print(f"  T+{plan['total_s']:.0f} s  transmission stopped")
    finally:
        try:
            t.off()
        except Exception as e:
            print(f"  ! could not confirm the output is off ({e})")
        rec["t_end"] = time.time()
        json.dump(rec, open(os.path.join(d, "tx.json"), "w"), indent=1)
    rec["operator_note"] = input("  Did the watch complete the 30 s ECG? (y / n / note) ").strip()
    json.dump(rec, open(os.path.join(d, "tx.json"), "w"), indent=1)
    st = rec.get("tx_status", {})
    ok = st.get("state") == "done" and str(st.get("events")) == str(plan["n_steps"])
    print(f"  transmitter: state {st.get('state')}, events {st.get('events')}/{plan['n_steps']}"
          + ("" if ok else "   ** schedule did not complete **"))
    if monitor:
        m = monitor_decode(d)
        if m:
            print(f"  transmitter self-check (its own A0, not the watch): {m['summary']}")
    if rec.get("voltage"):
        print_voltage(dict(rec["voltage"], carrier=rec["voltage"].get("carrier") or carrier_info(plan),
                           watch_input=rec["voltage"].get("watch_input") or watch_input(rec)))


def ingest(a):
    """import every exported PDF of a session (paired with the runs by recording time) and analyse the session"""
    check_frozen(a.allow_modified_decoder)
    sess = os.path.normpath(a.session)
    runs = sorted((json.load(open(p))["t_go"], os.path.dirname(p)) for p in glob.glob(os.path.join(sess, "ecg_*", "tx.json"))
                  if "t_go" in json.load(open(p)))
    files = []
    for f in a.files:
        files += sorted(glob.glob(os.path.join(f, "*.pdf"))) if os.path.isdir(f) else [f]
    seen, readings = set(), []
    for f in files:
        try:
            e = IO.load(f)
        except Exception as ex:
            print(f"  skipped {os.path.basename(f)}: not readable as a Fitbit ECG export ({ex})"); continue
        key = hash(e.mv.tobytes())
        if key in seen:
            print(f"  skipped {os.path.basename(f)}: same reading as an earlier file"); continue
        seen.add(key)
        when = parse_recorded(e.meta.get("recorded_on", ""))
        if when is None:
            print(f"  skipped {os.path.basename(f)}: no 'Recorded on' time"); continue
        readings.append((when.timestamp(), f, e))
    # pairing (ecg_pairing.py, 2026-11-06): by 'Recorded on' time; where two runs fit one minute, by signal (tones
    # present/absent + the sent tone pattern), never by file order. No unique consistent pairing -> not imported.
    import ecg_pairing as PA
    recs = []
    for t0, d in runs:
        r_ = json.load(open(os.path.join(d, "tx.json"))); r_["_dir"] = d; recs.append((t0, d, r_))
    readings.sort(key=lambda r: r[0])
    for i, d, method, ev in PA.resolve(readings, recs, ECG_LATEST_S):
        w, f, e = readings[i]
        tag = f"{os.path.basename(f)} ({datetime.datetime.fromtimestamp(w):%H:%M})"
        if method == "none":
            print(f"  skipped {tag}: no run of this session was transmitting then"); continue
        if method == "ambiguous":
            print(f"  NOT imported {tag}: {len(ev)} runs fit this minute and the signal check finds no unique pairing")
            PA.print_evidence(ev)
            print(f"      check it, then import it by hand: python ecg_ber.py import '{f}' --run <run folder>")
            continue
        if os.path.exists(os.path.join(d, "ecg.npz")):
            src = os.path.join(d, "ecg_source" + os.path.splitext(f)[1].lower())
            same = os.path.exists(src) and PA.sha(src) == PA.sha(f)
            print(f"  {os.path.basename(d)} already has " + ("this ECG - kept" if same else
                  f"a DIFFERENT ECG - kept, {os.path.basename(f)} not imported; check with: "
                  f"python ecg_pairing.py {sess}"))
            continue
        shutil.copy2(f, os.path.join(d, "ecg_source" + os.path.splitext(f)[1].lower()))
        IO.save_npz(e, os.path.join(d, "ecg.npz"))
        json.dump(dict(e.meta, samples=len(e.mv), imported=datetime.datetime.now().isoformat(timespec="seconds"),
                       original=os.path.abspath(f), pairing=dict(method=method, evidence=ev)),
                  open(os.path.join(d, "ecg_import.json"), "w"), indent=1, default=str)
        print(f"  {tag} -> {os.path.basename(d)}"
              + ("   (two or more runs fit this minute: resolved by the signal check)" if method == "signal" else ""))
        if method == "signal" or not ev[0]["ok"]:
            PA.print_evidence(ev)
    return analyse([d for _, d in runs], os.path.basename(sess), outroot=sess)


def check_frozen(allow=False):
    if L.FROZEN_AT and L.RULES_SHA != FROZEN_SHA:
        msg = (f"ecg_link.py changed since it was frozen (sha {L.RULES_SHA}, frozen {FROZEN_SHA} at {L.FROZEN_AT}): "
               "results would not be comparable")
        if not allow:
            sys.exit(msg + " - restore it, or pass --allow-modified-decoder for a labelled exploratory run")
        print("  ! " + msg)


def monitor_decode(d):
    """decode the transmitter's own A0 stream with the same decoder (bits, timing, tones without the watch)"""
    p = os.path.join(d, "monitor.npz")
    if not os.path.exists(p):
        return None
    from scipy import signal
    rec = json.load(open(os.path.join(d, "tx.json")))
    z = np.load(p)
    y, sps = z["counts"] * 3.3 / 4095 * 1e3, float(z["sps"])     # mV at A0
    if len(y) < sps * 5:
        return dict(summary=f"only {len(y)} samples", ok=False)
    q = int(round(sps / 250))
    x = signal.decimate(y - y.mean(), q, ftype="fir", zero_phase=True)
    tt = np.arange(len(x)) / (sps / q)
    k = int(30 * 250)                            # the middle 30 s, like one ECG recording
    i0 = max(0, (len(x) - k) // 2)
    rep, _ = L.decode(x[i0:i0 + k], tt[i0:i0 + k] - tt[i0], rec["plan"])
    fr = rep["frames"]
    e = sum(f["errors"] for f in fr if f["valid"]); n = sum(f["bits"] for f in fr if f["valid"])
    q_ = L.quality(x, 250.0, [f for f in (rec["plan"]["f1_line_hz"], rec["plan"]["f0_line_hz"]) if f])
    return dict(ok=bool(n and e == 0), errors=e, bits=n, frames_valid=sum(f["valid"] for f in fr),
                frames=len(fr), clock_ppm=rep["clock_ppm"], a0_ptp_mv=float(np.ptp(y)), quality=q_,
                summary=f"{e}/{n} errors in {sum(f['valid'] for f in fr)}/{len(fr)} frames, clock {rep['clock_ppm']:+.0f} ppm, "
                        f"A0 {np.ptp(y):.0f} mV p-p")


# ================================================================ voltage report
ADC_LSB_MV = 3.3 / 4095 * 1e3                               # Teensy 4.0 ADC: 12 bits on the 3.3 V supply
AD9833_DC_MV = 340.0                                        # VOUT mid-scale, datasheet typ. - measure the real one
A0_DETECT_RATIO = 3.0                                       # tone Vpp >= 3 x the noise floor (~10 dB) counts as present
A0_POINT = ("TeensyTRX A0 (pin 14) = the board's own electrode node E+ through its RX path, single-ended to Teensy GND. "
            "NOT the watch input: the attenuator and the watch come after it")
WATCH_POINT = "watch ECG input electrodes: crown (signal) - back (ground)"
WATCH_NOT_MEASURED = ("nothing in this setup measures the crown-back voltage: A0 is wired to E+ on the board, and the "
                      f"12-bit A0 ({ADC_LSB_MV:.2f} mV/LSB) could not resolve ~1 mVpp anyway")


def voltage_inputs(a):
    """what the operator declared about the attenuator and any external measurement (all optional)"""
    g = lambda k, dflt=None: getattr(a, k, dflt)
    return dict(r1_ohm=g("r1"), r2_ohm=g("r2"), source_point=g("source_point", "vout"),
                source_vpp=g("source_vpp") if g("source_vpp") is not None else g("vout_vpp"),
                source_measured=bool(g("source_measured", False)), source_dc_mv=g("source_dc_mv"),
                dc_block=bool(g("dc_block", False)), watch_vpp_mv=g("watch_vpp_mv"), watch_dc_mv=g("watch_dc_mv"),
                watch_instrument=g("watch_instrument"), declared_amp_mvpp=g("amp_mvpp"))


def carrier_info(plan):
    if plan["kind"] != "ber":
        return dict(wave="sine", tones_hz=plan.get("tones_hz"),
                    tones_ad9833_hz=[L.actual_hz(f) for f in plan.get("tones_hz") or []])
    return dict(wave="sine", modulation=plan["modulation"], bps=plan["bps"],
                f1_hz=plan["f1_hz"], f1_ad9833_hz=L.actual_hz(plan["f1_hz"]),
                f0_hz=plan["f0_hz"] or None, f0_ad9833_hz=L.actual_hz(plan["f0_hz"]) or None,
                f0_note=None if plan["f0_hz"] else "'0' = output at mid-scale (tone off)")


def watch_input(rec, vin=None):
    """the voltage across the watch electrodes, labelled by how it is known: measured / calculated / declared /
    simulated / unavailable. Never filled in from A0."""
    vin = vin or {}
    base = dict(point=WATCH_POINT, vpp_mv=None, dc_mv=None, directly_measured=False)
    if rec.get("synthetic"):
        return dict(base, status="simulated", vpp_mv=rec.get("amp_mvpp"), dc_mv=0.0,
                    basis="synthetic channel model (ecg_synth.py), no hardware")
    if vin.get("watch_vpp_mv") is not None:
        return dict(base, status="measured", directly_measured=True, vpp_mv=vin["watch_vpp_mv"],
                    dc_mv=vin.get("watch_dc_mv"),
                    basis=f"external instrument across crown-back, entered by the operator: "
                          f"{vin.get('watch_instrument') or 'instrument not stated'}"
                          + ("" if vin.get("watch_dc_mv") is not None else "; DC not entered"))
    r1, r2 = vin.get("r1_ohm"), vin.get("r2_ohm")
    if r1 is None and r2 is None and rec.get("attenuator"):          # 'tx' records its divider as 'attenuator'
        r1, r2 = rec["attenuator"]["r1_ohm"], rec["attenuator"]["r2_ohm"]
    if r1 and r2:
        src_vpp = vin.get("source_vpp") or AD9833_VPP
        rout = AD9833_ROUT if vin.get("source_point", "vout") == "vout" else 0.0
        ratio = r2 / (r1 + r2 + rout)
        src_dc = vin.get("source_dc_mv") if vin.get("source_dc_mv") is not None else AD9833_DC_MV
        what = ("measured by the operator" if vin.get("source_measured") and vin.get("source_vpp") else
                "NOT measured" + (" - datasheet nominal" if src_vpp == AD9833_VPP else ""))
        return dict(base, status="calculated", vpp_mv=src_vpp * ratio * 1e3, dc_mv=0.0 if vin.get("dc_block") else src_dc * ratio,
                    basis=f"divider R1 {r1:g} / R2 {r2:g} Ohm (+{rout:g} Ohm source) from {src_vpp:g} Vpp at "
                          f"{'AD9833 VOUT' if rout else 'E+'} ({what}); DC "
                          + ("blocked by a series capacitor (taken as 0)" if vin.get("dc_block") else
                             f"{src_dc:g} mV at the source x the divider"
                             + ("" if vin.get("source_dc_mv") is not None else " (nominal)"))
                          + "; watch input impedance taken as >> R2", ratio=ratio)
    if vin.get("declared_amp_mvpp") is not None or rec.get("amp_mvpp") is not None:
        v = vin.get("declared_amp_mvpp") if vin.get("declared_amp_mvpp") is not None else rec.get("amp_mvpp")
        return dict(base, status="declared", vpp_mv=v, basis="--amp-mvpp as stated by the operator: neither measured "
                    "nor calculated by this program")
    return dict(base, status="unavailable", reason=WATCH_NOT_MEASURED + "; no attenuator declared (--r1/--r2) to "
                "calculate it and no external measurement entered (--watch-vpp-mv)")


def a0_levels(rec, d):
    """tone Vpp and DC at A0 per transmitted frequency: least-squares sine + DC fit in every scheduled step (aligned by
    the firmware's t0), median over the steps. floor = the Vpp that noise alone gives the same fit: residual std s over
    n samples -> mean fitted amplitude s * sqrt(pi / n) (Rayleigh), x 2 for peak-to-peak"""
    base = dict(point=A0_POINT, directly_measured=True, adc_lsb_mv=ADC_LSB_MV)
    p = os.path.join(d, "monitor.npz")
    if not os.path.exists(p):
        return dict(base, status="unavailable", directly_measured=False,
                    reason="no A0 monitor stream in this run (--no-monitor, or the run was stopped)")
    st = rec.get("tx_status") or {}
    t0 = int(st.get("t0", -1))
    if t0 < 0 or "schedule" not in rec:
        return dict(base, status="unavailable", directly_measured=False,
                    reason="no firmware t0 (the schedule did not complete), so the steps cannot be aligned")
    z = np.load(p)
    y, sps = z["counts"] * ADC_LSB_MV, float(z["sps"])
    s = rec["schedule"]
    ns = int(round(s["step_us"] * sps / 1e6))
    seq = list(s["words"]) * int(s["reps"])
    edge = max(1, ns // 10)
    def fit(seg, f):
        tt = np.arange(len(seg)) / sps
        M = np.c_[np.ones_like(tt), np.sin(2 * np.pi * f * tt), np.cos(2 * np.pi * f * tt)]
        c = np.linalg.lstsq(M, seg, rcond=None)[0]
        return 2 * np.hypot(c[1], c[2]), 2 * np.std(seg - M @ c) * np.sqrt(np.pi / len(seg))
    per = {}
    for k, w in enumerate(seq):
        a, b = t0 + k * ns + edge, t0 + (k + 1) * ns - edge
        if b > len(y) or not w:
            continue
        per.setdefault(w, []).append(fit(y[a:b], w * STX.MCLK / 2 ** 28))
    e_end = min(len(y), t0 + len(seq) * ns)
    tones = []
    for w, v in sorted(per.items()):
        v = np.array(v)
        vpp, fl = float(np.median(v[:, 0])), float(np.median(v[:, 1]))
        tones.append(dict(ad9833_hz=w * STX.MCLK / 2 ** 28, vpp_mv=vpp, floor_vpp_mv=fl, steps=len(v),
                          detected=bool(vpp >= A0_DETECT_RATIO * fl)))
    idle = y[:t0] if t0 > sps * 0.1 else y[e_end + int(0.2 * sps):]
    return dict(base, status="measured", sps=sps, tones=tones,
                dc_tx_mv=float(y[t0:e_end].mean()) if e_end > t0 else None,
                dc_idle_mv=float(idle.mean()) if len(idle) else None,
                ptp_tx_mv=float(np.ptp(y[t0:e_end])) if e_end > t0 else None,
                dc_note="A0's DC is set by the board's A0 input network; it is not shown to be the DC offset at E+, "
                        "and it is not the DC at the watch")


def voltage_report(rec, d, vin=None):
    """carrier, watch-input voltage (labelled by how it is known) and the A0 measurement of one run"""
    v = dict(rec.get("voltage") or {})
    v.setdefault("carrier", carrier_info(rec["plan"]))
    v.setdefault("watch_input", watch_input(rec, vin))
    if "a0" not in v and not rec.get("synthetic"):
        v["a0"] = a0_levels(rec, d)
    return v


def fmt_hz(c):
    if c.get("transmitter") == "arduino":
        if "f1_hz" not in c:
            return f"{len(c.get('tones_hz') or [])} tones, sine, Arduino DAC (exact frequencies)"
        return (f"'1' {c['f1_hz']:g} Hz, '0' " + (f"{c['f0_hz']:g} Hz" if c.get("f0_hz") else "off (mid-scale)")
                + f", sine, Arduino DAC (exact frequencies), tone set {c.get('toneset')}")
    if "f1_hz" not in c:
        return f"{len(c.get('tones_hz') or [])} tones"
    f0 = f"{c['f0_hz']:g} Hz (AD9833 {c['f0_ad9833_hz']:.3f})" if c.get("f0_hz") else "off"
    return f"'1' {c['f1_hz']:g} Hz (AD9833 {c['f1_ad9833_hz']:.3f}), '0' {f0}, {c['wave']}"


def fmt_watch(w):
    if w["status"] == "unavailable":
        return "UNAVAILABLE (not measured, not calculated)"
    dc = "DC -" if w.get("dc_mv") is None else f"DC {w['dc_mv']:.2f} mV"
    tag = dict(measured="MEASURED", calculated="CALCULATED, not measured", declared="DECLARED by operator, not measured",
               simulated="SIMULATED")[w["status"]]
    return f"{w['vpp_mv']:.3f} mVpp, {dc}  [{tag}]"


def fmt_a0(a):
    if a is None:
        return "-"
    if a["status"] != "measured":
        return f"unavailable ({a['reason']})"
    t = ", ".join(f"{x['ad9833_hz']:.2f} Hz {x['vpp_mv']:.2f} mVpp (floor {x['floor_vpp_mv']:.2f}) "
                  f"{'present' if x['detected'] else 'NOT DETECTED'}" for x in a["tones"]) or "no tone steps"
    dc = "" if a.get("dc_tx_mv") is None else f"; DC {a['dc_tx_mv']:.0f} mV (output off: {a['dc_idle_mv']:.0f} mV)"
    return t + dc


def print_voltage(v, indent="  "):
    print(f"{indent}carrier (configured): {fmt_hz(v['carrier'])}")
    w = v["watch_input"]
    print(f"{indent}watch input, {w['point']}: {fmt_watch(w)}")
    print(f"{indent}    {w.get('basis') or w.get('reason')}")
    if "a0" in v and v["a0"].get("point") != ARD_NO_MONITOR:
        print(f"{indent}A0 = E+ node, measured, NOT the watch input: {fmt_a0(v['a0'])}")


# ================================================================ Arduino transmitter (--tx arduino), added 2026-11-06
# The TeensyTRX path above is unchanged; --tx teensy (the default) never reaches this section.
# Tone sets: 'baseline' = ecg_link.TONES (frozen). 'exp' = 2 bps only, chosen from a SIMULATED tone-set study
# (data_watch_ecg/exploratory_tonesim_*), not from measurements. Both decode with the frozen ecg-v1 decoder unchanged:
# the tones reach it through the plan (ber_plan f1 / f0) and every tone has a whole number of cycles per bit.
EXP_TONES = {2: dict(ook=(10.0, None), fsk=(12.0, 8.0))}
EXPLORATORY_BPS = (16,)                              # simulation: weak at 16 bps (both tone sets)
ARD_NO_MONITOR = "none (Arduino transmitter)"
ARD_WIRING = "A0 (DAC) -> R1 (series) -> crown node -> R2 -> GND = watch back"
ARD_SAFETY = [
    "Nobody is touching the watch, its electrodes, the leads or the divider (the watch is NOT on a wrist).",
    "The DAC output reaches the watch only through the divider: series R1 = two resistors (e.g. 2 x 470 kOhm), "
    "R2 = 470 Ohm across crown-back.",
    "Laptop on battery with the charger unplugged; no mains-powered instrument connected to the circuit.",
    "The watch is off its charger.",
]
HUMAN_CHECKS = [
    "Laptop on battery, charger unplugged, and NO other cable to the laptop (no monitor/HDMI, dock, Ethernet, "
    "powered USB hub) - the Arduino's GND is the laptop's GND.",
    "PicoScope probes, function generator and every other instrument are disconnected from the circuit.",
    "Measured today with the power off: crown node to A0 >= 940 kOhm (two series resistors, each >= 470 kOhm), "
    "crown node to GND = R2 (~470 Ohm).",
    "Bench fault test passed today (README section 12): <= 11 uA into a 10 kOhm test load with the DAC held at "
    "maximum, R2 removed AND one series resistor shorted.",
    "Level checked on the PicoScope this session (ard_bench.py tone): <= 10 mVpp at the divider output.",
    "Watch off its charger; the participant has no implanted electronic device and has agreed to the test.",
]


def ber_plan_for(mod, bps, toneset="baseline", seconds=90.0, transmitter="teensy"):
    """ecg_link.ber_plan with a named tone set; transmitter='arduino' adds the Arduino's exact tone frequencies"""
    f1 = f0 = None
    if toneset == "exp":
        if int(bps) not in EXP_TONES or bps != int(bps):
            raise SystemExit(f"tone set 'exp' is defined for {sorted(EXP_TONES)} bps only (the rest use the baseline)")
        f1, f0 = EXP_TONES[int(bps)][mod]
    elif toneset != "baseline":
        raise SystemExit(f"unknown tone set {toneset!r}")
    plan = dict(L.ber_plan(mod, bps, f1, f0, seconds), toneset=toneset, exploratory=int(bps) in EXPLORATORY_BPS)
    return arduino_plan(plan) if transmitter == "arduino" else plan


def arduino_plan(plan):
    """the plan as the Arduino plays it: exact nominal frequencies (no AD9833 word quantisation). Refuses tones the
    frozen decoder would look for at another frequency (not a whole number of cycles per bit)"""
    p = dict(plan, transmitter="arduino")
    if p["kind"] == "ber":
        for k in ("f1", "f0"):
            f = p[f"{k}_hz"] or 0.0
            if f and abs(p[f"{k}_line_hz"] - f) > 1e-9:
                raise SystemExit(f"{f:g} Hz at {p['bps']:g} bps is not a whole number of cycles per bit: the frozen "
                                 f"decoder would look for it at {p[f'{k}_line_hz']:.4f} Hz")
            p[f"{k}_actual_hz"] = f
        p["cycles_per_bit"] = [round(f / p["bps"], 3) for f in (p["f1_hz"], p["f0_hz"]) if f]
        p["phase_continuous"] = all(abs(c - round(c)) < 1e-9 for c in p["cycles_per_bit"])
    elif p["kind"] == "alias":
        raise SystemExit("alias probe: not possible with the Arduino (4 kHz DAC update, 400 Hz limit)")
    else:
        p["tones_actual_hz"] = [float(f) for f in p["tones_hz"]]
    return p


def ard_settings(a):
    """calibration sent to the firmware; None = the nominal default (recorded as not measured)"""
    import arduino_tx as AT
    g = lambda k: getattr(a, k, None)
    s = dict(vfull=g("vfull") or AT.DEFAULTS["vfull"], r1=g("r1") or AT.DEFAULTS["r1"],
             r2=g("r2") or AT.DEFAULTS["r2"], amp_mvpp=g("amp_mvpp") or AT.DEFAULTS["amp_mvpp"],
             tx_ppm=g("tx_ppm") if g("tx_ppm") is not None else AT.DEFAULTS["tx_ppm"])
    s["nominal"] = [k for k in ("vfull", "r1", "r2", "tx_ppm") if g(k) is None]
    return s


def carrier_arduino(plan):
    base = dict(wave="sine", transmitter="arduino", toneset=plan.get("toneset"))
    if plan["kind"] != "ber":
        return dict(base, tones_hz=plan.get("tones_hz"))
    return dict(base, modulation=plan["modulation"], bps=plan["bps"], f1_hz=plan["f1_hz"], f0_hz=plan["f0_hz"] or None,
                f0_note=None if plan["f0_hz"] else "'0' = DAC at mid-scale (tone off)")


def watch_input_arduino(rec, cfg, st, vin):
    """measured (external instrument, --watch-vpp-mv) if entered, else CALCULATED from the firmware's settings.
    calc_vpp_mv / calc_dc_mv always carry the calculated (nominal) value, so a measured run still shows both."""
    calc = dict(calc_vpp_mv=cfg.get("amp_mvpp"), calc_dc_mv=cfg.get("dc_mv"))
    if (vin or {}).get("watch_vpp_mv") is not None:
        return dict(watch_input(rec, vin), **calc)
    nominal = ", ".join(st["nominal"]) or "none"
    return dict(point=WATCH_POINT, status="calculated", directly_measured=False, vpp_mv=cfg.get("amp_mvpp"), **calc,
                dc_mv=cfg.get("dc_mv"), ratio=st["r2"] / (st["r1"] + st["r2"]),
                basis=f"Arduino DAC {cfg.get('dac_vpp', float('nan')):.4f} Vpp around mid-scale x divider R1 {st['r1']:g} / "
                      f"R2 {st['r2']:g} Ohm, DAC swing {st['vfull']:g} V; values left at the nominal default (not "
                      f"measured): {nominal}; DC = DAC mid-scale x the divider; watch input impedance taken as >> R2")


def transmit_arduino(t, d, rec, plan, cues=False):
    """upload, start, give the ECG cues, wait for '# done', stop; writes tx.json"""
    import arduino_tx as AT
    step_us, reps, hz = AT.steps_of(plan)
    rec["schedule"] = dict(step_us=step_us, reps=reps, hz=hz)
    try:
        rec["upload"] = t.load(step_us, reps, hz)
        t0, r = t.go(AT.GO_DELAY_MS)
        rec.update(t_go=t0, go_reply=r, t_go_local=datetime.datetime.fromtimestamp(t0).isoformat(timespec="seconds"))
        end = t0 + plan["total_s"]
        cue = {0: "TRANSMITTING - do not start the ECG yet",
               ECG_START_S: ">>> NOW: start the 30 s ECG recording <<<",
               ECG_START_S + 35: "the ECG should be finished (if it is still recording, let it finish)"} if cues else \
              {0: "TRANSMITTING - start the ECG recording now"}
        done, shown, ticks = None, set(), set()
        while time.time() < end + 3 and not done:
            ln = t.readline(0.1)
            if ln and ln.startswith("# done"):
                done = ln
            el = time.time() - t0
            for k, msg in cue.items():
                if el >= k and k not in shown:
                    shown.add(k); print(f"  T+{k:3d} s  {msg}")
            left = int(end - time.time())
            if left % 10 == 0 and left not in ticks and left > 0:
                ticks.add(left); print(f"           ({left} s of transmission left)")
        rec["done_reply"] = done
        rec["tx_status"] = {k: v for k, v in t.status().items() if k != "raw"}
        print(f"  T+{plan['total_s']:.0f} s  transmission stopped")
    except KeyboardInterrupt:
        rec["stopped"] = "Ctrl+C"
        print("  stopped (Ctrl+C)")
    finally:
        try:
            t.off()
        except Exception as e:
            print(f"  ! could not confirm the output is off ({e})")
        rec["t_end"] = time.time()
        json.dump(rec, open(os.path.join(d, "tx.json"), "w"), indent=1)
    rec["operator_note"] = input("  Did the watch complete the 30 s ECG? (y / n / note) ").strip()
    json.dump(rec, open(os.path.join(d, "tx.json"), "w"), indent=1)
    st = rec.get("tx_status", {})
    ok = st.get("state") == "done" and st.get("events") == plan["n_steps"]
    print(f"  transmitter: state {st.get('state')}, events {st.get('events')}/{plan['n_steps']}"
          + ("" if ok else "   ** schedule did not complete **"))
    if rec.get("voltage"):
        print_voltage(rec["voltage"])


def ard_record(a, plan, st, cfg, t, condition, contact):
    vin = dict(voltage_inputs(a), r1_ohm=st["r1"], r2_ohm=st["r2"], declared_amp_mvpp=st["amp_mvpp"])
    rec = dict(kind=plan["kind"], label=a.label, condition=condition, plan=plan, amp_mvpp=st["amp_mvpp"],
               contact=contact, toneset=plan.get("toneset"), attenuator=dict(r1_ohm=st["r1"], r2_ohm=st["r2"]),
               wiring=ARD_WIRING, transmitter=dict(cfg, kind="arduino", settings=st), synthetic=False,
               decoder_sha=L.RULES_SHA, voltage_inputs=vin)
    rec["voltage"] = dict(carrier=carrier_arduino(plan), watch_input=watch_input_arduino(rec, cfg, st, vin),
                          a0=dict(point=ARD_NO_MONITOR, status="unavailable", directly_measured=False,
                                  reason="Arduino transmitter: A0 is the DAC output, there is no receive monitor"))
    return rec


def tx_arduino(a):
    import arduino_tx as AT
    if a.kind == "ber":
        plan = ber_plan_for(a.mod, a.bps, a.toneset, a.seconds, "arduino") if not (a.f1 or a.f0) else \
            arduino_plan(dict(L.ber_plan(a.mod, a.bps, a.f1, a.f0, a.seconds), toneset="custom",
                              exploratory=int(a.bps) in EXPLORATORY_BPS))
    else:
        plan = arduino_plan(dict(L.tone_plan(a.kind, a.seconds), toneset="sweep"))
    st = ard_settings(a)
    print(f"Arduino {plan['kind']}: {fmt_hz(carrier_arduino(plan))}; {plan['total_s']:.0f} s"
          + (" - EXPLORATORY rate" if plan.get("exploratory") else ""))
    worn = bool(getattr(a, "worn", False))
    print("\n" + ("Human-connected run (--worn): isolation and fault-protection checks (README section 12)" if worn
                  else "Bench checklist") + " - answer y to each:")
    answers = []
    for q in (HUMAN_CHECKS if worn else ARD_SAFETY):
        if input(f"  {q} [y/N] ").strip().lower() != "y":
            sys.exit("  stopped: checklist not confirmed - nothing was transmitted")
        answers.append(q)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    t = AT.ArduinoTX(a.port)
    d = None
    try:
        cfg = t.configure(st["vfull"], st["r1"], st["r2"], st["amp_mvpp"], st["tx_ppm"])
        cond = f"{a.condition}, ard-{plan['toneset']}"
        d = run_dir(a.label, plan)
        os.makedirs(d)
        rec = dict(ard_record(a, plan, st, cfg, t, cond, "worn" if worn else "bench"), checklist=answers)
        print_voltage(rec["voltage"])
        if cfg.get("amp_mvpp", 99) > 10.0:
            sys.exit("calculated level above 10 mVpp - stopped")
        if worn:
            input("\nECG app ready (not yet recording). Press Enter to start transmitting ")
        else:
            input(f"\nWatch: open the ECG app, contacts in place. Press Enter to start the transmitter "
                  f"({plan['total_s']:.0f} s), then start the ECG recording within ~{max(plan['total_s'] - 40, 5):.0f} s ")
        transmit_arduino(t, d, rec, plan, cues=worn)
    finally:
        t.close()
    if d:
        print(f"saved {os.path.relpath(d, HERE)}")
        print(f"next: python ecg_ber.py import <file.pdf> --run {os.path.relpath(d, HERE)}")


def run_session_arduino(a):
    """run_session for the Arduino: OOK then FSK at one rate and one tone set, worn watch. Asks the isolation and
    fault-protection checks once per session (recorded in session.json)."""
    import arduino_tx as AT
    check_frozen(a.allow_modified_decoder)
    mods = [m.strip() for m in a.mods.split(",")]
    order = [m for _ in range(a.repeats) for m in mods]
    plans = [ber_plan_for(m, a.bps, a.toneset, a.seconds, "arduino") for m in order]
    st = ard_settings(a)
    sess = os.path.join(DATADIR, f"session_{a.label}_{round(a.bps):g}bps_ard-{a.toneset}_{stamp()}")
    print(f"Session {os.path.relpath(sess, HERE)}: Arduino, tone set {a.toneset}, {len(order)} runs at {a.bps:g} bps, "
          f"{plans[0]['total_s']:.0f} s each, decoder {L.VERSION} sha {L.RULES_SHA} (frozen {L.FROZEN_AT})"
          + ("\n  ** 16 bps is EXPLORATORY (simulation: weak for both tone sets) **" if plans[0]["exploratory"] else ""))
    for m, p in zip(mods, plans):
        print(f"  {L.MOD_NAMES[m]}: {fmt_hz(carrier_arduino(p))}, frame {p['frame_bits']} bits = {p['frame_s']:.2f} s")
    print("\nHuman-connected run: isolation and fault-protection checks (README section 12) - answer y to each:")
    answers = []
    for q in HUMAN_CHECKS:
        if input(f"  {q} [y/N] ").strip().lower() != "y":
            sys.exit("  stopped: not confirmed - nothing was transmitted")
        answers.append(q)
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    t = AT.ArduinoTX(a.port)
    done = []
    try:
        cfg = t.configure(st["vfull"], st["r1"], st["r2"], st["amp_mvpp"], st["tx_ppm"])
        if cfg.get("amp_mvpp", 99) > 10.0:
            sys.exit("calculated level above 10 mVpp - stopped")
        os.makedirs(sess)
        cond = f"{a.condition}, ard-{a.toneset}"
        meta = dict(session=os.path.basename(sess), created=datetime.datetime.now().isoformat(timespec="seconds"),
                    bps=a.bps, order=order, condition=cond, toneset=a.toneset, interface=a.interface,
                    approval=a.approval, decoder=L.VERSION, rules_sha=L.RULES_SHA, transmitter=dict(cfg, settings=st),
                    human_checks=answers, watch_input_point=WATCH_POINT, runs=[])
        json.dump(meta, open(os.path.join(sess, "session.json"), "w"), indent=1, default=float)
        for i, (m, plan) in enumerate(zip(order, plans)):
            print(f"\n[{i + 1}/{len(order)}] {L.MOD_NAMES[m]} {a.bps:g} bps, tone set {a.toneset}")
            input("  ECG app ready (not yet recording). Press Enter to start transmitting ")
            d = os.path.join(sess, f"ecg_{a.label}_ber_{m}_{round(a.bps):g}bps_{stamp()}")
            os.makedirs(d)
            rec = dict(ard_record(a, plan, st, cfg, t, cond, "worn"), session=os.path.basename(sess),
                       approval=a.approval, interface=a.interface)
            print_voltage(rec["voltage"])
            transmit_arduino(t, d, rec, plan, cues=True)
            done.append(d)
            meta["runs"].append(dict(run=os.path.basename(d), modulation=m, voltage=rec.get("voltage")))
            json.dump(meta, open(os.path.join(sess, "session.json"), "w"), indent=1, default=float)
    except KeyboardInterrupt:
        print("stopped (Ctrl+C)")
    finally:
        t.close()
    print(f"\n{len(done)} runs saved in {os.path.relpath(sess, HERE)}")
    print("Export each reading as a PDF in the Fitbit app, copy the PDFs to this laptop, then:")
    print(f"  ~/.venvs/galvanic/bin/python ecg_ber.py ingest {os.path.relpath(sess, HERE)} <pdf files or folder>")


# ================================================================ import
def parse_recorded(s):
    for fmt in ("%b %d, %Y %I:%M %p", "%B %d, %Y %I:%M %p", "%d %b %Y %H:%M", "%b %d, %Y %H:%M",
                "%b %d, %Y, %I:%M %p", "%B %d, %Y, %I:%M %p"):          # app export 2026-11: comma before the time
        try:
            return datetime.datetime.strptime(s.strip(), fmt)
        except (ValueError, AttributeError):
            pass
    return None


def find_run(when):
    """the tx run whose transmit window overlaps [when, when + 90 s] (the PDF gives the minute only)"""
    hits = []
    for tj in glob.glob(os.path.join(DATADIR, "ecg_*", "tx.json")):
        r = json.load(open(tj))
        if r.get("synthetic") or "t_go" not in r:
            continue
        a, b = r["t_go"] - 60, r.get("t_end", r["t_go"] + r["plan"]["total_s"]) + 60
        w = when.timestamp()
        if w <= b and w + 90 >= a:
            hits.append(os.path.dirname(tj))
    return hits


def do_import(a):
    e = IO.load(a.file, scale=tuple(map(float, a.scale.split(","))) if a.scale else None, index=a.index)
    print(IO.describe(e))
    d = a.run
    if not d:
        when = parse_recorded(e.meta.get("recorded_on", "")) if e.meta.get("recorded_on") else None
        if when is None:
            sys.exit("no recording time in the file - give --run <run folder>")
        hits = find_run(when)
        if len(hits) != 1:
            sys.exit(f"recorded {when}: {len(hits)} transmit runs match ({[os.path.basename(h) for h in hits]}) - "
                     "give --run <run folder>")
        d = hits[0]
        print(f"paired by time ({when:%Y-%m-%d %H:%M}) with {os.path.relpath(d, HERE)}")
    if os.path.exists(os.path.join(d, "ecg.npz")) and not a.force:
        sys.exit(f"{d} already holds an ECG (ecg.npz) - not overwritten (--force replaces it, keeping a copy)")
    if os.path.exists(os.path.join(d, "ecg.npz")):
        keep = os.path.join(d, f"previous_{stamp()}")
        os.makedirs(keep)
        for f in glob.glob(os.path.join(d, "ecg*")):
            shutil.move(f, keep)
    ext = os.path.splitext(a.file)[1].lower()
    shutil.copy2(a.file, os.path.join(d, "ecg_source" + ext))
    IO.save_npz(e, os.path.join(d, "ecg.npz"))
    json.dump(dict(e.meta, samples=len(e.mv), imported=datetime.datetime.now().isoformat(timespec="seconds"),
                   original=os.path.abspath(a.file)), open(os.path.join(d, "ecg_import.json"), "w"), indent=1, default=str)
    print(f"imported into {os.path.relpath(d, HERE)}")


# ================================================================ analysis
def usability(e, rep_line_snr=None):
    dur = float(e.t[-1] - e.t[0] + 1 / e.fs)
    x = e.mv - np.median(e.mv)
    d = np.diff(e.mv)
    flat = np.zeros(len(e.mv), bool)
    run = 0
    for i, v in enumerate(d):
        run = run + 1 if v == 0 else 0
        if run >= int(0.1 * e.fs):
            flat[i - run + 1:i + 2] = True
    bad = (flat | (np.abs(x) > MAX_ABS_MV)).mean()
    why = []
    if dur < MIN_DURATION_S:
        why.append(f"too short ({dur:.1f} s)")
    if bad > MAX_BAD_FRAC:
        why.append(f"flat/saturated {bad:.0%}")
    if rep_line_snr is not None and rep_line_snr < MIN_LINE_SNR_DB:
        why.append(f"no tone (line {rep_line_snr:.1f} dB)")
    return dict(duration_s=dur, bad_fraction=float(bad), usable=not why, reason="; ".join(why))


def analyse_ber(d, rec, e):
    plan = rec["plan"]
    rep, tr = L.decode(e.mv, e.t, plan)
    _, truth, _ = L.frame(plan["bps"])
    lines = [f for f in (plan["f1_line_hz"], plan["f0_line_hz"]) if f]
    q = L.quality(e.mv, e.fs, [f / (1 + rep["clock_ppm"] * 1e-6) for f in lines])
    u = usability(e, rep["line_snr_db"])
    fr = rep["frames"]
    res = dict(run=os.path.basename(os.path.normpath(d)), kind="ber", condition=rec.get("condition"),
               modulation=plan["modulation"], bps=plan["bps"], f1_hz=plan["f1_hz"], f0_hz=plan["f0_hz"],
               amp_mvpp=rec.get("amp_mvpp"), synthetic=rec.get("synthetic", False), usability=u, quality=q,
               blind=dict(L.frames_summary(fr, rep["frames_expected"], len(truth)), errors_by_type=L.error_types(fr, truth)),
               v4_accounting=dict(L.frames_summary(fr, rep["frames_expected"], len(truth), "valid_v4"),
                                  errors_by_type=L.error_types(fr, truth, "valid_v4")),
               **{k: rep[k] for k in ("decoder", "rules_sha", "frozen_at", "clock_ppm", "residual_ppm", "line_snr_db",
                                      "header_score", "tau_s", "frames_complete", "frames_expected", "guard_ms",
                                      "es_n0_db")},
               frames=[{k: v for k, v in f.items() if k != "norm"} for f in fr],
               source=json.load(open(os.path.join(d, "ecg_import.json"))).get("source")
               if os.path.exists(os.path.join(d, "ecg_import.json")) else None)
    if rec.get("synthetic"):
        res["synth_truth"] = rec.get("synth")
    if os.path.exists(os.path.join(d, "monitor.npz")):
        res["monitor"] = monitor_decode(d)
    res["voltage"] = voltage_report(rec, d, rec.get("voltage_inputs"))   # runs from before 2026-11-03: rebuilt here
    return res, rep, tr


def tone_amp(x, t, f, win):
    """peak amplitude of a sine at f in a centred window of `win` samples, per sample (alignment only)"""
    return 2 * np.sqrt(L.tone_power(x, t, f, win // 2))


def step_windows(x, t, plan, edge_s=STEP_EDGE_S):
    """every complete schedule step inside the recording: (tone index, samples) with edge_s trimmed at both ends
    (switching transients). Aligned on the step of the in-band reference tone (16 Hz, else the first tone)."""
    step, cyc, hz = plan["step_s"], plan["cycle_s"], plan["tones_actual_hz"]
    iref = plan["tones_hz"].index(16) if 16 in plan["tones_hz"] else 0
    fs = (len(t) - 1) / (t[-1] - t[0])
    A = tone_amp(x, t, hz[iref], int(0.8 * step * fs))
    c0 = float(t[int(np.argmax(A))]) - (iref + 0.5) * step          # start of the cycle holding the reference peak
    out = []
    for n in range(-int(t[-1] // cyc) - 2, int(t[-1] // cyc) + 3):
        for i in range(len(hz)):
            a, b = c0 + n * cyc + i * step + edge_s, c0 + n * cyc + (i + 1) * step - edge_s
            if a >= t[0] and b <= t[-1]:
                k = (t >= a) & (t < b)
                out.append((i, x[k], t[k]))
    return out


def hann_amp(seg, tt, f):
    w = np.hanning(len(seg))
    return float(2 * abs(np.sum(w * seg * np.exp(-2j * np.pi * f * tt))) / np.sum(w))


def peak_hz(own, f, span=0.02, n=401):
    """frequency maximising the summed Hann-weighted tone power over the tone's own windows (+-span x f grid, then
    parabolic): ~0.001 Hz from 1 s windows at 30 dB, where an FFT bin is 1 Hz"""
    g = f * (1 + np.linspace(-span, span, n))
    P = np.array([sum(hann_amp(s_, tt, fg) ** 2 for s_, tt in own) for fg in g])
    k = int(np.argmax(P))
    if 0 < k < n - 1:
        y0, y1, y2 = np.log(P[k - 1:k + 2])
        den = y0 - 2 * y1 + y2
        return float(g[k] + (0.5 * (y0 - y2) / den if den else 0) * (g[1] - g[0]))
    return float(g[k])


def analyse_sweep(d, rec, e):
    """per tone: Hann-weighted amplitude in its own step windows; noise = the same frequency in the steps of tones
    more than 2.5 frequency bins away (median); observed frequency (fine fit, tones with SNR > 20 dB)"""
    plan = rec["plan"]
    x = e.mv - e.mv.mean(); t = e.t
    W = step_windows(x, t, plan)
    hz = plan["tones_actual_hz"]
    ref = (rec.get("amp_mvpp") or 1.0) / 2
    rows = []
    for i, (f, fa) in enumerate(zip(plan["tones_hz"], hz)):
        own = [(s_, tt) for j, s_, tt in W if j == i]
        other = [(s_, tt) for j, s_, tt in W if abs(hz[j] - fa) > 2.5 / (len(s_) / e.fs)]     # > 2.5 bins away
        if not own:
            rows.append(dict(f_hz=f, f_actual_hz=fa, windows=0)); continue
        A = float(np.mean([hann_amp(s_, tt, fa) for s_, tt in own]))
        N = float(np.median([hann_amp(s_, tt, fa) for s_, tt in other])) if other else float("nan")
        snr = float(20 * np.log10(A / max(N, 1e-12)))
        fo = peak_hz([(s_ - s_.mean(), tt) for s_, tt in own], fa) if snr > 20 and fa < e.fs / 2 else None
        rows.append(dict(f_hz=f, f_actual_hz=fa, windows=len(own), cycles_per_window=fa * len(own[0][0]) / e.fs,
                         amp_mv=A, gain_db=float(20 * np.log10(A / ref)), snr_db=snr,
                         f_observed_hz=fo))
    strong = [r for r in rows if r.get("windows") and 3 <= r["f_hz"] <= 30 and r["f_observed_hz"]]
    delta = float(np.median([r["f_actual_hz"] / r["f_observed_hz"] - 1 for r in strong])) if strong else 0.0
    for r in rows:
        r["shift_hz"] = r["f_observed_hz"] * (1 + delta) - r["f_actual_hz"] if r.get("f_observed_hz") else None
        r["resolved"] = bool(r.get("windows") and r["cycles_per_window"] >= 2)
        r["near_mains"] = bool(min(abs(r["f_actual_hz"] - m) for m in (50.0, 60.0)) < 1.5)
    flat = [r["gain_db"] for r in rows if r.get("windows") and 3 <= r["f_hz"] <= 25]
    g0 = float(np.median(flat)) if flat else 0.0
    for r in rows:
        if r.get("windows"):
            r["rel_db"] = r["gain_db"] - (g0 if flat else 0.0)
    passband = [r["f_hz"] for r in rows if r.get("windows") and r["rel_db"] > -3 and r["snr_db"] > 10 and r["resolved"]
                and not r["near_mains"]]
    return dict(run=os.path.basename(os.path.normpath(d)), kind=plan["kind"], condition=rec.get("condition"),
                amp_mvpp=rec.get("amp_mvpp"), synthetic=rec.get("synthetic", False), clock_ppm=delta * 1e6,
                flat_gain_db=g0 if flat else None, minus3db_tones=[min(passband), max(passband)] if passband else None,
                usability=usability(e), tones=rows)


def analyse_alias(d, rec, e):
    """per probe tone: the strongest line (0.5 Hz - Nyquist, mains excluded) in its own step windows, its level re the
    16 Hz reference step, and the level at the alias expected for plain 250 Hz sampling"""
    plan = rec["plan"]
    x = e.mv - e.mv.mean(); t = e.t
    W = step_windows(x, t, plan)
    ref = np.mean([hann_amp(s_, tt, plan["tones_actual_hz"][0]) for j, s_, tt in W if j == 0]) if W else float("nan")
    rows = []
    for i, (f, fa) in enumerate(zip(plan["tones_hz"], plan["tones_actual_hz"])):
        segs = [s_ - s_.mean() for j, s_, tt in W if j == i]
        if not segs:
            rows.append(dict(f_hz=f, windows=0)); continue
        n = len(segs[0]); nf = n * 8
        F = np.fft.rfftfreq(nf, 1 / e.fs)
        S = np.mean([np.abs(np.fft.rfft(s_[:n] * np.hanning(n), nf)) for s_ in segs], axis=0) * 2 / (0.5 * n)
        ok = (F > 0.5) & ~(np.abs(F - 50) < 1) & ~(np.abs(F - 60) < 1)
        j = int(np.argmax(np.where(ok, S, 0)))
        fal = abs(fa - round(fa / e.fs) * e.fs)
        ja = int(np.argmin(np.abs(F - fal)))
        rows.append(dict(f_hz=f, windows=len(segs), expected_alias_hz=fal, strongest_hz=float(F[j]),
                         strongest_rel_db=float(20 * np.log10(S[j] / ref)),
                         strongest_snr_db=float(20 * np.log10(S[j] / np.median(S[ok]))),
                         at_alias_rel_db=float(20 * np.log10(S[max(ja - 2, 0):ja + 3].max() / ref))))
    return dict(run=os.path.basename(os.path.normpath(d)), kind="alias", condition=rec.get("condition"),
                amp_mvpp=rec.get("amp_mvpp"), synthetic=rec.get("synthetic", False), ref_amp_mv=float(ref),
                usability=usability(e), tones=rows)


def plot_ber_run(res, rep, tr, e, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1, 1]))
    c = COLORS[res["modulation"]]
    ax[0].plot(tr["t"], tr["x"], lw=0.6, color="#33312c")
    ax[1].plot(tr["t"], tr["feat"], lw=0.8, color=c)
    nb, Tb, P = tr["nb"], tr["Tb"], tr["nb"] * tr["Tb"] * (1 + tr["eps"])
    _, truth, _ = L.frame(res["bps"])
    for f in rep["frames"]:
        s0 = f["s0"]
        for a_ in ax:
            a_.axvspan(s0, s0 + len(L.HDR) * Tb * (1 + tr["eps"]), color="#e4e3df", lw=0)
        tb = s0 + (len(L.HDR) + np.arange(len(truth)) + 0.5) * Tb * (1 + tr["eps"])
        nv = np.array(f["got"])
        y = np.array([v for v in rep_norm(rep, f["m"])])
        ax[2].scatter(tb, y, c=np.where(np.array(truth) == 1, c, "#8a8780"), s=12, zorder=3)
        bad = nv != np.array(truth)
        ax[2].scatter(tb[bad], y[bad], facecolors="none", edgecolors="#d62728", s=70, lw=1.5, zorder=4)
    ax[2].axhline(0, color="#52514e", lw=0.8)
    ax[0].set_ylabel("ECG [mV]"); ax[1].set_ylabel("timing feature"); ax[2].set_ylabel("bit statistic\n(0 = threshold)")
    ax[2].set_xlabel("time in the recording [s] (watch clock corrected)")
    b = res["blind"]
    ax[0].set_title(f"{res['run']}  -  {L.MOD_NAMES[res['modulation']]} {res['bps']:g} bps, "
                    f"{b['errors']}/{b['bits']} errors, {b['frames_valid']}/{b['frames_sent']} frames, "
                    f"Es/N0 {res['es_n0_db'] if res['es_n0_db'] is None else round(res['es_n0_db'], 1)} dB, "
                    f"clock {res['clock_ppm']:+.0f} ppm" + ("  [SYNTHETIC]" if res.get("synthetic") else ""), fontsize=9)
    ax[2].text(0.01, 0.95, "dots: sent 1 coloured, sent 0 grey; red rings = errors; grey bands = headers",
               transform=ax[2].transAxes, fontsize=8, va="top", color="#52514e")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def rep_norm(rep, m):
    return next(f["norm"] for f in rep["frames"] if f["m"] == m)


def plot_sweep(res, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = res["tones"]
    f = np.array([x["f_hz"] for x in r]); g = np.array([x["rel_db"] for x in r]); s = np.array([x["snr_db"] for x in r])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].semilogx(f, g, "o-", color="#1f5fbf"); ax[0].axhline(-3, color="#8a8780", ls=":")
    for x_, y_, s_ in zip(f, g, s):
        if s_ < 10:
            ax[0].plot(x_, y_, "o", mfc="white", color="#1f5fbf")
    ax[0].set_xlabel("scheduled tone [Hz]"); ax[0].set_ylabel("level re 3-25 Hz median [dB]")
    ax[0].set_title("passband (open = SNR < 10 dB)", fontsize=10)
    sh = [x["shift_hz"] if x["shift_hz"] is not None else np.nan for x in r]
    ax[1].semilogx(f, sh, "o-", color="#e66100"); ax[1].axhline(0, color="#8a8780", lw=0.8)
    ax[1].set_xlabel("scheduled tone [Hz]"); ax[1].set_ylabel("observed - scheduled [Hz] (clock corrected)")
    ax[1].set_title(f"frequency shift; watch clock {res['clock_ppm']:+.0f} ppm", fontsize=10)
    fig.suptitle(res["run"] + ("  [SYNTHETIC]" if res.get("synthetic") else ""), fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def collect(specs):
    dirs = []
    for s in specs:
        if os.path.isdir(s) and os.path.exists(os.path.join(s, "tx.json")):
            dirs.append(s)
        elif os.path.isdir(s):                                  # a session folder
            dirs += sorted(os.path.dirname(p) for p in glob.glob(os.path.join(s, "ecg_*", "tx.json")))
        else:                                                   # a label: its runs, in sessions or not
            dirs += sorted(os.path.dirname(p) for p in glob.glob(os.path.join(DATADIR, "**", f"ecg_{s}_*", "tx.json"),
                                                                 recursive=True) if "/synth_" not in p)
    return sorted(set(os.path.normpath(d) for d in dirs))


def analyse(specs, label, outroot=None, quiet=False):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dirs = collect(specs)
    if not dirs:
        sys.exit(f"no run folders for {specs}")
    out = os.path.join(outroot or DATADIR, f"ecgber_{label}_{stamp()}")
    os.makedirs(os.path.join(out, "runs"))
    rows, dropped, unusable, sweeps, aliases = [], [], [], [], []
    for d in dirs:
        rec = json.load(open(os.path.join(d, "tx.json")))
        name = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "ecg.npz")):
            dropped.append(dict(run=name, kind=rec["kind"], reason="no ECG imported", note=rec.get("operator_note"),
                                voltage=voltage_report(rec, d, rec.get("voltage_inputs")) if rec["kind"] == "ber" else None))
            continue
        e = IO.load(os.path.join(d, "ecg.npz"))
        if rec["kind"] == "ber":
            res, rep, tr = analyse_ber(d, rec, e)
            plot_ber_run(res, rep, tr, e, os.path.join(out, "runs", name + ".png"))
        elif rec["kind"] in ("sweep", "sweeplow"):
            res = analyse_sweep(d, rec, e); sweeps.append(res)
            plot_sweep(res, os.path.join(out, "runs", name + ".png"))
        else:
            res = analyse_alias(d, rec, e); aliases.append(res)
        json.dump(res, open(os.path.join(d, f"result_{L.VERSION}_{L.RULES_SHA}.json"), "w"), indent=1, default=float)
        if rec["kind"] != "ber":
            continue
        u = res["usability"]
        row = dict(run=name, condition=res["condition"], modulation=res["modulation"], bps=res["bps"],
                   amp_mvpp=res["amp_mvpp"], usable=u["usable"], reason=u["reason"], duration_s=round(u["duration_s"], 2),
                   line_snr_db=round(res["line_snr_db"], 1), es_n0_db=None if res["es_n0_db"] is None else round(res["es_n0_db"], 1),
                   clock_ppm=round(res["clock_ppm"]), frames_valid=res["blind"]["frames_valid"],
                   frames_expected=res["frames_expected"], bits=res["blind"]["bits"], errors=res["blind"]["errors"],
                   ber=res["blind"]["ber"], ber_conservative=res["blind"]["ber_conservative"],
                   sent1_read0=res["blind"]["errors_by_type"]["sent1_read0"],
                   sent0_read1=res["blind"]["errors_by_type"]["sent0_read1"],
                   v4acc_frames_valid=res["v4_accounting"]["frames_valid"], v4acc_bits=res["v4_accounting"]["bits"],
                   v4acc_errors=res["v4_accounting"]["errors"], monitor=(res.get("monitor") or {}).get("summary"),
                   synthetic=res.get("synthetic", False), **voltage_row(res["voltage"]))
        if res.get("synthetic"):
            row["synth_clock_ppm"] = res["synth_truth"]["clock_ppm"]
        rows.append(row)
        if not u["usable"]:
            unusable.append(dict(run=name, reason=u["reason"]))
    if rows:
        with open(os.path.join(out, "per_run.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader(); w.writerows(rows)
    summ = summarise(rows)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        if summ:
            w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    json.dump(dict(label=label, created=datetime.datetime.now().isoformat(timespec="seconds"), decoder=L.VERSION,
                   rules_sha=L.RULES_SHA, frozen_at=L.FROZEN_AT, runs=dirs, summary=summ, rows=rows,
                   dropped=dropped, unusable=unusable, sweeps=sweeps, aliases=aliases),
              open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    if summ:
        plot_summary(summ, os.path.join(out, "ber.png"), any(r["synthetic"] for r in rows))
    for s in sweeps:
        if not quiet:
            print(f"\nsweep {s['run']}: clock {s['clock_ppm']:+.0f} ppm, -3 dB tones {s['minus3db_tones']}")
            print("  f [Hz]  rel [dB]  SNR [dB]  shift [Hz]")
            for r in s["tones"]:
                if not r.get("windows"):
                    print(f"  {r['f_hz']:6g}  not in the recording"); continue
                sh = "-" if r["shift_hz"] is None else f"{r['shift_hz']:+.3f}"
                print(f"  {r['f_hz']:6g}  {r['rel_db']:7.1f}  {r['snr_db']:7.1f}  {sh}"
                      + ("" if r["resolved"] else "   (< 2 cycles per window: use sweeplow)")
                      + ("   (mains frequency: level includes mains pickup)" if r["near_mains"] else ""))
    for s in aliases:
        if not quiet:
            print(f"\nalias probe {s['run']} (reference 16 Hz = {s['ref_amp_mv']:.3f} mV):")
            for r in s["tones"]:
                if r.get("windows"):
                    print(f"  {r['f_hz']:>9g} Hz: strongest line {r['strongest_hz']:6.2f} Hz at {r['strongest_rel_db']:+6.1f} dB "
                          f"(SNR {r['strongest_snr_db']:4.1f} dB); at the 250 Hz alias {r['expected_alias_hz']:6.2f} Hz: "
                          f"{r['at_alias_rel_db']:+6.1f} dB")
    if not quiet:
        report(summ, dropped, unusable)
        print(f"\n-> {os.path.relpath(out, HERE)}")
    return out, summ, rows


def voltage_row(v):
    """flat voltage columns for per_run.csv / summary (watch input and A0 kept apart, each with its status)"""
    c, w, a = v["carrier"], v["watch_input"], v.get("a0") or {}
    t1 = next((x for x in a.get("tones", []) if abs(x["ad9833_hz"] - c.get("f1_ad9833_hz", -1)) < 1e-6), None)
    t0 = next((x for x in a.get("tones", []) if c.get("f0_hz") and abs(x["ad9833_hz"] - c["f0_ad9833_hz"]) < 1e-6), None)
    return dict(f1_hz=c.get("f1_hz"), f0_hz=c.get("f0_hz"), f1_ad9833_hz=c.get("f1_ad9833_hz"),
                watch_v_point=w["point"], watch_v_status=w["status"], watch_vpp_mv=w.get("vpp_mv"), watch_dc_mv=w.get("dc_mv"),
                watch_v_directly_measured=w.get("directly_measured", False),
                watch_vpp_calc_mv=w.get("calc_vpp_mv"), watch_dc_calc_mv=w.get("calc_dc_mv"),
                a0_v_status=a.get("status", "n/a"), a0_f1_vpp_mv=t1 and t1["vpp_mv"], a0_f1_floor_vpp_mv=t1 and t1["floor_vpp_mv"],
                a0_f1_detected=t1 and t1["detected"], a0_f0_vpp_mv=t0 and t0["vpp_mv"], a0_f0_detected=t0 and t0["detected"],
                a0_dc_mv=a.get("dc_tx_mv"), a0_dc_idle_mv=a.get("dc_idle_mv"))


def med(rr, k):
    v = [r[k] for r in rr if r.get(k) is not None]
    return float(np.median(v)) if v else None


def summarise(rows):
    # the watch-voltage status is part of the key: measured and calculated (nominal) runs are never pooled together
    vs = lambda r: str(r.get("watch_v_status"))
    keys = sorted({(r["condition"] or "", r["modulation"], r["bps"], vs(r)) for r in rows})
    out = []
    for cond, mod, bps, status in keys:
        rr = [r for r in rows if (r["condition"] or "") == cond and r["modulation"] == mod and r["bps"] == bps
              and vs(r) == status]
        ok = [r for r in rr if r["usable"]]
        e, n = sum(r["errors"] for r in ok), sum(r["bits"] for r in ok)
        fv, fe = sum(r["frames_valid"] for r in ok), sum(r["frames_expected"] for r in ok)
        lost = max(fe - fv, 0)
        pl = L.frame(bps)[1]
        missing = lost * len(pl)
        lo, hi = cp(e, n)
        es = [r["es_n0_db"] for r in ok if r["es_n0_db"] is not None]
        out.append(dict(condition=cond, modulation=mod, bps=bps, recordings=len(rr), usable=len(ok),
                        bits=n, errors=e, ber=e / n if n else None, ci95_lo=lo, ci95_hi=hi,
                        ber_conservative=(e + 0.5 * missing) / (n + missing) if n + missing else None,
                        frames_valid=fv, frames_expected=fe, es_n0_db_median=float(np.median(es)) if es else None,
                        line_snr_db_median=float(np.median([r["line_snr_db"] for r in ok])) if ok else None,
                        clock_ppm_median=float(np.median([r["clock_ppm"] for r in ok])) if ok else None,
                        v4acc_errors=sum(r["v4acc_errors"] for r in ok), v4acc_bits=sum(r["v4acc_bits"] for r in ok),
                        f1_hz=rr[0].get("f1_hz"), f0_hz=rr[0].get("f0_hz"),
                        watch_v_status="/".join(sorted({str(r.get("watch_v_status")) for r in rr})),
                        watch_vpp_mv_median=med(rr, "watch_vpp_mv"), watch_dc_mv_median=med(rr, "watch_dc_mv"),
                        watch_vpp_calc_mv_median=med(rr, "watch_vpp_calc_mv"),
                        watch_vpp_meas_mv_median=med([r for r in rr if r.get("watch_v_status") == "measured"],
                                                     "watch_vpp_mv"),
                        a0_f1_vpp_mv_median=med(rr, "a0_f1_vpp_mv"), a0_f1_floor_vpp_mv_median=med(rr, "a0_f1_floor_vpp_mv"),
                        a0_f1_detected=f"{sum(bool(r.get('a0_f1_detected')) for r in rr)}/"
                                       f"{sum(r.get('a0_f1_detected') is not None for r in rr)}",
                        a0_dc_mv_median=med(rr, "a0_dc_mv")))
    return out


def report(summ, dropped, unusable):
    print(f"\n{'condition':26s} {'mod':4s} {'bps':>4s} {'rec':>4s} {'use':>4s} {'errors/bits':>14s} {'BER':>8s} "
          f"{'95% CI':>19s} {'frames':>8s} {'Es/N0':>6s} {'line':>6s}")
    for s in summ:
        ber = f"{s['ber']:.2%}" if s["ber"] is not None else "-"
        ci = f"{s['ci95_lo']:.2%}-{s['ci95_hi']:.2%}" if s["ci95_lo"] is not None else "-"
        es = f"{s['es_n0_db_median']:.1f}" if s["es_n0_db_median"] is not None else "-"
        print(f"{s['condition'][:26]:26s} {L.MOD_NAMES[s['modulation']]:4s} {s['bps']:4g} {s['recordings']:4d} "
              f"{s['usable']:4d} {s['errors']:>6d}/{s['bits']:<7d} {ber:>8s} {ci:>19s} "
              f"{s['frames_valid']:>3d}/{s['frames_expected']:<4d} {es:>6s} "
              f"{'-' if s['line_snr_db_median'] is None else format(s['line_snr_db_median'], '6.1f'):>6s}")
    print("  BER = blind decoding, usable recordings pooled; frames = recovered / complete frames in the recordings; "
          "Es/N0 = median per recording [dB]; line = tone line over the noise 2.5 Hz away [dB]")
    print(f"\nvoltage  (watch input = {WATCH_POINT}; A0 = the board's E+ node, NOT the watch input)")
    print(f"{'condition':26s} {'mod':4s} {'bps':>4s} {'carrier [Hz]':>13s} {'watch: calculated (nominal)':>28s} "
          f"{'watch: measured':>16s} {'A0 tone':>22s} {'A0 DC':>8s}")
    for s in summ:
        wv = s.get("watch_v_status")
        calc = s.get("watch_vpp_calc_mv_median")
        if calc is None and wv in ("calculated", "simulated", "declared"):
            calc = s.get("watch_vpp_mv_median")
        c_txt = "-" if calc is None else f"{calc:.3f} mVpp" + (f" ({wv})" if wv in ("simulated", "declared") else "")
        meas = s.get("watch_vpp_meas_mv_median")
        m_txt = f"{meas:.3f} mVpp" if meas is not None else ("unavailable" if wv == "unavailable" else "not measured")
        a = ("-" if s.get("a0_f1_vpp_mv_median") is None else
             f"{s['a0_f1_vpp_mv_median']:.2f} mVpp, {s['a0_f1_detected']} det")
        car = f"{s['f1_hz']:g}/" + (f"{s['f0_hz']:g}" if s.get("f0_hz") else "off") if s.get("f1_hz") else "-"
        print(f"{s['condition'][:26]:26s} {L.MOD_NAMES[s['modulation']]:4s} {s['bps']:4g} {car:>13s} {c_txt:>28s} "
              f"{m_txt:>16s} {a:>22s} "
              f"{'-' if s.get('a0_dc_mv_median') is None else format(s['a0_dc_mv_median'], '.0f') + ' mV':>8s}")
    print("  watch input: measured = external instrument across crown-back (--watch-vpp-mv); calculated (nominal) = "
          "divider arithmetic from the settings, not a measurement; measured and calculated runs are never pooled in one "
          "row; unavailable = neither (README section 8 gives the minimum measurement setup). "
          "A0 'det' = recordings whose A0 shows the '1' tone at >= 3x its noise floor")
    for d in dropped:
        print(f"  dropped: {d['run']} ({d['reason']}; operator note: {d.get('note')})")
        if d.get("voltage"):
            print(f"           watch input {fmt_watch(d['voltage']['watch_input'])}; A0 {fmt_a0(d['voltage'].get('a0'))}")
    for u in unusable:
        print(f"  unusable: {u['run']} ({u['reason']})")


def plot_summary(summ, path, synthetic):
    """a Es/N0, b BER (pooled, 95 % Clopper-Pearson), c frames recovered, vs bit rate; colour = modulation.
    Up to 4 conditions share one row (line style = condition); more get one row each."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = [c for c in dict.fromkeys(s["condition"] for s in summ) if c != "impaired"]
    rates = sorted({s["bps"] for s in summ})
    groups = [conds] if len(conds) <= 4 else [[c] for c in conds]
    fig, axs = plt.subplots(len(groups), 3, figsize=(15, 4.4 * len(groups)), squeeze=False)
    styles = ["-", "--", ":", "-."]
    x = {b: i for i, b in enumerate(rates)}
    for g, ax in zip(groups, axs):
        for ci, cond in enumerate(g):
            for mod in ("ook", "fsk"):
                ss = [s for s in summ if s["condition"] == cond and s["modulation"] == mod and s["bits"]]
                if not ss:
                    continue
                xs = np.array([x[s["bps"]] for s in ss]) + (0.06 if mod == "fsk" else -0.06) + 0.03 * ci
                c, lab = COLORS[mod], L.MOD_NAMES[mod] + (f", {cond}" if cond and len(g) > 1 else "")
                ls = styles[ci % 4]
                y = np.array([s["ber"] for s in ss]); n = np.array([s["bits"] for s in ss])
                yd = np.maximum(y, 0.5 / n)
                lo = np.array([s["ci95_lo"] for s in ss]); hi = np.array([s["ci95_hi"] for s in ss])
                nz = y > 0                      # line and filled marker only where errors were seen
                ax[1].errorbar(xs, yd, yerr=[np.maximum(yd - lo, 0), np.maximum(hi - yd, 0)], fmt="none", ecolor=c,
                               lw=1.2, capsize=3)
                if nz.any():
                    ax[1].plot(xs[nz], yd[nz], ls + MARKERS[mod], color=c, ms=7, lw=1.6, label=lab)
                if (~nz).any():
                    ax[1].plot(xs[~nz], yd[~nz], "v", color=c, ms=10, mfc="white", mew=1.6,
                               label=None if nz.any() else lab)
                es = [np.nan if s["es_n0_db_median"] is None else s["es_n0_db_median"] for s in ss]
                ax[0].plot(xs, es, ls + MARKERS[mod], color=c, ms=7, label=lab)
                ax[2].plot(xs, [100 * s["frames_valid"] / max(s["frames_expected"], 1) for s in ss], ls + MARKERS[mod],
                           color=c, ms=7, label=lab)
        for a_ in ax:
            a_.set_xticks(range(len(rates))); a_.set_xticklabels([f"{b:g}" for b in rates]); a_.set_xlabel("bit rate [bps]")
            a_.grid(color="#e4e3df", lw=0.8)
        ax[1].set_yscale("log"); ax[1].set_ylabel("BER (blind, pooled; 95 % Clopper-Pearson)")
        ax[0].set_ylabel("Es/N0 [dB] (median per recording)"); ax[2].set_ylabel("frames recovered [%]")
        ax[2].set_ylim(-5, 105)
        ax[1].legend(fontsize=7, loc="lower right")
        ax[1].text(0.01, 0.01, "open v = 0 errors (drawn at 0.5/bits)", transform=ax[1].transAxes, fontsize=7,
                   color="#52514e")
        if len(g) == 1 and g[0]:
            ax[0].set_title(g[0], loc="left", fontsize=10)
    fig.suptitle("Pixel Watch 2 ECG receiver" + ("  [SYNTHETIC DRY RUN - not watch data]" if synthetic else ""), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(path, dpi=110); plt.close(fig)


# ================================================================ synthetic dry run
def synth(a):
    import ecg_synth as Y
    rng = np.random.default_rng(a.seed)
    root = os.path.join(DATADIR, f"synth_{a.label}_{stamp()}")
    runs = os.path.join(root, "runs")
    os.makedirs(runs)
    mods = a.mods.split(","); rates = [float(r) for r in a.rates.split(",")]
    amps = [float(v) for v in a.amps.split(",")]
    made = 0
    for amp in amps:
        for mod in mods:
            for bps in rates:
                plan = L.ber_plan(mod, bps)
                for i in range(a.n):
                    t, x, info = Y.record(plan, rng, amp_mvpp=amp, white_uv=a.white_uv)
                    made += write_synth_run(runs, f"a{amp:g}", plan, x, info, amp, i, Y)
    if a.impaired:                               # recordings the pipeline must flag, not decode into nonsense
        plan = L.ber_plan("ook", 4)
        for i, kw in enumerate([dict(no_signal=True), dict(dropout_s=6.0), dict(dropout_s=0.5)]):
            t, x, info = Y.record(plan, rng, amp_mvpp=1.0, white_uv=a.white_uv, **kw)
            made += write_synth_run(runs, "impaired", plan, x, info, 1.0, i, Y)
        d = os.path.join(runs, f"ecg_synth_dropped_ber_ook_4bps_{0:02d}")
        os.makedirs(d)
        json.dump(dict(kind="ber", label="synth", condition="impaired", plan=plan, amp_mvpp=1.0, synthetic=True,
                       operator_note="n - app stopped (synthetic example of a dropped recording)"),
                  open(os.path.join(d, "tx.json"), "w"), indent=1)
    sw = L.tone_plan("sweep", 60.0)
    t, x, info = Y.record(sw, rng, amp_mvpp=1.0, white_uv=a.white_uv)
    made += write_synth_run(runs, "sweep", sw, x, info, 1.0, 0, Y)
    print(f"{made} synthetic recordings written as Fitbit-style PDFs and imported through the PDF reader")
    out, summ, rows = analyse([os.path.join(runs, d) for d in sorted(os.listdir(runs))], f"synth_{a.label}", root)
    check_synth(out, summ, rows, root)


def write_synth_run(runs, cond, plan, x, info, amp, i, Y):
    tag = plan["kind"] if plan["kind"] != "ber" else f"ber_{plan['modulation']}_{round(plan['bps']):g}bps"
    d = os.path.join(runs, f"ecg_synth_{cond}_{tag}_{i:02d}")
    os.makedirs(d)
    json.dump(dict(kind=plan["kind"], label="synth", condition=cond, plan=plan, amp_mvpp=amp, synthetic=True,
                   synth=info), open(os.path.join(d, "tx.json"), "w"), indent=1)
    pdf = os.path.join(d, "synthetic_export.pdf")
    Y.write_pdf(pdf, x)
    e = IO.load(pdf)                             # the real import path
    if len(e.mv) != len(x) or np.max(np.abs(e.mv - x)) > 1e-9:
        raise RuntimeError(f"PDF round trip changed the samples in {d}")
    IO.save_npz(e, os.path.join(d, "ecg.npz"))
    json.dump(dict(e.meta, samples=len(e.mv)), open(os.path.join(d, "ecg_import.json"), "w"), indent=1, default=str)
    return 1


CLOCK_TRUE = 1227.0                                         # ecg_synth.CLOCK_PPM


def check_synth(out, summ, rows, root):
    """dry-run acceptance: clean high-level cells error-free, clock recovered, impaired ones flagged, BER vs theory"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, L.MAXTOUCH)
    ok = True
    hi = [s for s in summ if s["condition"] == "a1"]
    for s in hi:
        good = s["errors"] == 0 and s["frames_valid"] == s["frames_expected"] and s["usable"] == s["recordings"]
        ok &= good
        print(f"  check 1 mVpp {s['modulation']} {s['bps']:g} bps: {s['errors']} errors, frames {s['frames_valid']}/"
              f"{s['frames_expected']} -> {'OK' if good else 'FAIL'}")
    clk = [abs(r["clock_ppm"] - r["synth_clock_ppm"]) for r in rows if r.get("synth_clock_ppm") and r["usable"]]
    if clk:
        print(f"  check clock recovery: |error| median {np.median(clk):.1f} ppm, max {max(clk):.1f} ppm (must stay inside "
              f"the +-200 ppm residual search; 100 ppm = 3 ms over 30 s) -> {'OK' if max(clk) < 200 else 'FAIL'}")
        ok &= max(clk) < 200
    imp = [r for r in rows if r["condition"] == "impaired"]
    for r in imp:
        print(f"  impaired {r['run']}: usable={r['usable']} {r['reason']}  frames {r['frames_valid']}/{r['frames_expected']}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    g = np.linspace(-5, 25, 200)
    for k, mod in enumerate(("ook", "fsk")):
        rr = [r for r in rows if r["modulation"] == mod and r["usable"] and r["es_n0_db"] is not None and r["bits"]
              and r["condition"] not in ("impaired",)]
        bl = sorted({r["bps"] for r in rr})
        for b, mk in zip(bl, "os^D"):
            q = [r for r in rr if r["bps"] == b]
            col = plt.cm.viridis(bl.index(b) / max(len(bl), 1))
            nz = [r for r in q if r["errors"]]; z = [r for r in q if not r["errors"]]
            ax[k].semilogy([r["es_n0_db"] for r in nz], [r["ber"] for r in nz], mk, ms=5, alpha=0.75, color=col,
                           label=f"{b:g} bps")
            ax[k].semilogy([r["es_n0_db"] for r in z], [0.5 / r["bits"] for r in z], "v", ms=6, mfc="white", color=col)
        th = 0.5 * np.exp(-10 ** (g / 10) / {"ook": 4, "fsk": 2}[mod])
        ax[k].semilogy(g, th, "k--", lw=1, label=f"noncoherent {L.MOD_NAMES[mod]}, AWGN")
        ax[k].set_ylim(1e-3, 0.7); ax[k].set_xlabel("Es/N0 measured per recording [dB]"); ax[k].set_title(L.MOD_NAMES[mod])
        ax[k].grid(color="#e4e3df", lw=0.8); ax[k].legend(fontsize=7)
    ax[0].set_ylabel("BER per recording (open v = 0 errors, at 0.5/bits)")
    fig.suptitle("SYNTHETIC dry run: decoder BER vs theory (Es/N0 here is a lower bound: ISI and leakage count as noise)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "synth_ber_vs_esn0.png"), dpi=120); plt.close(fig)
    sj = json.load(open(os.path.join(out, "summary.json")))
    for s in sj["sweeps"]:
        import ecg_synth as Y
        from scipy import signal
        f = np.array([r["f_hz"] for r in s["tones"]])
        _, h = signal.sosfreqz(Y.watch_filter(Y.FS_SIM), worN=f, fs=Y.FS_SIM)     # the filter the model applied
        model = 20 * np.log10(np.abs(h)); model -= np.median(model[(f >= 3) & (f <= 25)])
        meas = np.array([r["rel_db"] for r in s["tones"]])
        sel = np.array([r["resolved"] and r["snr_db"] > 20 and not r["near_mains"] for r in s["tones"]])
        err = np.max(np.abs(meas[sel] - model[sel]))
        ce = abs(s["clock_ppm"] - CLOCK_TRUE)
        print(f"  check passband sweep recovers the model response ({f[sel].min():g}-{f[sel].max():g} Hz, SNR > 20 dB): "
              f"max |error| {err:.2f} dB, clock {s['clock_ppm']:+.0f} ppm (true {CLOCK_TRUE:+.0f}) -> "
              f"{'OK' if err < 1 and ce < 300 else 'FAIL'}  (sweep clock: 1 s per tone, +-~200 ppm; info only)")
        ok &= err < 1 and ce < 300
    print(f"\nDRY RUN {'PASSED' if ok else 'FAILED'}  ({os.path.relpath(root, HERE)})")


# ================================================================ maXTouch vs ECG
def compare(specs, label):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = []
    for s in specs:
        name, d = s.split("=", 1)
        j = json.load(open(os.path.join(d, "summary.json")))
        if "rows" in j and j["rows"] and "blind_bits" in j["rows"][0]:          # stent_modcomp summary
            rr = [dict(modulation=r["modulation"], bps=r["bps"], e=r.get("blind_errors") or 0, n=r.get("blind_bits") or 0)
                  for r in j["rows"] if r["modulation"] in ("ook", "fsk")]
        else:                                                                 # ecgber summary
            rr = [dict(modulation=r["modulation"], bps=r["bps"], e=r["errors"], n=r["bits"]) for r in j["summary"]]
        pooled = {}
        for r in rr:
            k = (r["modulation"], round(r["bps"]))
            pooled.setdefault(k, [0, 0]); pooled[k][0] += r["e"]; pooled[k][1] += r["n"]
        sets.append((name, pooled))
    out = os.path.join(DATADIR, f"compare_{label}_{stamp()}")
    os.makedirs(out)
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for k, (name, pooled) in enumerate(sets):
        for mod in ("ook", "fsk"):
            pts = sorted((b, e, n) for (m, b), (e, n) in pooled.items() if m == mod and n)
            if not pts:
                continue
            b = np.array([p[0] for p in pts]); e = np.array([p[1] for p in pts]); n = np.array([p[2] for p in pts])
            y = np.maximum(e / n, 0.5 / n)
            ci = np.array([cp(int(ee), int(nn)) for ee, nn in zip(e, n)])
            ax.errorbar(b, y, yerr=[np.maximum(y - ci[:, 0], 0), np.maximum(ci[:, 1] - y, 0)], fmt=["-", "--", ":"][k % 3]
                        + MARKERS[mod], color=COLORS[mod], capsize=3, label=f"{L.MOD_NAMES[mod]}, {name}")
            rows += [dict(receiver=name, modulation=mod, bps=int(bb), errors=int(ee), bits=int(nn), ber=ee / nn,
                          ci95_lo=c[0], ci95_hi=c[1]) for bb, ee, nn, c in zip(b, e, n, ci)]
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("bit rate [bps]"); ax.set_ylabel("BER (blind, pooled, 95 % CI)")
    ax.grid(color="#e4e3df", lw=0.8); ax.legend(fontsize=8)
    ax.set_title("Same transmitter, two receivers: maXTouch panel vs Pixel Watch ECG", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, "compare.png"), dpi=120)
    with open(os.path.join(out, "compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> {os.path.relpath(out, HERE)}")


# ================================================================ self-test
def selftest():
    import stent_frames as SF
    import decode_v4 as V4
    ok = True
    p7 = L.prbs(7)
    same = np.array_equal(np.roll(SF.prbs7(), 0), p7)
    print(f"PRBS7 = stent_frames.prbs7: {same}"); ok &= same
    rng = np.random.default_rng(3)
    Tb, nb = 0.05, len(SF.frame_bits()[0])
    t = np.sort(rng.uniform(0, 30, 3000)); f = rng.normal(0, 1, len(t))
    frames = np.arange(0, 3)
    mine = L.header_score(f, t, 0.37, Tb, 1e-4, nb, frames, np.array(V4.HDR), "sum")
    theirs = V4.header_score(f, t, 0.37, Tb, 1e-4, nb, frames)
    print(f"header_score(sum, v4 header) = decode_v4.header_score: {mine:.12f} vs {theirs:.12f}")
    ok &= abs(mine - theirs) < 1e-12
    af = lambda tau, eps: np.arange(0, 3)
    a1 = L.search(f, t, Tb, nb, 0.0, 0.2, af, V4.HDR, "sum")[:2]
    a2 = V4.search(f, t, Tb, nb, 0.0, 0.2, af)
    print(f"search(sum, v4 header) = decode_v4.search: {a1} vs {a2}"); ok &= np.allclose(a1, a2)
    for b in (2, 4, 8, 16):
        for m in ("ook", "fsk"):
            p = L.ber_plan(m, b)
            print(f"  {m} {b:2d} bps: frame {p['frame_bits']} bits = {p['frame_s']:.2f} s, PRBS{p['prbs_order']}, "
                  f"cycles/bit {p['cycles_per_bit']}, phase-continuous {p['phase_continuous']}, "
                  f"lines {p['f1_line_hz']:g}/{p['f0_line_hz']:g} Hz")
            ok &= p["phase_continuous"] and p["frame_s"] <= L.FRAME_MAX_S
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))


def plan_cmd(a):
    print(f"ECG frames: header {L.HDR} (preamble 1010 + Barker-7), payload PRBS-n, gap {L.GAP}; frame <= {L.FRAME_MAX_S:g} s")
    for b in (2, 4, 8, 16):
        for m in ("ook", "fsk"):
            p = L.ber_plan(m, b)
            print(f"  {L.MOD_NAMES[m]} {b:2d} bps: f1 {p['f1_hz']:g} Hz (word -> {p['f1_actual_hz']:.3f} Hz, line "
                  f"{p['f1_line_hz']:g})" + (f", f0 {p['f0_hz']:g} Hz" if p["f0_hz"] else ", 0 = off (mid-scale)")
                  + f"; frame {p['frame_bits']} bits = {p['frame_s']:.2f} s, payload {p['payload_bits']} bits, "
                  f"{max(int(29.5 // p['frame_s']) - 1, 0) * p['payload_bits']}-{int(29.5 // p['frame_s']) * p['payload_bits']} "
                  f"payload bits per 29.5 s recording (complete frames only)")
    s = L.tone_plan("sweep"); al = L.tone_plan("alias")
    print(f"  sweep: {len(s['tones_hz'])} tones x {s['step_s']:g} s = {s['cycle_s']:g} s cycle: {s['tones_hz']}")
    print(f"  alias probe: {len(al['tones_hz'])} tones x {al['step_s']:g} s (16 Hz reference first): {al['tones_hz']}")


def add_arduino_args(p):
    """--tx arduino (README section 12). The defaults keep the TeensyTRX path exactly as before."""
    p.add_argument("--tx", choices=("teensy", "arduino"), default="teensy", help="transmitter (default teensy)")
    p.add_argument("--toneset", choices=("baseline", "exp"), default="baseline",
                   help="--tx arduino: baseline = ecg_link tones; exp = 2 bps OOK 10 Hz / FSK 12/8 Hz")
    p.add_argument("--vfull", type=float, help="--tx arduino: DAC swing V(hold max) - V(hold min), volts (measured)")
    p.add_argument("--tx-ppm", type=float, help="--tx arduino: board clock error, ppm (ard_bench.py tone)")
    p.add_argument("--worn", action="store_true", help="tx --tx arduino: the watch is worn (human-connected checks, "
                   "ECG cue at T+5 s); 'run' is always worn")


def add_voltage_args(p, divider=True):
    """attenuator and external-measurement declarations for the voltage report (none of them changes the transmission)"""
    if divider:
        p.add_argument("--r1", type=float, help="attenuator series resistor [Ohm] (for the CALCULATED watch voltage)")
        p.add_argument("--r2", type=float, help="attenuator shunt resistor across crown-back [Ohm]")
        p.add_argument("--source-vpp", type=float, help=f"Vpp at the attenuator input (default: nominal {AD9833_VPP:g})")
    p.add_argument("--source-point", choices=("vout", "eplus"), default="vout", help="attenuator input: AD9833 VOUT or E+")
    p.add_argument("--source-measured", action="store_true", help="--source-vpp/--vout-vpp was measured (scope)")
    p.add_argument("--source-dc-mv", type=float, help=f"DC at the attenuator input [mV] (default nominal {AD9833_DC_MV:g})")
    p.add_argument("--dc-block", action="store_true", help="a series capacitor blocks DC before the watch")
    p.add_argument("--watch-vpp-mv", type=float, help="MEASURED Vpp across crown-back with an external instrument")
    p.add_argument("--watch-dc-mv", type=float, help="MEASURED DC across crown-back [mV]")
    p.add_argument("--watch-instrument", help="what measured --watch-vpp-mv / --watch-dc-mv, and how")


def main():
    ap = argparse.ArgumentParser(description="Pixel Watch 2 ECG receiver: transmit, import, decode, BER")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan"); sub.add_parser("selftest")
    p = sub.add_parser("divider")
    p.add_argument("--vout-vpp", type=float, default=AD9833_VPP); p.add_argument("--r1", type=float, default=320e3)
    p.add_argument("--r2", type=float, default=560.0)
    p = sub.add_parser("tx")
    p.add_argument("--kind", choices=("ber", "sweep", "sweeplow", "alias"), required=True)
    p.add_argument("--mod", choices=("ook", "fsk"), default="ook"); p.add_argument("--bps", type=float, default=4.0)
    p.add_argument("--f1", type=float); p.add_argument("--f0", type=float)
    p.add_argument("--seconds", type=float, default=90.0, help="transmit this long (frames repeat); default 90")
    p.add_argument("--label", required=True); p.add_argument("--condition", default="bench-direct")
    p.add_argument("--amp-mvpp", type=float, required=True, help="tone amplitude at the watch electrodes (from the divider)")
    p.add_argument("--vout-vpp", type=float, default=AD9833_VPP); p.add_argument("--r1", type=float); p.add_argument("--r2", type=float)
    p.add_argument("--wiring", default="crown <- attenuator out, back <- TX ground (WatchLink orientation)")
    p.add_argument("--monitor", action="store_true", help="stream the transmitter's A0 at 2 kS/s and decode it too")
    p.add_argument("--port"); add_voltage_args(p, divider=False); add_arduino_args(p)
    p = sub.add_parser("run", help="worn watch: OOK + FSK at one rate")
    p.add_argument("--bps", type=float, required=True); p.add_argument("--label", required=True)
    p.add_argument("--mods", default="ook,fsk"); p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seconds", type=float, default=60.0, help="transmission per run (frames repeat); default 60")
    p.add_argument("--approval", help="optional: safety review reference, recorded in tx.json")
    p.add_argument("--interface", default="same TX setup as the maXTouch through-body runs",
                   help="transmitter / electrode setup, recorded in tx.json")
    p.add_argument("--condition", default="worn-still")
    p.add_argument("--no-monitor", dest="monitor", action="store_false"); p.add_argument("--port")
    p.add_argument("--allow-modified-decoder", action="store_true"); add_voltage_args(p)
    add_arduino_args(p); p.add_argument("--amp-mvpp", type=float, help="--tx arduino: level at the watch (default 2)")
    p = sub.add_parser("ingest", help="import a session's exported PDFs and analyse it")
    p.add_argument("session"); p.add_argument("files", nargs="+", help="PDF files or a folder of them")
    p.add_argument("--allow-modified-decoder", action="store_true")
    p = sub.add_parser("import")
    p.add_argument("file"); p.add_argument("--run"); p.add_argument("--force", action="store_true")
    p.add_argument("--scale", help="mm_per_s,mm_per_mV,fs (only if the PDF page text cannot be read)")
    p.add_argument("--index", type=int, default=0, help="json/csv: which reading")
    p = sub.add_parser("analyse")
    p.add_argument("runs", nargs="+", help="labels or run folders"); p.add_argument("--label")
    p = sub.add_parser("synth")
    p.add_argument("--label", default="dry"); p.add_argument("--n", type=int, default=6)
    p.add_argument("--mods", default="ook,fsk"); p.add_argument("--rates", default="2,4,8,16")
    p.add_argument("--amps", default="1,0.1,0.05,0.03,0.02", help="mVpp at the watch; 1 = the bench start level")
    p.add_argument("--white-uv", type=float, default=10.0); p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--no-impaired", dest="impaired", action="store_false")
    p = sub.add_parser("compare")
    p.add_argument("sets", nargs="+", help="NAME=<modcomp folder or ecgber folder>"); p.add_argument("--label", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan_cmd(a)
    elif a.cmd == "selftest":
        selftest()
    elif a.cmd == "divider":
        dv = divider(a.vout_vpp, a.r1, a.r2)
        for k, v in dv.items():
            print(f"  {k:26s} {v:.4g}")
    elif a.cmd == "tx":
        tx_arduino(a) if a.tx == "arduino" else tx(a)
    elif a.cmd == "run":
        run_session_arduino(a) if a.tx == "arduino" else run_session(a)
    elif a.cmd == "ingest":
        ingest(a)
    elif a.cmd == "import":
        do_import(a)
    elif a.cmd == "analyse":
        analyse(a.runs, a.label or "_".join(os.path.basename(os.path.normpath(r)) for r in a.runs)[:60])
    elif a.cmd == "synth":
        synth(a)
    elif a.cmd == "compare":
        compare(a.sets, a.label)


if __name__ == "__main__":
    main()
