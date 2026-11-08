"""Driver + bench CLI for the EXPERIMENTAL watch_psk_tx x2 firmware (psk-x1 modulations). Added 2026-11-06.

This file has no worn-watch mode (that is psk_ber.py run, gated on the safety review).

  python psk_tx.py id
  python psk_tx.py off
  python psk_tx.py cw 16 --seconds 60                       (carrier only, for the scope)
  python psk_tx.py frame --bps 16 [--mod dbpsk] [--shape rect|rc|rrc] [--fc 16] [--seconds 45] [calibration]
  python psk_tx.py sweep --rates 16,32,64,100 [--fcs 16] [--shape rect] [--seg 10] --label s1 [calibration]
        plays every (rate, carrier) for --seg seconds, 2 s of silence between, all inside ONE scope capture; the
        record (data_watch_ecg/benchsweep_<label>_<time>.json) tells psk_bench.py sweep what was played and when.
calibration: --vfull V --r1 OHM --r2 OHM --amp-mvpp 2 --tx-ppm P   (nominal defaults if not given; recorded as such)
"""
from __future__ import annotations

import argparse
import re
import time

import numpy as np

import arduino_tx as AT
import psk_link as P

FW_ID = "watch_psk_tx x2"
SYM_SCALE, PULSE_SCALE = 120, 8192          # watch_psk_tx.ino SYM_SCALE / PULSE_SCALE
LINE_SYMS = 64


def fw_payload(p):
    """what the firmware plays: int8 symbols, int16 pulse table, the peak norm of the QUANTISED waveform, and the
    loop (identical repeated frames stored once: DBPSK frames have an even number of flips, so they all match)"""
    a = P.symbols(p)
    si = np.round(a.real * SYM_SCALE).astype(int); sq = np.round(a.imag * SYM_SCALE).astype(int)
    ls, nf, F = p["lead_symbols"], p["frame_symbols"], p["frames"]
    fr = [(tuple(si[ls + m * nf: ls + (m + 1) * nf]), tuple(sq[ls + m * nf: ls + (m + 1) * nf])) for m in range(F)]
    if F > 1 and all(f == fr[0] for f in fr):
        keep = np.r_[np.arange(ls + nf), np.arange(ls + F * nf, len(si))]
        loop = (ls, nf, F)
    else:
        keep, loop = np.arange(len(si)), (0, 0, 1)
    if np.any(np.abs(si) > 127) or np.any(np.abs(sq) > 127):
        raise ValueError("symbol outside int8")
    span, table = P.pulse_table(p)
    pt = np.round(table * PULSE_SCALE).astype(int)
    if np.any(np.abs(pt) > 32767):
        raise ValueError("pulse value outside int16")
    aq = (si + 1j * sq) / SYM_SCALE
    tq = pt / PULSE_SCALE
    tt = np.arange(0, p["total_s"], 1 / 4000.0)
    norm = float(np.max(np.abs(P.baseband(p, aq, span, tq, tt))))
    return dict(si=si[keep], sq=sq[keep], span=span, pulse=pt, norm=norm, aq=aq, tq=tq, loop=loop, n_play=len(si))


def model(p, t_tx, fw=None):
    """the firmware's output (relative, peak 1) at TX times t - quantised exactly as uploaded"""
    fw = fw or fw_payload(p)
    nm = int(round(fw["norm"] * 1e6)) * 1e-6
    b = P.baseband(p, fw["aq"], fw["span"], fw["tq"], t_tx)
    fc = int(round(p["fc_hz"] * 1000)) / 1000.0
    return np.clip(np.real(b * np.exp(2j * np.pi * fc * t_tx)) / nm, -1.05, 1.05)


def checksum(sym_us, fc_mhz, si, sq, span, pulse, pts=P.PTS, loop=(0, 0, 1)):
    n = len(si)
    s = (sym_us * 7 + n * 13 + fc_mhz * 3 + span * 5 + pts + loop[0] * 17 + loop[1] * 19 + loop[2] * 23) & 0xFFFFFFFF
    for i, (a, b) in enumerate(zip(si, sq)):
        s = (s + (i + 1) * (((a & 0xFF) << 8) | (b & 0xFF))) & 0xFFFFFFFF
    for j, v in enumerate(pulse):
        s = (s + (j + 1) * (v & 0xFFFF)) & 0xFFFFFFFF
    return s


def hex16(vals):
    return "".join(f"{v & 0xFFFF:04x}" for v in vals)


class PskTX(AT.ArduinoTX):
    def __init__(self, port=None, timeout=1.0):                 # same as ArduinoTX, other firmware id
        import serial
        self.port = port or AT.find_port()
        try:
            self.dev = serial.Serial(self.port, 115200, timeout=timeout)
        except serial.SerialException as e:
            raise SystemExit(f"{self.port}: {e}\n(close the Arduino IDE Serial Monitor - only one program can use the port)")
        time.sleep(1.5)
        self.dev.reset_input_buffer()
        self.ident = self.ask("id", r"^# watch_", timeout=4.0)
        if FW_ID not in self.ident:
            raise SystemExit(f"{self.port}: firmware '{self.ident}' - need {FW_ID}: upload arduino/watch_psk_tx")
        self.cfg = {}

    def upload(self, p):
        fw = fw_payload(p)
        n = len(fw["si"])
        if n > 4096:
            raise SystemExit(f"{n} symbols > 4096 (firmware buffer): shorten --seconds")
        fc_mhz = int(round(p["fc_hz"] * 1000))
        ls, ll, lr = fw["loop"]
        r = self.ask(f"cfg {p['sym_us']} {fc_mhz} {n} {fw['span']} {P.PTS} {int(round(fw['norm'] * 1e6))} {ls} {ll} {lr}",
                     r"^# cfg ok")
        if int(r.split()[-1]) != fw["n_play"]:
            raise RuntimeError(f"board plays {r.split()[-1]} symbols, expected {fw['n_play']}")
        packed = [((int(a) & 0xFF) << 8) | (int(b) & 0xFF) for a, b in zip(fw["si"], fw["sq"])]
        for i in range(0, n, LINE_SYMS):
            self.ask(f"s {i} {hex16(packed[i:i + LINE_SYMS])}", r"^# ok")
        pl = [int(v) for v in fw["pulse"]]
        for i in range(0, len(pl), LINE_SYMS):
            self.ask(f"p {i} {hex16(pl[i:i + LINE_SYMS])}", r"^# ok")
        got = int(self.ask("sum", r"^# sum").split()[-1])
        want = checksum(p["sym_us"], fc_mhz, fw["si"], fw["sq"], fw["span"], fw["pulse"], loop=fw["loop"])
        if got != want:
            raise RuntimeError(f"upload check failed: board sum {got}, expected {want}")
        return dict(n_stored=n, n_played=fw["n_play"], loop=list(fw["loop"]), checksum=want, norm=fw["norm"],
                    span=fw["span"], fc_mhz=fc_mhz, sym_us=p["sym_us"], firmware=self.ident)

    def play(self, p, delay_ms=AT.GO_DELAY_MS):
        """upload, start; returns (upload record, t_go = laptop time of the first symbol)"""
        up = self.upload(p)
        self.send(f"go {delay_ms}")
        t_send = time.time()
        r = self.readline(3.0)
        if not r or not r.startswith("# armed"):
            raise RuntimeError(f"go: {r}")
        return up, t_send + delay_ms / 1000.0

    def wait_done(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            r = self.readline(1.0)
            if r and r.startswith("# done"):
                return r
        return None


def main():
    import datetime
    import json
    import os
    ap = argparse.ArgumentParser(description="watch_psk_tx x2 bench driver (no worn-watch mode)")
    ap.add_argument("cmd", choices=("id", "off", "cw", "frame", "sweep"))
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--port")
    ap.add_argument("--mod", choices=sorted(P.MODS), default="dbpsk")
    ap.add_argument("--bps", type=float, default=16.0)
    ap.add_argument("--rates", default="16,32,64,100")
    ap.add_argument("--fcs", default="16")
    ap.add_argument("--shape", choices=("rect", "rc", "rrc"), default="rect")
    ap.add_argument("--fc", type=float, default=16.0)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--seg", type=float, default=10.0, help="sweep: seconds per segment")
    ap.add_argument("--label", default="bench")
    for k, v in AT.DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=None)
    a = ap.parse_args()
    cal = {k: (getattr(a, k) if getattr(a, k) is not None else v) for k, v in AT.DEFAULTS.items()}
    nominal = [k for k in AT.DEFAULTS if getattr(a, k) is None]
    t = PskTX(a.port)
    try:
        print(t.ident)
        if a.cmd == "id":
            return
        cfg = t.configure(cal["vfull"], cal["r1"], cal["r2"], cal["amp_mvpp"], cal["tx_ppm"])
        print(f"  {cal['amp_mvpp']:g} mVpp peak-to-peak at the divider output: CALCULATED, not measured"
              + (f" (nominal defaults: {', '.join(nominal)})" if nominal else ""))
        if a.cmd == "off":
            print(t.ask("x", r"^# off")); return
        if a.cmd == "cw":
            print(t.ask(f"cw {float(a.arg or a.fc):g} {int(a.seconds)}", r"^# cw"))
            print("  carrier on - 'python psk_tx.py off' to stop")
            return
        if a.cmd == "frame":
            segs = [(a.bps, a.fc)]
            seg_s = a.seconds
        else:
            segs = [(float(r), float(f)) for f in a.fcs.split(",") for r in a.rates.split(",")]
            seg_s = a.seg
        rec = dict(kind="benchsweep", label=a.label, firmware=t.ident, cal=cal, nominal=nominal, cfg=cfg,
                   watch_connected=False, segments=[])
        print(f"  {len(segs)} segment(s): start the PicoScope capture NOW (it must cover about "
              f"{len(segs) * (seg_s + 3.5) + 2:.0f} s); transmitting in 3 s")
        time.sleep(3)
        for bps, fc in segs:
            p = P.plan(a.mod, bps, fc=fc, shape=a.shape, seconds=seg_s)
            up, t_go = t.play(p)
            print(f"  {p['modulation']} {p['bps']:g} bps (= {p['baud']:g} baud), fc {p['fc_hz']:g} Hz "
                  f"({p['cycles_per_symbol']:.2f} cycles/symbol), {p['shape']}, {p['frames']} frames x {p['frame_s']:.2f} s, "
                  f"{up['n_played']} symbols ({up['n_stored']} stored), checksum ok")
            r = t.wait_done(p["total_s"] + 10)
            rec["segments"].append(dict(plan=p, upload=up, t_go=t_go, done=r))
            if not r:
                print("  ! no '# done' from the board"); break
            time.sleep(2.0)                                   # silence between segments (output at mid-scale)
        if a.cmd == "sweep":
            path = os.path.join(P_DATADIR(), f"benchsweep_{a.label}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
            json.dump(rec, open(path, "w"), indent=1, default=float)
            print(f"  stop the capture, save it as CSV, then:\n    python psk_bench.py sweep <capture.csv> --record {path} "
                  f"--zero <zero.csv>")
    finally:
        try:
            t.ask("x", r"^# off")
        except Exception:
            pass
        t.dev.close()


def P_DATADIR():
    import ecg_ber as B
    return B.DATADIR


if __name__ == "__main__":
    main()
