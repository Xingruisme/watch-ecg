#!/usr/bin/env python3
"""
Host driver for the Arduino UNO R4 transmitter (arduino/watch_ber_tx/watch_ber_tx.ino, firmware "watch_ber_tx v1"):
the R4's 12-bit DAC on A0 -> R_TOP -> crown node -> R_BOT -> GND (watch back). Used by ecg_ber.py --tx arduino and,
for the PicoScope bench checks (README section 12), directly:

   python arduino_tx.py id                                   # is the sketch there? prints its settings
   python arduino_tx.py hold max | min | mid                 # DAC at full scale / zero / mid (measure A0 to GND)
   python arduino_tx.py cw 16 --seconds 60 --vfull 4.98      # steady tone at the divider output
   python arduino_tx.py frame --mod ook --bps 2 --toneset exp --seconds 40    # BER frames, as ecg_ber.py sends them
   python arduino_tx.py sweep --seconds 40                   # the ecg_link sweep schedule (2-100 Hz, 1.25 s steps)
   python arduino_tx.py off
Every command takes the calibration: --vfull (V, = V(hold max) - V(hold min)), --r1 / --r2 (Ohm, measured),
--amp-mvpp (at the watch, calculated), --tx-ppm (board clock error from 'ard_bench.py tone'). The firmware keeps
them only until it is reset, so each command sends them again.

The output is at mid-scale (~1.17 mV DC at the watch with 1 MOhm / 470 Ohm) whenever nothing is playing; 'frame',
'sweep' and 'cw' turn it back to mid-scale on exit, error or Ctrl+C.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import signal
import sys
import time

import numpy as np

FW_ID = "watch_ber_tx v1"
ARDUINO_VID = 0x2341
UPDATE_US = 250                       # watch_ber_tx.ino UPDATE_US
MID = 2048
GO_DELAY_MS = 500
DEFAULTS = dict(vfull=5.0, r1=1.0e6, r2=470.0, amp_mvpp=2.0, tx_ppm=0.0)


# ================================================================ schedules (shared with ecg_ber.py and the bench tool)
def steps_of(plan):
    """plan (ecg_link.ber_plan / tone_plan, Arduino) -> (step_us, reps, [Hz per step]); 0 Hz = mid-scale"""
    import ecg_link as L
    if plan["kind"] == "ber":
        bits, _, _ = L.frame(plan["bps"])
        return plan["bit_us"], plan["frames"], [plan["f1_hz"] if b else (plan["f0_hz"] or 0.0) for b in bits]
    return int(round(plan["step_s"] * 1e6)), plan["frames"], list(plan["tones_hz"])


def checksum(step_us, reps, mhz):
    """watch_ber_tx.ino checksum(): uint32 arithmetic"""
    s = (step_us * 7 + reps * 13 + len(mhz)) & 0xFFFFFFFF
    for i, m in enumerate(mhz):
        s = (s + (i + 1) * m) & 0xFFFFFFFF
    return s


def model_output(step_us, reps, hz, t, amp_mvpp=2.0, vfull=5.0, r1=1e6, r2=470.0, board_ppm=0.0, clk_ppm=0.0,
                 t_start=0.0):
    """MODEL (not a measurement) of the voltage at the divider output [mV] at true times t [s] when the firmware
    plays the schedule: 4 kHz DAC updates (zero-order hold), 12-bit codes, phase restarted every step, mid-scale
    between steps and outside the run. board_ppm = the board's real clock error, clk_ppm = the correction set."""
    ratio = r2 / (r1 + r2)
    half = amp_mvpp / 1000 / ratio / 2 / vfull * 4095
    micros = (t - t_start) * 1e6 * (1 + board_ppm * 1e-6)           # what the board's micros() reads
    upd = np.floor(micros / UPDATE_US) * UPDATE_US                   # last DAC update
    el = upd / (1 + clk_ppm * 1e-6)                                  # firmware's corrected elapsed time [us]
    k = np.floor(el / step_us).astype(np.int64)
    f = np.asarray(hz, float)[np.clip(k, 0, None) % len(hz)]
    on = (k >= 0) & (k < reps * len(hz)) & (f > 0)
    ph = f * (el - k * step_us) * 1e-6
    code = np.where(on, MID + np.round(half * np.sin(2 * np.pi * (ph - np.floor(ph)))), MID)
    return code / 4095 * vfull * ratio * 1e3


# ================================================================ driver
def find_port():
    from serial.tools import list_ports
    ps = [p for p in list_ports.comports() if p.vid == ARDUINO_VID] or \
         [p for p in list_ports.comports() if "ACM" in p.device]
    if not ps:
        raise SystemExit("no Arduino serial port found (plug in the UNO R4; Arduino IDE Serial Monitor closed?)")
    return ps[0].device


class ArduinoTX:
    def __init__(self, port=None, timeout=1.0):
        import serial
        self.port = port or find_port()
        try:
            self.dev = serial.Serial(self.port, 115200, timeout=timeout)
        except serial.SerialException as e:
            raise SystemExit(f"{self.port}: {e}\n(close the Arduino IDE Serial Monitor - only one program can use the port)")
        time.sleep(1.5)
        self.dev.reset_input_buffer()
        self.ident = self.ask("id", r"^# watch_ber_tx", timeout=4.0)
        if FW_ID not in self.ident:
            raise SystemExit(f"{self.port}: unexpected firmware '{self.ident}' (need {FW_ID}: upload watch_ber_tx.ino)")
        self.cfg = {}

    def close(self):
        try:
            self.off()
        finally:
            self.dev.close()

    def send(self, line):
        self.dev.write((line + "\n").encode())
        self.dev.flush()

    def readline(self, timeout):
        end = time.time() + timeout
        buf = b""
        while time.time() < end:
            c = self.dev.readline()
            if c:
                buf += c
                if buf.endswith(b"\n"):
                    return buf.decode(errors="replace").strip()
        return buf.decode(errors="replace").strip() or None

    def ask(self, line, want, timeout=3.0):
        self.send(line)
        end = time.time() + timeout
        while time.time() < end:
            r = self.readline(max(0.05, end - time.time()))
            if r is None:
                break
            if r.startswith("# error"):
                raise RuntimeError(f"'{line[:60]}': {r}")
            if re.search(want, r):
                return r
        raise TimeoutError(f"no reply matching {want!r} to '{line[:60]}'")

    @staticmethod
    def parse_kv(r):
        tok = r.lstrip("# ").split()
        out = {}
        for k, v in zip(tok[1::2], tok[2::2]):
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
        return out

    def configure(self, vfull, r1, r2, amp_mvpp, tx_ppm=0.0):
        self.ask(f"cal {vfull:.5f} {r1:.1f} {r2:.4f}", r"^# cfg")
        self.ask(f"clk {tx_ppm:.1f}", r"^# cfg")
        r = self.ask(f"a {amp_mvpp:.5f}", r"^# cfg")
        self.cfg = dict(self.parse_kv(r), firmware=self.ident, port=self.port)
        return self.cfg

    def load(self, step_us, reps, hz):
        mhz = [int(round(f * 1000)) for f in hz]
        r = self.ask(f"load {int(step_us)} {int(reps)} " + " ".join(map(str, mhz)), r"^# loaded", timeout=5.0)
        got = self.parse_kv(r)
        want = checksum(int(step_us), int(reps), mhz)
        if int(got.get("n", -1)) != len(mhz) or int(got.get("sum", -1)) != want:
            raise RuntimeError(f"schedule upload check failed: {r} (expected n {len(mhz)} sum {want})")
        return dict(n=len(mhz), checksum=want, step_us=int(step_us), reps=int(reps), mhz=mhz)

    def go(self, delay_ms=GO_DELAY_MS):
        t_cmd = time.time()
        r = self.ask(f"go {int(delay_ms)}", r"^# armed in")
        return t_cmd + delay_ms / 1e3, r

    def status(self):
        self.dev.reset_input_buffer()
        r = self.ask("?", r"^# state")
        m = re.match(r"# state (\w+) events (\d+) of (\d+)", r)
        if not m:
            raise RuntimeError(f"unexpected status line: {r}")
        return dict(state=m.group(1), events=int(m.group(2)), total=int(m.group(3)), raw=r)

    def wait_done(self, timeout):
        end = time.time() + timeout
        while time.time() < end:
            r = self.readline(0.5)
            if r and r.startswith("# done"):
                return r
        return None

    def tone(self, hz, seconds=300):
        return self.ask(f"cw {hz:.4f} {int(seconds)}", r"^# cw ")

    def hold(self, which):
        return self.ask({"max": "h", "min": "l", "mid": "m"}[which], r"^# hold")

    def off(self):
        return self.ask("x", r"^# off")


# ================================================================ CLI (bench)
def plan_for(a):
    import ecg_ber as B
    if a.cmd == "frame":
        return B.ber_plan_for(a.mod, a.bps, a.toneset, a.seconds, transmitter="arduino")
    import ecg_link as L
    return B.arduino_plan(L.tone_plan(a.cmd, a.seconds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("id", "off", "hold", "cw", "frame", "sweep", "sweeplow"))
    ap.add_argument("arg", nargs="?", help="hold: max|min|mid; cw: frequency in Hz")
    ap.add_argument("--port")
    ap.add_argument("--vfull", type=float, default=DEFAULTS["vfull"], help="DAC swing V(hold max) - V(hold min), volts")
    ap.add_argument("--r1", type=float, default=DEFAULTS["r1"], help="series resistance R_TOP, Ohm (measured)")
    ap.add_argument("--r2", type=float, default=DEFAULTS["r2"], help="resistor across the watch R_BOT, Ohm (measured)")
    ap.add_argument("--amp-mvpp", type=float, default=DEFAULTS["amp_mvpp"], help="at the watch, calculated")
    ap.add_argument("--tx-ppm", type=float, default=DEFAULTS["tx_ppm"], help="board clock error (ard_bench.py tone)")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--mod", choices=("ook", "fsk"), default="ook"); ap.add_argument("--bps", type=float, default=2.0)
    ap.add_argument("--toneset", choices=("baseline", "exp"), default="baseline")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    tx = ArduinoTX(a.port)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    try:
        cfg = tx.configure(a.vfull, a.r1, a.r2, a.amp_mvpp, a.tx_ppm)
        print(f"{tx.ident} on {tx.port}")
        print(f"  CALCULATED at the divider output: {cfg['amp_mvpp']:.3f} mVpp, DC {cfg['dc_mv']:.3f} mV "
              f"(DAC {cfg['dac_vpp']:.3f} Vpp, divider 1/{(a.r1 + a.r2) / a.r2:.1f}, vfull {a.vfull:g} V, clk {a.tx_ppm:+g} ppm)")
        if a.cmd == "id":
            print(" ", tx.status()["raw"])
            return
        if a.cmd == "off":
            print(" ", tx.off()); return
        if a.cmd == "hold":
            print(" ", tx.hold(a.arg or "mid"), "- stays until 'python arduino_tx.py off'")
            tx.dev.close(); return
        if a.cmd == "cw":
            hz = float(a.arg or 16)
            print(" ", tx.tone(hz, a.seconds), "- Ctrl+C stops it")
            time.sleep(a.seconds + 0.2)
            return
        plan = plan_for(a)
        step_us, reps, hz = steps_of(plan)
        up = tx.load(step_us, reps, hz)
        what = (f"{plan['modulation'].upper()} {plan['bps']:g} bps, '1' {plan['f1_hz']:g} Hz, '0' "
                + (f"{plan['f0_hz']:g} Hz" if plan["f0_hz"] else "off") + f", tone set {plan['toneset']}"
                if plan["kind"] == "ber" else f"{plan['kind']} {plan['tones_hz']}")
        print(f"  {what}: {up['n']} steps x {reps} = {plan['total_s']:.1f} s (checksum {up['checksum']})")
        t0, _ = tx.go()
        print(f"  start at {time.strftime('%H:%M:%S', time.localtime(t0))} - Ctrl+C stops it")
        r = tx.wait_done(plan["total_s"] + 5)
        print(" ", r or "no '# done' reply", "|", tx.status()["raw"])
    except KeyboardInterrupt:
        print("stopped")
    finally:
        if a.cmd != "hold" and not tx.dev.closed:
            tx.close()


if __name__ == "__main__":
    main()
