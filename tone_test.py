#!/usr/bin/env python3
"""
Continuous test tone from the TeensyTRX (Teensy 4.0 + AD9833), in place of the function generator.

Uses the firmware's own 'cw <hz>' command (no reflash, no schedule): the AD9833 runs one frequency until stopped.
Ctrl+C (or the end of --seconds) always turns the output off.

   python tone_test.py                      # 16 Hz sine until Ctrl+C
   python tone_test.py --hz 10 --seconds 60 # 10 Hz sine for 60 s
   python tone_test.py --hz 16 --onoff 2    # 2 s on / 2 s off (easy to see on the ECG trace)
   python tone_test.py --off                # just turn the output off

The AD9833 frequency is quantised to 25 MHz / 2^28 = 0.093 Hz steps; the line printed by the Teensy is the real one.
Level at the electrodes = AD9833 VOUT (~0.6 Vpp) x the external attenuator (README section 3) - the script cannot set it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "maxtouch-screen-main"))
import stent_tx as STX                          # TeensyTRX host driver, unchanged

CHECK_SPS, CHECK_S = 10000, 1.0


def a0_check(t, hz):
    """stream A0 for CHECK_S and report the tone line (A0 = the board's electrode node, not necessarily VOUT)"""
    t.dev.reset_input_buffer()
    t.send(f"mono {CHECK_SPS}")
    need, buf, end = int(2 * CHECK_SPS * (CHECK_S + 0.1)), b"", time.time() + CHECK_S + 3
    while len(buf) < need and time.time() < end:
        buf += t.dev.read(1 << 14)
    t.send("x")
    time.sleep(0.3)
    t.dev.reset_input_buffer()
    a = np.frombuffer(buf[int(0.2 * CHECK_SPS):], np.uint8)          # skip the first ~0.1 s
    j = next((i for i in range(len(a) - 1) if a[i] & 0x80 and not a[i + 1] & 0x80), None)
    if j is None:
        print("  A0 check: no stream data"); return
    pr = a[j:j + (len(a) - j) // 2 * 2].reshape(-1, 2)
    ok = (pr[:, 0] & 0x80 != 0) & (pr[:, 1] & 0x80 == 0)
    v = ((((pr[:, 0] & 0x7F).astype(np.uint16) << 7) | pr[:, 1]) >> 2)[ok].astype(float)
    sp = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(len(v))))
    fr = np.fft.rfftfreq(len(v), 1 / CHECK_SPS)
    k = int(np.argmin(np.abs(fr - hz)))
    db = 20 * np.log10(sp[max(k - 2, 1):k + 3].max() / (np.median(sp[1:]) + 1e-12))
    print(f"  A0 check: {v.min():.0f}..{v.max():.0f} counts ({(v.max() - v.min()) * 3.3 / 4095:.3f} Vpp), "
          f"line at {hz:g} Hz {db:.1f} dB over the median"
          + ("" if db >= 20 else "  (weak/absent on A0 - A0 reads the electrode node; check VOUT with a scope)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hz", type=float, default=16.0)
    ap.add_argument("--wave", default="sine", choices=("sine", "square"))
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    ap.add_argument("--onoff", type=float, default=0, help="toggle on/off every this many seconds (0 = continuous)")
    ap.add_argument("--no-check", action="store_true", help="skip the A0 check at the start")
    ap.add_argument("--off", action="store_true", help="turn the output off and exit")
    ap.add_argument("--port")
    a = ap.parse_args()

    t = STX.StentTX(a.port)
    try:
        print(f"Teensy on {t.port}: {t.configure(wave=a.wave)['line'][:60]}")
        if a.off:
            print(t.off()); return
        r = t.tone(a.hz)
        print(r)
        if not a.no_check:
            a0_check(t, a.hz)
        print(("continuous" if not a.onoff else f"{a.onoff:g} s on / {a.onoff:g} s off")
              + (f" for {a.seconds:g} s" if a.seconds else " - Ctrl+C to stop"))
        t0, on, nxt = time.time(), True, a.onoff
        while not a.seconds or time.time() - t0 < a.seconds:
            el = time.time() - t0
            if a.onoff and el >= nxt:
                on = not on
                t.tone(a.hz) if on else t.off()
                nxt += a.onoff
            print(f"\r  T+{el:6.1f} s  {'ON ' if on else 'off'}", end="", flush=True)
            time.sleep(0.05)
        print()
    except KeyboardInterrupt:
        print()
    finally:
        t.close()                                # sends 'off' and confirms it
        print("output off")


if __name__ == "__main__":
    main()
