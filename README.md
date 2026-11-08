# Pixel Watch 2 ECG as a second receiver for the TeensyTRX link

Same transmitter, frames and BER methodology as the maXTouch experiments (`../maxtouch-screen-main/stent_modcomp.py`).
The receiver is the Pixel Watch 2 ECG (back electrode + crown), read out through the Fitbit ECG app's export.
**Status 2026-11-02:** offline pipeline and synthetic dry run done (`python ecg_ber.py synth` PASSED). No watch has been
connected and nothing has been transmitted. The decoder is **experimental, not frozen**.

## 0. Worn-watch BER run (the short path)

Same transmitter setup as the maXTouch through-body runs (operator's decision, 2026-11-02). The decoder is frozen:
`ecg_link.py` ecg-v1, sha `b422eaa49c6a`, 2026-11-02 16:34:17. `run` and `ingest` stop if it changes.

**One run per rate, OOK then FSK** (61–70 s of transmission each, whole frames):
```
cd ~/xingru/watch-ecg
~/.venvs/galvanic/bin/python ecg_ber.py run --bps 2 --label wear1
```
For each run:
1. Watch on the wrist, ECG app open on "hold your finger on the crown", finger OFF the crown.
2. Press Enter. The transmitter starts (T+0).
3. At **T+5 s** the script says NOW: put your finger on the crown. The 30 s ECG runs inside the transmission, which
   continues ≥ 25 s past its end.
4. Answer whether the recording completed.

Then export both readings as PDFs and run:
```
~/.venvs/galvanic/bin/python ecg_ber.py ingest data_watch_ecg/session_wear1_2bps_<time> ~/Downloads
```
`ingest` pairs each PDF with its run by the recording time. It skips other readings and duplicates, and prints per
modulation: BER (blind, 95 % CI), frames recovered, Es/N0, tone-line SNR, and unusable or dropped recordings.

**Next rate:** if 2 bps recovers its frames with low BER, repeat with `--bps 4`, then 8, then 16. Use `--repeats 3`
for more bits per cell. `python ecg_ber.py analyse wear1` pools every session of the label.

## 1. What is verified, what is not

| Item | Status | Evidence |
|---|---|---|
| Fitbit ECG PDF holds the full 250 Hz waveform as a vector polyline (not an image) | **verified** on one real Pixel Watch 2 export | WatchLink repo `pdf_parser/pixel_1_Hz.pdf` (Wear OS 4.0.0, feature 6-2.14.2-2.13-2.16): 7376 points, x step 0.5 units = 4 ms, y to 0.01 unit = 0.2 µV, 1 mV pulse = 50 units, rows continuous |
| Inconclusive readings (no heartbeat) are still saved and exportable | verified (same file: "Inconclusive", 120 bpm from a 1 Hz test signal) | same |
| Export passband ≈ 1 Hz high-pass (1st order), flat 3–25 Hz, −2 dB 31 Hz, −8 dB 41 Hz, −25 dB 51 Hz | **estimate** from the harmonics of WatchLink's test signal, assumes it was a square wave at the electrodes (its spectrum says so: odd harmonics only, 1/n + 3 dB) | `ecg_io.py` on the sample; rms misfit of the 1 Hz HP + 5th-order 31.5 Hz LP model = 3 dB |
| Watch sample clock is not exactly 250 Hz | estimate: +1227 ppm vs. a 1.00000 Hz source (or the generator was off) | same sample |
| Recording starts / runs with copper contacts and no person | **NOT verified** - WatchLink characterised both watches with copper tape + a function generator, but does not say whether anyone wore the watch | paper §4.1 |
| ECG app available in your country / on your account | **NOT verified** | check in the Fitbit app first |
| Google Health API v4 `electrocardiogram` (waveformSamples, millivoltsScalingFactor, samplingFrequencyHertz) | documented, **not tried** (needs a Google Cloud project + OAuth; reports say new projects are not onboarded at the moment) | Google Health API reference |
| Fitbit Web API ECG endpoint | **gone**: switched off 2026-10-30 | Google developer newsletter |
| Google Takeout `Afib ECG Readings.csv` with a `waveform_samples` column | community report only, **unverified** | Fitbit community forum |
| What the watch does with tones above Nyquist (125 Hz), e.g. the 159.585 kHz maXTouch carrier | **unknown** - rejected or aliased; measured by the alias probe | - |
| How the TeensyTRX board's output chain (Ra/Rb current network, coupling, TX/RX switch) passes 1–30 Hz | **unknown** (wiring artifact not readable from here) | measured by `--monitor` |

**Data access chosen:** the PDF export (Fitbit app → Today/Discover → Heart rhythm assessment → reading → *Export a PDF
for your doctor*). Per recording, no API keys, and it carries the exact samples. `ecg_io.py` also reads Google Health
API JSON, legacy Fitbit Web API JSON and Takeout-style CSV, but those layouts are untested on real files. No root, ADB
or undocumented raw-sensor access is assumed (WatchLink: no real-time ECG API on Pixel Watch).

## 2. Frequencies and bit rates

Three frequencies are involved, and they differ:

1. **Generated:** the AD9833 word frequency, quantised to 0.093 Hz (16 Hz → 16.019 Hz).
2. **Transmitted line:** TeensyTRX restarts the AD9833 phase at every schedule step (= every bit). With an integer
   number of cycles per bit, the waveform repeats every bit, so the spectral line sits at exactly (cycles × bps) on
   the Teensy crystal: 16.000 Hz.
3. **Observed in the export:** the line ÷ (1 + watch clock error), e.g. 15.98 Hz in nominal-250 Hz time at
   +1227 ppm. The decoder measures this offset from the tone lines and corrects for it. v4's ±200 ppm search alone
   would leave ~37 ms of drift over 30 s.

Anything above ~45 Hz arrives > 15 dB down. Tones above 125 Hz (including 159.585 kHz) are either rejected or aliased;
the `alias` probe measures which. Use **sine mode only**. In square mode, OOK '0' = 0 V and '1' = 0/3.3 V, so every
edge is a 1.65 V DC step, and the odd harmonics land inside the band and above Nyquist. In sine mode, '0' is word 0 at
DAC mid-scale, which equals the sine's mean, so there is no step.

**Initial plan** (`python ecg_ber.py plan`):

| Rate | OOK tone | FSK '1' / '0' | Cycles per bit | Frame | Payload bits per 29.5 s recording |
|---|---|---|---|---|---|
| 2 bps | 16 Hz | 16 / 8 Hz | 8 / 4 | 28 bits = 14 s (PRBS4) | 15–30 |
| 4 bps | 16 Hz | 16 / 8 Hz | 4 / 2 | 44 bits = 11 s (PRBS5) | 31–62 |
| 8 bps | 16 Hz | 16 / 8 Hz | 2 / 1 | 76 bits = 9.5 s (PRBS6) | 63–189 |
| 16 bps | 16 Hz | 16 / 32 Hz | 1 / 2 | 140 bits = 8.75 s (PRBS7) | 254–381 |

Why these tones:
- Inside the estimated flat band (3–25 Hz).
- Above the 1 Hz high-pass and baseline wander.
- Clear of 50/60 Hz.
- Integer cycles per bit at every rate, so the waveform is phase-continuous.
- The two FSK tones are a multiple of the bit rate apart (orthogonal).

16 bps needs a tone ≥ 32 Hz for two cycles per bit, and 32 Hz is already at the band edge. So **16 bps is the
expected practical ceiling**. Start at 2 bps and step up (§4, phase 3). **Re-pick the tones after the passband sweep
on your watch.**

Why a new frame: the v4 frame (170 bits) lasts 21–85 s at these rates and would not fit in a 30 s recording.

Amplitude: 1 mVpp at the electrodes to start. That's the calibration-pulse size, and inside WatchLink's window (≥ 0.1 mV
needed; the Pixel stopped showing data above 5 mV). The ladder is 0.2 / 0.5 / 1 / 2 mVpp.

## 3. Bench setup (no person anywhere in the circuit)

```
   laptop (on battery, charger unplugged)
     │ USB
   [USB isolator ≥ 2.5 kV]  (or no isolator only if the laptop is on battery and touches nothing earthed)
     │ USB
   TeensyTRX (Teensy 4.0 + AD9833, original firmware, 'wave sine')
     │ AD9833 VOUT (0.04–0.65 V sine, ~0.34 V DC)  ── or the E+ node if --monitor shows the tones there
     │
     ├── C1 10 µF film (non-polar) ── R1a 160 kΩ ── R1b 160 kΩ ──┬──── W+ ── copper tape ── CROWN (side electrode)
     │                                                             │
     │                                                            R2 560 Ω   (0.2 / 0.5 / 1 / 2 mVpp:
     │                                                             │          110 / 270 / 560 / 1.1 kΩ)
   TeensyTRX GND ──────────────────────────────────────────────────┴──── W− ── copper tape ── BACK electrode
                                                                   Pixel Watch 2 lying on a non-conductive
   A0 (TeensyTRX ADC, = E+ node) → `--monitor` stream, 2 kS/s     stand, off its charger, not on a wrist
```

Orientation (crown = signal, back = ground) is WatchLink's. The maXTouch stent / ankle electrodes and the
current-setting network are **disconnected** from the output. At these frequencies their ~1 mA is about 100× the
10 µA a low-frequency patient circuit may carry (IEC 60601-1 patient auxiliary current, normal condition; the lab
must confirm the applicable limit). The 160 kHz allowance does not apply at 16 Hz.

`python ecg_ber.py divider --vout-vpp <measured> --r1 320e3 --r2 560` prints the level at the watch and the fault
currents (attenuator values above, 0.6 Vpp):
- **Level at the watch:** 1.05 mVpp.
- **Normal current:** 0.94 µA peak.
- **VOUT stuck at the 3.3 V rail with C1 shorted:** 10.3 µA.
- **One R1 also shorted:** 20.6 µA.

The capacitor, the two series resistors and the isolator are there to fail safe. None of it is an approval for human
contact.

**Checks before the first transmission** (`tx` asks for the first four, and won't transmit without a 'y' to each):
1. Nobody touches the watch, its electrodes, the leads or the attenuator while the transmitter runs. Start the ECG app
   by tapping the glass with a stylus, never the crown or the case.
2. The attenuator is in place: R1 ≥ 320 kΩ as two resistors, measured with a DMM, plus C1.
3. The TeensyTRX is isolated from mains earth.
4. The watch is off its charger.
5. Scope on VOUT: a ~0.6 Vpp sine at the scheduled frequency, no DC step at OOK edges. Use `python ../maxtouch-screen-main/stent_tx.py tone 16 --wave sine`, then `... off`.
6. `--monitor` decodes the transmitter's own A0 with the same decoder. 0 errors there means the bits, timing and tones
   leave the transmitter correctly, independent of the watch. If A0 shows no 16 Hz tone, the board's E+ chain blocks
   low frequencies: tap VOUT directly as drawn.

## 4. Protocol

**Phase 0 – access (no transmitter connected)**
- Check that the ECG app works on this account in this country.
- Export any ECG PDF and run `python ecg_io.py <file>.pdf`.
- The reader must print `cal pulse 50.0 units`, `x step 0.5`, `gaps none`. If your app version uses another layout,
  the reader stops instead of guessing; send me the file.

**Phase 1 – app gate (transmitter physically disconnected; R2 alone across W+/W−)**
- Tape the contacts, open the ECG app, and see whether a 30 s recording runs and saves.
  - Runs and saves: go on.
  - Refused (e.g. "watch not on wrist"): try a saline-soaked sponge or a gel pad on the back electrode, still with no person.
  - Still refused: **stop**. A protocol with a person in the loop needs the lab's approval first. Do not improvise.
- Record the outcome as the `tx` operator note, or in your lab book.

**Phase 2 – channel (1 mVpp)**
```
python ecg_ber.py tx --kind sweep    --label bench1 --amp-mvpp 1.0 --r1 320e3 --r2 560 --monitor
python ecg_ber.py tx --kind sweeplow --label bench1 --amp-mvpp 1.0 --r1 320e3 --r2 560 --monitor
python ecg_ber.py tx --kind alias    --label bench1 --amp-mvpp 1.0 --r1 320e3 --r2 560 --monitor
```
For each run:
1. Press Enter at the prompt.
2. Start the ECG recording while "TRANSMITTING" is shown (90 s window).
3. Export the PDF.
4. Run `python ecg_ber.py import <pdf>`. It pairs the PDF with the run by the PDF's "Recorded on" minute; add
   `--run <folder>` if two runs share that minute.

`python ecg_ber.py analyse bench1` prints and plots:
- the passband (level, SNR and frequency shift per tone);
- the watch clock;
- where each out-of-band tone shows up.

Then pick the BER tones and the amplitude: re-run the sweep at 0.2 and 2 mVpp by swapping R2.

**Phase 3 – BER ladder (OOK and FSK, interleaved at each rate)**
```
python ecg_ber.py tx --kind ber --mod ook --bps 2 --label bench1 --amp-mvpp 1.0 --r1 320e3 --r2 560 --monitor
python ecg_ber.py tx --kind ber --mod fsk --bps 2 --label bench1 --amp-mvpp 1.0 --r1 320e3 --r2 560 --monitor
... then 4, 8, 16 bps
```
- **Recordings per cell:** about 3 at 2–4 bps (a link check: 15–60 bits each), about 5 at 8 bps and 3–4 at 16 bps
  (≈ 1000 payload bits).
- **When to step up:** go to the next rate once the current one has 100 % frames and < 1 % BER over ≥ 3 usable
  recordings.
- **Before the first BER session:** freeze the decoder (set `FROZEN_AT` in `ecg_link.py`). Every BER recording after
  that is validation data.

**Phase 4 – compare receivers**
```
python ecg_ber.py compare MAXTOUCH=../maxtouch-screen-main/data_maxtouch/modcomp_<...> ECG=data_watch_ecg/ecgber_<...> --label both
```
Blind BER on both sides. The ECG has no known-timing variant: the watch clock is unrelated to the laptop's, and the
PDF only gives the minute.

## 5. Pipeline

`tx` (`tx.json`, `monitor.npz`) → watch → PDF → `import` → ecg.npz → `analyse` →:
- per-run `result_<decoder>_<sha>.json`, which is never overwritten;
- a new `ecgber_<label>_<time>/` folder holding `per_run.csv`, `summary.csv/json`, `ber.png` and per-run figures.

Decoder (`ecg_link.py`, rules in its docstring):

| Step | Same as maXTouch v4? |
|---|---|
| Frame: preamble + Barker SFD + PRBS payload + zeros | same family, shorter (11 + PRBS + 2) |
| Clock estimate from the tone lines (bit-free) | new: needed for the watch clock |
| Timing feature: tone power over a half-bit window | new: tones replace node energy |
| Timing search: τ over one frame, then ±6 ms × ±200 ppm, header correlation | v4 rule 3, but **mean** over frames, not sum (a 30 s window holds 2 or 3 frames; the sum chose a 20-bit false lock in the dry run) |
| Bit statistic: tone energy of the guarded bit samples, guard max(2 ms, 10 % Tb) | analogous to v4 rule 4 |
| Threshold halfway between the frame's known 1s and 0s | v4 rule 5 |
| Frame accepted when header r ≥ 0.3 | v4 rule 6 without its truth-based 'BER < 0.35' (reported separately as `v4acc_*`) |
| Accounting: frames_summary, error types, conservative BER, Clopper-Pearson | decode_v2 / decode_v3 / stent_modcomp, imported or copied unchanged |

`selftest` confirms that the search code, run with v4's header and `reduce='sum'`, reproduces `decode_v4.search` and
`decode_v4.header_score` exactly, and that PRBS7 = `stent_frames.prbs7`.

Recordings are judged usable or not before decoding, without the bits. A recording is **unusable** if any of these
holds, and it's listed with its reason:
- shorter than 25 s;
- tone line < 6 dB over the same lines 2.5 Hz away;
- > 10 % of samples in flat runs (≥ 0.1 s) or beyond ±4.9 mV.

Runs with no ECG imported are listed as dropped, with your note.

## 6. Dry run (synthetic, `python ecg_ber.py synth`)

What the dry run covers:
- **Channel model:** the firmware's exact schedule (word quantisation, phase restart per bit), the 1 Hz HP + 31.5 Hz LP
  watch model, +1227 ppm clock, white noise 10 µV, 5 µV of 50 Hz, 30 µV baseline wander, 0.2 µV quantisation.
- **Path:** written as Fitbit-style PDFs and read back through the same PDF reader.
- **Checks, all passing:**
  - 1 mVpp: 0 errors and 100 % frames for OOK and FSK at 2–16 bps.
  - Clock recovered within 9 ppm (median).
  - The no-signal and 6 s dropout recordings are flagged unusable; the dropped run is listed.
  - The sweep recovers the model response within 0.2 dB.

Reduced amplitudes give the expected BER-vs-Es/N0 behaviour (`synth_ber_vs_esn0.png`). These numbers describe the
pipeline, not the watch.

## 7. Files
- `ecg_io.py`: readers (PDF vector extraction, Google Health / Fitbit JSON, CSV).
- `ecg_link.py`: frames, TeensyTRX schedules, decoder.
- `ecg_synth.py`: synthetic channel and Fitbit-style PDF writer.
- `ecg_ber.py`: CLI (plan, divider, selftest, tx, run, ingest, import, analyse, synth, compare).
- `tone_test.py`: one continuous tone from the Teensy (`cw`), for checking the wiring against the function generator.
- `att_check.py`: attenuator sizing (`design`, calculated) and PicoScope / watch-export verification (`check`, measured); section 11.

## 8. Voltage report (added 2026-11-03)

Every `run` / `tx` transmission reports its voltages:
- **Where:** in the terminal, in `tx.json` and `session.json` (key `voltage`), and in `result_*.json`, `per_run.csv`, `summary.csv`/`.json` and the printed summary.
- **Older runs:** they get the same report, rebuilt at analysis time from `tx.json` and `monitor.npz`. Their files are not rewritten.
- **Decoder:** the report has no effect on decoding or BER. On the dry run, all 243 runs gave the same results in every existing column as before.

| Item | Where it comes from | Status label |
|---|---|---|
| Carrier | The plan: '1' and '0' frequencies, plus the AD9833's quantised frequency. | configured |
| Watch input (crown − back) | `--watch-vpp-mv` / `--watch-dc-mv` from an external instrument. | **measured** |
| | `--r1 --r2` (optionally `--source-vpp --source-measured --source-dc-mv --dc-block`): divider arithmetic. | **calculated**, not measured |
| | None of the above. | **unavailable**, no value invented |
| A0 | The firmware's own ADC stream. | measured, but **not the watch input** |

**What A0 is.**
- The firmware streams pin 14, the receive input. Per the TeensyTRX notes it is the board's own electrode node E+, read through the board's RX path.
- It is single-ended to Teensy GND and has 12 bits on 3.3 V (0.81 mV per LSB).
- The attenuator and the watch sit after it, so A0 cannot show the voltage the watch receives.
  - At E+, A0 is about 1000 times too coarse to resolve 1 mVpp.
  - Even if moved to the crown node, A0 would be at the edge of what it can resolve.
  - Moving it would also put the board's A0 input network (about 1.94 V DC on the `gen1` run) onto the watch electrode.
- The report fits the tone in each scheduled step, aligned by the firmware's `t0`. It gives each tone's Vpp, a noise floor and "present" when the tone is at least 3 times the floor.
- Its DC value is that of A0's input network. It is not shown to be the DC at E+ or at the watch.

**Finding from the existing `gen1` 5 bps OOK run (2026-11-03 10:40).**
- A0 showed no 16 Hz tone: 1.13 mVpp fitted against a 0.76 mVpp floor, line 4.7 dB, against about 1 Vpp expected at E+.
- The DC was 1936 mV with the output on and with it off.
- So in that run the tone did not reach the E+ node, or A0 does not see it. This is a property of the board's output and monitor path, independent of the watch.

**Minimum extra setup to measure the watch input directly.**
- **Scope, recommended:**
  - What: one channel with a ×1 probe (a ×10 probe makes 1 mV into 0.1 mV), tip on the crown node (the R1/R2 junction), ground clip on the back-electrode lead.
  - Scope settings: 1 or 2 mV/div, 20 MHz bandwidth limit, averaging 16–64, triggered on the tone, or use the scope's Vpp measurement over a long sweep.
  - Laptop on battery, so the scope earth and USB ground don't form a loop.
  - Measure while `tone_test.py --hz 16` runs, with the watch connected as in the BER run (the electrodes and wiring as used). Then pass `--watch-vpp-mv <value> --watch-dc-mv <value> --watch-instrument "scope ..."` to `run`.
- **DC offset:** a DMM on mV DC across crown–back during the tone. That is enough; 0.1 mV resolution is typical.
- **Without measuring 1 mV directly:**
  - Measure the attenuator input (about 0.6 Vpp at VOUT, easy on any scope) and R1 and R2 with a DMM.
  - Pass `--r1 --r2 --source-vpp <measured> --source-measured`.
  - The report then says "calculated from measured source and resistors". It still is not a measurement at the watch.
- **Automatic measurement (not built):**
  - An instrumentation amplifier across crown–back (e.g. INA333, gain about 1000, low bias current, output referenced to 1.65 V) into a spare Teensy ADC pin.
  - It also needs a firmware change so `mono` streams that pin. Today the firmware streams only A0.
- Reused unchanged from `../maxtouch-screen-main`: `stent_tx.py` (TeensyTRX driver), `stent_frames.py`, `decode_v2.py`, `decode_v3.py`, `decode_v4.py` (self-test only).

Run everything with `~/.venvs/galvanic/bin/python` (or `source ~/.venvs/galvanic/bin/activate` first). Plain `python3` lacks hidapi and scipy.

Sources:
- WatchLink paper (UIST '24): https://ubicomplab.cs.washington.edu/pdfs/watchlink.pdf
- WatchLink repository: https://github.com/ubicomplab/WatchLink
- Google Health API, Electrocardiogram type: https://googleapis.dev/dotnet/Google.Apis.GoogleHealthAPI.v4/latest/api/Google.Apis.GoogleHealthAPI.v4.Data.Electrocardiogram.html
- Legacy Fitbit ECG endpoint: https://dev.fitbit.com/build/reference/web-api/electrocardiogram/get-ecg-log-list/
- Fitbit Web API shutdown: https://developers.google.com/health/newsletters

## 9. Hardware isolation: Teensy not seen, function generator seen (2026-11-04)

Do this on the bench only: the watch is off the wrist and nobody touches the watch, the leads or the attenuator. The ECG decoder is not involved.

**What is already known.**
- **2026-10-22:** `cw` square at 20–400 kHz gave about 2 Vpp on A0. So the AD9833, the E+ drive and A0 all worked at high frequency.
- **2026-11-03, `gen1` 5 bps OOK, schedule mode:**
  - A0 stayed at 1936 mV DC with the output on and off.
  - It showed no 16 Hz tone: 1.13 mVpp against a 0.76 mVpp floor.
- **What this fits:** a high-pass, probably a coupling capacitor, somewhere between the AD9833 and A0.
- **What it doesn't settle:**
  - A0 can't tell whether the high-pass is in the E+ drive (then the watch got nothing) or only in A0's own input network (then the watch may have got the tone).
  - That run used the schedule, not `cw`.

**Step 1: automatic A0 test.** Takes about 30 s. Run `~/.venvs/galvanic/bin/python bench_isolate.py --label b1`.
- **Output off:** the A0 DC level and noise.
- **Sine ladder, `cw` at 16, 50 and 160 Hz up to 160 kHz:** the A0 tone Vpp at each frequency, relative to 1.6–50 kHz. The program estimates a first-order corner from it.
- **16 Hz square:** the folded step response at A0.
  - Flat tops mean no high-pass above about 0.3 Hz.
  - Decaying tops give τ and the corner, fc = 1/(2πτ).
  - "Spikes only" means a corner above about 20 Hz.
- **Validation:** on a simulated board the program recovered a 300 Hz corner from the ladder (16 Hz at 5.3 %, predicted 5.3 %), and 0.5, 3 and 10 Hz corners from the square.

| Step 1 says | Meaning | Next |
|---|---|---|
| No tone even at 1.6–50 kHz | Nothing reaches A0: dead output or A0 not connected. Not a low-frequency effect. | Step 2 (VOUT) |
| 16 Hz passes (≥ 70 % of the reference) | The AD9833 and the path up to A0 pass 16 Hz. | If `gen1` still showed nothing: schedule mode vs `cw`. Otherwise: the attenuator, leads or contacts (step 4). |
| High frequencies pass, 16 Hz is lost | A high-pass before A0. The AD9833 is working, since it uses the same DAC at every frequency. | Steps 3–4 decide whether the watch gets anything |

**Steps 2–6: scope.** Run `bench_isolate.py --label b1 --scope`. The script holds each tone while you read the scope, and saves every reading as `measured`, or `not measured` if you skip it.
- **Scope setup:** DC coupling, 20 MHz bandwidth limit, scope ground on Teensy GND (the back lead), laptop on battery.

| Step | Point | Tone | Probe / range | Expected if healthy |
|---|---|---|---|---|
| 2 | AD9833 VOUT pin | 16 Hz sine | ×10, 100 mV/div | about 600 mVpp around about +300 mV |
| 3 | E+ output wire | 16 Hz, then 160 kHz | ×10 | the same at both frequencies, unless the drive is AC-coupled |
| 4 | Attenuator output (crown node, across R2) | 16 Hz | ×1, 1–2 mV/div, averaging 16–64 | about 1 mVpp if fed from VOUT with 30 kΩ / 51 Ω |
| 5 | Function generator at the same crown–back point | its working setting | same as step 4 | the reference level that the watch is known to see |
| 6 | Multimeter, mV DC, crown–back | 16 Hz running | — | the DC offset at the watch |

**How to read the results.**
- **VOUT is about 600 mVpp:** the AD9833 produces 16 Hz.
- **E+ is below 10 % of VOUT at 16 Hz but normal at 160 kHz:** the E+ current drive blocks low frequencies. Feed the attenuator from VOUT, not E+.
- **E+ carries 16 Hz but A0 doesn't:** this is an A0 monitoring limitation, and the signal was transmitted.
- **Step 4 against step 5:** this compares the Teensy's level with the generator's. It is the only measured watch-input number.
- **Readings you skip** stay "not measured". The divider value from `ecg_ber.py` is a calculation and is never shown as a measurement.

Results go to `data_watch_ecg/bench_<label>_<time>/`: `bench.json` and the raw A0 captures in `a0_captures.npz`. They are kept apart from the BER sessions.

## 10. Known low-SNR failures of the frozen decoder ecg-v1 (documented 2026-11-04, decoder unchanged)

**Found by:** `ecg_ber.py synth --n 2 --amps 1,0.05` (seed 20261102) on 2026-11-03. Both symptoms are on one recording, `ecg_synth_impaired_ber_ook_4bps_00`. It is a **no-signal** recording made with the dry-run noise model: white 10 µV, mains 5 µV, wander 30 µV.

**F1: clock lock on noise.**
- **Symptom:** the clock search reports +688 ppm (the true clock is +1227) with a line SNR of 11.2 dB. The recording passes the usability rule (6 dB or more) although no tone was sent.
- **Cause, in `clock_delta`:**
  - The search takes the strongest of 501 candidate clocks (±5000 ppm in 20 ppm steps). At 16 Hz over 29.5 s that is about 5 independent noise bins.
  - The SNR divides that maximum by a single bin 2.5 Hz away.
  - The maximum of several noise bins over one noise bin often clears 6 dB, so the rule is biased upward and noisy.

**F2: false frame acceptance.**
- **Symptom:** one frame passes the truth-free acceptance rule (header r = 0.41, threshold 0.3) and adds 17 errors in 31 bits to the pooled BER.
- **Cause:** with an 11-bit header, the correlation of noise with the header pattern has a spread of about 0.3. The timing search also picks the τ/ε that maximises the mean header score, which pushes it higher still.

**How often, on noise-only recordings.** `tests/false_alarm_scan.py --n 100`, seed 20261104, exploratory, output in `data_watch_ecg/exploratory_falsealarm_20261104_113518/`:

| | 2 bps | 4 bps | 8 bps | 16 bps |
|---|---|---|---|---|
| OOK: usable / with ≥ 1 accepted frame | 45 / 13 | 37 / 15 | 38 / 20 | 41 / 26 |
| FSK: usable / with ≥ 1 accepted frame | 29 / 11 | 24 / 11 | 17 / 5 | 28 / 16 |

The BER of the accepted false frames is 0.46–0.54. So a false frame doesn't make a BER look good, but a recording with no tone can show up as "usable, BER ≈ 50 %" instead of "no tone".

**What this means.**
- **The original dry run (`synth_dry1`)** flagged its no-signal recording because that draw happened to fall below 6 dB. Its results remain valid as recorded.
- **Real recordings:** "usable" and "frames recovered" are not proof that a tone was received. Before reading a BER, check the line SNR, Es/N0 (None means no tone above the off level), and, after the hardware fix, the measured voltage at the watch.

**Regression tests.** Run `~/.venvs/galvanic/bin/python -m unittest discover -s tests -v` (about 25 s, read-only). It currently gives 11 tests, OK, with 5 expected failures.
- `tests/fixtures/nosignal_ook4.{npz,json}` holds the failing recording. `tests/make_fixture.py` rebuilds it from its seed through the PDF path; it was verified identical to the recording that failed.
- `TestBenchmarkUnchanged` checks:
  - that `ecg_link.py` is the frozen sha `b422eaa49c6a`;
  - that the saved `synth_dry1` `per_run.csv`, `summary.csv` and `summary.json` match their pinned sha256;
  - that the fixture reproduces from its seed.
- `TestCurrentBehaviour` pins today's F1/F2 values, plus a seeded batch of 20 noise-only OOK 4 bps recordings: 7 usable, 5 with false frames, 8 false frames.
- `TestKnownFailures` holds the desired behaviour as `expectedFailure`: the recording is unusable, no frame is accepted, and the clock is reported only when recovered.
  - A future decoder that fixes F1/F2 turns these into unexpected successes, which fails the run.
  - That decoder must be a new version. ecg-v1 and its results stay as they are.

## 11. Low-level output for the fixed 5 Hz watch test (2026-11-04)

**Finding (user's bench test, 2026-11-04).**
- Function generator, 5 Hz, wired straight to the watch: 2, 5 and 10 mVpp record; 20 mVpp and above the ECG recording fails.
- Teensy output as previously wired: 892.2 mVpp, 385.1 mV DC mean, 5.013 Hz (PicoScope). The 1 / 2.5 / 4 mA configurations were also far above 20 mVpp.
- So the earlier Teensy-to-watch tests were roughly 450× over the watch's input window. This matches section 2 (WatchLink saw no data above 5 mV).

**Hardware change (firmware unchanged).** Use the section 3 attenuator with R2 sized for 2 mVpp:

```
AD9833 VOUT ── C1 10 µF film ── R1a 160 kΩ ── R1b 160 kΩ ──┬── W+ ── crown
                                                           R2 715 Ω 1 %  (or 680 Ω + 39 Ω in series)
TeensyTRX GND ─────────────────────────────────────────────┴── W− ── back
```

- `python att_check.py design --source-vpp 892.2 --source-dc 385.1` (calculated, not measured):
  - 2 mVpp: R2 = 715 Ω (1.988 mVpp) or 680 + 39 Ω (1.999 mVpp);
  - 5 mVpp: 1.8 kΩ; 10 mVpp: 3.6 kΩ.
  - Output resistance ≈ R2 (≈ 713 Ω; the generator's is 50 Ω).
  - DC: C1 blocks the 385 mV; without C1 the output would sit at +0.86 mV.
  - C1 corner 0.05 Hz (no loss at 5 Hz), but it takes about 16 s to settle after power-up or reconnecting.
- Feed the attenuator from **VOUT** (a voltage source) and disconnect the 1 / 2.5 / 4 mA current network. From a current-driven node, the level would depend on the load, not on the divider.
- If the 892.2 mVpp was not measured at VOUT, measure VOUT, then rerun `design` with that value. A scope's max−min reading includes noise. `att_check.py check` on the VOUT capture gives the tone's own Vpp (it will flag the 2 mVpp target, which you can ignore for this one).
- Tone: `python tone_test.py --hz 5 --no-check`. The AD9833 gives 5.029 Hz (word 54); Ctrl+C turns it off.

**Alternative: the TeensyTRX board's own transmit network, with low-value parts only** (`trx_wiring`: AD9833 OUT → Ra → tap (Rb to GND) → 10 µF → E+). Revised 2026-11-04 for parts ≤ 2 kΩ.
- **Ra = 2 kΩ** (replaces Ra 1 kΩ). **Rb = 4.5 Ω** = four ~1 Ω in series plus two ~1 Ω in parallel (replaces Rb 1 kΩ).
- **Take the watch from the tap, not from E+.** The crown lead goes to the top of Rb, and the back lead to Rb's ground leg itself.
  - E+ sits behind the 10 µF, so the crown would have no DC path to the back: any DC bias or lead-off current from the watch would charge the 10 µF.
  - Disconnect E+ from all electrodes. The 10 µF and the receive front-end can stay on the board; they load the tap by < 0.01 %.
- **Calculated level and loads** (full network model: receive branch, 1 MΩ scope, ≥ 1 MΩ watch assumed):
  - level: 2.003 mVpp from 892.2 mVpp at the pin; each 50 mΩ of contact resistance in Rb adds +1 %;
  - Rout 4.5 Ω (generator: 50 Ω);
  - AD9833 load 2004 Ω, the same as the original 1 k + 1 k, so the pin voltage and current are unchanged (0.22 mA AC + 0.19 mA DC).
- **DC at the watch:** +0.86 mV (the pin's 385 mV × Rb / (Ra + Rb)). The 10 µF can't block it here: in front of Ra it would make a 7 Hz high-pass.
  - This is far inside the ±300 mV electrode offset ECG front-ends are built for, but it is not 0 like the generator.
  - Run `check` with `--dc-tol-mv 1.5`.
  - To remove it: a ≥ 100 µF capacitor between the AD9833 OUT and Ra (+ toward the AD9833) gives a 0.7 Hz corner.
- **Rb depends on the real tone at the pin.** A scope's max−min can include noise, and a bare AD9833 can't swing 892 mVpp around 385 mV DC.
  - Measure the pin first (procedure step 3), then Rb = 2000 × 2 / (V_pin − 2) Ω.
  - V_pin 600 mVpp → 6.7 Ω; 700 → 5.7; 800 → 5.0; 892 → 4.5; 1000 → 4.0.
- **Ground (Kelvin) connection:** the Teensy's supply current in the GND rail can drop millivolts. Clip the back lead and the scope ground on Rb's ground leg, and run a separate wire from that leg to the GND rail.
- **Not for worn tests.** If Rb opens, the watch (or a person) gets about 0.86 Vpp from a 2 kΩ source, up to about 0.4 mA. A worn test needs the series-resistor design above (≥ 2 × 160 kΩ, which needs high-value resistors).
- **Check command:** `--r1 2000 --r2 4.5 --rs 0`. A 1 Ω load step across Rb reads about 0.18 and confirms Rb is connected and about 4.5 Ω.
- For the galvanic link afterwards, put Ra and Rb back to 1 kΩ.

**Verification before the watch is connected.** The test load is the scope's 1 MΩ input. The watch's input impedance is not known. Any input ≥ 1 MΩ changes a 713 Ω source by < 0.1 %. The 1 kΩ load step (capture 3) measures the output resistance to prove the attenuator is wired as designed.

- **Scope:** channel A, ×1 probe or BNC-to-clip lead, **DC coupling**, smallest range (±10 / ±20 / ±50 mV by model), bandwidth limit on, resolution enhancement ≥ 2 bits (or 12-16-bit mode), 1 s/div (10 s), ≥ 1 kS/s. Export each capture as CSV. Don't read Vpp off the scope at 2 mV: its max−min is mostly noise.
- **Grounding** (as it will be for the watch):
  - laptop on battery; Teensy and PicoScope on that laptop;
  - ground clip on W− (attenuator bottom = Teensy GND), tip on W+, at the ends of the leads that will go to the watch;
  - DMM: W− to the Teensy GND pin ≈ 0 Ω, and W− connected to nothing else.
- **Captures**, 10 s each, with the tone on and at least 20 s after connecting:
  1. `teensy.csv`: tip on W+, clip on W−.
  2. `zero.csv`: tip **and** clip on W− (scope offset; a tone here means the ground lead carries signal).
  3. `teensy_1k.csv`: as 1, with a DMM-measured 1 kΩ across W+/W−.
  4. `gen_2mV.csv`: the function generator in place of the attenuator, on the same leads, set exactly as in the successful watch test. This is the measured target. If the generator's load setting is "50 Ω", a high-impedance load gets twice the displayed amplitude; this capture shows what the watch actually got.
- `python att_check.py check --label att1 teensy.csv --zero zero.csv --ref gen_2mV.csv --loaded teensy_1k.csv --rl 1000 --r2 715`

| Check | Limit | Required |
|---|---|---|
| frequency | 5 Hz ± 1 % | yes |
| amplitude | generator capture ± 10 % (2.000 mVpp without `--ref`) | yes |
| DC offset (zero-corrected) | generator's DC ± 0.2 mV (0 without `--ref`) | yes |
| no clipping | 0 overrange samples | yes |
| ground: tone with tip on W− | ≤ 2 % of the tone | yes, with `--zero` |
| output resistance | R2 ∥ R1 ± 30 % | yes, with `--loaded` |
| length, scope resolution, fit precision, mains pickup | ≥ 10 cycles; step ≤ Vpp/8; ± 3 %; ≤ 2× the generator's | warning only |

Results go to `data_watch_ecg/attcheck_<label>_<stamp>/` (the CSVs are copied in, `check.json` holds every value as measured). The command prints `READY for the watch` only when every required check passes.

**Then the fixed 5 Hz watch test.**
- Same leads, laptop on battery, watch off its charger, nobody touching the leads.
- The scope can stay on W+/W− as a live measurement at the electrodes.
- Run `python tone_test.py --hz 5 --no-check`, wait 20 s, record a 30 s ECG and export the PDF.
- `python att_check.py check --label w1 <export>.pdf [--ref <generator recording>.pdf]` reports the tone found in the ECG (tone search ±5 %, detected at ≥ 12 dB; 0 / 300 false alarms on noise-only synthetic recordings).

**Before OOK/FSK BER.** Give the voltage report the real numbers: `--r1 320e3 --r2 715 --source-vpp 0.8922 --source-measured --source-dc-mv 385.1 --dc-block`, plus `--watch-vpp-mv <check value>` to mark the watch level as measured. Scope one OOK run on W+/W− first: on/off edges must not step the DC (section 3, check 5). Note `--source-vpp` is in volts, unlike `att_check.py --source-vpp` (mV).

## 12. Arduino UNO R4 transmitter: DAC, bench validation, worn-watch runs (added 2026-11-06)

**What was added.** `arduino/watch_ber_tx/watch_ber_tx.ino` (firmware "watch_ber_tx v1"), `arduino_tx.py` (host driver + bench CLI), `ard_bench.py` (PicoScope analysis), `--tx arduino` in `ecg_ber.py tx` / `run`, `tests/test_arduino.py`. The TeensyTRX path is unchanged (`--tx teensy` is the default), and `ecg_link.py` (decoder ecg-v1, sha b422eaa49c6a) is unchanged. The regression tests and selftest give the same results as before.

**Three kinds of result, kept apart:**
- **SIMULATED:** `data_watch_ecg/exploratory_tonesim_*` (tone-set study, synthetic channel) and `tests/test_arduino.py` (synthetic captures, fake board). These say nothing about the real board.
- **MEASURED on the bench:** `data_watch_ecg/bench_ard_*` (`ard_bench.py`, PicoScope at the divider output, no watch).
- **MEASURED with the watch:** `session_*` / `ecg_*` runs. Their watch-input voltage is CALCULATED from the firmware settings unless `--watch-vpp-mv` (from the PicoScope) is given.

**Firmware.**
- Sine output from the 12-bit DAC on A0, updated at 4 kHz. The waveform is computed from elapsed time, and `clk` corrects the board clock.
- Phase restarts at every bit. The output sits at mid-scale when idle, for an OOK '0', and before and after a run, so OOK has no DC step.
- Limits: at most 10 mVpp calculated at the watch, and a DAC swing of at most 96 %. `cw` stops on its own after 600 s. Nothing plays at power-up.

**Tone sets** (`--toneset`). Every tone has a whole number of cycles per bit, so the frozen decoder needs no change. Tones without whole cycles per bit (e.g. 5 Hz at 2 bps) are refused.

| rate | baseline (ecg_link) | exp (from the simulation only) |
|---|---|---|
| 2 bps | OOK 16 Hz, FSK '1' 16 / '0' 8 Hz | OOK 10 Hz, FSK '1' 12 / '0' 8 Hz |
| 4, 8 bps | OOK 16, FSK 16/8 | (baseline) |
| 16 bps | OOK 16, FSK 16/32, EXPLORATORY | (baseline) |

Arduino runs are recorded with condition `<condition>, ard-<toneset>`, so `analyse` keeps the baseline and the exp tone set in separate rows and lines.

**Wiring.** A0 (DAC) → R1 → crown node → R2 470 Ω → GND = watch back. For any human-connected run, R1 is **two resistors in series** (2 × 470 kΩ; see the safety part below). Pass the measured values as `--r1 940000 --r2 470`.

### Upload
Arduino IDE 2.3.9 (`~/Downloads/arduino-ide_2.3.9_Linux_64bit/arduino-ide`):
1. File → Open `~/xingru/watch-ecg/arduino/watch_ber_tx/watch_ber_tx.ino`.
2. Tools → Board → UNO R4 WiFi; Port → /dev/ttyACM0.
3. Upload, then **close the Serial Monitor**.

Or from the terminal:
`~/Downloads/arduino-ide_2.3.9_Linux_64bit/resources/app/lib/backend/resources/arduino-cli compile --upload -b arduino:renesas_uno:unor4wifi -p /dev/ttyACM0 --config-file ~/.arduinoIDE/arduino-cli.yaml ~/xingru/watch-ecg/arduino/watch_ber_tx`

Check it: `python arduino_tx.py id` prints the firmware id and the CALCULATED level.

### Bench validation (no watch connected; PicoScope tip on the crown node, ground clip on R2's GND leg)
Use `~/.venvs/galvanic/bin/python`. CAL below means `--vfull <V> --r1 <Ohm> --r2 <Ohm>`, all measured.

0. **Calibration.**
   - Power off: measure R1 (crown node to A0) and R2.
   - Power on: run `python arduino_tx.py hold max`, measure A0 to GND with a multimeter; then `hold min`, measure again; then `off`. vfull = V(max) − V(min).
1. **+ 2. Carrier, amplitude, DC, waveform quality.**
   - Scope: ±50 mV, DC coupling, resolution enhancement ≥ +2 bits, 200 ms/div, 100 kS.
   - Run `python arduino_tx.py cw 16 --seconds 120 CAL`, then Stop → Save as CSV `ard_cw16.csv`.
   - Zero capture: tip and clip both on GND → `ard_zero.csv`.
   - Run `python ard_bench.py tone ard_cw16.csv --hz 16 --zero ard_zero.csv --label cw16 CAL`.
   - It gives f → the **board clock error in ppm**, the tone mVpp against the calculated value, DC against the calculated mid-scale DC (~1.2 mV), and H2/H3. Repeat for 8, 10 and 12 Hz (all the tones used).
   - Then repeat 16 Hz with `--tx-ppm <ppm>`. It should read within ±100 ppm, the scope's timebase accuracy.
3. **+ 4. + 5. Transients, symbol timing, offline decoding.**
   - Scope: 5 s/div (50 s), 500 kS (10 kS/s), the other settings as above.
   - Start the capture, then run `python arduino_tx.py frame --mod ook --bps 2 --toneset exp --seconds 45 CAL --tx-ppm <ppm>`. Stop when it says done, then Save → `ard_ook2exp.csv`.
   - Run `python ard_bench.py frame ard_ook2exp.csv --mod ook --bps 2 --toneset exp --zero ard_zero.csv --label ook2exp CAL`.
   - Do all four 2 bps cases (baseline / exp × OOK / FSK). FSK 4 and 8 bps are optional.
   - It reports: frozen-decoder BER on the capture resampled to 250 S/s; bit length → transmitter clock against the scope; per-frame start jitter; drift; '1' / '0' tone levels; the DC step between '1' and '0'; and the residual within 5 ms of a bit change against mid-bit. Plot: `bench_ard_<label>_*/frame.png`.
   - **Pass:** 0 errors in all complete frames; |clock| ≤ 1000 ppm after correction; jitter ≤ 2 % of a bit; DC step ≤ 0.05 mV; edge residual ≤ 1.5 × mid-bit.

### Watch runs (after the bench passes; worn = human-connected, see below)
- **Calibrated sweep, one recording:**
  - Run `python ecg_ber.py tx --tx arduino --worn --kind sweep --label sweep_ard1 --amp-mvpp 2 --seconds 75 CAL --tx-ppm <ppm>`. Start the ECG at the T+5 s cue.
  - Then `python ecg_ber.py import <pdf> --run <run>` and `python ecg_ber.py analyse sweep_ard1`.
  - Every tone has the same DAC level (checked in step 1), so the per-tone dB values are the watch's response. If the response differs from the model used in the simulation, rerun `exploratory_tonesim` before the A/B.
- **BER A/B at 2 bps:**
  - Run `python ecg_ber.py run --tx arduino --toneset baseline --bps 2 --label ab2 --repeats 3 CAL --tx-ppm <ppm>`, then the same with `--toneset exp`. Alternate the order between sessions.
  - Run `ecg_ber.py ingest` for each session, then `python ecg_ber.py analyse ab2`.
  - At 2 mVpp the simulation predicts 0 errors for both tone sets. Compare Es/N0 and line SNR as well as the BER.
  - Then run 4 and 8 bps with the baseline. 16 bps is exploratory.

### Isolation and fault protection before any human-connected run
Asked by `run --tx arduino` and `tx --tx arduino --worn`, and recorded:

- **Two series resistors.**
  - With R1 = 2 × 470 kΩ and the DAC at 0–5 V, the DAC can drive at most 5 V / 940 kΩ = **5.3 µA**, whatever the body does. The electrodes see ≤ 2 mVpp + 1.2 mV DC while R2 is in place.
  - Single fault, one resistor shorted: ≤ 5 V / 470 kΩ = **10.6 µA**. Single fault, R2 open: ≤ 5.3 µA (2.7 µA DC at mid-scale).
  - For reference, IEC 60601-1 patient auxiliary current, type BF: 10 µA DC / 100 µA AC in normal condition, 50 µA DC / 500 µA AC in single fault. This is a reference point, not a compliance claim: this is not a certified device.
  - **With a single 1 MΩ, one short puts the DAC directly on the electrodes: milliamps.**
- **No mains path.**
  - Laptop on battery, charger unplugged. No monitor, dock, Ethernet or powered hub (the Arduino's GND is the laptop's GND).
  - PicoScope and function generator disconnected from the circuit.
  - Watch off its charger.
- **Bench fault test, each session, before the person is connected:**
  - Remove R2 and put a 10 kΩ test resistor across crown and back. Run `python arduino_tx.py hold max` and measure DC mV across the 10 kΩ: expect ~53 mV (5.3 µA).
  - Short one series resistor: expect ~104 mV (10.4 µA). **Pass: ≤ 110 mV (≤ 11 µA).**
  - Then `off`, remove the short and the 10 kΩ, and put R2 back. Check R1 ≥ 940 kΩ and R2 ≈ 470 Ω again with the power off.
- **Level:** `ard_bench.py tone` this session ≤ 10 mVpp. The firmware refuses more than 10 mVpp calculated, but it cannot limit current if a resistor is shorted. **The resistors are the protection.**
- **Participant:** no implanted electronic device, intact skin, informed and free to stop. Follow the lab's approval for human-connected tests.

## 13. EXPERIMENTAL psk-x1: DBPSK / BPSK / DQPSK / 4-ASK (added 2026-11-06, bench and simulation only)

Separate from the frozen ecg-v1 and from the OOK/FSK results, which are unchanged. Same frame bits as ecg-v1
(1010 + Barker-7, PRBS payload, 00, frame <= 14 s), so BER counts stay comparable.

| file | what |
|---|---|
| `channel_est.py` | the real watch channel, fitted to the 10 real worn recordings (measurement-derived) |
| `psk_link.py` | psk-x1 modulator model + receiver (clock from the power line, matched filter, timing/frame sync, differential or coherent detection, optional decision-directed equaliser) |
| `arduino/watch_psk_tx/watch_psk_tx.ino` | firmware `watch_psk_tx x1`: plays uploaded complex symbols with an uploaded pulse table on a continuous carrier; peak clamped to the configured amplitude (<= 10 mVpp) |
| `psk_tx.py` | bench driver (`id`, `off`, `cw`, `frame`). **No worn-watch mode.** |
| `psk_bench.py` | PicoScope validation: amplitude/DC/clock, waveform vs the exact firmware model, per-symbol phase steps and EVM, full-frame decode |
| `psk_sim.py` | **SIMULATED** BER through the measured channel with the wearer's real ECG residual as interference -> `data_watch_ecg/SIMULATED_psk_*` |
| `tests/test_psk.py` | round trip, noise-only false frames, firmware parity (gcc compiles the firmware's own checksum/sampleAt), bench tool pass and fail cases |

Frame validity (truth-free, set from noise-only runs on real residuals before any PSK measurement): header correlation
>= 0.70, recording sync score >= 0.70, and at most 1 of the 13 known bits (header + gap) wrong after detection.

### Bench validation (watch NOT connected, nobody touching the circuit)
1. Upload `arduino/watch_psk_tx` (Arduino IDE 2, board UNO R4 WiFi). `python psk_tx.py id` must print `watch_psk_tx x1`.
2. PicoScope: tip on the crown node, clip on GND; +-50 mV, DC, resolution enhancement >= +2 bits, 5 s/div, >= 10 kS/s.
   Zero capture (tip and clip on GND) -> `zero.csv`.
3. Per test: press Run, then
   `python psk_tx.py frame --mod dbpsk --bps 16 --shape rect --seconds 45 $CAL`; Stop after `# done`, save `d16.csv`;
   `python psk_bench.py d16.csv --mod dbpsk --bps 16 --shape rect --seconds 45 --label d16 --zero zero.csv $CAL`.
   READY = amplitude within 10 % of calculated, DC within 0.5 mV, waveform residual within 5 % (or 1.5 x scope noise),
   no extra residual at symbol boundaries, every phase step within 10 deg, EVM <= -25 dB, all frames decoded, 0 errors.
4. Order: dbpsk 8, 16, 24, 32 (rect, then rrc); then the alternatives the simulation ranks best.

Worn-watch psk-x1 runs are NOT provided: they need the independent electrical-safety review first (section 12 checks).

## 14. DBPSK maximum-bitrate experiment: real BER workflow (added 2026-11-06)

Pre-registered protocol: `psk_protocol.json` (thresholds, ladder 16-100 bps, stepping rule, carrier and pulse arms;
its sha is stored with every result). Receiver `psk-x3` (frozen; `psk_ber.py` refuses a modified `psk_link.py`):
primary `x3-wl` (widely-linear, header-trained, leave-one-frame-out decisions), secondary `x2-diff` (the earlier
complex matched-filter receiver), both on the same recordings. `psk_diag.py` adds a known-transmission diagnosis per
recording (frames transmitted while recording; ideal linear receiver on the unseen half) and classifies failures as
channel-limited, receiver-limited (sync / false lock / errors), transmission not found, or not usable.

| step | command |
|---|---|
| upload | Arduino IDE: `arduino/watch_psk_tx` (UNO R4 WiFi); `python psk_tx.py id` -> `watch_psk_tx x2` |
| bench sweep (watch NOT connected) | `python psk_tx.py sweep --rates 16,24,32,40,50,64,80,100 --shape rrc --seg 8 --label rrc1 $CAL` during ONE PicoScope capture, then `python psk_bench.py sweep <csv> --record <json printed> --zero zero.csv` (repeat with `--shape rect`, and `--fcs 20,24 --rates 80,100`) |
| worn session | `python psk_ber.py run --bps 16 --label d16` (3 recordings, 80 s apart, cue at T+5 s; the six checks are still asked) |
| import + BER | `python psk_ber.py ingest data_watch_ecg/psk_session_d16_* ~/Downloads/d16/` |
| curve + next rate | `python psk_ber.py analyse d16 d24 d32 ... --label curve` ; `python psk_ber.py next d16 d24 ...` |

Voltage: the 2 mVpp is CALCULATED; `--bench-report` stores an off-body PicoScope level separately, labelled off-body.

## 15. psk-x4 "x4-mb": improved DBPSK receiver (added 2026-11-07, EXPERIMENTAL)

`psk_rx4.py` is a separate receiver, frozen at sha `96d8ccdb9e7c` (2026-11-07 19:20). The psk-x3 receiver x3-wl
(`psk_link.py`, sha `04dbb95193d8`) and ecg-v1 are unchanged.

**Why it was needed.** At 40 bps x3-wl failed with 453 errors in 2667 bits, while the known-transmission ideal
receiver made 0 errors. The 40 bps recordings showed three separate causes:

| recording | x3-wl | cause |
|---|---|---|
| r1 | the same 2 errors in 4 of 7 frames | bias from decision-directed training of a ~165-parameter filter |
| r2 | 57 of 127 bits wrong in every frame | false lock: the training converged on a wrong but self-consistent pattern |
| r3 | errors growing from 6 to 10 per frame | clock 208 ppm off, i.e. 0.25 symbol of timing drift |

**How x4 works.** It uses the physical model instead of training an equaliser:
x(w) = Σ a_n Re{e^{j2πfc(n+½)T} v_b(w/sc − τ0 − (n+½)T)} + ECG + noise, where
- v is the known pulse on its carrier, passed through the fixed channel model;
- b is a short complex correction (10 real parameters);
- sc is the clock and τ0 the timing.

The decoding steps:
1. **Clock and timing:** a blind search with no known bits, minimising the model residual.
2. **Noise model:** an AR(12) model of the ECG noise, then generalised least squares.
3. **Symbols:** ridge least squares through the model, so the ISI is inverted rather than learned.
4. **Frame sync:** the 13 known bits searched in the decided bit stream. Each frame has its own sign, so the identical
   test frames are not exploited.
5. **Validity:** the psk-x3 rules, unchanged.

The payload truth is used only to count errors.

**Development and held-out data.** x4 was developed on d40 r1, d40 r2 and simulations (random payload per frame, real
ECG residuals). These two recordings are reported as `x4-mb (development)` and never count toward an x4 verdict.
x4 then decoded the held-out recordings for the first time:

| held out | x3-wl | x4-mb |
|---|---|---|
| 16 bps (4 recordings) | 0/1143 | 0/1143, sync 9/9 |
| 24 bps (3 recordings) | 0/1524 | 0/1524, sync 12/12 |
| 32 bps (3 recordings) | 0/2159 | 0/2159, sync 17/17 |
| 40 bps r3 (1 recording) | 52/889 | 0/889, sync 7/7 |

Report: `data_watch_ecg/pskber_x4_heldout_20261107_191324/`.

**Simulation (SIMULATED, `data_watch_ecg/SIMULATED_x4_20261107/`).**
- 64 bps (fc 16 Hz, rrc): sync 95 %, BER 0.8 %. Almost all errors come from one noisy residual.
- 80 bps: best at fc 20 Hz with the rect pulse (sync 96 %, 0.35 %). At fc 16 or 24 Hz it fails.
- 100 bps: fails at every carrier.
- Noise only: 0 false frames at every rate from 16 to 80 bps.

**x4 test arm** (`psk_protocol.json` "x4_arm"): written before the held-out decode. It uses the same reliability rule.
- Ladder: 40/16/rrc → 50/16/rrc → 64/16/rrc → 80/20/rect → 100/24/rect, with bisection.
- `psk_ber.py analyse` prints both stepping rules: the pre-registered x3-wl one and the x4 one.
- `python psk_ber.py next <labels> --receiver x4-mb` shows the next x4 step.
