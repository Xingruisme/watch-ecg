SIMULATED - not measured BER. psk-x4 'x4-mb' development study, 2026-11-07 (sim4.py, copied here unchanged).
Each simulated recording: DBPSK with a RANDOM payload in every frame (so repetition cannot help), exact transmitter
model, watch channel model (lin27 = the receiver's own model; min22 = a different, minimum-phase 22.5 Hz model to test
mismatch), x the real gain, + the REAL residual (ECG + noise) of one of the 10 OOK/FSK recordings; clock 3400 +-300 ppm.
s1.json   40, 64 bps fc 16 rrc, lin27 + min22, no whitening
s2.json   80, 100 bps fc 16/20/24, rrc + rect, lin27, no whitening (s2.txt = summary)
w64/w80   64 bps fc16 rrc, 80 bps fc20 rect, whitening order 12 (adopted) (w.txt = summary)
n1/n2     noise only (residual, nothing transmitted), whitening 12: 0 false valid frames at 16-80 bps
Residual #1 (FSK 16 bps run) holds most remaining errors: strongest in-band noise, likely partly leftover FSK tones.
