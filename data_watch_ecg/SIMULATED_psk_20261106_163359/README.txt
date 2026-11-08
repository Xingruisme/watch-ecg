SIMULATED - NOT MEASURED BER.
SIMULATED evaluation of the experimental psk-x1 modulations through the MEASURED watch channel (added 2026-11-06).

Nothing here is a measured BER. Each simulated recording = the psk-x1 transmitter waveform (exact firmware model)
  -> the watch channel fitted to the 10 real worn recordings (channel_est.py parametric: 1 Hz HP, low-pass ~25-27.5 Hz,
     linear-phase best; minimum-phase 2nd order 22.5 Hz as the alternative)
  -> x the gain measured on one real recording, + THE REAL RESIDUAL of that recording (the wearer's ECG and noise as
     the watch recorded it, after removing the fitted OOK/FSK signal), at a random start and clock (+-5000 ppm).
Pre-registered (before any PSK measurement): thresholds psk_link.R_MIN / G_MIN are checked on residual-only input
('falsealarm'); 'reliable' = frame sync >= 95 % of complete frames AND pooled BER upper 95 % CI < 1 %.

  python psk_sim.py                      -> data_watch_ecg/SIMULATED_psk_<time>/ (results.jsonl, summary.csv, README.txt)
  python psk_sim.py --quick              (fewer cells)

decoder psk-x1; R_MIN 0.7, G_MIN 0.7; seed 20261106
channel: data_watch_ecg/channel_est_* (fitted to the real recordings); residuals: the real recordings minus the fitted OOK/FSK signal.
