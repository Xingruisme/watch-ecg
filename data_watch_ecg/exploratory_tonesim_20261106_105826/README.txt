Tone-set simulation for the Arduino (UNO R4 DAC) BER design, 2026-11-06. Exploratory, synthetic only.
Decoder: frozen ecg-v1 (ecg_link.decode, unchanged; tones passed through ber_plan(f1=, f0=)).
Channel: user's preliminary measured Pixel response fit (HP 1 Hz 1st order + Butterworth LP 3rd order 21 Hz:
20 Hz 0.75, 30 Hz 0.32, 50 Hz 0.07 of 10 Hz). TX clock -2400 ppm, watch +1227 ppm. Noise as ecg_synth
(10 uV white, 30 uV wander, 5 uV 50 Hz). 'heart' = ecg_synth heartbeat 70 bpm (~1 mV R). n = 8 recordings per point.
Rerun one config: python tonesim.py '<json line from jobs.txt>'
