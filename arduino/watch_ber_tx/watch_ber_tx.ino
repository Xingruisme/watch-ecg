/*
  watch_ber_tx.ino - Arduino UNO R4 (WiFi or Minima) transmitter for the Pixel Watch ECG BER tests.
  Plays the frames / tone schedules that ecg_ber.py (--tx arduino) or arduino_tx.py send, as a sine from the
  12-bit DAC on A0. Firmware id: "watch_ber_tx v1".

  Wiring (bench):  A0 (DAC) --[ R_TOP ]--+--[ R_BOT 470 Ohm ]-- GND -- watch BACK
                                         |
                                     watch CROWN
  R_TOP is the series resistance: for any human-connected run it must be TWO resistors in series (e.g. 2 x 470 kOhm),
  so that one shorted resistor still leaves >= 470 kOhm (README section 12).

  Output:  mid-scale (code 2048) when idle, for an OOK '0' and between runs, so the watch sees no DC step when a tone
           starts or stops. Tone = mid-scale +- amplitude, sine, phase restarted at the start of every step (bit).
           DAC updated every UPDATE_US (4 kHz); waveform computed from elapsed time, so timing does not drift with
           loop jitter. Clock correction 'clk' scales the board's micros() to true time.
  Limits:  amplitude at the watch (calculated from cal) <= MAX_WATCH_MVPP; the DAC swing <= 96 % of full scale.
           'cw' stops by itself after at most CW_MAX_S. Nothing starts at power-up.

  Serial 115200, one command per line, replies start with '#'. Errors start with '# error'.
    id                              -> # watch_ber_tx v1 ...
    cal <vfull_V> <rtop_ohm> <rbot_ohm>   DAC full-scale swing (measure with h / l) and the divider
    a <mVpp>                        amplitude at the watch (crown - back), calculated from cal
    clk <ppm>                       board clock error measured on the scope: (f_measured / f_set - 1) x 1e6
    ?                               -> # cfg ... and # state ...
    load <step_us> <reps> <mHz_1> ... <mHz_n>   schedule: n steps, played reps times; 0 mHz = mid-scale (off)
                                    -> # loaded n <n> sum <checksum>
    go <delay_ms>                   -> # armed in <delay_ms> ms ; then # start ; at the end # done events <N>
    cw <Hz> [seconds]               steady tone (default 300 s, max CW_MAX_S)
    x                               stop: output to mid-scale -> # off
    h / l / m                       hold the DAC at max / min / mid (multimeter or scope on A0: vfull = V(h) - V(l))
*/

#define FW_ID "watch_ber_tx v1"

const float    MAX_WATCH_MVPP = 10.0f;    // the watch recorded 2-10 mVpp; >= 20 mVpp failed (2026-11-04)
const float    MAX_SWING_FRAC = 0.96f;
const uint32_t UPDATE_US      = 250;      // 4 kHz DAC update
const uint32_t CW_MAX_S       = 600;
const int      MAX_STEPS      = 400;
const int      MID            = 2048;

float vFull = 5.00f, rTop = 1.0e6f, rBot = 470.0f, ampMvpp = 2.0f, clkPpm = 0.0f;
float halfCodes = 0;                      // tone peak in DAC codes

uint32_t stepUs = 0, reps = 0, nSteps = 0;
uint32_t stepMilliHz[MAX_STEPS];

enum State { IDLE, ARMED, RUN, DONE, CW, HOLD };
State state = IDLE;
uint32_t tArm = 0, armDelayUs = 0, tStart = 0, nextUs = 0, cwHz1000 = 0, cwEndS = 0;
uint32_t lastEvents = 0;
char line[2600];
uint16_t len = 0;

float ratio() { return rBot / (rTop + rBot); }

void dac(int code) {
  if (code < 0) code = 0;
  if (code > 4095) code = 4095;
  analogWrite(DAC, code);
}

bool setAmp(float mvpp) {
  if (!(mvpp >= 0.0f) || mvpp > MAX_WATCH_MVPP) {
    Serial.print("# error amplitude "); Serial.print(mvpp, 3); Serial.print(" mVpp outside 0-");
    Serial.print(MAX_WATCH_MVPP, 1); Serial.println(" mVpp (calculated at the watch)");
    return false;
  }
  float dacVpp = mvpp / 1000.0f / ratio();
  if (dacVpp > MAX_SWING_FRAC * vFull) {
    Serial.print("# error amplitude needs "); Serial.print(dacVpp, 3); Serial.println(" Vpp from the DAC - too much");
    return false;
  }
  ampMvpp = mvpp;
  halfCodes = dacVpp / 2.0f / vFull * 4095.0f;
  return true;
}

void printCfg() {
  float dacVpp = ampMvpp / 1000.0f / ratio();
  Serial.print("# cfg vfull "); Serial.print(vFull, 4);
  Serial.print(" rtop "); Serial.print(rTop, 1);
  Serial.print(" rbot "); Serial.print(rBot, 3);
  Serial.print(" amp_mvpp "); Serial.print(ampMvpp, 4);
  Serial.print(" dac_vpp "); Serial.print(dacVpp, 4);
  Serial.print(" dc_mv "); Serial.print(vFull * MID / 4095.0f * ratio() * 1000.0f, 4);
  Serial.print(" clk_ppm "); Serial.print(clkPpm, 1);
  Serial.print(" update_us "); Serial.print(UPDATE_US);
  Serial.print(" limit_mvpp "); Serial.println(MAX_WATCH_MVPP, 1);
}

uint32_t checksum() {
  uint32_t s = stepUs * 7u + reps * 13u + nSteps;
  for (uint32_t i = 0; i < nSteps; i++) s += (i + 1) * stepMilliHz[i];
  return s;
}

const char *stateName() {
  switch (state) {
    case IDLE: return "idle"; case ARMED: return "armed"; case RUN: return "run";
    case DONE: return "done"; case CW: return "cw"; default: return "hold";
  }
}

void printState() {
  Serial.print("# state "); Serial.print(stateName());
  Serial.print(" events "); Serial.print(lastEvents);
  Serial.print(" of "); Serial.print(reps * nSteps);
  Serial.print(" n "); Serial.print(nSteps);
  Serial.print(" step_us "); Serial.print(stepUs);
  Serial.print(" reps "); Serial.println(reps);
}

// elapsed true microseconds since t0 (board clock corrected)
double elapsedUs(uint32_t t0, uint32_t now) {
  return (double)(uint32_t)(now - t0) / (1.0 + clkPpm * 1e-6);
}

void stopOutput(const char *msg) {
  state = IDLE;
  dac(MID);
  if (msg) Serial.println(msg);
}

void handle(char *s) {
  char *cmd = strtok(s, " \t");
  if (!cmd) return;
  if (!strcmp(cmd, "id")) {
    Serial.print("# " FW_ID " board UNO-R4 dac A0 bits 12 update_us "); Serial.println(UPDATE_US);
  } else if (!strcmp(cmd, "cal")) {
    char *a = strtok(NULL, " "), *b = strtok(NULL, " "), *c = strtok(NULL, " ");
    if (!a || !b || !c) { Serial.println("# error usage: cal <vfull_V> <rtop_ohm> <rbot_ohm>"); return; }
    float v = atof(a), rt = atof(b), rb = atof(c);
    if (v < 1.0f || v > 6.0f || rt < 1000.0f || rb <= 0.0f) { Serial.println("# error cal values out of range"); return; }
    if (state == RUN || state == ARMED || state == CW) { Serial.println("# error busy: send x first"); return; }
    float oV = vFull, oT = rTop, oB = rBot;
    vFull = v; rTop = rt; rBot = rb;
    if (!setAmp(ampMvpp)) { vFull = oV; rTop = oT; rBot = oB; setAmp(ampMvpp); return; }
    printCfg();
  } else if (!strcmp(cmd, "a")) {
    char *a = strtok(NULL, " ");
    if (!a) { Serial.println("# error usage: a <mVpp>"); return; }
    if (state == RUN || state == ARMED) { Serial.println("# error busy: send x first"); return; }
    if (setAmp(atof(a))) printCfg();
  } else if (!strcmp(cmd, "clk")) {
    char *a = strtok(NULL, " ");
    float p = a ? atof(a) : 0.0f;
    if (!a || p < -20000.0f || p > 20000.0f) { Serial.println("# error usage: clk <ppm> (|ppm| <= 20000)"); return; }
    if (state == RUN || state == ARMED) { Serial.println("# error busy: send x first"); return; }
    clkPpm = p; printCfg();
  } else if (!strcmp(cmd, "?")) {
    printCfg(); printState();
  } else if (!strcmp(cmd, "load")) {
    if (state == RUN || state == ARMED || state == CW) { Serial.println("# error busy: send x first"); return; }
    char *a = strtok(NULL, " "), *b = strtok(NULL, " ");
    if (!a || !b) { Serial.println("# error usage: load <step_us> <reps> <mHz...>"); return; }
    uint32_t su = strtoul(a, NULL, 10), rp = strtoul(b, NULL, 10), n = 0;
    char *f;
    while ((f = strtok(NULL, " ")) != NULL) {
      if (n >= (uint32_t)MAX_STEPS) { Serial.println("# error too many steps"); nSteps = 0; return; }
      uint32_t mhz = strtoul(f, NULL, 10);
      if (mhz > 400000) { Serial.println("# error step frequency above 400 Hz"); nSteps = 0; return; }
      stepMilliHz[n++] = mhz;
    }
    if (su < 10000 || rp < 1 || n < 1) { Serial.println("# error step_us >= 10000, reps >= 1, n >= 1"); nSteps = 0; return; }
    stepUs = su; reps = rp; nSteps = n; lastEvents = 0; state = IDLE;
    Serial.print("# loaded n "); Serial.print(nSteps); Serial.print(" sum "); Serial.println(checksum());
  } else if (!strcmp(cmd, "go")) {
    char *a = strtok(NULL, " ");
    if (!nSteps) { Serial.println("# error nothing loaded"); return; }
    if (state == RUN || state == ARMED || state == CW) { Serial.println("# error busy: send x first"); return; }
    uint32_t ms = a ? strtoul(a, NULL, 10) : 500;
    if (ms > 10000) ms = 10000;
    dac(MID);
    tArm = micros(); armDelayUs = ms * 1000u; lastEvents = 0; state = ARMED;
    Serial.print("# armed in "); Serial.print(ms); Serial.println(" ms");
  } else if (!strcmp(cmd, "cw")) {
    char *a = strtok(NULL, " "), *b = strtok(NULL, " ");
    float hz = a ? atof(a) : -1.0f;
    uint32_t sec = b ? strtoul(b, NULL, 10) : 300;
    if (hz <= 0.0f || hz > 400.0f) { Serial.println("# error usage: cw <Hz 0-400> [seconds]"); return; }
    if (state == RUN || state == ARMED) { Serial.println("# error busy: send x first"); return; }
    if (sec < 1) sec = 1;
    if (sec > CW_MAX_S) sec = CW_MAX_S;
    cwHz1000 = (uint32_t)(hz * 1000.0f + 0.5f); cwEndS = sec;
    tStart = micros(); nextUs = tStart; state = CW;
    Serial.print("# cw "); Serial.print(cwHz1000 / 1000.0f, 3); Serial.print(" Hz for "); Serial.print(sec);
    Serial.println(" s");
  } else if (!strcmp(cmd, "x")) {
    stopOutput("# off");
  } else if (!strcmp(cmd, "h") || !strcmp(cmd, "l") || !strcmp(cmd, "m")) {
    if (state == RUN || state == ARMED) { Serial.println("# error busy: send x first"); return; }
    state = HOLD;
    int code = cmd[0] == 'h' ? 4095 : (cmd[0] == 'l' ? 0 : MID);
    dac(code);
    Serial.print("# hold code "); Serial.println(code);
  } else {
    Serial.print("# error unknown command: "); Serial.println(cmd);
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[len] = 0;
      if (len) handle(line);
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    } else {
      len = 0;
      Serial.println("# error line too long");
    }
  }
}

void setup() {
  analogWriteResolution(12);
  dac(MID);
  setAmp(ampMvpp);
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 2000) {}
  Serial.println("# " FW_ID " ready (output at mid-scale)");
}

void loop() {
  readSerial();
  uint32_t now = micros();
  if (state == ARMED) {
    if ((uint32_t)(now - tArm) >= armDelayUs) {
      tStart = tArm + armDelayUs; nextUs = tStart; state = RUN;
      Serial.println("# start");
    } else {
      return;
    }
  }
  if (state != RUN && state != CW) return;
  if ((int32_t)(now - nextUs) < 0) return;
  nextUs += UPDATE_US;
  if ((int32_t)(now - nextUs) > (int32_t)(20 * UPDATE_US)) nextUs = now + UPDATE_US;   // after a long stall: resync
  double el = elapsedUs(tStart, now);
  if (state == CW) {
    if (el >= cwEndS * 1e6) { stopOutput("# cw ended"); return; }
    double ph = cwHz1000 * 1e-3 * el * 1e-6;
    ph -= floor(ph);
    dac(MID + (int)lroundf(halfCodes * sinf(6.2831853f * (float)ph)));
    return;
  }
  uint32_t total = reps * nSteps;
  uint32_t k = (uint32_t)(el / stepUs);
  if (k >= total) {
    lastEvents = total; state = DONE; dac(MID);
    Serial.print("# done events "); Serial.println(total);
    return;
  }
  lastEvents = k + 1;
  uint32_t mhz = stepMilliHz[k % nSteps];
  if (!mhz) { dac(MID); return; }
  double ph = mhz * 1e-3 * (el - (double)k * stepUs) * 1e-6;
  ph -= floor(ph);
  dac(MID + (int)lroundf(halfCodes * sinf(6.2831853f * (float)ph)));
}
