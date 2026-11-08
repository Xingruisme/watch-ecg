/*
  watch_psk_tx.ino - EXPERIMENTAL Arduino UNO R4 transmitter for the psk-x1 modulations (DBPSK, BPSK, DQPSK, 4-ASK).
  Firmware id: "watch_psk_tx x2" (x2, 2026-11-06: stored-frame looping, so long runs at high rates fit in RAM).
  Separate from watch_ber_tx v1 (OOK/FSK, unchanged).

  Wiring and limits are the same as watch_ber_tx v1:
      A0 (DAC) --[ R_TOP ]--+--[ R_BOT 470 Ohm ]-- GND -- watch BACK
                            |
                        watch CROWN
  R_TOP = TWO resistors in series for any human-connected use (README section 12).

  Output = mid-scale + halfCodes * clamp( Re{ b(t) e^{j 2 pi fc t} } / norm , +-1.05 )
      b(t) = sum_k a_k p(t/T - k - 1/2)       a_k = (I_k + j Q_k) / 120 (int8 pairs uploaded by psk_tx.py)
      p    = pulse table, span symbols x pts per symbol (+1 point), int16 / 8192, linear interpolation,
             zero outside [-span/2, +span/2)  (identical to psk_link.pulse_at / baseband)
  halfCodes is set by 'a' (peak-to-peak at the watch, calculated from cal) and limited to MAX_WATCH_MVPP, so the clamp
  guarantees the peak never exceeds the configured amplitude by more than 5 %, whatever is uploaded.
  The carrier phase is continuous from the start of the run; the symbol clock and carrier share the corrected time base.
  Mid-scale when idle, before and after the run. Nothing starts at power-up.

  Serial 115200, one command per line, replies start with '#', errors with '# error'.
    id | cal <vfull_V> <rtop_ohm> <rbot_ohm> | a <mVpp> | clk <ppm> | ?
    cfg <sym_us> <fc_mHz> <n_stored> <span> <pts> <norm_x1e6> <loop_start> <loop_len> <loop_reps>  -> # cfg ok
                          played symbols: stored[0 .. loop_start) once, stored[loop_start .. +loop_len) loop_reps
                          times, then the rest once (loop_len 0 = no loop). Clears the buffers.
    s <start> <hex>       symbols from index start: 4 hex chars each (I byte, Q byte, two's complement)
    p <start> <hex>       pulse table values from index start: 4 hex chars each (int16 two's complement)
    sum                   -> # sum <checksum>   (psk_tx.checksum)
    go <delay_ms>         -> # armed in .. ms ; # start ; # done symbols N
    cw <Hz> [s] | x | h / l / m      as watch_ber_tx v1
*/

#define FW_ID "watch_psk_tx x2"

const float    MAX_WATCH_MVPP = 10.0f;
const float    MAX_SWING_FRAC = 0.96f;
const uint32_t UPDATE_US      = 250;
const uint32_t CW_MAX_S       = 600;
const int      MAX_SYMS       = 4096;
const int      MAX_PULSE      = 8 * 64 + 1;
const int      MID            = 2048;
const float    SYM_SCALE      = 120.0f;
const float    PULSE_SCALE    = 8192.0f;

float vFull = 5.00f, rTop = 1.0e6f, rBot = 470.0f, ampMvpp = 2.0f, clkPpm = 0.0f;
float halfCodes = 0;

int8_t   symI[MAX_SYMS], symQ[MAX_SYMS];
int16_t  pulse[MAX_PULSE];
uint32_t symUs = 0, fcMilliHz = 0, nSyms = 0, span = 0, pts = 0;      // nSyms = stored symbols
uint32_t loopStart = 0, loopLen = 0, loopReps = 1, nPlay = 0;           // nPlay = played symbols
float    norm = 1.0f;

enum State { IDLE, ARMED, RUN, DONE, CW, HOLD };
State state = IDLE;
uint32_t tArm = 0, armDelayUs = 0, tStart = 0, nextUs = 0, cwHz1000 = 0, cwEndS = 0, lastSym = 0;
char line[600];
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
  if (dacVpp > MAX_SWING_FRAC * vFull / 1.05f) {
    Serial.print("# error amplitude needs "); Serial.print(dacVpp, 3); Serial.println(" Vpp from the DAC - too much");
    return false;
  }
  ampMvpp = mvpp;
  halfCodes = dacVpp / 2.0f / vFull * 4095.0f;
  return true;
}

void printCfg() {
  Serial.print("# cfg vfull "); Serial.print(vFull, 4);
  Serial.print(" rtop "); Serial.print(rTop, 1);
  Serial.print(" rbot "); Serial.print(rBot, 3);
  Serial.print(" amp_mvpp "); Serial.print(ampMvpp, 4);
  Serial.print(" dac_vpp "); Serial.print(ampMvpp / 1000.0f / ratio(), 4);
  Serial.print(" dc_mv "); Serial.print(vFull * MID / 4095.0f * ratio() * 1000.0f, 4);
  Serial.print(" clk_ppm "); Serial.print(clkPpm, 1);
  Serial.print(" update_us "); Serial.print(UPDATE_US);
  Serial.print(" limit_mvpp "); Serial.println(MAX_WATCH_MVPP, 1);
}

uint32_t checksum() {
  uint32_t s = symUs * 7u + nSyms * 13u + fcMilliHz * 3u + span * 5u + pts + loopStart * 17u + loopLen * 19u
               + loopReps * 23u;
  for (uint32_t i = 0; i < nSyms; i++) s += (i + 1) * ((((uint32_t)(uint8_t)symI[i]) << 8) | (uint8_t)symQ[i]);
  uint32_t np = span * pts + 1;
  for (uint32_t j = 0; j < np; j++) s += (j + 1) * (uint32_t)(uint16_t)pulse[j];
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
  Serial.print(" symbols "); Serial.print(lastSym);
  Serial.print(" of "); Serial.print(nPlay);
  Serial.print(" sym_us "); Serial.print(symUs);
  Serial.print(" fc_mhz "); Serial.print(fcMilliHz);
  Serial.print(" span "); Serial.print(span);
  Serial.print(" pts "); Serial.println(pts);
}

double elapsedUs(uint32_t t0, uint32_t now) { return (double)(uint32_t)(now - t0) / (1.0 + clkPpm * 1e-6); }

void stopOutput(const char *msg) { state = IDLE; dac(MID); if (msg) Serial.println(msg); }

int hexv(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// 4 hex chars -> 16-bit value; false on a bad character
bool hex16(const char *h, uint16_t &v) {
  v = 0;
  for (int i = 0; i < 4; i++) { int d = hexv(h[i]); if (d < 0) return false; v = (v << 4) | d; }
  return true;
}

bool busy() { return state == RUN || state == ARMED || state == CW; }

void handle(char *s) {
  char *cmd = strtok(s, " \t");
  if (!cmd) return;
  if (!strcmp(cmd, "id")) {
    Serial.print("# " FW_ID " board UNO-R4 dac A0 bits 12 update_us "); Serial.print(UPDATE_US);
    Serial.print(" max_syms "); Serial.print(MAX_SYMS); Serial.print(" max_pulse "); Serial.println(MAX_PULSE);
  } else if (!strcmp(cmd, "cal")) {
    char *a = strtok(NULL, " "), *b = strtok(NULL, " "), *c = strtok(NULL, " ");
    if (!a || !b || !c) { Serial.println("# error usage: cal <vfull_V> <rtop_ohm> <rbot_ohm>"); return; }
    float v = atof(a), rt = atof(b), rb = atof(c);
    if (v < 1.0f || v > 6.0f || rt < 1000.0f || rb <= 0.0f) { Serial.println("# error cal values out of range"); return; }
    if (busy()) { Serial.println("# error busy: send x first"); return; }
    float oV = vFull, oT = rTop, oB = rBot;
    vFull = v; rTop = rt; rBot = rb;
    if (!setAmp(ampMvpp)) { vFull = oV; rTop = oT; rBot = oB; setAmp(ampMvpp); return; }
    printCfg();
  } else if (!strcmp(cmd, "a")) {
    char *a = strtok(NULL, " ");
    if (!a) { Serial.println("# error usage: a <mVpp>"); return; }
    if (busy()) { Serial.println("# error busy: send x first"); return; }
    if (setAmp(atof(a))) printCfg();
  } else if (!strcmp(cmd, "clk")) {
    char *a = strtok(NULL, " ");
    float p = a ? atof(a) : 0.0f;
    if (!a || p < -20000.0f || p > 20000.0f) { Serial.println("# error usage: clk <ppm> (|ppm| <= 20000)"); return; }
    if (busy()) { Serial.println("# error busy: send x first"); return; }
    clkPpm = p; printCfg();
  } else if (!strcmp(cmd, "?")) {
    printCfg(); printState();
  } else if (!strcmp(cmd, "cfg")) {
    if (busy()) { Serial.println("# error busy: send x first"); return; }
    char *a[9];
    for (int i = 0; i < 9; i++) a[i] = strtok(NULL, " ");
    if (!a[8]) { Serial.println("# error usage: cfg <sym_us> <fc_mHz> <n_stored> <span> <pts> <norm_x1e6> <loop_start> <loop_len> <loop_reps>"); return; }
    uint32_t ls = strtoul(a[6], NULL, 10), ll = strtoul(a[7], NULL, 10), lr = strtoul(a[8], NULL, 10);
    uint32_t su = strtoul(a[0], NULL, 10), fc = strtoul(a[1], NULL, 10), n = strtoul(a[2], NULL, 10);
    uint32_t sp = strtoul(a[3], NULL, 10), pt = strtoul(a[4], NULL, 10);
    float nm = strtoul(a[5], NULL, 10) * 1e-6f;
    if (su < 5000 || fc < 1000 || fc > 100000 || n < 1 || n > (uint32_t)MAX_SYMS || sp < 1 || pt < 2
        || sp * pt + 1 > (uint32_t)MAX_PULSE || nm < 0.05f || lr < 1 || ls + ll > n || (ll == 0 && lr != 1)) {
      Serial.println("# error cfg: sym_us >= 5000, fc 1-100 Hz, 1 <= n <= max_syms, span*pts+1 <= max_pulse, norm >= 0.05");
      nSyms = 0; return;
    }
    symUs = su; fcMilliHz = fc; nSyms = n; span = sp; pts = pt; norm = nm; lastSym = 0; state = IDLE;
    loopStart = ls; loopLen = ll; loopReps = lr; nPlay = n + ll * (lr - 1);
    memset(symI, 0, sizeof(symI)); memset(symQ, 0, sizeof(symQ)); memset(pulse, 0, sizeof(pulse));
    Serial.print("# cfg ok played "); Serial.println(nPlay);
  } else if (!strcmp(cmd, "s") || !strcmp(cmd, "p")) {
    if (busy()) { Serial.println("# error busy: send x first"); return; }
    bool isSym = cmd[0] == 's';
    char *a = strtok(NULL, " "), *h = strtok(NULL, " ");
    if (!a || !h) { Serial.println("# error usage: s|p <start> <hex>"); return; }
    uint32_t st = strtoul(a, NULL, 10), cnt = strlen(h) / 4, lim = isSym ? nSyms : span * pts + 1;
    if (strlen(h) % 4 || st + cnt > lim) { Serial.println("# error s/p: bad length or beyond the configured size"); return; }
    for (uint32_t i = 0; i < cnt; i++) {
      uint16_t v;
      if (!hex16(h + 4 * i, v)) { Serial.println("# error s/p: bad hex"); return; }
      if (isSym) { symI[st + i] = (int8_t)(v >> 8); symQ[st + i] = (int8_t)(v & 0xFF); }
      else pulse[st + i] = (int16_t)v;
    }
    Serial.print("# ok "); Serial.println(st + cnt);
  } else if (!strcmp(cmd, "sum")) {
    Serial.print("# sum "); Serial.println(checksum());
  } else if (!strcmp(cmd, "go")) {
    char *a = strtok(NULL, " ");
    if (!nSyms) { Serial.println("# error nothing loaded"); return; }
    if (busy()) { Serial.println("# error busy: send x first"); return; }
    uint32_t ms = a ? strtoul(a, NULL, 10) : 500;
    if (ms > 10000) ms = 10000;
    dac(MID);
    tArm = micros(); armDelayUs = ms * 1000u; lastSym = 0; state = ARMED;
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
    Serial.print("# cw "); Serial.print(cwHz1000 / 1000.0f, 3); Serial.print(" Hz for "); Serial.print(sec); Serial.println(" s");
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
    if (c == '\n') { line[len] = 0; if (len) handle(line); len = 0; }
    else if (len < sizeof(line) - 1) line[len++] = c;
    else { len = 0; Serial.println("# error line too long"); }
  }
}

float pulseAt(float tau) {                       // psk_link.pulse_at
  float u = (tau + span * 0.5f) * pts;
  if (u < 0.0f) return 0.0f;
  int i = (int)floorf(u);
  if (i >= (int)(span * pts)) return 0.0f;
  float f = u - i;
  return (pulse[i] * (1.0f - f) + pulse[i + 1] * f) / PULSE_SCALE;
}

// stored index of played symbol k (the loop)
uint32_t stored(uint32_t k) {
  if (k < loopStart || loopLen == 0) return k;
  uint32_t r = k - loopStart;
  if (r < loopLen * loopReps) return loopStart + r % loopLen;
  return k - loopLen * (loopReps - 1);
}

// relative output (peak 1, clamped to +-1.05) at el corrected microseconds after the start (psk_tx.model)
float sampleAt(double el) {
  double u = el / symUs;
  int k0 = (int)floor(u);
  float bi = 0.0f, bq = 0.0f;
  int h = span / 2 + 1;
  for (int dk = -h; dk <= h; dk++) {
    int k = k0 + dk;
    if (k < 0 || k >= (int)nPlay) continue;
    float w = pulseAt((float)(u - k - 0.5));
    uint32_t j = stored((uint32_t)k);
    bi += symI[j] * w; bq += symQ[j] * w;
  }
  double cyc = fcMilliHz * 1e-3 * el * 1e-6; cyc -= floor(cyc);
  float ph = 6.2831853f * (float)cyc;
  float v = (bi * cosf(ph) - bq * sinf(ph)) / SYM_SCALE / norm;   // Re{b e^{j ph}}
  if (v > 1.05f) v = 1.05f;
  if (v < -1.05f) v = -1.05f;
  return v;
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
    if ((uint32_t)(now - tArm) >= armDelayUs) { tStart = tArm + armDelayUs; nextUs = tStart; state = RUN; Serial.println("# start"); }
    else return;
  }
  if (state != RUN && state != CW) return;
  if ((int32_t)(now - nextUs) < 0) return;
  nextUs += UPDATE_US;
  if ((int32_t)(now - nextUs) > (int32_t)(20 * UPDATE_US)) nextUs = now + UPDATE_US;
  double el = elapsedUs(tStart, now);
  if (state == CW) {
    if (el >= cwEndS * 1e6) { stopOutput("# cw ended"); return; }
    double ph = cwHz1000 * 1e-3 * el * 1e-6; ph -= floor(ph);
    dac(MID + (int)lroundf(halfCodes * sinf(6.2831853f * (float)ph)));
    return;
  }
  double u = el / symUs;                         // symbols since the start
  if (u >= (double)nPlay + span * 0.5 + 1.0) {
    lastSym = nPlay; state = DONE; dac(MID);
    Serial.print("# done symbols "); Serial.println(nPlay);
    return;
  }
  int k0 = (int)floor(u);
  lastSym = k0 < (int)nPlay ? k0 + 1 : nPlay;
  dac(MID + (int)lroundf(halfCodes * sampleAt(el)));
}
