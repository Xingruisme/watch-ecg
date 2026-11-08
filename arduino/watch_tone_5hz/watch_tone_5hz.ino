/*
  watch_tone_5hz.ino - Arduino UNO R4 (WiFi or Minima): 5 Hz, 2 mVpp test tone for the Pixel Watch ECG.

  Wiring (Siqi's setup 2):
      OUT pin --[ R_TOP 1 MOhm ]--+--[ R_BOT 470 Ohm ]-- Arduino GND -- watch BACK
                                  |
                              watch CROWN
  At the watch:  V_watch = V_pin x R_BOT / (R_TOP + R_BOT) = V_pin / 2128.7
  so 2 mVpp at the watch needs 4.257 Vpp at the pin.

  Three output modes (set MODE below):
    MODE_PWM_SINE  (default)  D9, 20 kHz PWM whose duty follows a 5 Hz sine.
                   Uses the wiring as drawn. ADD 1 uF ACROSS THE 470 Ohm (crown to back).
                   It removes the 20 kHz PWM ripple (2.35 mVpp -> ~0.04 mVpp) and
                   changes the 5 Hz level by < 0.01 %.
    MODE_DAC_SINE  A0 (the R4's 12-bit DAC), a clean sine with no ripple and no capacitor needed.
                   Move the 1 MOhm from D9 to A0 (and the thermistor of setup 1 to A1).
    MODE_SQUARE    D9 toggled 0/5 V at 5 Hz. The level is fixed by the resistors:
                   5 V / 2128.7 = 2.35 mVpp square (fundamental ~3.0 mVpp sine).
                   For exactly 2.00 mVpp use R_BOT = 400 Ohm (e.g. 470 || 2.7k).

  Serial monitor at 115200, commands end with Enter:
    o          output on (it starts ON)
    x          output off (holds the mid level, so the watch sees no step)
    f 5        frequency in Hz
    a 2        amplitude in mVpp at the watch (sine modes)
    h / l      hold the pin at its maximum / minimum (measure them with a multimeter, then use v)
    v 4.93     the measured (max - min) pin voltage, in volts; this calibrates the level
    ?          status
  Level accuracy depends on V_FULL (the default is 5.00 V; USB power on an R4 is often 4.7-5.0 V)
  and the real resistor values. Always check on the scope (+-50 mV range) before the watch.
*/

#include "pwm.h"

#define MODE_PWM_SINE 0
#define MODE_DAC_SINE 1
#define MODE_SQUARE   2

// ---------------- settings ----------------
#define MODE MODE_SQUARE

const int   OUT_PIN       = (MODE == MODE_DAC_SINE) ? DAC : 9;  // DAC = A0 on the UNO R4
float       toneHz        = 5.0f;       // tone frequency
float       targetMvpp    = 2.0f;       // at the watch (crown - back)
float       vFull         = 5.00f;      // pin swing at 100 % duty / full DAC code, volts (measure: h, l, v)
const float R_TOP         = 1.0e6f;     // series resistor, Ohm (measured value is better)
const float R_BOT         = 470.0f;     // resistor across the watch, Ohm
const float PWM_HZ        = 20000.0f;   // PWM carrier (MODE_PWM_SINE)
const uint32_t UPDATE_US  = 500;        // sine update every 0.5 ms = 2 kHz
// ------------------------------------------

#if MODE == MODE_PWM_SINE
PwmOut pwm(9);
#endif

enum State { ON, OFF, HOLD_HI, HOLD_LO };
State state = ON;
float swing = 0;            // 0..1 fraction of full scale, peak-to-peak
uint32_t t0 = 0, nextUs = 0;
bool lastSq = false;

float divider() { return R_BOT / (R_TOP + R_BOT); }

void computeSwing() {
  float pinVpp = targetMvpp / 1000.0f / divider();
  swing = pinVpp / vFull;
  if (swing > 0.98f) {
    Serial.print("! ");
    Serial.print(targetMvpp, 3);
    Serial.println(" mVpp needs more than the pin can swing; clamped");
    swing = 0.98f;
  }
}

void writeLevel(float frac) {          // frac 0..1 of full scale
  if (frac < 0) frac = 0;
  if (frac > 1) frac = 1;
#if MODE == MODE_PWM_SINE
  pwm.pulse_perc(frac * 100.0f);
#elif MODE == MODE_DAC_SINE
  analogWrite(DAC, (int)(frac * 4095.0f + 0.5f));
#else
  digitalWrite(OUT_PIN, frac >= 0.5f ? HIGH : LOW);
#endif
}

void status() {
  const char *names[] = {"PWM sine on D9", "DAC sine on A0", "square on D9"};
  Serial.print("mode: "); Serial.println(names[MODE]);
  Serial.print("state: "); Serial.println(state == ON ? "ON" : state == OFF ? "OFF" : state == HOLD_HI ? "HOLD HIGH" : "HOLD LOW");
  Serial.print("frequency Hz: "); Serial.println(toneHz, 3);
  Serial.print("divider: 1/"); Serial.println(1.0f / divider(), 1);
  Serial.print("V_FULL V: "); Serial.println(vFull, 3);
  if (MODE == MODE_SQUARE) {
    Serial.print("watch level (square, fixed by resistors) mVpp: ");
    Serial.println(vFull * divider() * 1000.0f, 3);
  } else {
    Serial.print("pin Vpp: "); Serial.println(swing * vFull, 3);
    Serial.print("watch level (calculated) mVpp: "); Serial.println(swing * vFull * divider() * 1000.0f, 3);
    Serial.print("watch DC offset (calculated) mV: "); Serial.println(0.5f * vFull * divider() * 1000.0f, 3);
  }
}

void handleSerial() {
  static char buf[32];
  static uint8_t n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c != '\n' && c != '\r') {
      if (n < sizeof(buf) - 1) buf[n++] = c;
      continue;
    }
    if (n == 0) continue;
    buf[n] = 0;
    n = 0;
    float val = atof(buf + 1);
    switch (buf[0]) {
      case 'o': state = ON; t0 = micros(); Serial.println("ON"); break;
      case 'x': state = OFF; writeLevel(MODE == MODE_SQUARE ? 0.0f : 0.5f); Serial.println("OFF"); break;
      case 'h': state = HOLD_HI; writeLevel(1.0f); Serial.println("holding MAX - measure pin to GND"); break;
      case 'l': state = HOLD_LO; writeLevel(0.0f); Serial.println("holding MIN - measure pin to GND"); break;
      case 'f': if (val > 0.1f && val < 100.0f) { toneHz = val; t0 = micros(); } status(); break;
      case 'a': if (val > 0.0f) { targetMvpp = val; computeSwing(); } status(); break;
      case 'v': if (val > 1.0f && val < 6.0f) { vFull = val; computeSwing(); } status(); break;
      default: status();
    }
  }
}

void setup() {
  Serial.begin(115200);
#if MODE == MODE_PWM_SINE
  pwm.begin(PWM_HZ, 50.0f);
#elif MODE == MODE_DAC_SINE
  analogWriteResolution(12);
#else
  pinMode(OUT_PIN, OUTPUT);
#endif
  computeSwing();
  writeLevel(MODE == MODE_SQUARE ? 0.0f : 0.5f);
  delay(1500);                 // time to open the serial monitor
  status();
  t0 = micros();
  nextUs = t0;
}

void loop() {
  handleSerial();
  if (state != ON) return;
  uint32_t now = micros();
  if ((int32_t)(now - nextUs) < 0) return;
  nextUs += UPDATE_US;
  double t = (now - t0) * 1e-6;
  double ph = t * toneHz;
  ph -= floor(ph);                                   // 0..1 cycles
#if MODE == MODE_SQUARE
  bool hi = ph < 0.5;
  if (hi != lastSq) { digitalWrite(OUT_PIN, hi ? HIGH : LOW); lastSq = hi; }
#else
  writeLevel(0.5f + 0.5f * swing * (float)sin(2.0 * PI * ph));
#endif
}
