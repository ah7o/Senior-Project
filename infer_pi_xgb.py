# /home/ali/healthproj/src/infer_pi_xgb.py
#

import json
import math
import time
from collections import deque
from pathlib import Path

import board, busio
import adafruit_mlx90614
from adafruit_bme280.basic import Adafruit_BME280_I2C
import adafruit_mpu6050
from smbus2 import SMBus
import joblib
import numpy as np
import paho.mqtt.client as mqtt

# ---- MODEL BUNDLE (fall-only binary model) ----
BUNDLE_PATH = Path("/home/ali/healthproj/models/xgb_model.joblib")

# ---- WINDOW SETTINGS (5 s windows for live monitoring) ----
FS       = 10.0      # Hz
WIN_SEC  = 5.0
STEP_SEC = 2.5

WIN_SAMPLES  = int(WIN_SEC * FS)
STEP_SAMPLES = int(STEP_SEC * FS)

# ---- FALL RULE PARAMETERS ----
# Impact detection (from IMU features in each 5 s window):
G_IMPACT_THRESH       = 20.0    # acc_max threshold for possible impact (m/s^2)
JERK_IMPACT_THRESH    = 80.0    # jerk_max threshold for possible impact

# Low-activity thresholds (post-impact stillness):
ACC_VAR_LOW           = 0.5     # variance close to stationary
ACC_RMS_LOW_CENTER    = 9.8     # around 1 g
ACC_RMS_LOW_TOL       = 0.7     # tolerance around 1 g

# How long we remember an impact and inactivity:
IMPACT_MEMORY_SEC     = 5.0     # impact stays "recent" this long
LOW_ACTIVITY_SEC_REQ  = 20.0    # inactivity duration required to declare fall

IMPACT_MEMORY_WINDOWS = max(1, int(IMPACT_MEMORY_SEC / STEP_SEC))
LOW_ACTIVITY_WINDOWS  = max(1, int(LOW_ACTIVITY_SEC_REQ / STEP_SEC))

impact_recent_counter = 0
low_activity_counter  = 0

# ---- MIC COUGH RULE ----
MIC_RMS_THRESH = 0.02      # volts, after DC removal
MIC_MAX_THRESH = 0.05      # volts, after DC removal

# ---- ADS1115 (A0=mic, A2=GSR) ----
I2C_BUS  = 1
ADS_ADDR = 0x48

REG_CONVERSION = 0x00
REG_CONFIG     = 0x01

OS_SINGLE    = 0x8000
PGA_4V096    = 0x0200
MODE_SINGLE  = 0x0100
DR_250SPS    = 0x00A0
COMP_DISABLE = 0x0003

MUX = {
    "A0": 0x4000,  # MIC
    "A2": 0x6000,  # GSR
}


def ads1115_read(channel_name: str) -> float:
    mux_bits = MUX[channel_name]
    config = (
        OS_SINGLE
        | mux_bits
        | PGA_4V096
        | MODE_SINGLE
        | DR_250SPS
        | COMP_DISABLE
    )
    with SMBus(I2C_BUS) as bus:
        bus.write_i2c_block_data(
            ADS_ADDR,
            REG_CONFIG,
            [(config >> 8) & 0xFF, config & 0xFF],
        )
        for _ in range(30):
            hi, lo = bus.read_i2c_block_data(ADS_ADDR, REG_CONFIG, 2)
            if hi & 0x80:
                break
            time.sleep(0.001)
        hi, lo = bus.read_i2c_block_data(ADS_ADDR, REG_CONVERSION, 2)
        code = (hi << 8) | lo
        if code & 0x8000:
            code -= 1 << 16
        return code * 0.000125  # volts


# ---- GSR conversion (voltage divider model) ----
def gsr_from_voltage(v: float, vref: float = 3.3, r_fixed: float = 100_000.0):
    """
    Convert Grove GSR output voltage to:
      - R_skin in ohms
      - conductance in µS

    Assumes a simple divider: Vout = Vref * R_skin / (R_fixed + R_skin)
    => R_skin = R_fixed * Vout / (Vref - Vout)
    """
    # sanity checks
    if v <= 0.0 or v >= vref:
        return float("inf"), 0.0

    R_ohm = r_fixed * (v / (vref - v))
    if R_ohm <= 0.0:
        return float("inf"), 0.0

    G_uS = 1e6 / R_ohm
    return R_ohm, G_uS


# ---- MAX30102 (PPG + SpO2) low-level ----
MAX30102_ADDR = 0x57

REG_INTR_STATUS_1 = 0x00
REG_INTR_STATUS_2 = 0x01
REG_INTR_ENABLE_1 = 0x02
REG_INTR_ENABLE_2 = 0x03
REG_FIFO_WR_PTR   = 0x04
REG_OVF_COUNTER   = 0x05
REG_FIFO_RD_PTR   = 0x06
REG_FIFO_DATA     = 0x07
REG_FIFO_CONFIG   = 0x08
REG_MODE_CONFIG   = 0x09
REG_SPO2_CONFIG   = 0x0A
REG_LED1_PA       = 0x0C
REG_LED2_PA       = 0x0D

max_bus = None  # shared I2C bus for MAX30102


def max_write_reg(reg, value):
    global max_bus
    max_bus.write_byte_data(MAX30102_ADDR, reg, value)


def max_read_reg(reg, length=1):
    global max_bus
    return max_bus.read_i2c_block_data(MAX30102_ADDR, reg, length)


def max30102_init():
    max_write_reg(REG_MODE_CONFIG, 0x40)  # reset
    time.sleep(0.1)
    try:
        chip_id = max_read_reg(0xFF, 1)[0]
        print(f"MAX30102 Chip ID = 0x{chip_id:02X}")
    except Exception:
        print("MAX30102: cannot read chip ID.")

    max_write_reg(REG_INTR_ENABLE_1, 0xC0)
    max_write_reg(REG_INTR_ENABLE_2, 0x00)

    max_write_reg(REG_FIFO_WR_PTR, 0x00)
    max_write_reg(REG_OVF_COUNTER, 0x00)
    max_write_reg(REG_FIFO_RD_PTR, 0x00)

    max_write_reg(REG_FIFO_CONFIG, 0x4F)  # avg=4, almost full=17, no rollover
    max_write_reg(REG_MODE_CONFIG, 0x03)  # SpO2 mode
    max_write_reg(REG_SPO2_CONFIG, 0x27)  # 100 Hz, 411 µs
    max_write_reg(REG_LED1_PA, 0x24)      # ~7 mA
    max_write_reg(REG_LED2_PA, 0x24)      # ~7 mA
    print("MAX30102 initialized.")


def max30102_read_fifo_sample():
    _ = max_read_reg(REG_INTR_STATUS_1, 1)
    _ = max_read_reg(REG_INTR_STATUS_2, 1)
    d = max_read_reg(REG_FIFO_DATA, 6)
    red = (d[0] << 16 | d[1] << 8 | d[2]) & 0x03FFFF
    ir  = (d[3] << 16 | d[4] << 8 | d[5]) & 0x03FFFF
    return red, ir


class PulseOxProcessor:
    """Simple BPM + SpO2 processing from RED/IR (non-medical)."""

    def __init__(self, window_sec=8.0, min_hr_bpm=45.0, max_hr_bpm=130.0):
        self.window_sec = window_sec
        self.min_hr = min_hr_bpm
        self.max_hr = max_hr_bpm
        self.samples = []   # list of (t, red, ir)

    def add_sample(self, t, red, ir):
        self.samples.append((t, float(red), float(ir)))
        t_min = t - self.window_sec
        while self.samples and self.samples[0][0] < t_min:
            self.samples.pop(0)

    def _bandpass_ir(self, ir):
        n = len(ir)
        if n == 0:
            return ir
        mean_ir = sum(ir) / n
        x = [v - mean_ir for v in ir]
        window = 5
        if n <= window:
            return x
        y = [0.0] * n
        s = sum(x[0:window])
        y[window - 1] = s / window
        for i in range(window, n):
            s += x[i] - x[i - window]
            y[i] = s / window
        for i in range(window - 1):
            y[i] = y[window - 1]
        return y

    def _compute_hr(self, ts, ir):
        n = len(ir)
        if n < 25:
            return None
        ir_f = self._bandpass_ir(ir)
        mean_ir = sum(ir_f) / n
        var_ir = sum((x - mean_ir) ** 2 for x in ir_f) / n
        std_ir = math.sqrt(var_ir) if var_ir > 0 else 0.0
        if std_ir < 1e-4:
            return None
        thresh = mean_ir + 0.5 * std_ir
        peak_times = []
        last_peak_t = None
        min_rr = 0.5  # seconds (~120 bpm max)
        for i in range(1, n - 1):
            if ir_f[i] > ir_f[i - 1] and ir_f[i] >= ir_f[i + 1] and ir_f[i] > thresh:
                t_i = ts[i]
                if last_peak_t is None or (t_i - last_peak_t) >= min_rr:
                    peak_times.append(t_i)
                    last_peak_t = t_i
        if len(peak_times) < 2:
            return None
        intervals = [
            peak_times[i + 1] - peak_times[i]
            for i in range(len(peak_times) - 1)
            if peak_times[i + 1] > peak_times[i]
        ]
        intervals = [dt for dt in intervals if dt > 0]
        if not intervals:
            return None
        intervals.sort()
        mid = len(intervals) // 2
        if len(intervals) % 2 == 1:
            rr = intervals[mid]
        else:
            rr = 0.5 * (intervals[mid - 1] + intervals[mid])
        if rr <= 0:
            return None
        bpm = 60.0 / rr
        if bpm < self.min_hr or bpm > self.max_hr:
            return None
        return bpm

    def _compute_spo2(self, red, ir):
        n = len(red)
        if n < 25:
            return None
        dc_red = sum(red) / n
        dc_ir  = sum(ir) / n
        if dc_red <= 0 or dc_ir <= 0:
            return None
        ac_red_sq = sum((x - dc_red) ** 2 for x in red) / n
        ac_ir_sq  = sum((x - dc_ir) ** 2 for x in ir) / n
        ac_red = math.sqrt(ac_red_sq) if ac_red_sq > 0 else 0.0
        ac_ir  = math.sqrt(ac_ir_sq) if ac_ir_sq > 0 else 0.0
        if ac_red / dc_red < 0.001 or ac_ir / dc_ir < 0.001:
            return None
        R = (ac_red / dc_red) / (ac_ir / dc_ir)
        if R < 0.2 or R > 2.0:
            return None
        spo2 = 110.0 - 25.0 * R
        if spo2 < 0 or spo2 > 100:
            return None
        return spo2

    def compute(self):
        n = len(self.samples)
        if n < 25:
            return None, None
        ts  = [s[0] for s in self.samples]
        red = [s[1] for s in self.samples]
        ir  = [s[2] for s in self.samples]
        bpm  = self._compute_hr(ts, ir)
        spo2 = self._compute_spo2(red, ir)
        if bpm is None:
            spo2 = None
        return bpm, spo2


# ---- Feature helpers (match make_windows.py) ----
def imu_feats(ax, ay, az):
    ax = np.asarray(ax, float)
    ay = np.asarray(ay, float)
    az = np.asarray(az, float)
    res = np.sqrt(ax**2 + ay**2 + az**2)
    jerk = np.diff(res, prepend=res[0]) * FS
    return {
        "acc_rms": float(np.sqrt(np.mean(res**2))),
        "acc_max": float(np.max(res)),
        "acc_var": float(np.var(res)),
        "jerk_max": float(np.max(np.abs(jerk))),
    }


def slope_per_min(y):
    y = np.asarray(y, float)
    if len(y) < 2:
        return 0.0
    n = len(y)
    lo = int(n * 0.2)
    hi = int(n * 0.8)
    if hi <= lo:
        return 0.0
    return 60.0 * (np.nanmean(y[hi:]) - np.nanmean(y[:lo])) / (n / FS)


def ok_flag(vals, frac=0.8):
    vals = np.asarray(vals, float)
    finite = np.isfinite(vals)
    if finite.sum() < frac * len(vals):
        return 0
    if np.nanmax(vals) == np.nanmin(vals):
        return 0
    return 1


def mic_rms_to_db(rms, ref=0.01):
    """Convert mic RMS voltage to dB relative to ref (default 10 mV)."""
    try:
        r = float(rms)
    except Exception:
        return float("nan")
    if not math.isfinite(r) or r <= 0.0:
        return float("nan")
    return 20.0 * math.log10(r / ref)


# ---- ThingsBoard setup ----
THINGSBOARD_HOST = "demo.thingsboard.io"
ACCESS_TOKEN = "85qlk0u9e26hp3gmwtmp"

tb_client = mqtt.Client()
tb_client.username_pw_set(ACCESS_TOKEN)
tb_client.connect(THINGSBOARD_HOST, 1883, 60)
tb_client.loop_start()


def clean_value(v):
    try:
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
    except Exception:
        return None
    return v


def send_tb_telemetry(data: dict) -> None:
    """Publish telemetry to ThingsBoard and print debug info."""
    try:
        clean = {k: clean_value(v) for k, v in data.items()}
        info = tb_client.publish("v1/devices/me/telemetry", json.dumps(clean), qos=1)
        # info.wait_for_publish()  # optional, can uncomment if needed
        print("Sent TB:", clean, flush=True)
    except Exception as e:
        print("ThingsBoard send error:", repr(e), flush=True)



# ---- Init I2C sensors ----
def init_sensors():
    global max_bus
    i2c = busio.I2C(board.SCL, board.SDA)

    # MLX90614 (skin temp)
    try:
        mlx = adafruit_mlx90614.MLX90614(i2c)
    except Exception:
        mlx = None

    # BME280 (ambient + humidity)
    try:
        bme = Adafruit_BME280_I2C(i2c, address=0x76)
        print("BME280 OK at 0x76")
    except Exception as e:
        print("BME280 init failed:", e)
        bme = None

    # MPU6050
    try:
        mpu = adafruit_mpu6050.MPU6050(i2c)
    except Exception:
        mpu = None

    # MAX30102
    max_bus = None
    max_ok = False
    try:
        max_bus = SMBus(I2C_BUS)
        _ = max_bus.read_i2c_block_data(MAX30102_ADDR, REG_INTR_STATUS_1, 1)
        max30102_init()
        max_ok = True
    except Exception as e:
        print("MAX30102 init failed:", e)
        if max_bus is not None:
            max_bus.close()
        max_bus = None
        max_ok = False

    print("Sensors detected:")
    print("  MLX90614:", "OK" if mlx else "NONE")
    print("  BME280  :", "OK" if bme else "NONE")
    print("  MPU6050 :", "OK" if mpu else "NONE")
    print("  MAX30102:", "OK" if max_ok else "NONE")
    print("  ADS1115 :", "assumed at 0x48 (A0=mic, A2=GSR)")

    return mlx, bme, mpu, max_ok


def main():
    global impact_recent_counter, low_activity_counter, max_bus

    if not BUNDLE_PATH.exists():
        raise SystemExit(f"Fall model bundle not found: {BUNDLE_PATH}")
    obj = joblib.load(BUNDLE_PATH)

    # Expect a bundle dict
    if isinstance(obj, dict) and "model" in obj and "features" in obj:
        model      = obj["model"]
        feat_names = obj["features"]
        label_map  = obj.get("label_map", {0: "normal", 1: "fall_like"})
    else:
        # Fallback: treat obj as bare model, and use default feature names
        model      = obj
        feat_names = ["acc_rms", "acc_max", "acc_var", "jerk_max"]
        label_map  = {0: "normal", 1: "fall_like"}

    mlx, bme, mpu, max_ok = init_sensors()
    pulseox = PulseOxProcessor(window_sec=8.0)

    print("\nRunning live hybrid inference. Ctrl+C to stop.\n")

    # Sliding buffers
    buf = {
        "timestamp": deque(maxlen=WIN_SAMPLES),
        "bpm": deque(maxlen=WIN_SAMPLES),
        "spo2": deque(maxlen=WIN_SAMPLES),
        "ax": deque(maxlen=WIN_SAMPLES),
        "ay": deque(maxlen=WIN_SAMPLES),
        "az": deque(maxlen=WIN_SAMPLES),
        "skinT": deque(maxlen=WIN_SAMPLES),
        "ambT": deque(maxlen=WIN_SAMPLES),   # from BME (preferred)
        "humid": deque(maxlen=WIN_SAMPLES),
        "gsr": deque(maxlen=WIN_SAMPLES),    # µS
        "mic": deque(maxlen=WIN_SAMPLES),    # volts
    }

    sample_count = 0
    PERIOD = 1.0 / FS
    next_t = time.time()

    try:
        while True:
            ts = time.time()

            # MAX30102: multiple FIFO reads per loop for smoother HR/SpO2
            bpm = spo2 = None
            if max_bus is not None:
                try:
                    for _ in range(3):  # ~30 Hz PPG inside 10 Hz loop
                        red, ir = max30102_read_fifo_sample()
                        pulseox.add_sample(ts, red, ir)
                    bpm, spo2 = pulseox.compute()
                except Exception:
                    bpm = spo2 = None

            bpm_val  = float(bpm)  if bpm  is not None else float("nan")
            spo2_val = float(spo2) if spo2 is not None else float("nan")

            # MPU6050 (acc only for logic)
            if mpu:
                try:
                    ax, ay, az = mpu.acceleration
                except Exception:
                    ax = ay = az = float("nan")
            else:
                ax = ay = az = float("nan")

            # MLX90614: skin temperature only
            if mlx:
                try:
                    skinT = float(mlx.object_temperature)
                except Exception:
                    skinT = float("nan")
            else:
                skinT = float("nan")

            # BME280: ambient temp + humidity
            if bme:
                try:
                    ambT  = float(bme.temperature)
                    humid = float(bme.humidity)
                except Exception:
                    ambT = humid = float("nan")
            else:
                ambT = humid = float("nan")

            # ADS1115: mic + gsr
            try:
                gsr_v = ads1115_read("A0")
                mic_v = ads1115_read("A2")
            except Exception:
                mic_v = 0.0
                gsr_v = 0.0
            _, gsr_GuS = gsr_from_voltage(gsr_v)

            # update buffers
            buf["timestamp"].append(ts)
            buf["bpm"].append(bpm_val)
            buf["spo2"].append(spo2_val)
            buf["ax"].append(ax)
            buf["ay"].append(ay)
            buf["az"].append(az)
            buf["skinT"].append(skinT)
            buf["ambT"].append(ambT)
            buf["humid"].append(humid)
            buf["gsr"].append(gsr_GuS)
            buf["mic"].append(mic_v)

            sample_count += 1

            # defaults for printing/telemetry
            state          = "normal"
            p_fall         = 0.0
            cough_flag     = 0
            rule_heat_flag = False

            hr_win = spo2_mean = skin_mean = amb_mean = humid_mean = acc_rms = np.nan
            mic_rms = mic_max = mic_db = np.nan
            fall_rule_flag = False
            ml_fall_flag   = False
            ml_state       = "normal"

            # only process a window when we have enough samples and reached step
            if len(buf["timestamp"]) == WIN_SAMPLES and sample_count >= STEP_SAMPLES:
                sample_count = 0  # step

                w = {k: np.array(v, float) for k, v in buf.items()}
                feats = {}

                # HR / SpO2 (window averages)
                bpm_vals  = w["bpm"]
                spo2_vals = w["spo2"]
                feats["hr"]        = float(np.nanmean(bpm_vals))  if bpm_vals.size else np.nan
                feats["spo2_mean"] = float(np.nanmean(spo2_vals)) if spo2_vals.size else np.nan
                feats["spo2_min"]  = float(np.nanmin(spo2_vals))  if spo2_vals.size else np.nan

                # IMU features
                imu_dict = imu_feats(w["ax"], w["ay"], w["az"])
                feats.update(imu_dict)

                acc_rms   = imu_dict["acc_rms"]
                acc_max   = imu_dict["acc_max"]
                acc_var   = imu_dict["acc_var"]
                jerk_max  = imu_dict["jerk_max"]

                # Temps / humidity / gsr means
                for col in ["skinT", "ambT", "humid", "gsr"]:
                    arr = w[col] if col in w else np.array([])
                    feats[f"{col}_mean"] = float(np.nanmean(arr)) if arr.size else np.nan

                # skin vs ambient
                skin = w["skinT"]
                amb  = w["ambT"]
                if skin.size and amb.size:
                    feats["skin_amb"]   = float(np.nanmean(skin) - np.nanmean(amb))
                    feats["skin_slope"] = float(slope_per_min(skin))
                else:
                    feats["skin_amb"]   = np.nan
                    feats["skin_slope"] = np.nan

                gsr_arr = w["gsr"]
                feats["gsr_slope"] = float(slope_per_min(gsr_arr)) if gsr_arr.size else np.nan

                # MIC features + cough flag + dB
                mic_arr = w["mic"]
                if mic_arr.size:
                    mic_ac  = mic_arr - np.nanmean(mic_arr)
                    mic_rms = float(np.sqrt(np.nanmean(mic_ac**2)))
                    mic_max = float(np.nanmax(np.abs(mic_ac)))
                    mic_db  = mic_rms_to_db(mic_rms)
                    cough_flag = 1 if (mic_rms > MIC_RMS_THRESH and mic_max > MIC_MAX_THRESH) else 0
                else:
                    mic_rms = mic_max = mic_db = np.nan
                    cough_flag = 0

                feats["mic_rms"]    = mic_rms
                feats["mic_max"]    = mic_max
                feats["mic_db"]     = mic_db
                feats["cough_flag"] = cough_flag

                # sensor_ok flags
                feats["sensor_ok_mlx"]      = ok_flag(w["skinT"])
                feats["sensor_ok_bme"]      = ok_flag(w["ambT"])
                feats["sensor_ok_max30102"] = ok_flag(w["bpm"])
                feats["sensor_ok_gsr"]      = ok_flag(w["gsr"])
                feats["sensor_ok_mpu"]      = ok_flag(w["ax"])
                feats["sensor_ok_mic"]      = ok_flag(w["mic"])

                # ---- IMPACT DETECTION (ML) ----
                x_vec = [feats.get(name, np.nan) for name in feat_names]
                X_window = np.asarray(x_vec, float).reshape(1, -1)

                try:
                    proba = model.predict_proba(X_window)[0]
                    # for binary model: class 1 = fall_like
                    p_fall = float(proba[1])
                    ml_pred_class = int(np.argmax(proba))  # 0 or 1
                    ml_state = label_map.get(ml_pred_class, f"class_{ml_pred_class}")
                    ml_fall_flag = (ml_pred_class == 1)
                except Exception as e:
                    print("Model inference error:", e)
                    p_fall = 0.0
                    ml_state = "model_error"
                    ml_fall_flag = False

                # Impact now: either ML says fall, or raw features show high impact.
                impact_now = (
                    ml_fall_flag or
                    (acc_max > G_IMPACT_THRESH) or
                    (jerk_max > JERK_IMPACT_THRESH)
                )

                if impact_now:
                    impact_recent_counter = IMPACT_MEMORY_WINDOWS
                else:
                    impact_recent_counter = max(impact_recent_counter - 1, 0)

                # ---- LOW ACTIVITY DETECTION ----
                low_activity_now = (
                    (acc_var < ACC_VAR_LOW) and
                    (abs(acc_rms - ACC_RMS_LOW_CENTER) < ACC_RMS_LOW_TOL)
                )

                if low_activity_now:
                    low_activity_counter += 1
                else:
                    low_activity_counter = 0

                fall_rule_flag = (
                    (impact_recent_counter > 0) and
                    (low_activity_counter >= LOW_ACTIVITY_WINDOWS)
                )

                # For reference / printing:
                hr_win      = feats["hr"]
                spo2_mean   = feats["spo2_mean"]
                skin_mean   = feats["skinT_mean"]
                amb_mean    = feats["ambT_mean"]
                humid_mean  = feats["humid_mean"]
                gsr_mean    = feats["gsr_mean"]

                # ---- HEAT STRAIN RULE (non-medical) ----
                try:
                    if (
                        math.isfinite(amb_mean)
                        and math.isfinite(humid_mean)
                        and math.isfinite(hr_win)
                        and math.isfinite(skin_mean)
                    ):
                        if (
                            amb_mean  >= 30.0 and
                            humid_mean >= 60.0 and
                            hr_win    >= 100.0 and
                            skin_mean >= 35.0
                        ):
                            rule_heat_flag = True
                except Exception:
                    rule_heat_flag = False

                # ---- FINAL STATE LOGIC ----
                state = "normal"

                # Highest priority: fall rule
                if fall_rule_flag:
                    state = "possible_fall"
                # Next: heat strain
                elif rule_heat_flag:
                    state = "possible_heat_strain"
                # Next: cough events (only override normal)
                elif cough_flag == 1:
                    state = "possible_cough_event"
                # Else, if ML strongly says fall but inactivity not yet met
                elif ml_fall_flag:
                    state = "impact_only"

                # DEBUG: prove we reached the window block
                print("DEBUG: building tb_data for ThingsBoard", flush=True)
                
                # telemetry to ThingsBoard
                               # telemetry to ThingsBoard (clean set)
                tb_data = {
                    # overall state
                    "state": state,
                    "p_fall": p_fall,
                    "ml_state": ml_state,

                    # main mean values
                    "hr_mean": hr_win,
                    "spo2_mean": spo2_mean,
                    "skin_temp_mean": skin_mean,
                    "amb_temp_mean": amb_mean,
                    "humidity_mean": humid_mean,
                    "gsr_mean": gsr_mean,

                    # motion features
                    "acc_rms": acc_rms,
                    "jerk_max": jerk_max,

                    # event flags for alarms
                    "flag_cough": int(cough_flag),
                    "flag_heat": int(rule_heat_flag),
                    "flag_fall": int(fall_rule_flag),
                }
                send_tb_telemetry(tb_data)


                print(
                    f"{state:22s} "
                    f"p_fall={p_fall:5.3f}  "
                    f"hr={hr_win:5.1f}  spo2={spo2_mean:5.1f}  "
                    f"skin={skin_mean:4.1f}C  amb={amb_mean:4.1f}C  "
                    f"humid={humid_mean:5.1f}%  "
                    f"acc_rms={acc_rms:5.2f}  acc_max={acc_max:6.2f}  "
                    f"jerk_max={jerk_max:7.2f}  "
                    f"impact_mem={impact_recent_counter:2d}  "
                    f"low_act_win={low_activity_counter:2d}  "
                    f"heat_flag={int(rule_heat_flag)}  "
                    f"cough_flag={cough_flag}  "
                    f"mic_db={mic_db:6.1f}",
                    flush=True,
                )

            # pacing at ~FS Hz
            next_t += PERIOD
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.time()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if max_bus is not None:
            max_bus.close()


if __name__ == "__main__":
    main()
