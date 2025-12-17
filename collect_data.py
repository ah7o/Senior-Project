# /home/ali/Codes/collect_data.py
# Collect raw data for ML training:
#   - MAX30102 (HR, SpO2 via PulseOxProcessor)
#   - MPU6050 (ax, ay, az)
#   - MLX90614 (skinT, ambT IR)
#   - BME280 (ambT_bme, humidity)
#   - ADS1115: A0 = mic (MAX9814), A2 = GSR
#
# Saves CSV: /home/ali/Codes/raw_<label>_<subject>_<epoch>.csv

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import board, busio
import adafruit_mlx90614
from adafruit_bme280.basic import Adafruit_BME280_I2C
import adafruit_mpu6050
from smbus2 import SMBus

# --------- MAX30102 low-level (same as monitor) ----------
I2C_BUS = 1
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

max_bus = None


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
    max_write_reg(REG_FIFO_CONFIG, 0x4F)  # avg=4, almost full=17
    max_write_reg(REG_MODE_CONFIG, 0x03)  # SpO2 mode
    max_write_reg(REG_SPO2_CONFIG, 0x27)  # 100 Hz, 411 µs
    max_write_reg(REG_LED1_PA, 0x24)      # ~7 mA
    max_write_reg(REG_LED2_PA, 0x24)      # ~7 mA
    print("MAX30102 initialized for collect.")


def max30102_read_fifo_sample():
    _ = max_read_reg(REG_INTR_STATUS_1, 1)
    _ = max_read_reg(REG_INTR_STATUS_2, 1)
    d = max_read_reg(REG_FIFO_DATA, 6)
    red = (d[0] << 16 | d[1] << 8 | d[2]) & 0x03FFFF
    ir  = (d[3] << 16 | d[4] << 8 | d[5]) & 0x03FFFF
    return red, ir


class PulseOxProcessor:
    def __init__(self, window_sec=8.0, min_hr_bpm=45.0, max_hr_bpm=130.0):
        self.window_sec = window_sec
        self.min_hr = min_hr_bpm
        self.max_hr = max_hr_bpm
        self.samples = []   # (t, red, ir)

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
        min_rr = 0.5
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


# ---------- ADS1115 (mic, GSR) ----------
ADS_ADDR = 0x48

REG_CONVERSION = 0x00
REG_CONFIG     = 0x01

OS_SINGLE    = 0x8000
PGA_4V096    = 0x0200
MODE_SINGLE  = 0x0100
DR_250SPS    = 0x00A0
COMP_DISABLE = 0x0003

MUX = {
    "A0": 0x4000,  # mic
    "A2": 0x6000,  # gsr
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
        return code * 0.000125


def gsr_from_voltage(v: float, vref: float = 3.3, calib: int = 700):
    """
    Grove GSR modeled as simple divider:

        Vout = Vcc * R_skin / (R_fixed + R_skin)
        => R_skin = R_fixed * Vout / (Vcc - Vout)

    Returns (R_ohm, G_uS).
    """
    R_FIXED = 10_000.0  # ≈10 kΩ series resistor on Grove GSR

    if v <= 0.0 or v >= vref:
        return float("inf"), 0.0

    R_ohm = R_FIXED * (v / (vref - v))
    if R_ohm <= 0.0:
        return float("inf"), 0.0

    G_uS = 1e6 / R_ohm
    return R_ohm, G_uS


def init_sensors():
    global max_bus
    i2c = busio.I2C(board.SCL, board.SDA)

    try:
        mlx = adafruit_mlx90614.MLX90614(i2c)
    except Exception:
        mlx = None

    try:
        bme = Adafruit_BME280_I2C(i2c, address=0x76)
    except Exception:
        bme = None

    try:
        mpu = adafruit_mpu6050.MPU6050(i2c)
    except Exception:
        mpu = None

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

    print("Sensors:")
    print("  MLX90614:", "OK" if mlx else "NONE")
    print("  BME280  :", "OK" if bme else "NONE")
    print("  MPU6050 :", "OK" if mpu else "NONE")
    print("  MAX30102:", "OK" if max_ok else "NONE")
    print("  ADS1115 :", "assumed at 0x48 (A0=GSR, A2=mic)")

    return mlx, bme, mpu, max_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, required=True, help="class label (0=normal,1=heat,2=breathing,3=fall)")
    parser.add_argument("--subject_id", type=str, default="S0")
    parser.add_argument("--duration", type=float, default=300.0, help="seconds")
    args = parser.parse_args()

    mlx, bme, mpu, max_ok = init_sensors()
    pulseox = PulseOxProcessor(window_sec=8.0)

    FS = 10.0
    PERIOD = 1.0 / FS

    out_dir = Path("/home/ali/Codes")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"raw_{args.label}_{args.subject_id}_{int(time.time())}.csv"

    fields = [
        "timestamp", "label", "subject_id",
        "bpm", "spo2",
        "ax", "ay", "az",
        "skinT", "ambT_ir",
        "ambT_bme", "humid",
        "gsr_uS",
        "mic_v",
    ]

    print(f"Writing to: {fname}")
    print("Ctrl+C to stop early.")

    t_start = time.time()
    next_t = t_start

    last_hr = None
    last_spo2 = None

    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        try:
            while True:
                now = time.time()
                if now - t_start >= args.duration:
                    break

                ts = now

                # MAX30102
                if max_ok and max_bus is not None:
                    try:
                        for _ in range(3):  # a few samples per loop
                            red, ir = max30102_read_fifo_sample()
                            pulseox.add_sample(ts, red, ir)
                    except Exception:
                        pass

                hr, spo2 = pulseox.compute()
                if hr is not None:
                    last_hr = float(hr)
                if spo2 is not None:
                    last_spo2 = float(spo2)

                bpm_val  = last_hr   if last_hr   is not None else float("nan")
                spo2_val = last_spo2 if last_spo2 is not None else float("nan")

                # MPU6050
                if mpu:
                    try:
                        ax, ay, az = mpu.acceleration
                    except Exception:
                        ax = ay = az = float("nan")
                else:
                    ax = ay = az = float("nan")

                # MLX90614
                if mlx:
                    try:
                        skinT = float(mlx.object_temperature)
                        ambT_ir = float(mlx.ambient_temperature)
                    except Exception:
                        skinT = ambT_ir = float("nan")
                else:
                    skinT = ambT_ir = float("nan")

                # BME280
                if bme:
                    try:
                        ambT_bme = float(bme.temperature)
                        humid = float(bme.humidity)
                    except Exception:
                        ambT_bme = humid = float("nan")
                else:
                    ambT_bme = humid = float("nan")

                # ADS1115: mic + GSR
                try:
                    mic_v = ads1115_read("A2")
                    gsr_v = ads1115_read("A0")
                except Exception:
                    mic_v = 0.0
                    gsr_v = 0.0
                _, gsr_uS = gsr_from_voltage(gsr_v)

                row = {
                    "timestamp": ts,
                    "label": args.label,
                    "subject_id": args.subject_id,
                    "bpm": bpm_val,
                    "spo2": spo2_val,
                    "ax": ax,
                    "ay": ay,
                    "az": az,
                    "skinT": skinT,
                    "ambT_ir": ambT_ir,
                    "ambT_bme": ambT_bme,
                    "humid": humid,
                    "gsr_uS": gsr_uS,
                    "mic_v": mic_v,
                }
                writer.writerow(row)

                next_t += PERIOD
                sleep_for = next_t - time.time()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_t = time.time()

        except KeyboardInterrupt:
            print("\nStopped by user.")

    if max_bus is not None:
        max_bus.close()
    print("Done.")


if __name__ == "__main__":
    main()
