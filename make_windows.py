# /home/ali/healthproj/src/make_windows.py
# Convert raw CSV logs into 5 s overlapping feature windows for XGBoost training.
#
# Compatible with raw files from:
#   /home/ali/Codes/collect_data.py
#
# Expected raw columns (new collector):
#   timestamp, label, subject_id,
#   bpm, spo2,
#   ax, ay, az,
#   skinT,
#   ambT_bme, ambT_ir (optional),
#   humid,
#   gsr_uS,
#   mic_v
#
# It also tolerates some older names:
#   ambT, gsr, mic

from pathlib import Path
import glob
import math

import numpy as np
import pandas as pd

# ---- SETTINGS ----
FS = 10.0                       # Hz (must match collect_data + infer)
WIN = int(5 * FS)               # 5 s window
STEP = int(2.5 * FS)            # 50% overlap

RAW_GLOB = "/home/ali/Codes/raw_0_S0_1765392935.csv"
OUT_PATH = Path("/home/ali/healthproj/data/windows/windows_features.csv")

# mic cough thresholds (must match infer)
MIC_RMS_THRESH = 0.02
MIC_MAX_THRESH = 0.05

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


def ok_flag(vals, frac=0.5):
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
        return np.nan
    if not math.isfinite(r) or r <= 0.0:
        return np.nan
    return float(20.0 * math.log10(r / ref))


# ---- LOAD RAW ----
files = sorted(glob.glob(RAW_GLOB))
if not files:
    raise SystemExit(f"No raw CSV files found matching: {RAW_GLOB}")

dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

# basic cleaning: convert obvious zeros to NaN for these columns
for col in ["bpm", "spo2", "skinT", "ambT_bme", "ambT", "humid", "gsr_uS", "gsr"]:
    if col in df.columns:
        vals = df[col].astype(float).values
        vals[vals == 0.0] = np.nan
        df[col] = vals

rows = []
n = len(df)

for start in range(0, n - WIN + 1, STEP):
    w = df.iloc[start : start + WIN]
    f = {}

    # ----- label and subject -----
    if "label" in w.columns:
        try:
            f["label"] = int(w["label"].mode().iloc[0])
        except Exception:
            f["label"] = int(w["label"].iloc[0])
    else:
        f["label"] = 0

    f["subject_id"] = (
        w["subject_id"].mode().iloc[0] if "subject_id" in w.columns else "S0"
    )

    # ----- t_center (optional, not used by model) -----
    if "timestamp" in w.columns:
        f["t_center"] = float(w["timestamp"].iloc[len(w) // 2])
    else:
        f["t_center"] = float(start / FS)

    # ----- HR / SpO2 -----
    if "bpm" in w.columns:
        bpm_vals = w["bpm"].astype(float).values
        f["hr"] = float(np.nanmean(bpm_vals))
    else:
        f["hr"] = np.nan

    if "spo2" in w.columns:
        spo2_vals = w["spo2"].astype(float).values
        f["spo2_mean"] = float(np.nanmean(spo2_vals))
        f["spo2_min"]  = float(np.nanmin(spo2_vals))
    else:
        f["spo2_mean"] = np.nan
        f["spo2_min"]  = np.nan

    # ----- IMU feats -----
    if {"ax", "ay", "az"}.issubset(w.columns):
        f.update(imu_feats(w["ax"].values, w["ay"].values, w["az"].values))
    else:
        f.update(
            {"acc_rms": np.nan, "acc_max": np.nan, "acc_var": np.nan, "jerk_max": np.nan}
        )

    # ----- Temps / humidity / GSR means -----
    # skinT
    if "skinT" in w.columns:
        f["skinT_mean"] = float(np.nanmean(w["skinT"].astype(float).values))
    else:
        f["skinT_mean"] = np.nan

    # ambient temp: prefer ambT_bme, fallback to ambT
    if "ambT_bme" in w.columns:
        amb_arr = w["ambT_bme"].astype(float).values
    elif "ambT" in w.columns:
        amb_arr = w["ambT"].astype(float).values
    else:
        amb_arr = np.array([], float)

    f["ambT_mean"] = float(np.nanmean(amb_arr)) if amb_arr.size else np.nan

    # humidity
    if "humid" in w.columns:
        f["humid_mean"] = float(np.nanmean(w["humid"].astype(float).values))
    else:
        f["humid_mean"] = np.nan

    # GSR (µS): prefer gsr_uS, fallback to gsr
    if "gsr_uS" in w.columns:
        gsr_arr = w["gsr_uS"].astype(float).values
    elif "gsr" in w.columns:
        gsr_arr = w["gsr"].astype(float).values
    else:
        gsr_arr = np.array([], float)

    f["gsr_mean"] = float(np.nanmean(gsr_arr)) if gsr_arr.size else np.nan

    # ----- skin vs ambient gradient + slope -----
    if "skinT" in w.columns and amb_arr.size:
        skin = w["skinT"].astype(float).values
        # amb_arr already chosen above
        f["skin_amb"]   = float(np.nanmean(skin) - np.nanmean(amb_arr))
        f["skin_slope"] = float(slope_per_min(skin))
    else:
        f["skin_amb"]   = np.nan
        f["skin_slope"] = np.nan

    # ----- GSR slope -----
    if gsr_arr.size:
        f["gsr_slope"] = float(slope_per_min(gsr_arr))
    else:
        f["gsr_slope"] = np.nan

    # ----- MIC features + cough flag + dB -----
    mic_col_name = "mic_v" if "mic_v" in w.columns else ("mic" if "mic" in w.columns else None)
    if mic_col_name is not None:
        mic_arr = w[mic_col_name].astype(float).values
        mic_ac  = mic_arr - np.nanmean(mic_arr)
        mic_rms = float(np.sqrt(np.nanmean(mic_ac**2)))
        mic_max = float(np.nanmax(np.abs(mic_ac)))
        mic_db  = mic_rms_to_db(mic_rms)
        cough_flag = int(mic_rms > MIC_RMS_THRESH and mic_max > MIC_MAX_THRESH)
    else:
        mic_rms = mic_max = mic_db = np.nan
        cough_flag = 0

    f["mic_rms"]    = mic_rms
    f["mic_max"]    = mic_max
    f["mic_db"]     = mic_db
    f["cough_flag"] = cough_flag

    # ----- sensor_ok flags (only live sensors) -----
    def ok_col(name):
        if name not in w.columns:
            return 0
        return ok_flag(w[name].astype(float).values)

    # No DS18B20 anymore
    f["sensor_ok_mlx"]      = ok_col("skinT")
    f["sensor_ok_bme"]      = ok_col("ambT_bme") if "ambT_bme" in w.columns else ok_col("ambT")
    f["sensor_ok_max30102"] = ok_col("bpm")
    f["sensor_ok_gsr"]      = ok_col("gsr_uS") if "gsr_uS" in w.columns else ok_col("gsr")
    f["sensor_ok_mpu"]      = ok_col("ax")
    f["sensor_ok_mic"]      = ok_col(mic_col_name) if mic_col_name is not None else 0

    rows.append(f)

out_df = pd.DataFrame(rows)

# round for readability
out_df = out_df.round(4)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUT_PATH, index=False)
print("Saved features to", OUT_PATH)
