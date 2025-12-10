# /home/ali/healthproj/src/train_xgb.py
# Train XGBoost on window features and save a bundle:
#   {
#       "model":     XGBClassifier(...),
#       "features":  [list of feature names in order],
#       "label_map": {class_idx: label_str, ...}
#   }
#
# Works for:
#   - Full multi-class: labels 0,1,2,3
#   - Fall-only: labels only 0 and 3 (remapped to 0/1)

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
WIN_PATH   = Path("/home/ali/healthproj/data/windows/windows_featuresfinal.csv")
MODEL_PATH = Path("/home/ali/healthproj/models/xgb_model.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Load windows
# ------------------------------------------------------------------
if not WIN_PATH.exists():
    raise SystemExit(f"Window file not found: {WIN_PATH}")

df = pd.read_csv(WIN_PATH)
print(f"Loaded windows: {len(df)}")

if "label" not in df.columns:
    raise SystemExit("No 'label' column found in windows file.")

labels_present = sorted(set(df["label"].unique()))
print("Labels present in file:", labels_present)

# ------------------------------------------------------------------
# Decide training mode: fall-only vs multi-class
# ------------------------------------------------------------------
labels_set = set(labels_present)

# Case 1: fall-only (only 0 and 3 exist)
if labels_set.issubset({0, 3}):
    print("\nDetected fall-only dataset (labels 0 and 3).")
    print("Training binary normal-vs-fall_like model.\n")

    # Remap: 0 -> 0 (normal), 3 -> 1 (fall_like)
    y = df["label"].map(lambda v: 0 if v == 0 else 1).values.astype(int)
    label_map = {0: "normal", 1: "fall_like"}

# Case 2: multi-class (0,1,2,3 or subset including at least one non-0/3)
else:
    print("\nDetected multi-class dataset.")
    print("Training on original labels (0=normal,1=heat,2=breathing,3=fall_like).\n")

    y = df["label"].values.astype(int)

    # You can adjust this if you use different label semantics
    label_map = {
        0: "normal",
        1: "heat_risk",
        2: "breathing_risk",
        3: "fall_like",
    }

# ------------------------------------------------------------------
# Select features (drop non-feature columns)
# ------------------------------------------------------------------
NON_FEATURE_COLS = {"label", "subject_id", "t_center"}

feat_names = [c for c in df.columns if c not in NON_FEATURE_COLS]
print(f"Number of features: {len(feat_names)}")
print("Example features:", feat_names[:10])

X = df[feat_names].values.astype(float)

# ------------------------------------------------------------------
# Train / test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ------------------------------------------------------------------
# Define and train XGBoost model
# ------------------------------------------------------------------
num_classes = len(sorted(set(y_train)))

if num_classes == 2:
    # Binary logistic
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
else:
    # Multi-class softmax
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

print("\nTraining model...")
clf.fit(X_train, y_train)
print("Training done.\n")

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
y_pred = clf.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred, digits=3))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------------
# Save bundle (model + features + label_map)
# ------------------------------------------------------------------
bundle = {
    "model": clf,
    "features": feat_names,
    "label_map": label_map,
}

joblib.dump(bundle, MODEL_PATH)
print(f"\nSaved model bundle to: {MODEL_PATH}")
