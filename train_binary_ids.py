from pathlib import Path
import gc
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

DATA_DIR = Path("pcaps")
OUT_DIR = Path("output/binary_xgb")

# At most this many benign and attack flows per source file.
MAX_PER_CLASS_PER_FILE = 100_000
RANDOM_STATE = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
frames = []
common_columns = None

files = sorted(DATA_DIR.glob("*.parquet"))

if not files:
    raise FileNotFoundError(f"No Parquet files found in {DATA_DIR}")

# Load and sample each source file separately.
for path in files:
    print(f"\nReading {path.name}", flush=True)
    df = pd.read_parquet(path)

    label_col = next(
        (c for c in df.columns if c.lower() == "label"), None
    )
    if label_col is None:
        raise ValueError(f"No Label column in {path.name}")

    labels = df[label_col].astype("string").str.strip()
    valid = labels.notna() & labels.ne("")
    df = df.loc[valid].copy()
    labels = labels.loc[valid]

    # Binary target: Benign=0, every other label=1.
    df["target"] = (labels.str.casefold() != "benign").astype("int8")
    print("Original labels:")
    print(labels.value_counts().to_string())

    # Remove identifiers that could encourage memorization.
    excluded = {"label", "target"}
    for col in df.columns:
        name = col.lower().strip()
        if (
            "flow id" in name
            or "timestamp" in name
            or "source ip" in name
            or "destination ip" in name
            or name in {"src ip", "dst ip", "src address", "dst address"}
        ):
            excluded.add(col)

    candidates = [c for c in df.columns if c not in excluded]
    if common_columns is None:
        common_columns = set(candidates)
    else:
        common_columns &= set(candidates)

    parts = []
    for target_value in (0, 1):
        part = df.loc[df["target"] == target_value]
        n = min(len(part), MAX_PER_CLASS_PER_FILE)
        if n:
            parts.append(
                part.sample(n=n, random_state=RANDOM_STATE)
            )

    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled[candidates + ["target"]]
    frames.append(sampled)

    print(
        f"Sampled {len(sampled):,} rows "
        f"(benign={(sampled.target == 0).sum():,}, "
        f"attack={(sampled.target == 1).sum():,})"
    )

    del df, labels, sampled, parts
    gc.collect()

# Use only columns present in every source dataset.
feature_names = sorted(common_columns)
data = pd.concat(frames, ignore_index=True)
del frames
gc.collect()

# Convert candidate features to numeric and discard unusable columns.
for col in feature_names:
    data[col] = pd.to_numeric(data[col], errors="coerce")

feature_names = [
    col for col in feature_names
    if data[col].notna().any()
    and np.isfinite(data[col].dropna()).all()
]

X = data[feature_names].replace([np.inf, -np.inf], np.nan)
y = data["target"].astype("int8")

# Remove columns that are entirely missing after conversion.
feature_names = [c for c in feature_names if X[c].notna().any()]
X = X[feature_names]

# XGBoost supports missing feature values directly.
print(f"\nTotal sampled rows: {len(data):,}")
print(f"Benign: {(y == 0).sum():,}")
print(f"Attack: {(y == 1).sum():,}")
print(f"Features: {len(feature_names)}")

# 70% train, 15% validation, 15% test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=RANDOM_STATE, stratify=y_temp
)

model = xgb.XGBClassifier(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

print("\nTraining XGBoost...", flush=True)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=25,
)

# Select a threshold using validation data, not the test set.
val_prob = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, val_prob)

if len(thresholds):
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_idx = int(np.argmax(f1_values))
    threshold = float(thresholds[best_idx])
else:
    threshold = 0.5

test_prob = model.predict_proba(X_test)[:, 1]

results = {}
for name, cutoff in [
    ("default_0.5", 0.5),
    ("validation_tuned", threshold),
]:
    pred = (test_prob >= cutoff).astype("int8")
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    results[name] = {
        "threshold": cutoff,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision_attack": float(precision_score(
            y_test, pred, zero_division=0
        )),
        "recall_attack": float(recall_score(
            y_test, pred, zero_division=0
        )),
        "f1_attack": float(f1_score(y_test, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, pred,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            output_dict=True,
            zero_division=0,
        ),
    }

    print(f"\n=== {name} (threshold={cutoff:.6f}) ===")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(cm)
    print(classification_report(
        y_test, pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        zero_division=0,
    ))
    print(f"Balanced accuracy: {results[name]['balanced_accuracy']:.4f}")
    print(f"False-positive rate: {results[name]['false_positive_rate']:.4f}")

# Save model and the exact feature order required for inference.
joblib.dump(model, OUT_DIR / "xgb_binary_model.joblib")
(OUT_DIR / "features.txt").write_text(
    "\n".join(feature_names) + "\n"
)
(OUT_DIR / "metrics.json").write_text(
    json.dumps(results, indent=2)
)

pd.DataFrame({
    "actual": y_test.to_numpy(),
    "attack_probability": test_prob,
    "prediction_default": (test_prob >= 0.5).astype("int8"),
    "prediction_tuned": (test_prob >= threshold).astype("int8"),
}).to_csv(OUT_DIR / "test_predictions.csv", index=False)

print(f"\nSaved artifacts in {OUT_DIR}/")
print("  xgb_binary_model.joblib")
print("  features.txt")
print("  metrics.json")
print("  test_predictions.csv")
