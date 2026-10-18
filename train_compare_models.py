from pathlib import Path
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("pcaps")
OUT_DIR = Path("output/binary_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Limits memory use while retaining samples from every source file.
MAX_PER_CLASS_PER_FILE = 3000

frames = []
common_features = None
files = sorted(DATA_DIR.glob("*.parquet"))

if not files:
    raise FileNotFoundError(f"No Parquet files found in {DATA_DIR}")

# 1. Load and sample each file.
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

    df["target"] = (
        labels.str.casefold() != "benign"
    ).astype("int8")

    excluded = {"target", label_col}
    for col in df.columns:
        name = col.lower().strip()
        if (
            "flow id" in name
            or "timestamp" in name
            or "source ip" in name
            or "destination ip" in name
            or name in {
                "src ip", "dst ip", "src address", "dst address"
            }
        ):
            excluded.add(col)

    candidates = [c for c in df.columns if c not in excluded]

    if common_features is None:
        common_features = set(candidates)
    else:
        common_features &= set(candidates)

    sampled_parts = []
    for target in (0, 1):
        part = df.loc[df["target"] == target]
        n = min(len(part), MAX_PER_CLASS_PER_FILE)

        if n:
            sampled_parts.append(
                part.sample(n=n, random_state=RANDOM_STATE)
            )

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled[candidates + ["target"]]
    frames.append(sampled)

    print(
        f"Selected {len(sampled):,} rows; "
        f"benign={(sampled.target == 0).sum():,}, "
        f"attack={(sampled.target == 1).sum():,}"
    )

    del df, labels, sampled, sampled_parts
    gc.collect()

# 2. Use a consistent feature set across every dataset.
feature_names = sorted(common_features)
data = pd.concat(frames, ignore_index=True)
del frames
gc.collect()

for col in feature_names:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].replace([np.inf, -np.inf], np.nan)

feature_names = [
    col for col in feature_names
    if data[col].notna().any()
]

X = data[feature_names]
y = data["target"].astype("int8")

# Remove rows without any usable feature values.
valid_rows = X.notna().any(axis=1)
X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

# Drop constant features.
feature_names = [
    col for col in X.columns
    if X[col].nunique(dropna=True) > 1
]
X = X[feature_names]

print("\nDataset summary")
print(f"Rows: {len(X):,}")
print(f"Benign: {(y == 0).sum():,}")
print(f"Attack: {(y == 1).sum():,}")
print(f"Features: {len(feature_names)}")

# 70% training, 15% validation, 15% test.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30,
    random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=RANDOM_STATE, stratify=y_temp
)

print(
    f"Split sizes: train={len(X_train):,}, "
    f"validation={len(X_val):,}, test={len(X_test):,}"
)

# All models use the same rows and feature columns.
# Imputation is fitted on training data only.
models = {
    "knn": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=-1,
        )),
    ]),
    "random_forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ]),
    "xgboost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", xgb.XGBClassifier(
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
        )),
    ]),
}

results = {}
prediction_columns = {
    "actual": y_test.to_numpy()
}

for name, model in models.items():
    print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}", flush=True)
    start = time.time()

    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]

    # Select a threshold using validation F1 only.
    precision, recall, thresholds = precision_recall_curve(
        y_val, val_prob
    )
    if len(thresholds):
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )
        threshold = float(thresholds[int(np.argmax(f1_values))])
    else:
        threshold = 0.5

    # Final evaluation on the untouched test split.
    test_prob = model.predict_proba(X_test)[:, 1]
    pred_default = (test_prob >= 0.5).astype("int8")
    pred_tuned = (test_prob >= threshold).astype("int8")

    model_results = {
        "threshold_selected_on_validation": threshold,
        "training_seconds": round(time.time() - start, 2),
    }

    for version, pred in [
        ("default_0.5", pred_default),
        ("validation_tuned", pred_tuned),
    ]:
        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        model_results[version] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, pred)
            ),
            "attack_precision": float(
                precision_score(y_test, pred, zero_division=0)
            ),
            "attack_recall": float(
                recall_score(y_test, pred, zero_division=0)
            ),
            "attack_f1": float(
                f1_score(y_test, pred, zero_division=0)
            ),
            "false_positive_rate": float(
                fp / (fp + tn) if fp + tn else 0
            ),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "classification_report": classification_report(
                y_test, pred,
                labels=[0, 1],
                target_names=["Benign", "Attack"],
                output_dict=True,
                zero_division=0,
            ),
        }

        print(f"\n{name} — {version}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(model_results[version]["confusion_matrix"])
        print(classification_report(
            y_test, pred,
            labels=[0, 1],
            target_names=["Benign", "Attack"],
            zero_division=0,
        ))
        print(
            f"Balanced accuracy: "
            f"{model_results[version]['balanced_accuracy']:.4f}"
        )
        print(
            f"False-positive rate: "
            f"{model_results[version]['false_positive_rate']:.4f}"
        )

    results[name] = model_results
    joblib.dump(model, OUT_DIR / f"{name}_pipeline.joblib")

    prediction_columns[f"{name}_probability"] = test_prob
    prediction_columns[f"{name}_default"] = pred_default
    prediction_columns[f"{name}_tuned"] = pred_tuned

    print(f"Saved {name} pipeline.")

# Save feature order, metrics and predictions.
(OUT_DIR / "features.txt").write_text(
    "\n".join(feature_names) + "\n"
)
(OUT_DIR / "metrics.json").write_text(
    json.dumps(results, indent=2)
)
pd.DataFrame(prediction_columns).to_csv(
    OUT_DIR / "test_predictions.csv", index=False
)

summary_rows = []
for name, result in results.items():
    for version in ("default_0.5", "validation_tuned"):
        m = result[version]
        summary_rows.append({
            "model": name,
            "threshold_type": version,
            "threshold": (
                0.5 if version == "default_0.5"
                else result["threshold_selected_on_validation"]
            ),
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "attack_precision": m["attack_precision"],
            "attack_recall": m["attack_recall"],
            "attack_f1": m["attack_f1"],
            "false_positive_rate": m["false_positive_rate"],
            "training_seconds": result["training_seconds"],
        })

pd.DataFrame(summary_rows).to_csv(
    OUT_DIR / "comparison.csv", index=False
)

print(f"\nAll artifacts saved in: {OUT_DIR}")
print(pd.DataFrame(summary_rows).to_string(index=False))
