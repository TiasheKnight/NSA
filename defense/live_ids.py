import time
from pathlib import Path

import joblib
import pandas as pd

from predict_live_flows import FEATURE_MAP

PROJECT = Path(__file__).resolve().parents[1]

FLOW_FILE = PROJECT / "defense/flows/live_flows.csv"
MODEL_FILE = PROJECT / "defense/output/binary_comparison/xgboost_pipeline.joblib"
ALERT_DIR = PROJECT / "defense/alerts"
RESULT_FILE = ALERT_DIR / "live_predictions.csv"
ALERT_FILE = ALERT_DIR / "live_alerts.csv"

POLL_SECONDS = 2

METADATA = [
    "src_ip", "dst_ip", "src_port",
    "dst_port", "protocol", "timestamp"
]


def main():
    print("Loading IDS model...")
    model = joblib.load(MODEL_FILE)
    expected = list(model.feature_names_in_)

    missing_mapping = [name for name in expected if name not in FEATURE_MAP]
    if missing_mapping:
        raise ValueError(f"Missing feature mappings: {missing_mapping}")

    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    processed_rows = 0

    print(f"Monitoring flows: {FLOW_FILE}")
    print("Waiting for completed flows. Press Ctrl+C to stop.")

    while True:
        try:
            if not FLOW_FILE.exists():
                time.sleep(POLL_SECONDS)
                continue

            flows = pd.read_csv(FLOW_FILE, on_bad_lines="skip")

            if flows.empty or len(flows) <= processed_rows:
                time.sleep(POLL_SECONDS)
                continue

            # Process only rows not previously handled.
            new_flows = flows.iloc[processed_rows:].copy()
            processed_rows = len(flows)

            required = set(METADATA)
            missing = required - set(new_flows.columns)
            if missing:
                raise ValueError(f"Missing metadata columns: {sorted(missing)}")

            feature_columns = [FEATURE_MAP[name] for name in expected]
            missing_features = set(feature_columns) - set(new_flows.columns)
            if missing_features:
                raise ValueError(
                    f"Missing flow features: {sorted(missing_features)}"
                )

            X = new_flows[feature_columns].copy()
            X.columns = expected

            predictions = model.predict(X)

            results = new_flows[METADATA].copy()
            results["predicted_class"] = predictions
            results["predicted_label"] = pd.Series(
                predictions, index=results.index
            ).map({0: "Benign", 1: "Malicious"})

            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)
                if 1 in classes:
                    probabilities = model.predict_proba(X)
                    results["attack_probability"] = probabilities[:, classes.index(1)]

            # Append all predictions to the history file.
            results.to_csv(
                RESULT_FILE,
                mode="a",
                header=not RESULT_FILE.exists(),
                index=False,
            )

            # Keep a separate file containing only predicted attacks.
            alerts = results[results["predicted_class"] == 1]
            if not alerts.empty:
                alerts.to_csv(
                    ALERT_FILE,
                    mode="a",
                    header=not ALERT_FILE.exists(),
                    index=False,
                )

                print(f"\nALERT: {len(alerts)} flow(s) predicted malicious")
                print(alerts.to_string(index=False))
            else:
                print(f"Processed {len(results)} flow(s): no malicious predictions")

        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            # The capture tool may be writing a row while we read the file.
            pass

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nIDS monitoring stopped.")
