#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ALIASES = {
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Fwd Packets Length Total",
    "TotLen Bwd Pkts": "Bwd Packets Length Total",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
}

def main():
    parser = argparse.ArgumentParser(
        description="Run IDS predictions on CICFlowMeter CSV or Parquet data."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--features-file", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--limit", type=int, default=0,
                        help="Test only the first N rows; 0 processes all rows.")
    args = parser.parse_args()

    try:
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_path, low_memory=False)
        elif input_path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(input_path)
        else:
            raise ValueError("Input must be CSV or Parquet.")

        if df.empty:
            raise ValueError("Input dataset is empty.")

        if args.limit > 0:
            df = df.head(args.limit).copy()
        else:
            df = df.copy()

        features = [
            line.strip()
            for line in Path(args.features_file).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        model = joblib.load(args.model)
        scaler = joblib.load(args.scaler)

        # Build the model input in the exact order used during training.
        selected = {}
        missing = []
        for feature in features:
            source = feature if feature in df.columns else ALIASES.get(feature)
            if source is None or source not in df.columns:
                missing.append(feature)
            else:
                selected[feature] = pd.to_numeric(df[source], errors="coerce")

        if missing:
            raise ValueError(
                "Missing required features: " + ", ".join(missing)
                + ". Do not substitute arbitrary values for missing features."
            )

        X = pd.DataFrame(selected, index=df.index)[features]

        if X.isna().any().any():
            bad = X.columns[X.isna().any()].tolist()
            raise ValueError(
                "Missing or non-numeric values in: " + ", ".join(bad)
                + ". Clean these rows using the same approach as training."
            )

        if getattr(scaler, "n_features_in_", None) != len(features):
            raise ValueError(
                f"Scaler expects {scaler.n_features_in_} features, "
                f"but feature list contains {len(features)}."
            )
        if getattr(model, "n_features_in_", None) != len(features):
            raise ValueError(
                f"Model expects {model.n_features_in_} features, "
                f"but feature list contains {len(features)}."
            )

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        output = df.copy()
        output["predicted_class"] = predictions

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)
            output["prediction_confidence"] = probabilities.max(axis=1)
            for i, label in enumerate(model.classes_):
                output[f"probability_class_{label}"] = probabilities[:, i]

        output.to_csv(args.output, index=False)

        print(f"Rows predicted: {len(output)}")
        print(f"Feature count: {len(features)}")
        print("\nPredicted class counts:")
        print(pd.Series(predictions).value_counts().sort_index().to_string())
        print(f"\nSaved results to: {args.output}")
        print("Predicted classes are model outputs; confirm their mapping to "
              "Benign/Attack from the training label encoding.")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
