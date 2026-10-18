from pathlib import Path

import joblib
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT / "defense/flows/lab_test.csv"
MODEL_FILE = PROJECT / "defense/output/binary_comparison/xgboost_pipeline.joblib"
OUTPUT_FILE = PROJECT / "defense/alerts/lab_test_predictions.csv"

# Map model feature names to CICFlowMeter Python output columns.
FEATURE_MAP = {
    "ACK Flag Count": "ack_flag_cnt",
    "Active Max": "active_max",
    "Active Mean": "active_mean",
    "Active Min": "active_min",
    "Active Std": "active_std",
    "Avg Bwd Segment Size": "bwd_seg_size_avg",
    "Avg Fwd Segment Size": "fwd_seg_size_avg",
    "Avg Packet Size": "pkt_size_avg",
    "Bwd Header Length": "bwd_header_len",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Min": "bwd_iat_min",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Bwd Packets Length Total": "totlen_bwd_pkts",
    "Bwd Packets/s": "bwd_pkts_s",
    "CWE Flag Count": "cwr_flag_count",
    "Down/Up Ratio": "down_up_ratio",
    "ECE Flag Count": "ece_flag_cnt",
    "FIN Flag Count": "fin_flag_cnt",
    "Flow Bytes/s": "flow_byts_s",
    "Flow Duration": "flow_duration",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Min": "flow_iat_min",
    "Flow IAT Std": "flow_iat_std",
    "Flow Packets/s": "flow_pkts_s",
    "Fwd Act Data Packets": "fwd_act_data_pkts",
    "Fwd Header Length": "fwd_header_len",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Min": "fwd_iat_min",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Fwd Packets Length Total": "totlen_fwd_pkts",
    "Fwd Packets/s": "fwd_pkts_s",
    "Fwd Seg Size Min": "fwd_seg_size_min",
    "Fwd URG Flags": "fwd_urg_flags",
    "Idle Max": "idle_max",
    "Idle Mean": "idle_mean",
    "Idle Min": "idle_min",
    "Idle Std": "idle_std",
    "Init Bwd Win Bytes": "init_bwd_win_byts",
    "Init Fwd Win Bytes": "init_fwd_win_byts",
    "PSH Flag Count": "psh_flag_cnt",
    "Packet Length Max": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Min": "pkt_len_min",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "Protocol": "protocol",
    "RST Flag Count": "rst_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "Subflow Bwd Bytes": "subflow_bwd_byts",
    "Subflow Bwd Packets": "subflow_bwd_pkts",
    "Subflow Fwd Bytes": "subflow_fwd_byts",
    "Subflow Fwd Packets": "subflow_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Fwd Packets": "tot_fwd_pkts",
    "URG Flag Count": "urg_flag_cnt",
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Flow CSV not found: {INPUT_FILE}")

    flows = pd.read_csv(INPUT_FILE)
    model = joblib.load(MODEL_FILE)

    # Use the exact feature order recorded by the trained pipeline.
    expected = list(model.feature_names_in_)

    missing_mapping = [name for name in expected if name not in FEATURE_MAP]
    if missing_mapping:
        raise ValueError(f"Features have no mapping: {missing_mapping}")

    missing_columns = [
        FEATURE_MAP[name]
        for name in expected
        if FEATURE_MAP[name] not in flows.columns
    ]
    if missing_columns:
        raise ValueError(f"Flow CSV is missing columns: {missing_columns}")

    X = flows[[FEATURE_MAP[name] for name in expected]].copy()
    X.columns = expected

    # The pipeline performs its own imputation.
    predictions = model.predict(X)

    results = flows[
        ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp"]
    ].copy()

    results["predicted_class"] = predictions

    # This assumes the training labels use 0 = benign and 1 = attack.
    results["predicted_label"] = pd.Series(predictions).map(
        {0: "Benign", 1: "Malicious"}
    ).values

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            probabilities = model.predict_proba(X)
            results["attack_probability"] = probabilities[:, classes.index(1)]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Flows analysed: {len(results)}")
    print(f"Model: {MODEL_FILE}")
    print(f"Predictions saved to: {OUTPUT_FILE}")
    print("\nPrediction counts:")
    print(results["predicted_label"].value_counts(dropna=False).to_string())
    print("\nResults:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
