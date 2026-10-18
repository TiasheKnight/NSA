from pathlib import Path

import kagglehub
import pandas as pd

SEED = 42
ROWS_PER_FILE = 2000

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "attack" / "flows"
OUTPUT_FILE = OUTPUT_DIR / "sampled_dataset_flows.csv"


def main():
    print("Downloading CSE-CIC-IDS2018 dataset...")
    dataset_path = Path(
        kagglehub.dataset_download("dhoogla/csecicids2018")
    )
    print(f"Dataset location: {dataset_path}")

    files = sorted(
        list(dataset_path.rglob("*.parquet"))
        + list(dataset_path.rglob("*.csv"))
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV or Parquet files found under {dataset_path}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = []

    for file in files:
        print(f"Sampling: {file.name}")

        if file.suffix.lower() == ".parquet":
            df = pd.read_parquet(file)
        else:
            df = pd.read_csv(file, low_memory=False)

        if df.empty:
            continue

        sample = df.sample(
            n=min(ROWS_PER_FILE, len(df)),
            random_state=SEED,
        ).copy()

        sample.insert(0, "source_file", file.name)
        samples.append(sample)

        print(f"  Sampled {len(sample):,} of {len(df):,} rows")

        del df

    if not samples:
        raise RuntimeError("No rows were sampled.")

    result = pd.concat(samples, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"\nTotal sampled flows: {len(result):,}")
    print(f"Saved to: {OUTPUT_FILE}")

    label_col = next(
        (c for c in result.columns if c.strip().lower() == "label"),
        None,
    )
    if label_col:
        print("\nSampled labels:")
        print(result[label_col].value_counts().to_string())


if __name__ == "__main__":
    main()
