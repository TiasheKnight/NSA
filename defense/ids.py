
#!/usr/bin/env python3

import argparse
import hashlib
import os
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = ROOT / "defense/alerts/live_predictions.csv"

REFRESH_SECONDS = 2
MAX_ROWS = 25

REQUIRED_COLUMNS = [
    "src_ip", "dst_ip", "src_port", "dst_port",
    "protocol", "timestamp", "predicted_class",
    "predicted_label", "attack_probability",
]


def load_flows():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_FILE}\n"
            "Start the IDS and generate some traffic first."
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    df = df.dropna(subset=["timestamp"]).copy()
    df["predicted_class"] = pd.to_numeric(
        df["predicted_class"], errors="coerce"
    )
    df["attack_probability"] = pd.to_numeric(
        df["attack_probability"], errors="coerce"
    )

    # Reproducible IDs for the current CSV contents.
    identity_columns = [
        "src_ip", "dst_ip", "src_port", "dst_port",
        "protocol", "timestamp",
    ]
    occurrences = {}
    flow_ids = []

    for _, row in df.iterrows():
        identity = "|".join(str(row[col]) for col in identity_columns)
        occurrence = occurrences.get(identity, 0)
        occurrences[identity] = occurrence + 1

        digest = hashlib.sha256(
            f"{identity}|{occurrence}".encode()
        ).hexdigest()[:10].upper()

        date = str(row["timestamp"])[:10].replace("-", "")
        flow_ids.append(f"FLOW-{date}-{digest}")

    df.insert(0, "flow_id", flow_ids)
    return df


def filter_flows(df, attacks_only=False):
    if attacks_only:
        df = df[
            (df["predicted_class"] == 1)
            | df["predicted_label"].astype(str).str.lower().isin(
                ["malicious", "attack", "attacking"]
            )
        ]

    return df.sort_values("timestamp", ascending=False)


def report_once(df, attacks_only=False):
    df = filter_flows(df, attacks_only)

    if df.empty:
        print("No matching flows found.")
        return

    columns = [
        "flow_id", "timestamp", "src_ip", "src_port",
        "dst_ip", "dst_port", "protocol",
        "predicted_label", "attack_probability",
    ]

    print(f"\nFlows found: {len(df)}")
    print(df[columns].to_string(index=False))


def live_report(attacks_only=False):
    try:
        while True:
            df = load_flows()
            df = filter_flows(df, attacks_only)
            df = df.head(MAX_ROWS)

            # Refresh the display in place.
            print("\033[2J\033[H", end="")
            title = (
                "PREDICTED ATTACKS"
                if attacks_only else "ALL TRAFFIC"
            )

            print(f"IDS REPORT — {title}")
            print("=" * 100)
            print(
                f"Showing latest {len(df)} matching flows"
                f" | Refresh: {REFRESH_SECONDS}s | Press q to exit"
            )
            print("-" * 100)

            if df.empty:
                print("No matching flows yet.")
            else:
                for _, row in df.iterrows():
                    label = str(row["predicted_label"]).upper()
                    source = f'{row["src_ip"]}:{row["src_port"]}'
                    destination = f'{row["dst_ip"]}:{row["dst_port"]}'
                    probability = row["attack_probability"]

                    print(
                        f'{row["flow_id"]} | {row["timestamp"]} | '
                        f'{source} -> {destination} | '
                        f'{label} | score={probability}'
                    )

            # Wait in short intervals so q exits promptly.
            for _ in range(REFRESH_SECONDS * 10):
                time.sleep(0.1)
                # Check for q without requiring an extra package.
                if os.name == "posix":
                    import select
                    import sys
                    import termios
                    import tty

                    if not sys.stdin.isatty():
                        continue

                    # Switch to non-canonical mode once per refresh loop.
                    break

            if os.name == "posix" and sys.stdin.isatty():
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    deadline = time.time() + REFRESH_SECONDS
                    while time.time() < deadline:
                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if ready and sys.stdin.read(1).lower() == "q":
                            return
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    except KeyboardInterrupt:
        pass
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        print(f"\nError: {exc}")
        return
    finally:
        print("\nIDS viewer closed.")


def inspect_flow(df, flow_id):
    matches = df[df["flow_id"] == flow_id]

    if matches.empty:
        print(f"Flow not found: {flow_id}")
        print("Run --report --all to see available flow IDs.")
        return

    row = matches.iloc[0]
    print("\nFlow details")
    print("=" * 50)
    for column, value in row.items():
        print(f"{column}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Report and inspect live IDS flow predictions."
    )

    parser.add_argument("--report", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true")
    group.add_argument("--attack", action="store_true")
    parser.add_argument("--inspect", metavar="FLOW_ID")

    args = parser.parse_args()

    if not args.report and not args.inspect:
        parser.error("Specify --report or --inspect FLOW_ID.")
    if args.report and not (args.all or args.attack):
        parser.error("--report requires --all or --attack.")
    if args.report and args.inspect:
        parser.error("Use --report or --inspect, not both.")

    if args.report:
        live_report(attacks_only=args.attack)
        return

    try:
        inspect_flow(load_flows(), args.inspect)
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
