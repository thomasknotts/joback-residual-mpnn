#!/usr/bin/env python3
"""Regenerate a mean/std summary from comparison_ablations_results.csv."""

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results" / "comparison_ablations_results.csv"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path.with_name("comparison_ablations_summary.csv")
    frame = pd.read_csv(input_path)
    metrics = [column for column in frame.columns if column.startswith(("train_", "val_", "test_"))]
    summary = frame.groupby(["property", "variant", "fraction"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(column).strip("_") if isinstance(column, tuple) else column for column in summary.columns]
    summary.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
