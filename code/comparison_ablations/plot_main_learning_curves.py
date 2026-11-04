#!/usr/bin/env python3
"""Generate the compact learning-curve figure for the main manuscript.

This is intentionally limited to the primary comparison:

* Full JR-MPNN
* MPNN control
* Original Joback

The detailed ablation and multi-metric figures remain available through
``plot_learning_curves.py`` for the Supporting Information.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_learning_curves import load_summary_rows, plot_metric


MAIN_VARIANTS = ("jrmpnn_full", "mpnn_control", "joback_original")


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_dir, help="Directory containing the summary CSV files.")
    parser.add_argument("--output-dir", type=Path, default=default_dir / "plots_main", help="Directory for the main-manuscript figure.")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution. Default: 600.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows_by_property = load_summary_rows(input_dir)
    outputs = plot_metric(rows_by_property, "mae", MAIN_VARIANTS, output_dir, args.dpi)
    print("Main-manuscript variants:")
    print("  - Full JR-MPNN")
    print("  - MPNN")
    print("  - Original Joback")
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
