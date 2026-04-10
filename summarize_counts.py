#!/usr/bin/env python3.10
"""Aggregate CountSummaryPerClassificationTile.csv by Sample + AdapterGene.

Outputs a summary CSV and a log-log scatter plot with linear regression.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

INPUT_FILE = "CountSummaryPerClassificationTile.csv"
REQUIRED_COLUMNS = [
    "Run Name",
    "Run ID",
    "Sample",
    "AdapterGene",
    "ReadsPerMM2",
    "BulkTPM",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Aggregate {INPUT_FILE} by Sample + AdapterGene, "
            "perform log10 linear regression, and generate a scatter plot."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Base name for output files (default: output -> output.csv + output.png)",
    )
    return parser.parse_args()


def require_columns(
    filepath: str, fieldnames: list[str] | None, required: list[str]
) -> None:
    """Exit with a loud error if any required column is missing."""
    if not fieldnames:
        print(f"ERROR: {filepath} has no header row.", file=sys.stderr)
        sys.exit(1)
    missing = [col for col in required if col not in fieldnames]
    if missing:
        print(
            f"ERROR: {filepath} is missing required column(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        print(f"  Found columns: {', '.join(fieldnames)}", file=sys.stderr)
        sys.exit(1)


def read_and_aggregate(
    filepath: str,
) -> list[dict[str, str | float]]:
    """Read the input CSV, group by (Sample, AdapterGene), and aggregate.

    Returns:
        List of dicts with keys: Run Name, Run ID, Sample, AdapterGene,
        SumReads, BulkTPM.
    """
    GroupKey = tuple[str, str]
    groups: dict[GroupKey, list[dict[str, str]]] = defaultdict(list)
    row_indices: dict[GroupKey, list[int]] = defaultdict(list)

    with open(filepath, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require_columns(filepath, reader.fieldnames, REQUIRED_COLUMNS)
        for row_num, row in enumerate(reader, start=2):
            bulk_tpm_raw = row["BulkTPM"].strip()
            if not bulk_tpm_raw:
                print(
                    f"INFO: Skipping row {row_num} (empty BulkTPM).",
                    file=sys.stderr,
                )
                continue
            key: GroupKey = (row["Sample"], row["AdapterGene"])
            groups[key].append(row)
            row_indices[key].append(row_num)

    results: list[dict[str, str | float]] = []
    for (sample, adapter_gene), rows in groups.items():
        sum_reads = sum(float(r["ReadsPerMM2"]) for r in rows)

        tpm_values = [float(r["BulkTPM"]) for r in rows]
        if len(set(tpm_values)) > 1:
            indices_str = ", ".join(str(i) for i in row_indices[(sample, adapter_gene)])
            print(
                f"WARNING: BulkTPM mismatch for Sample={sample}, "
                f"AdapterGene={adapter_gene} at CSV rows: {indices_str}. "
                f"Values: {tpm_values}. Using mean.",
                file=sys.stderr,
            )
        bulk_tpm = sum(tpm_values) / len(tpm_values)

        results.append(
            {
                "Run Name": rows[0]["Run Name"],
                "Run ID": rows[0]["Run ID"],
                "Sample": sample,
                "AdapterGene": adapter_gene,
                "SumReads": sum_reads,
                "BulkTPM": bulk_tpm,
            }
        )

    return results


def write_csv(results: list[dict[str, str | float]], output_path: str) -> None:
    """Write aggregated results to a CSV file."""
    fieldnames = ["Run Name", "Run ID", "Sample", "AdapterGene", "SumReads", "BulkTPM"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {len(results)} rows to {output_path}")


def run_regression_and_plot(
    results: list[dict[str, str | float]], plot_path: str
) -> None:
    """Fit log10 linear regression (BulkTPM < 1000) and save scatter plot."""
    all_tpm = np.array([float(r["BulkTPM"]) for r in results])
    all_reads = np.array([float(r["SumReads"]) for r in results])

    # For log-scale plotting, only include points where both values are > 0.
    plot_mask = (all_tpm > 0) & (all_reads > 0)
    plot_tpm = all_tpm[plot_mask]
    plot_reads = all_reads[plot_mask]
    n_skipped = int(np.sum(~plot_mask))
    if n_skipped:
        print(
            f"INFO: Skipping {n_skipped} data point(s) with BulkTPM <= 0 or "
            f"SumReads <= 0 (cannot take log10).",
            file=sys.stderr,
        )

    # Regression subset: BulkTPM < 1000 AND both values > 0.
    fit_mask = (all_tpm > 0) & (all_reads > 0) & (all_tpm < 1000)
    fit_tpm = all_tpm[fit_mask]
    fit_reads = all_reads[fit_mask]

    if len(fit_tpm) < 2:
        print(
            "WARNING: Not enough valid data points with BulkTPM < 1000 for regression.",
            file=sys.stderr,
        )
        return

    log_tpm = np.log10(fit_tpm)
    log_reads = np.log10(fit_reads)

    coeffs = np.polyfit(log_tpm, log_reads, 1)
    slope, intercept = coeffs[0], coeffs[1]

    y_pred_fit = slope * log_tpm + intercept
    ss_res = np.sum((log_reads - y_pred_fit) ** 2)
    ss_tot = np.sum((log_reads - np.mean(log_reads)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    x_target = 3.0  # log10(1000)
    y_target = slope * x_target + intercept
    predicted_sum_reads = 10**y_target

    print(f"\n--- Linear Regression (log10 scale, BulkTPM < 1000) ---")
    print(f"  log10(SumReads) = {slope:.4f} * log10(BulkTPM) + {intercept:.4f}")
    print(f"  R-squared: {r_squared:.4f}")
    print(f"  Prediction at BulkTPM=1000: SumReads = {predicted_sum_reads:.4f}")

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(plot_tpm, plot_reads, alpha=0.6, edgecolors="k", linewidths=0.5, label="All data")

    x_line = np.linspace(max(fit_tpm.min(), 1e-10), 1000, 200)
    y_line = 10 ** (slope * np.log10(x_line) + intercept)
    ax.plot(x_line, y_line, color="blue", linewidth=2, label="Regression (BulkTPM < 1000)")

    ax.scatter(
        [1000],
        [predicted_sum_reads],
        color="red",
        s=120,
        zorder=5,
        marker="X",
        label=f"Predicted at 1000 = {predicted_sum_reads:.2f}",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("BulkTPM")
    ax.set_ylabel("SumReads")
    ax.set_title("SumReads vs BulkTPM (log10)")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


def main() -> None:
    args = parse_args()

    results = read_and_aggregate(INPUT_FILE)

    csv_path = f"{args.output}.csv"
    png_path = f"{args.output}.png"

    write_csv(results, csv_path)
    run_regression_and_plot(results, png_path)


if __name__ == "__main__":
    main()
