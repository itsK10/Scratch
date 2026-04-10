#!/usr/bin/env python3
"""Join two CSVs on Lot No, expanding rows with Tissue Section ID and Order."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Join CSV1 (unique Lot No) with CSV2 (semicolon-separated Lot No + Tissue Section ID)."
    )
    parser.add_argument("csv1", help="Path to the first CSV (unique Lot No per row)")
    parser.add_argument(
        "csv2",
        help="Path to the second CSV (Lot No may be semicolon-separated; has Tissue Section ID)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.csv",
        help="Output CSV path (default: output.csv)",
    )
    return parser.parse_args()


def require_columns(filepath: str, fieldnames: list[str] | None, required: list[str]) -> None:
    """Exit with a loud error if any required column is missing from the CSV."""
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


def build_csv2_index(csv2_path: str) -> dict[str, list[tuple[str, int]]]:
    """Build a mapping from lot_no -> list of (tissue_section_id, order).

    Each row in CSV2 has a Lot No cell that may contain multiple values
    separated by semicolons.  Order is the 1-based position within that cell.
    """
    index: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with open(csv2_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require_columns(csv2_path, reader.fieldnames, ["Lot No", "Tissue Section ID"])
        for row in reader:
            raw_lot = row["Lot No"]
            tissue_id = row["Tissue Section ID"]
            parts = [p.strip() for p in raw_lot.split(";")]
            for position, part in enumerate(parts, start=1):
                if part:
                    index[part].append((tissue_id, position))
    return index


def main():
    args = parse_args()

    csv2_index = build_csv2_index(args.csv2)

    with open(args.csv1, newline="", encoding="utf-8-sig") as f_in:
        reader = csv.DictReader(f_in)
        require_columns(args.csv1, reader.fieldnames, ["Lot No"])
        csv1_fields = list(reader.fieldnames or [])
        output_fields = csv1_fields + ["Tissue Section ID", "Order"]

        rows_out = []
        for row in reader:
            lot_no = row.get("Lot No", "").strip()
            matches = csv2_index.get(lot_no, [])
            if matches:
                for tissue_id, order in matches:
                    new_row = dict(row)
                    new_row["Tissue Section ID"] = tissue_id
                    new_row["Order"] = order
                    rows_out.append(new_row)
            else:
                new_row = dict(row)
                new_row["Tissue Section ID"] = ""
                new_row["Order"] = ""
                rows_out.append(new_row)

    with open(args.output, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {args.output}")


if __name__ == "__main__":
    main()
