"""Prepare UCI Communities and Crime for the regression agent."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


NON_PREDICTIVE_COLUMNS = [
    "state",
    "county",
    "community",
    "communityname",
    "fold",
]
TARGET_COLUMN = "ViolentCrimesPerPop"


def attribute_names(names_path: Path) -> list[str]:
    pattern = re.compile(r"^@attribute\s+(\S+)\s+", re.IGNORECASE)
    columns = []
    for line in names_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            columns.append(match.group(1))
    return columns


def prepare(raw_path: Path, names_path: Path, output_path: Path) -> pd.DataFrame:
    columns = attribute_names(names_path)
    frame = pd.read_csv(raw_path, header=None, names=columns, na_values="?")
    if len(columns) != 128 or frame.shape != (1994, 128):
        raise ValueError(
            f"Unexpected dataset shape: parsed {len(columns)} fields and {frame.shape} rows/columns"
        )
    frame = frame.drop(columns=NON_PREDICTIVE_COLUMNS)
    frame = frame.apply(pd.to_numeric, errors="raise")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file", type=Path)
    parser.add_argument("names_file", type=Path)
    parser.add_argument("output_file", type=Path)
    args = parser.parse_args()
    frame = prepare(args.raw_file, args.names_file, args.output_file)
    print(
        f"Prepared {len(frame)} rows, {frame.shape[1] - 1} predictors, "
        f"target={TARGET_COLUMN}, missing_values={int(frame.isna().sum().sum())}"
    )


if __name__ == "__main__":
    main()
