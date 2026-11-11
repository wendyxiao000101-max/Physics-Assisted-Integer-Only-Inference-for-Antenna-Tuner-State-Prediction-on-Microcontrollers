#!/usr/bin/env python3
"""Pack data/x_train.csv and y_train.csv into main/dataset.bin.

Layout, little-endian:
  int16 features[rows * 5]   selected columns [g1Re, g1Im, freq, g2Re, g2Im]
  uint8 labels[rows]
"""
import csv
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repository root
OUT = Path(__file__).resolve().parent / "main" / "dataset.bin"
ROWS_EXPECTED = 96768


def rows(path: Path):
    with path.open(encoding="gbk", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return [[int(v) for v in row] for row in reader if row]


def selected(row):
    if len(row) == 6:
        return [row[i] for i in (0, 1, 2, 4, 5)]
    if len(row) == 5:
        return row
    raise ValueError(f"expected 5 or 6 features, got {len(row)}")


def main():
    x = rows(ROOT / "data" / "x_train.csv")
    y = [v[0] for v in rows(ROOT / "data" / "y_train.csv")]
    if len(x) != len(y) or len(x) != ROWS_EXPECTED:
        raise SystemExit(f"row count {len(x)}/{len(y)}, expected {ROWS_EXPECTED}")
    feat = bytearray()
    labels = bytearray()
    for raw, label in zip(x, y):
        values = selected(raw)
        if any(v < -32768 or v > 32767 for v in values):
            raise SystemExit(f"feature does not fit int16: {values}")
        if not 0 <= label <= 14:
            raise SystemExit(f"label out of range: {label}")
        feat += struct.pack("<5h", *values)
        labels.append(label)
    OUT.write_bytes(feat + labels)
    print(f"wrote {OUT} bytes={OUT.stat().st_size} rows={len(x)}")


if __name__ == "__main__":
    sys.exit(main())
