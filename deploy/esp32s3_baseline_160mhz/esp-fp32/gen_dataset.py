#!/usr/bin/env python3
"""Pack the public CSV features and labels for the dataset flash partition."""
import csv
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # repository root
OUT = Path(__file__).resolve().parent / "main" / "dataset.bin"
ROWS_EXPECTED = 96768


def rows(path: Path) -> list[list[int]]:
    with path.open(encoding="gbk", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        return [[int(value) for value in row] for row in reader if row]


def selected(row: list[int]) -> list[int]:
    if len(row) == 6:
        return [row[i] for i in (0, 1, 2, 4, 5)]
    if len(row) == 5:
        return row
    raise ValueError(f"expected 5 or 6 features, got {len(row)}")


def main() -> None:
    x = rows(ROOT / "data" / "x_train.csv")
    y = [row[0] for row in rows(ROOT / "data" / "y_train.csv")]
    if len(x) != len(y) or len(x) != ROWS_EXPECTED:
        raise SystemExit(f"row count {len(x)}/{len(y)}, expected {ROWS_EXPECTED}")

    features = bytearray()
    labels = bytearray()
    for raw, label in zip(x, y):
        values = selected(raw)
        if any(value < -32768 or value > 32767 for value in values):
            raise SystemExit(f"feature does not fit int16: {values}")
        if not 0 <= label <= 14:
            raise SystemExit(f"label out of range: {label}")
        features += struct.pack("<5h", *values)
        labels.append(label)

    OUT.write_bytes(features + labels)
    print(f"wrote {OUT} bytes={OUT.stat().st_size} rows={len(x)}")


if __name__ == "__main__":
    main()
