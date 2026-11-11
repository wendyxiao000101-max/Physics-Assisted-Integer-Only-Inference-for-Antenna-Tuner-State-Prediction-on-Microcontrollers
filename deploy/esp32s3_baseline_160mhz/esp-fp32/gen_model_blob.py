#!/usr/bin/env python3
"""Extract the folded FP32 weights from the supplied TorchScript archive."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2] / "pc_reference"
MODEL = ROOT / "models" / "fp32" / "model_fp32_folded.pt"
OUT = Path(__file__).resolve().parent / "main" / "fp32_model.bin"
MAGIC = b"F32R32V1"
EXPECTED_SIZES = (640, 128, 4096, 128, 4096, 128, 1920)


def main() -> None:
    with zipfile.ZipFile(MODEL) as archive:
        chunks = [archive.read(f"archive/data/{i}") for i in range(len(EXPECTED_SIZES))]
    sizes = tuple(map(len, chunks))
    if sizes != EXPECTED_SIZES:
        raise SystemExit(f"unexpected tensor storage sizes: {sizes}")
    blob = MAGIC + b"".join(chunks)
    OUT.write_bytes(blob)
    print(f"wrote {OUT} bytes={len(blob)}")


if __name__ == "__main__":
    main()
