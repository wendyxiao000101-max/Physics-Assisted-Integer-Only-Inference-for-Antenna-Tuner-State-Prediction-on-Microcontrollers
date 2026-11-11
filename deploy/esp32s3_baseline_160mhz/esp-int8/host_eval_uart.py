#!/usr/bin/env python3
"""Send frames to ESP32-S3 over UART and score INT8 predictions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import serial

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fast_integer_runtime import Predictor  # noqa: E402
from pc_common import rows, selected  # noqa: E402

READY_MARK = "READY I8R32V02"
EXPECTED_SHA = "07f8ff773e70fb92070f6e205987f546860ceceea0371efacaadbdea8b85de54"


def open_serial(port: str, baud: int, timeout: float) -> serial.Serial:
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = timeout
    ser.dsrdtr = False
    ser.rtscts = False
    ser.dtr = False
    ser.rts = False
    ser.open()
    # USB-Serial-JTAG uses RTS as reset. Pulse it after the port is open so
    # the app boots while this handle is already reading.
    time.sleep(0.05)
    ser.rts = True
    time.sleep(0.1)
    ser.rts = False
    ser.dtr = False
    return ser


def wait_ready(ser: serial.Serial, seconds: float) -> str:
    deadline = time.time() + seconds
    last = ""
    ser.write(b"HELLO\n")
    ser.flush()
    while time.time() < deadline:
        line = ser.readline()
        if not line:
            ser.write(b"HELLO\n")
            ser.flush()
            continue
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        last = text
        if text.startswith(READY_MARK):
            if EXPECTED_SHA not in text:
                raise RuntimeError(f"unexpected READY line: {text}")
            return text
    raise TimeoutError(f"no READY from device; last line={last!r}")


def infer_one(ser: serial.Serial, feat: list[int], retries: int) -> tuple[int, int]:
    payload = (" ".join(str(v) for v in feat) + "\n").encode("ascii")
    last_err = "timeout"
    for _ in range(retries):
        ser.reset_input_buffer()
        ser.write(payload)
        ser.flush()
        line = ser.readline()
        if not line:
            continue
        text = line.decode("utf-8", errors="replace").strip()
        if text.startswith(READY_MARK):
            continue
        parts = text.split()
        if len(parts) >= 2 and parts[0] == "OK":
            cls = int(parts[1])
            t_us = int(parts[2]) if len(parts) > 2 else -1
            if not 0 <= cls <= 14:
                last_err = f"class out of range: {text}"
                continue
            return cls, t_us
        last_err = text or "empty"
    raise RuntimeError(last_err)


def main() -> None:
    p = argparse.ArgumentParser(description="ESP32-S3 INT8 UART frame evaluation")
    p.add_argument("--port", required=True, help="serial port, e.g. /dev/cu.usbmodem* or COM3")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--x", type=Path, default=HERE.parents[2] / "data/x_train.csv")
    p.add_argument("--y", type=Path, default=HERE.parents[2] / "data/y_train.csv")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows (0=all)")
    p.add_argument("--timeout", type=float, default=1.0)
    p.add_argument("--ready-timeout", type=float, default=8.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--no-pc-check", action="store_true")
    p.add_argument("--output", type=Path, default=HERE / "results/uart")
    args = p.parse_args()
    if args.retries < 1:
        raise ValueError("retries must be positive")

    x = [selected(v) for v in rows(args.x)]
    y = None if args.no_labels else [v[0] for v in rows(args.y)]
    if y is not None and len(x) != len(y):
        raise ValueError("X/Y row count mismatch")
    if args.limit > 0:
        x = x[: args.limit]
        if y is not None:
            y = y[: args.limit]

    pc = None if args.no_pc_check else Predictor(ROOT / "models/int8/model_int8.bin")
    ser = open_serial(args.port, args.baud, args.timeout)
    try:
        time.sleep(1.5)
        ready = wait_ready(ser, args.ready_timeout)
        pred: list[int] = []
        times: list[int] = []
        uart_errors = 0
        start = time.perf_counter()
        for i, feat in enumerate(x):
            try:
                cls, t_us = infer_one(ser, feat, args.retries)
            except Exception as exc:  # noqa: BLE001
                uart_errors += 1
                cls, t_us = -1, -1
                print(f"UART error at row {i}: {exc}", file=sys.stderr)
            pred.append(cls)
            times.append(t_us)
            if (i + 1) % 1000 == 0 or i == 0:
                print(f"progress {i + 1}/{len(x)}", flush=True)
        elapsed = time.perf_counter() - start
    finally:
        ser.close()

    ok_idx = [i for i, v in enumerate(pred) if 0 <= v <= 14]
    report = {
        "rows": len(pred),
        "received": len(ok_idx),
        "uart_errors": uart_errors,
        "seconds": elapsed,
        "port": args.port,
        "baud": args.baud,
        "ready": ready,
        "model_sha256": EXPECTED_SHA,
        "onchip_mean_us": (sum(times[i] for i in ok_idx) / len(ok_idx)) if ok_idx and all(times[i] >= 0 for i in ok_idx) else None,
        "scope": "full training-set refit over UART, not held-out test accuracy",
    }
    if y is not None and ok_idx:
        correct = sum(int(pred[i] == y[i]) for i in ok_idx)
        report.update(
            correct=correct,
            errors=len(ok_idx) - correct,
            accuracy_percent=100 * correct / len(ok_idx),
        )
    if pc is not None and ok_idx:
        import numpy as np

        batch = np.asarray([x[i] for i in ok_idx], dtype=np.int64)
        pc_pred = pc.predict_batch(batch).tolist()
        agree = sum(int(pred[ok_idx[j]] == int(pc_pred[j])) for j in range(len(ok_idx)))
        report.update(
            pc_agreement_count=agree,
            pc_agreement_percent=100 * agree / len(ok_idx),
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "prediction", "label", "correct", "t_us"])
        for i, pcls in enumerate(pred):
            label = "" if y is None else y[i]
            correct = "" if y is None or pcls < 0 else int(pcls == y[i])
            w.writerow([i, pcls, label, correct, times[i]])
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Results:", args.output)


if __name__ == "__main__":
    main()
