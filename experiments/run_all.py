"""Run matched five-fold FP32 experiments on the archived grouped folds.

Every variant in ``configs/variants.json`` changes only the listed options of
the main recipe in ``configs/main_recipe.json``; splits, seeds (42-46), training
budget and validation-based checkpoint selection are identical across variants.

    python experiments/run_all.py --variants baseline no_physics            # Table VII, physics control
    python experiments/run_all.py --variants no_ala no_pairwise no_traj     # Table VII, auxiliary losses
    python experiments/run_all.py --table VI                                # Table VI, nine settings
    python experiments/run_all.py --dry-run                                 # print commands only

Completed folds (``COMPLETE.json``) are skipped, so the command can be repeated
after an interruption.  Outputs go to ``runs/<variant>/fold<k>/``.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = REPO / "runs"


def build_command(spec, fold, out, args):
    recipe = json.loads((HERE / "configs/main_recipe.json").read_text(encoding="utf8"))
    options = {k: (v if isinstance(v, list) else [v]) for k, v in recipe.items()
               if not any(k.startswith(p) for p in spec.get("drop_prefix", []))}
    for key, value in spec.get("set", {}).items():
        options[key] = [str(value)]
    if args.epochs is not None:  # smoke tests only; the paper uses the recipe value
        options["--epochs"] = [str(args.epochs)]
    options.update({
        "--device": [args.device],
        "--x": [str(REPO / "data/x_train.csv")],
        "--y": [str(REPO / "data/y_train.csv")],
        "--output": [str(out)],
        "--ablate-export-dir": [str(out / "models")],
        "--sweep-fold": [str(fold)],
        "--teacher-cache-dir": [str(RUNS / "teacher_cache")],
        "--seeds": [str(42 + fold)],
    })
    command = [sys.executable, "-u", str(HERE / "run_fold.py")]
    for key, values in options.items():
        command += [key, *values]
    return command


def main():
    variants = json.loads((HERE / "configs/variants.json").read_text(encoding="utf8"))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", choices=list(variants))
    parser.add_argument("--table", choices=["VI", "VII"], help="run every variant used by this table")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)), choices=range(5))
    parser.add_argument("--workers", type=int, default=3, help="parallel training processes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, help="override the epoch budget (smoke tests only)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    names = args.variants or [n for n, s in variants.items() if args.table is None or args.table in s["table"].split(", ")]

    (RUNS / "teacher_cache").mkdir(parents=True, exist_ok=True)
    lock, state, started = threading.Lock(), {}, time.time()

    def save():
        (RUNS / "run_status.json").write_text(json.dumps({"elapsed_seconds": time.time() - started, "jobs": state}, indent=2), encoding="utf8")

    def run(name, fold):
        key = f"{name}/fold{fold}"
        out = RUNS / name / f"fold{fold}"
        if (out / "COMPLETE.json").exists():
            print("SKIP (complete)", key, flush=True)
            return
        command = build_command(variants[name], fold, out, args)
        if args.dry_run:
            print(key, " ".join(command[3:]), flush=True)
            return
        out.mkdir(parents=True, exist_ok=True)
        (out / "command.json").write_text(json.dumps(command, indent=2), encoding="utf8")
        with lock:
            state[key] = {"state": "running", "start": time.time()}
            save()
        print("START", key, flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="6", MKL_NUM_THREADS="6")
        with (out / "train.log").open("w", encoding="utf8") as log:
            code = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).wait()
        seconds = time.time() - state[key]["start"]
        with lock:
            state[key].update(state="complete" if code == 0 else "failed", seconds=seconds, exit_code=code)
            save()
        print("DONE", key, "exit", code, "seconds", round(seconds), flush=True)
        if code:
            raise RuntimeError(f"{key} failed; see {out / 'train.log'}")
        (out / "COMPLETE.json").write_text(json.dumps({"seconds": seconds, "returncode": 0}), encoding="utf8")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for job in as_completed([pool.submit(run, n, f) for n in names for f in args.folds]):
            job.result()
    if not args.dry_run:
        print(f"Finished {len(names)} variant(s) x {len(args.folds)} fold(s) in {round((time.time() - started) / 60)} min. "
              "Next: python experiments/aggregate.py", flush=True)


if __name__ == "__main__":
    main()
