"""Fine-tune a YOLO model from a frozen config.

    python -m src.train.train --config model_dev/c01_yolo26s_1280_v0val_pos/config.yaml --seed 0
    python -m src.train.train --config <...>/config.yaml --smoke    # 1 epoch on 10% of train

Writes model_dev/<config>/seed<N>/ (Ultralytics output: results.csv, args.yaml, weights/)
plus meta.json (git commit, config hash, versions) and train.log (all terminal output). Smoke runs go to <config>/smoke/ and
are overwritten each time.

A config is frozen once it has a real run: if config.yaml no longer matches the hash in
an existing run's meta.json, training refuses to start. Make a new config folder instead.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src import index
from src.paths import PRETRAINED, ROOT, split_yaml


def git_state():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def versions():
    import torch
    import ultralytics

    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def tee_output(log_path):
    """Copy everything written to stdout/stderr (including subprocesses and C extensions) to log_path."""
    sys.stdout.flush()
    sys.stderr.flush()
    tee = subprocess.Popen(["tee", "-a", str(log_path)], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())
    sys.stdout.reconfigure(line_buffering=True)  # keep prints and tracebacks in order in the log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 10%% of train, written to <config>/smoke/")
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    cfg_dir = cfg_path.parent
    cfg_text = cfg_path.read_text()
    cfg = yaml.safe_load(cfg_text)
    cfg_hash = hashlib.sha256(cfg_text.encode()).hexdigest()

    # Frozen-config check against every real run of this config
    for meta_path in sorted(cfg_dir.glob("seed*/meta.json")):
        if json.loads(meta_path.read_text())["config_sha256"] != cfg_hash:
            sys.exit(f"{cfg_path} changed since {meta_path.parent.name} was trained. Make a new config folder.")

    run_name = "smoke" if args.smoke else f"seed{args.seed}"
    run_dir = cfg_dir / run_name
    if args.smoke:
        shutil.rmtree(run_dir, ignore_errors=True)
    elif run_dir.exists():
        sys.exit(f"{run_dir} already exists. Use another --seed.")

    data = split_yaml(cfg["split"])
    if not data.exists():
        sys.exit(f"{data} not found. Run: python -m src.data.make_yolo_split (see its --help)")

    # Create the run dir up front so the log starts with the first line of output.
    # Uniqueness was checked above, so Ultralytics is told the dir may exist.
    run_dir.mkdir(parents=True)
    tee_output(run_dir / "train.log")

    overrides = dict(cfg.get("train", {}))
    if args.smoke:
        overrides.update(epochs=1, fraction=0.1)

    # Ultralytics downloads assets (pretrained weights, the AMP-check model) into the cwd,
    # so work from pretrained/. Every other path is absolute.
    PRETRAINED.mkdir(exist_ok=True)
    os.chdir(PRETRAINED)

    from ultralytics import YOLO

    model = YOLO(cfg["model"])  # e.g. yolo26s.pt, downloaded into pretrained/ on first use

    meta = {
        "config_name": cfg_dir.name,
        "run": run_name,
        "seed": args.seed,
        "smoke": args.smoke,
        "config_sha256": cfg_hash,
        "config": cfg,
        "data_yaml": str(data),
        "git": git_state(),
        "versions": versions(),
        "argv": sys.argv,
        "started": now(),
        "status": "running",
    }

    def write_meta(trainer):
        Path(trainer.save_dir, "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    def on_end(trainer):
        meta.update(status="finished", finished=now())
        write_meta(trainer)

    model.add_callback("on_train_start", write_meta)
    model.add_callback("on_train_end", on_end)

    if not args.smoke:
        index.upsert(cfg_dir.name, description=cfg.get("description", ""))

    model.train(data=str(data), project=str(cfg_dir), name=run_name, exist_ok=True, seed=args.seed, **overrides)
    print(f"\nrun dir: {run_dir}\nnext: python -m src.eval.evaluate --run {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
