"""Fine-tune a YOLO model from a frozen config.

    python -m src.train.train --config model_dev/c01_yolo26s_1280_v0val_pos/config.yaml --seed 0
    python -m src.train.train --config <...>/config.yaml --smoke    # 1 epoch on 10% of train

Writes model_dev/<config>/seed<N>/ (Ultralytics output: results.csv, args.yaml, weights/)
plus meta.json (git commit, config hash, versions) and train.log (all terminal output). Smoke runs go to <config>/smoke/ and
are overwritten each time.

Config keys: model, split, train (passed to Ultralytics), and optional fitness: f2, which
makes per-epoch val compute F2 and use it for best.pt and early stopping
(src/train/f2_validator.py). Without it, Ultralytics' default fitness (mAP50-95) is used.
Optional val_imgsz / val_batch run per-epoch validation at a different size than training
(for crop training: train on upscaled crops at 1280, validate on full frames at e.g. 3648).
Optional rot90: p adds an exact random 90-degree rotation (src/train/rot90.py).
model may be a pretrained file name (looked up in pretrained/) or a path relative to the repo
root, e.g. model_dev/<config>/seed0/weights/best.pt to fine-tune an earlier run; its sha256
is recorded in meta.json.

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


def with_val_overrides(base, val_imgsz=None, val_batch=None):
    """Trainer subclass whose per-epoch validation runs at val_imgsz with batch val_batch.

    For crop training (train imgsz 1280 on upscaled crops), val must run on full frames at
    the upscaled size, or F2, best.pt and early stopping would be measured at the wrong scale.
    Ultralytics builds the val loader from the training args, so imgsz is swapped while it is built.
    """

    class ValOverrideTrainer(base):
        def build_dataset(self, img_path, mode="train", batch=None):
            if mode != "val" or not val_imgsz:
                return super().build_dataset(img_path, mode, batch)
            train_imgsz, self.args.imgsz = self.args.imgsz, val_imgsz
            try:
                return super().build_dataset(img_path, mode, batch)
            finally:
                self.args.imgsz = train_imgsz

        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "val" and val_batch:
                batch_size = val_batch
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        def get_validator(self):
            validator = super().get_validator()
            if val_imgsz:
                validator.args.imgsz = val_imgsz
            return validator

    ValOverrideTrainer.__name__ = f"{base.__name__}ValOverride"
    return ValOverrideTrainer


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

    model_src, model_sha = cfg["model"], None
    if "/" in model_src:  # weights of an earlier run, relative to the repo root
        model_path = (ROOT / model_src).resolve()
        if not model_path.exists():
            sys.exit(f"model {model_path} not found")
        model_src, model_sha = str(model_path), hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = YOLO(model_src)  # e.g. yolo26s.pt, downloaded into pretrained/ on first use

    meta = {
        "config_name": cfg_dir.name,
        "run": run_name,
        "seed": args.seed,
        "smoke": args.smoke,
        "config_sha256": cfg_hash,
        "config": cfg,
        "data_yaml": str(data),
        "model_sha256": model_sha,
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

    fitness = cfg.get("fitness", "map")
    if fitness not in ("map", "f2"):
        sys.exit(f"unknown fitness '{fitness}' in {cfg_path} (use map or f2)")
    trainer = None
    if fitness == "f2":
        from src.train.f2_validator import F2DetectionTrainer

        trainer = F2DetectionTrainer
    if cfg.get("val_imgsz") or cfg.get("val_batch"):
        from ultralytics.models.yolo.detect import DetectionTrainer

        trainer = with_val_overrides(trainer or DetectionTrainer, cfg.get("val_imgsz"), cfg.get("val_batch"))

    if cfg.get("rot90"):
        from ultralytics.models.yolo.detect import DetectionTrainer

        from src.train.rot90 import with_rot90

        trainer = with_rot90(trainer or DetectionTrainer, float(cfg["rot90"]))

    model.train(trainer=trainer, data=str(data), project=str(cfg_dir), name=run_name, exist_ok=True, seed=args.seed, **overrides)
    print(f"\nrun dir: {run_dir}\nnext: python -m src.eval.evaluate --run {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
