"""Full-val F2 for a trained run: predict once at a low confidence, then sweep the cutoff.

    python -m src.eval.evaluate --run model_dev/c01_yolo26s_1280_v0val_pos/seed0
    python -m src.eval.evaluate --run <run dir> --weights best last epoch12
    python -m src.eval.evaluate --run <run dir> --limit 300     # quick check on a strided subset

Writes <run>/eval/val_preds_<weights>.csv (cached; --force to redo) and <weights>_f2.json,
and records the best result in model_dev/index.csv. Ground truth is the raw train.csv boxes
(unclipped), since that is what Kaggle scores against. Predicted coordinates are rounded
to integers, as in the submission string.
"""

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src import index
from src.eval.f2 import IOU_THRESHOLDS, f2_sweep
from src.paths import ROOT, TRAIN_CSV, YOLO_DATA, split_dir

CONF_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def val_frames(split):
    """[(image_id, image path, gt (M, 4))] for the full val set, in split.csv order."""
    gt = {}
    with open(TRAIN_CSV) as f:
        for r in csv.DictReader(f):
            boxes = ast.literal_eval(r["annotations"])
            gt[r["image_id"]] = np.array([[b["x"], b["y"], b["width"], b["height"]] for b in boxes], dtype=np.float64).reshape(-1, 4)
    out = []
    with open(split_dir(split) / "split.csv") as f:
        for r in csv.DictReader(f):
            if r["split"] == "val":
                img = YOLO_DATA / "images" / f"video_{r['video_id']}" / f"{r['video_frame']}.jpg"
                out.append((r["image_id"], img, gt[r["image_id"]]))
    return out


def predict_to_csv(weights_path, frames, imgsz, batch, out_csv):
    from src.inference.predict import Detector

    det = Detector(weights_path, imgsz=imgsz, conf=CONF_GRID[0])
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "conf", "x", "y", "w", "h"])
        for i in range(0, len(frames), batch):
            chunk = frames[i : i + batch]
            imgs = [np.asarray(Image.open(p).convert("RGB")) for _, p, _ in chunk]
            for (iid, _, _), dets in zip(chunk, det(imgs)):
                for c, x, y, bw, bh in dets:
                    w.writerow([iid, f"{c:.4f}", round(x), round(y), round(bw), round(bh)])
            print(f"\r  predicted {min(i + batch, len(frames))}/{len(frames)}", end="", flush=True)
    print()


def load_preds(path):
    preds = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            preds.setdefault(r["image_id"], []).append([float(r[k]) for k in ("conf", "x", "y", "w", "h")])
    return {k: np.array(v) for k, v in preds.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--weights", nargs="+", default=["best", "last"], help="names in <run>/weights/ without .pt")
    ap.add_argument("--batch", type=int, default=None, help="default: the config's val_batch, else 16")
    ap.add_argument("--limit", type=int, default=None, help="evaluate a strided subset of about N val frames")
    ap.add_argument("--force", action="store_true", help="re-run prediction even if cached")
    args = ap.parse_args()

    run = args.run.resolve()
    meta = json.loads((run / "meta.json").read_text())
    cfg = meta["config"]
    imgsz = cfg.get("val_imgsz") or cfg.get("train", {}).get("imgsz", 640)
    if args.batch is None:
        args.batch = cfg.get("val_batch") or 16
    frames = val_frames(cfg["split"])
    suffix = ""
    if args.limit:
        frames = frames[:: math.ceil(len(frames) / args.limit)]
        suffix = f"_limit{args.limit}"
    n_gt = sum(len(g) for *_, g in frames)
    print(f"{run.relative_to(ROOT)}: {len(frames)} val frames, {n_gt} boxes, imgsz {imgsz}")

    for wname in args.weights:
        wpath = run / "weights" / f"{wname}.pt"
        if not wpath.exists():
            print(f"skip {wname}: {wpath} not found", file=sys.stderr)
            continue
        preds_csv = run / "eval" / f"val_preds_{wname}{suffix}.csv"
        if args.force or not preds_csv.exists():
            print(f"{wname}: predicting")
            predict_to_csv(wpath, frames, imgsz, args.batch, preds_csv)
        preds = load_preds(preds_csv)

        sweep = f2_sweep([(preds.get(iid, np.zeros((0, 5))), g) for iid, _, g in frames], CONF_GRID)
        b = int(np.argmax(sweep["f2"]))
        tp, fp, fn = (sweep[k][:, b] for k in ("tp", "fp", "fn"))
        result = {
            "run": meta["run"],
            "config": meta["config_name"],
            "weights": wname,
            "n_frames": len(frames),
            "n_gt": n_gt,
            "best_conf": float(CONF_GRID[b]),
            "best_f2": float(sweep["f2"][b]),
            "at_best_conf": {
                f"{t:.2f}": {"f2": float(sweep["f2_per_iou"][i, b]), "precision": float(tp[i] / max(tp[i] + fp[i], 1)),
                             "recall": float(tp[i] / max(tp[i] + fn[i], 1))}
                for i, t in enumerate(IOU_THRESHOLDS)
            },
            "curve": {f"{c:.2f}": float(v) for c, v in zip(CONF_GRID, sweep["f2"])},
        }
        (run / "eval" / f"{wname}{suffix}_f2.json").write_text(json.dumps(result, indent=2) + "\n")
        r50 = result["at_best_conf"]["0.50"]
        print(f"{wname}: F2 {result['best_f2']:.4f} at conf {result['best_conf']:.2f}  "
              f"(IoU 0.5: P {r50['precision']:.3f}, R {r50['recall']:.3f})")
        if not args.limit and not meta["smoke"] and index.record_f2(meta["config_name"], result["best_f2"], result["best_conf"], meta["run"], wname):
            print(f"  new best for {meta['config_name']} in model_dev/index.csv")


if __name__ == "__main__":
    main()
