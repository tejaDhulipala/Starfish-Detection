"""Check the Kaggle ONNX pipeline (kaggle/cots_onnx.py) against the Ultralytics one, then score it.

    python -m src.eval.evaluate_onnx --run model_dev/c04_yolo26s_1920_seq20val_pos_aug/seed0

1. Box-level diff against src.inference.predict on --compare frames (max |conf| and |coord| gaps).
2. Full-val F2 sweep of the ONNX predictions, cached in <run>/eval/val_preds_<weights>_onnx.csv.
   It should reproduce src.eval.evaluate's F2 for the same weights. index.csv is not updated.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.evaluate import CONF_GRID, load_preds, val_frames
from src.eval.f2 import f2_sweep
from src.paths import ROOT

sys.path.insert(0, str(ROOT / "kaggle"))
from cots_onnx import OnnxDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--weights", default="best")
    ap.add_argument("--compare", type=int, default=50, help="frames for the box-level diff (0 to skip)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run = args.run.resolve()
    cfg = json.loads((run / "meta.json").read_text())["config"]
    imgsz = cfg.get("val_imgsz") or cfg["train"]["imgsz"]
    frames = val_frames(cfg["split"])
    det = OnnxDetector(run / "weights" / f"{args.weights}.onnx", conf=CONF_GRID[0])
    print(f"{run.relative_to(ROOT)}: {len(frames)} val frames, ONNX input {det.in_h}x{det.in_w}, "
          f"providers {det.sess.get_providers()}")

    if args.compare:
        from src.inference.predict import Detector

        ref = Detector(run / "weights" / f"{args.weights}.pt", imgsz=imgsz, conf=CONF_GRID[0])
        sub = frames[:: math.ceil(len(frames) / args.compare)]
        n_ref = n_onnx = 0
        dconf = dxy = 0.0
        unmatched = 0
        for _, path, _ in sub:
            img = np.asarray(Image.open(path).convert("RGB"))
            a, b = ref([img])[0], det(img)
            n_ref, n_onnx = n_ref + len(a), n_onnx + len(b)
            for row in a[a[:, 0] >= 0.1]:  # every confident reference box must have an ONNX twin
                if len(b) == 0:
                    unmatched += 1
                    continue
                d = np.abs(b[:, 1:] - row[1:]).max(1)
                j = int(np.argmin(d))
                if d[j] > 2:
                    unmatched += 1
                    continue
                dxy, dconf = max(dxy, d[j]), max(dconf, abs(b[j, 0] - row[0]))
        print(f"compare on {len(sub)} frames: boxes {n_ref} ultralytics vs {n_onnx} onnx; for ref boxes with conf >= 0.1: "
              f"max |dconf| {dconf:.4f}, max |dxy| {dxy:.2f} px, unmatched {unmatched}")

    cache = run / "eval" / f"val_preds_{args.weights}_onnx.csv"
    if args.force or not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        with open(cache, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "conf", "x", "y", "w", "h"])
            for n, (iid, path, _) in enumerate(frames):
                for c, x, y, bw, bh in det(np.asarray(Image.open(path).convert("RGB"))):
                    w.writerow([iid, f"{c:.4f}", round(x), round(y), round(bw), round(bh)])
                print(f"\r  predicted {n + 1}/{len(frames)}", end="", flush=True)
        print()
    preds = load_preds(cache)
    sweep = f2_sweep([(preds.get(iid, np.zeros((0, 5))), g) for iid, _, g in frames], CONF_GRID)
    b = int(np.argmax(sweep["f2"]))
    print(f"ONNX {args.weights}: F2 {sweep['f2'][b]:.4f} at conf {CONF_GRID[b]:.2f}")
    ref_json = run / "eval" / f"{args.weights}_f2.json"
    if ref_json.exists():
        r = json.loads(ref_json.read_text())
        print(f"Ultralytics {args.weights}: F2 {r['best_f2']:.4f} at conf {r['best_conf']:.2f}")


if __name__ == "__main__":
    main()
