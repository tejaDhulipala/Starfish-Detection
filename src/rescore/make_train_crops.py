"""Extra classifier training crops from the TRAIN frames (no detector needed, no val leakage).

    python -m src.rescore.make_train_crops

The val candidates (make_candidates) give only ~600 positive crops per training fold. This adds:
  - jittered ground truth: for each train GT box, --jitter boxes with random shift and scale,
    accepted so that their IoU with the GT is spread roughly uniformly over [0.1, 1.0], which
    covers all 7 IoU bins;
  - background: per train frame (with or without starfish), --neg random boxes with sides drawn
    from the GT side distribution, kept only if they overlap no GT (IoU 0).
Crops are cut exactly as in make_candidates (side 1.8 * max(w, h) incl. the rotation margin), with
target iou = max IoU with the frame's GT. fold = -1: always training data, never scored.
"""

import argparse
import csv
import random

import cv2
import numpy as np

from src.eval.f2 import iou_matrix
from src.paths import DATA, IMG_H, IMG_W, TRAIN_CSV, YOLO_DATA, split_dir
from src.rescore.make_candidates import MARGIN, VIEW, cut


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jitter", type=int, default=3)
    ap.add_argument("--neg", type=float, default=1.0, help="background boxes per train frame (fractional = probability)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="train_jitter")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    import ast

    gt = {}
    for r in csv.DictReader(open(TRAIN_CSV)):
        b = ast.literal_eval(r["annotations"])
        gt[r["image_id"]] = np.array([[d["x"], d["y"], d["width"], d["height"]] for d in b], dtype=np.float64).reshape(-1, 4)
    train = [r for r in csv.DictReader(open(split_dir("seq20val_pos") / "split.csv")) if r["split"] == "train"]
    sides = np.concatenate([np.maximum(g[:, 2], g[:, 3]) for g in gt.values() if len(g)])

    out = DATA / "cls" / args.name
    (out / "crops").mkdir(parents=True, exist_ok=True)
    rows, idx = [], 0
    for n, r in enumerate(train):
        g = gt[r["image_id"]]
        boxes = []
        for b in g:
            want = [rng.uniform(0.1, 1.0) for _ in range(args.jitter)]
            for target in want:  # rejection-sample a jitter whose IoU lands near the target
                for _ in range(200):
                    s = rng.uniform(0.5, 1.6)
                    w, h = b[2] * s * rng.uniform(0.85, 1.15), b[3] * s * rng.uniform(0.85, 1.15)
                    cx = b[0] + b[2] / 2 + rng.uniform(-0.8, 0.8) * b[2]
                    cy = b[1] + b[3] / 2 + rng.uniform(-0.8, 0.8) * b[3]
                    j = np.array([cx - w / 2, cy - h / 2, w, h])
                    if abs(iou_matrix(j[None], b[None])[0, 0] - target) < 0.05:
                        boxes.append(j)
                        break
        k = int(args.neg) + (rng.random() < args.neg - int(args.neg))
        for _ in range(k):
            for _ in range(50):
                s = float(rng.choice(sides)) * rng.uniform(0.8, 1.25)
                w, h = s * rng.uniform(0.7, 1.0), s * rng.uniform(0.7, 1.0)
                j = np.array([rng.uniform(0, IMG_W - w), rng.uniform(0, IMG_H - h), w, h])
                if not len(g) or iou_matrix(j[None], g).max() == 0:
                    boxes.append(j)
                    break
        if not boxes:
            continue
        img = cv2.imread(str(YOLO_DATA / "images" / f"video_{r['video_id']}" / f"{r['video_frame']}.jpg"))
        bx = np.array(boxes)
        ious = iou_matrix(bx, g).max(1) if len(g) else np.zeros(len(bx))
        for (x, y, w, h), iou in zip(bx, ious):
            cv2.imwrite(str(out / "crops" / f"{idx}.png"), cut(img, x + w / 2, y + h / 2, MARGIN * VIEW * max(w, h)))
            rows.append([idx, r["image_id"], r["sequence"], -1, "", f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}", f"{iou:.4f}"])
            idx += 1
        if n % 200 == 0:
            print(f"\r  frames {n + 1}/{len(train)}, crops {idx}", end="", flush=True)
    print()
    with open(out / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "image_id", "sequence", "fold", "conf", "x", "y", "w", "h", "iou"])
        w.writerows(rows)
    ious = np.array([float(r[-1]) for r in rows])
    print(f"wrote {len(rows)} crops to {out}; iou > 0.3: {(ious > 0.3).sum()}, > 0.5: {(ious > 0.5).sum()}, = 0: {(ious == 0).sum()}")


if __name__ == "__main__":
    main()
