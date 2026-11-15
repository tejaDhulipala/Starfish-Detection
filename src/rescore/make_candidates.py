"""Cut the ensemble's val candidate boxes into crops for the rescoring classifier.

    python -m src.rescore.make_candidates --models c04 c06 c07 c08

Candidates: every WBF-fused box (src.eval.ensemble, from each model's cached val predictions) with
fused conf > --min-conf. Each is cut as a square centred on the box, of side 1.2 * max(w, h) (the
classifier's view), but stored with a 1.5x margin around that (side 1.8 * max(w, h)) so training can
rotate up to 45 degrees and centre-crop without exposing corners. Out-of-frame parts are padded with
114 (Ultralytics' pad grey). Crops are stored unresized as PNG.

Target: iou = max IoU of the box with that frame's ground truth (raw train.csv boxes), 0 if none.
The classifier learns 7 cumulative bins [iou > 0.3, > 0.4, ..., > 0.9].

Folds: val sequences are assigned to --folds folds (seeded), greedily balancing ground-truth boxes,
so out-of-fold scores never come from a classifier that saw the same sequence.

Writes data/cls/<name>/crops/<idx>.png and data/cls/<name>/candidates.csv.
"""

import argparse
import csv
import random

import cv2
import numpy as np

from src.eval.ensemble import fused_val_preds
from src.eval.f2 import iou_matrix
from src.paths import DATA, split_dir

VIEW = 1.2  # classifier view: side = VIEW * max(w, h)
MARGIN = 1.5  # stored side = MARGIN * view side (>= sqrt(2) for 45-degree rotation)


def box_sharpness(gray, x, y, w, h):
    """Laplacian variance of the grey pixels inside the box (clipped to the frame), as src/eval/fail_analysis.py."""
    H, W = gray.shape[:2]
    c = gray[int(max(y, 0)) : int(min(y + h, H)), int(max(x, 0)) : int(min(x + w, W))]
    return float(cv2.Laplacian(c, cv2.CV_32F).var()) if c.size else 0.0


def cut(img, cx, cy, side):
    """Square of the given side centred on (cx, cy), padded with 114 outside the frame."""
    s = max(int(round(side)), 4)
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    h, w = img.shape[:2]
    pad = max(0, -x0, -y0, x0 + s - w, y0 + s - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x0, y0 = x0 + pad, y0 + pad
    return img[y0 : y0 + s, x0 : x0 + s]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["c04", "c06", "c07", "c08"])
    ap.add_argument("--min-conf", type=float, default=0.01)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="default: ens_<models>")
    args = ap.parse_args()

    name = args.name or "ens_" + "_".join(args.models)
    out = DATA / "cls" / name
    (out / "crops").mkdir(parents=True, exist_ok=True)

    fused, frames = fused_val_preds(args.models)
    seq = {r["image_id"]: r["sequence"] for r in csv.DictReader(open(split_dir("seq20val_pos") / "split.csv"))}

    gt_per_seq = {}
    for iid, _, g in frames:
        gt_per_seq[seq[iid]] = gt_per_seq.get(seq[iid], 0) + len(g)
    seqs = list(gt_per_seq)
    random.Random(args.seed).shuffle(seqs)
    fold_of, load = {}, [0] * args.folds
    for s in sorted(seqs, key=lambda s: -gt_per_seq[s]):  # biggest first, into the lightest fold
        k = int(np.argmin(load))
        fold_of[s] = k
        load[k] += gt_per_seq[s]
    print(f"{len(seqs)} val sequences in {args.folds} folds, GT boxes per fold {load}")

    rows, idx = [], 0
    for n, (iid, path, g) in enumerate(frames):
        p = fused[iid]
        p = p[p[:, 0] > args.min_conf]
        if not len(p):
            continue
        img = cv2.imread(str(path))
        ious = iou_matrix(p[:, 1:], g).max(1) if len(g) else np.zeros(len(p))
        for (c, x, y, w, h), iou in zip(p, ious):
            crop = cut(img, x + w / 2, y + h / 2, MARGIN * VIEW * max(w, h))
            cv2.imwrite(str(out / "crops" / f"{idx}.png"), crop)
            rows.append([idx, iid, seq[iid], fold_of[seq[iid]], f"{c:.5f}", f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}", f"{iou:.4f}"])
            idx += 1
        print(f"\r  frames {n + 1}/{len(frames)}, crops {idx}", end="", flush=True)
    print()
    with open(out / "candidates.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["idx", "image_id", "sequence", "fold", "conf", "x", "y", "w", "h", "iou"])
        wr.writerows(rows)
    ious = np.array([float(r[-1]) for r in rows])
    print(f"wrote {len(rows)} crops to {out}; iou > 0.3: {(ious > 0.3).sum()}, > 0.5: {(ious > 0.5).sum()}")


if __name__ == "__main__":
    main()
