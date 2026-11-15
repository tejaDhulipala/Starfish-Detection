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

    python -m src.rescore.make_train_crops --config model_dev/rescore/<run>/config.yaml

Config mode (blur-specialised set, see blur_synth): positives are jittered train GT boxes whose own
sharpness is below max_sharp (selected, never artificially blurred); negatives are neg_per_pos per
positive, a neg_loose_frac share of loose jitters on those boxes and the rest blur-matched random
background from every train-sequence frame (with or without starfish). Writes data/cls/<run>_train/.
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.eval.f2 import iou_matrix
from src.paths import DATA, IMG_H, IMG_W, TRAIN_CSV, YOLO_DATA, split_dir
from src.rescore.make_candidates import MARGIN, VIEW, box_sharpness, cut


def jitter_box(rng, b, lo, hi, tries=400, tol=0.05, targets=10):
    """Random shift/scale of GT box b (x y w h) whose IoU with b lands within tol of a target ~ U(lo, hi)
    and inside [lo, hi). Up to `targets` fresh targets are tried before giving up."""
    for _ in range(targets):
        target = rng.uniform(lo, hi)
        near_one = target > 0.85
        for _ in range(tries):
            if near_one:
                s, sh = rng.uniform(0.92, 1.08), 0.06
            elif target >= 0.7:
                s, sh = rng.uniform(0.8, 1.25), 0.25
            else:
                s, sh = rng.uniform(0.5, 1.6), 0.8
            w, h = b[2] * s * rng.uniform(0.95 if near_one else 0.85, 1.05 if near_one else 1.15), b[3] * s * rng.uniform(0.95 if near_one else 0.85, 1.05 if near_one else 1.15)
            cx = b[0] + b[2] / 2 + rng.uniform(-sh, sh) * b[2]
            cy = b[1] + b[3] / 2 + rng.uniform(-sh, sh) * b[3]
            j = np.array([cx - w / 2, cy - h / 2, w, h])
            iou = iou_matrix(j[None], b[None])[0, 0]
            if abs(iou - target) < tol and lo <= iou and (iou < hi or hi >= 1.0):
                return j
    return None


def blur_synth(cfg_path):
    import ast

    import yaml

    cfg = yaml.safe_load(open(cfg_path))
    tc, ms = cfg["train_crops"], float(cfg["max_sharp"])
    assert tc["hard_negatives"] == "none" and tc["neg_background"] == "random" and tc["neg_blur_match"]
    assert tc["pos_source"] == "train_gt" and tc["neg_frames"] == "all_train_seq"
    rng = random.Random(tc["seed"])
    T = list(csv.DictReader(open(TRAIN_CSV)))
    split = {r["image_id"]: r["split"] for r in csv.DictReader(open(split_dir("seq20val_pos") / "split.csv"))}
    val_seq = {r["sequence"] for r in T if split.get(r["image_id"]) == "val"}
    frames = [r for r in T if r["sequence"] not in val_seq]  # train-sequence frames, with or without starfish
    assert not any(split.get(r["image_id"]) == "val" for r in frames)
    gt = {}
    for r in frames:
        b = ast.literal_eval(r["annotations"])
        gt[r["image_id"]] = np.array([[d["x"], d["y"], d["width"], d["height"]] for d in b], dtype=np.float64).reshape(-1, 4)
    sides = np.concatenate([np.maximum(g[:, 2], g[:, 3]) for g in gt.values() if len(g)])
    path = lambda r: str(YOLO_DATA / "images" / f"video_{r['video_id']}" / f"{r['video_frame']}.jpg")

    # pass 1: blurry train GT boxes (sharpness of the GT box itself)
    blurry, n_gt = [], 0
    for r in frames:
        g = gt[r["image_id"]]
        if not len(g):
            continue
        gray = cv2.cvtColor(cv2.imread(path(r)), cv2.COLOR_BGR2GRAY)
        for k, b in enumerate(g):
            n_gt += 1
            if box_sharpness(gray, *b) < ms:
                blurry.append((r["image_id"], k))
    n_pos_target = len(blurry) * int(tc["pos_jitter_per_box"])
    n_neg = int(round(float(tc["neg_per_pos"]) * n_pos_target))
    n_loose = int(round(float(tc["neg_loose_frac"]) * n_neg))
    n_bg = n_neg - n_loose
    print(f"train-sequence frames {len(frames)}; GT boxes {n_gt}, blurry (< {ms}) {len(blurry)}")
    print(f"targets: positives {n_pos_target}, negatives {n_neg} (loose {n_loose}, background {n_bg})", flush=True)

    # plan boxes per frame: (box, kind)
    plan = {}
    by_frame = {}
    for iid, k in blurry:
        by_frame.setdefault(iid, []).append(k)
    loose_owner = [rng.choice(blurry) for _ in range(n_loose)]
    for iid, k in loose_owner:
        plan.setdefault(iid, []).append(("loose", k))
    for iid, ks in by_frame.items():
        for k in ks:
            plan.setdefault(iid, []).extend([("pos", k)] * int(tc["pos_jitter_per_box"]))
    per_frame = n_bg / len(frames)
    for r in frames:
        m = int(per_frame) + (rng.random() < per_frame - int(per_frame))
        plan.setdefault(r["image_id"], []).extend([("bg", None)] * m)

    out = DATA / "cls" / f"{Path(cfg_path).parent.name}_train"
    shutil.rmtree(out, ignore_errors=True)
    (out / "crops").mkdir(parents=True)
    rows, idx, cnt, fail = [], 0, {"pos": 0, "loose": 0, "bg": 0}, {"pos": 0, "loose": 0, "bg": 0}
    pos_lo, pos_hi = tc["pos_iou"]
    lo_lo, lo_hi = tc["neg_loose_iou"]
    for n, r in enumerate(frames):
        todo = plan.get(r["image_id"])
        if not todo:
            continue
        g = gt[r["image_id"]]
        img = cv2.imread(path(r))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for kind, k in todo:
            j = None
            if kind == "pos":
                j = jitter_box(rng, g[k], pos_lo, pos_hi)
            elif kind == "loose":
                for _ in range(20):  # label is the max IoU over all GT, so it must stay below the loose range's top
                    j = jitter_box(rng, g[k], lo_lo, lo_hi)
                    if j is not None and iou_matrix(j[None], g).max() < lo_hi:
                        break
                    j = None
            else:
                for _ in range(100):
                    s = float(rng.choice(sides)) * rng.uniform(0.8, 1.25)
                    w, h = s * rng.uniform(0.7, 1.0), s * rng.uniform(0.7, 1.0)
                    c = np.array([rng.uniform(0, IMG_W - w), rng.uniform(0, IMG_H - h), w, h])
                    if (not len(g) or iou_matrix(c[None], g).max() == 0) and box_sharpness(gray, *c) < ms:
                        j = c
                        break
            if j is None:
                fail[kind] += 1
                continue
            iou = float(iou_matrix(j[None], g).max()) if len(g) else 0.0
            x, y, w, h = j
            cv2.imwrite(str(out / "crops" / f"{idx}.png"), cut(img, x + w / 2, y + h / 2, MARGIN * VIEW * max(w, h)))
            rows.append([idx, r["image_id"], r["sequence"], -1, "", f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}", f"{iou:.4f}", kind, f"{box_sharpness(gray, *j):.1f}"])
            idx += 1
            cnt[kind] += 1
        if n % 500 == 0:
            print(f"\r  frames {n + 1}/{len(frames)}, crops {idx}", end="", flush=True)
    print()
    with open(out / "candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "image_id", "sequence", "fold", "conf", "x", "y", "w", "h", "iou", "kind", "sharp"])
        w.writerows(rows)
    ious = np.array([float(r[9]) for r in rows])
    print(f"wrote {len(rows)} crops to {out}: {cnt}; failed to place {fail}")
    print(f"  iou >= 0.5: {(ious >= 0.5).sum()}, 0 < iou < 0.5: {((ious > 0) & (ious < 0.5)).sum()}, iou = 0: {(ious == 0).sum()}; pos:neg = 1:{(ious < 0.5).sum() / max((ious >= 0.5).sum(), 1):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jitter", type=int, default=3)
    ap.add_argument("--neg", type=float, default=1.0, help="background boxes per train frame (fractional = probability)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="train_jitter")
    ap.add_argument("--config", default=None, help="blur-specialised set from a rescore config (see blur_synth)")
    args = ap.parse_args()
    if args.config:
        return blur_synth(args.config)
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
