"""Make a crop-based training split for upscaled training without full-frame memory cost.

Idea: to train at scale S (e.g. 2.85x, i.e. full frames at 1280 * 2.85 = 3648 px) without
fitting a 3648 px frame in memory, cut crops of (1280 / S) x (720 / S) source pixels and
train with `imgsz: 1280, rect: true`. Ultralytics resizes each crop so its long side is
1280, which is exactly the S-times upscale, and rect batches keep the 16:9 shape (no square
padding). Validation and inference then run on whole frames at imgsz 1280 * S, so the
starfish have the same pixel size in training and at test time.

Built on an existing full-frame split (--base): train crops come only from that split's
train sequences; val is the base split's val frames, unchanged.

Crops (source-pixel coordinates, seeded):
  - Positive crops, per train frame with boxes: take the first box not yet fully inside a
    crop, place a crop at a random position that fully contains it, then mark every box
    fully inside as covered. Repeat until every box is fully inside at least one crop.
  - Negative crops: about --neg-frac of the total, random crops that overlap no box,
    taken from any frame of the train sequences (with or without boxes).
  - Labels: every box overlapping a crop, clipped to it, kept if the clipped part is more
    than 2 px per side (in training pixels) and more than 10% of the box area. These are
    Ultralytics' own thresholds for boxes cut by its augmentations.
  - Saved as PNG (lossless), so crops do not pick up a second round of JPEG artifacts.

Layout:
    data/yolo/crops/<name>/images/video_N/<frame>_<k>.png   (+ labels/... .txt)
    data/yolo/splits/<name>/train.txt      crop paths
    data/yolo/splits/<name>/val.txt        copied from the base split (full frames)
    data/yolo/splits/<name>/val_fast.txt   every --val-fast-step-th val frame
    data/yolo/splits/<name>/split.csv      copied from the base split (used by src.eval.evaluate)
    data/yolo/splits/<name>/crops.csv      one row per crop: where it came from
    data/yolo/<name>.yaml

Usage (from the repo root):
    python -m src.data.make_crop_split --base seq20val_pos --scale 2.85
"""

import argparse
import ast
import csv
import random
import shutil
from pathlib import Path

import cv2

from src.paths import IMG_H, IMG_W, TRAIN_CSV, YOLO_DATA, split_dir

MIN_SIDE_TRAIN_PX = 2  # Ultralytics box_candidates wh_thr
MIN_AREA_FRAC = 0.1  # Ultralytics box_candidates area_thr


def clip_box(b, x0, y0, cw, ch):
    """Box (x, y, w, h) clipped to the crop at (x0, y0, cw, ch), in crop coordinates, or None."""
    x1, y1 = max(b[0], x0), max(b[1], y0)
    x2, y2 = min(b[0] + b[2], x0 + cw), min(b[1] + b[3], y0 + ch)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 - x0, y1 - y0, x2 - x1, y2 - y1)


def fully_inside(b, x0, y0, cw, ch):
    return b[0] >= x0 and b[1] >= y0 and b[0] + b[2] <= x0 + cw and b[1] + b[3] <= y0 + ch


def crop_labels(boxes, x0, y0, cw, ch, scale):
    """YOLO label lines (class cx cy w h, normalized to the crop) for boxes kept in this crop."""
    lines = []
    for b in boxes:
        c = clip_box(b, x0, y0, cw, ch)
        if c is None:
            continue
        if min(c[2], c[3]) * scale <= MIN_SIDE_TRAIN_PX or c[2] * c[3] <= MIN_AREA_FRAC * b[2] * b[3]:
            continue
        cx, cy = (c[0] + c[2] / 2) / cw, (c[1] + c[3] / 2) / ch
        lines.append(f"0 {cx:.6f} {cy:.6f} {c[2] / cw:.6f} {c[3] / ch:.6f}")
    return lines


def place(lo, hi, rng):
    """Random integer start in [lo, hi]; if the range is empty (object bigger than the crop), center it."""
    return rng.randint(lo, hi) if lo <= hi else (lo + hi) // 2


def positive_crops(boxes, cw, ch, rng):
    """Crop origins so that every box is fully inside at least one crop (if it fits)."""
    origins, covered = [], [False] * len(boxes)
    for i, b in enumerate(boxes):
        if covered[i]:
            continue
        # crop must contain [b.x, b.x + b.w]: x0 in [b.x + b.w - cw, b.x], and inside the frame
        x0 = place(max(0, int(b[0] + b[2]) - cw + 1), min(int(b[0]), IMG_W - cw), rng)
        y0 = place(max(0, int(b[1] + b[3]) - ch + 1), min(int(b[1]), IMG_H - ch), rng)
        x0, y0 = min(max(x0, 0), IMG_W - cw), min(max(y0, 0), IMG_H - ch)
        origins.append((x0, y0))
        covered[i] = True
        for j, bj in enumerate(boxes):
            if not covered[j] and fully_inside(bj, x0, y0, cw, ch):
                covered[j] = True
    return origins


def clip_to_frame(b):
    x1, y1 = max(0, b["x"]), max(0, b["y"])
    x2, y2 = min(IMG_W, b["x"] + b["width"]), min(IMG_H, b["y"] + b["height"])
    return (x1, y1, x2 - x1, y2 - y1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="seq20val_pos", help="existing full-frame split to take sequences and val from")
    ap.add_argument("--scale", type=float, default=2.85, help="training upscale; crops are (1280/scale) x (720/scale) px")
    ap.add_argument("--neg-frac", type=float, default=0.15, help="negative crops as a fraction of all crops")
    ap.add_argument("--val-fast-step", type=int, default=1, help="per-epoch val uses every Nth val frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="default: crops<scale*100>_<base>")
    args = ap.parse_args()

    name = args.name or f"crops{round(args.scale * 100)}_{args.base}"
    cw, ch = round(IMG_W / args.scale), round(IMG_H / args.scale)
    train_scale = IMG_W / cw  # actual upscale Ultralytics applies (long side -> 1280)
    rng = random.Random(args.seed)

    base_dir = split_dir(args.base)
    with open(base_dir / "split.csv") as f:
        base_rows = list(csv.DictReader(f))
    val_seqs = {(r["video_id"], r["sequence"]) for r in base_rows if r["split"] == "val"}

    frames = []  # every frame of the train sequences: (image_id, video_id, video_frame, boxes)
    with open(TRAIN_CSV) as f:
        for r in csv.DictReader(f):
            if (r["video_id"], r["sequence"]) in val_seqs:
                continue
            boxes = [clip_to_frame(b) for b in ast.literal_eval(r["annotations"])]
            boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]
            frames.append((r["image_id"], r["video_id"], r["video_frame"], boxes))
    base_train = {r["image_id"] for r in base_rows if r["split"] == "train"}
    pos_frames = [fr for fr in frames if fr[0] in base_train and fr[3]]

    plan = []  # (frame, x0, y0, kind)
    for fr in pos_frames:
        plan += [(fr, x0, y0, "pos") for x0, y0 in positive_crops(fr[3], cw, ch, rng)]
    n_neg = round(len(plan) * args.neg_frac / (1 - args.neg_frac))
    tries = 0
    while n_neg > 0 and tries < 100 * n_neg:
        tries += 1
        fr = rng.choice(frames)
        x0, y0 = rng.randint(0, IMG_W - cw), rng.randint(0, IMG_H - ch)
        if all(clip_box(b, x0, y0, cw, ch) is None for b in fr[3]):
            plan.append((fr, x0, y0, "neg"))
            n_neg -= 1

    crop_root = YOLO_DATA / "crops" / name
    shutil.rmtree(crop_root, ignore_errors=True)
    out_split = split_dir(name)
    shutil.rmtree(out_split, ignore_errors=True)
    out_split.mkdir(parents=True)

    plan.sort(key=lambda p: (p[0][1], int(p[0][2])))  # by frame, so each frame is decoded once
    rows, paths, cache_key, img = [], [], None, None
    per_frame = {}
    for (iid, vid, vframe, boxes), x0, y0, kind in plan:
        if cache_key != iid:
            img = cv2.imread(str(YOLO_DATA / "images" / f"video_{vid}" / f"{vframe}.jpg"))
            cache_key = iid
        k = per_frame[iid] = per_frame.get(iid, -1) + 1
        rel = Path(f"video_{vid}") / f"{vframe}_{k}.png"
        ip, lp = crop_root / "images" / rel, crop_root / "labels" / rel.with_suffix(".txt")
        ip.parent.mkdir(parents=True, exist_ok=True)
        lp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ip), img[y0 : y0 + ch, x0 : x0 + cw], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        lines = crop_labels(boxes, x0, y0, cw, ch, train_scale) if kind == "pos" else []
        lp.write_text("\n".join(lines) + ("\n" if lines else ""))
        paths.append(ip)
        rows.append({"crop": str(rel), "image_id": iid, "x0": x0, "y0": y0, "w": cw, "h": ch, "kind": kind, "n_boxes": len(lines)})
        if len(rows) % 500 == 0:
            print(f"\r  wrote {len(rows)}/{len(plan)} crops", end="", flush=True)
    print(f"\r  wrote {len(rows)}/{len(plan)} crops")

    rng.shuffle(paths)  # Ultralytics shuffles anyway; keeps --smoke's fraction=0.1 representative
    (out_split / "train.txt").write_text("".join(f"{p}\n" for p in paths))
    with open(out_split / "crops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shutil.copy(base_dir / "split.csv", out_split / "split.csv")
    val = (base_dir / "val.txt").read_text().splitlines()
    (out_split / "val.txt").write_text("".join(f"{p}\n" for p in val))
    (out_split / "val_fast.txt").write_text("".join(f"{p}\n" for p in val[:: args.val_fast_step]))

    (YOLO_DATA / f"{name}.yaml").write_text(
        f"# Generated by src/data/make_crop_split.py --base {args.base} --scale {args.scale} "
        f"--neg-frac {args.neg_frac} --val-fast-step {args.val_fast_step} --seed {args.seed}\n"
        f"# train: {cw}x{ch} crops (upscaled {train_scale:.4f}x at imgsz 1280, rect); "
        f"val: full frames of {args.base} (validate at imgsz {round(IMG_W * args.scale / 32) * 32})\n"
        f"path: {YOLO_DATA}\n"
        f"train: splits/{name}/train.txt\n"
        f"val: splits/{name}/val_fast.txt\n"
        "names:\n  0: cots\n"
    )

    n_pos = sum(r["kind"] == "pos" for r in rows)
    n_boxes = sum(r["n_boxes"] for r in rows)
    print(f"{name}: crops {cw}x{ch} (x{train_scale:.4f} at imgsz 1280)")
    print(f"  train: {len(rows)} crops = {n_pos} positive ({n_pos / len(pos_frames):.2f} per frame, "
          f"{len(pos_frames)} frames) + {len(rows) - n_pos} negative; {n_boxes} labels")
    print(f"  val: {len(val)} full frames, val_fast {len(val[:: args.val_fast_step])}")
    print(f"  yaml: {YOLO_DATA / f'{name}.yaml'}")


if __name__ == "__main__":
    main()
