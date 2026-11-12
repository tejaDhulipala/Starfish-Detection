"""Make a grid-tile training split: every training tile is one cell of a fixed overlapping grid.

Each 1280x720 frame is cut into a cols x rows grid of square T x T tiles (default 3 x 2 of
493 px), evenly spaced so the first and last tiles touch the frame edges. At 493 px that is
x = 0, 393, 787 and y = 0, 227: about 100 px of overlap across and 266 px down, so every
starfish under ~100 px wide is fully inside at least one tile. No other crop positions are used.

Train with `imgsz: 1408` (square, rect off): Ultralytics resizes each 493 px tile to 1408,
a 2.856x upscale, about c05's 2.85x. Validate and infer on full frames at `val_imgsz: 3648`.

Tiles (from the base split's train sequences only):
  - For each box overlapping a tile, visible = (area inside the tile) / (box area), with the
    box first clipped to the frame.
  - A tile with any box at 0 < visible < --min-visible (a "sliver") is dropped: that
    starfish is whole in a neighbouring tile, and labelling or not labelling the sliver
    both teach the wrong thing.
  - Positive tiles: every remaining tile with a box, from the base split's train frames.
    Boxes with visible >= --min-visible are labelled, clipped to the tile.
  - Empty tiles: grid tiles that overlap no box, from any frame of the train sequences
    (with or without starfish), sampled uniformly (seeded) so they make up --empty-frac of
    all tiles. The sample is fixed; it is not redrawn per epoch.
  - Saved as PNG (lossless).

Layout (same as make_crop_split):
    data/yolo/crops/<name>/images/video_N/<frame>_c<col>r<row>.png   (+ labels/... .txt)
    data/yolo/splits/<name>/train.txt, val.txt, val_fast.txt, split.csv, tiles.csv
    data/yolo/<name>.yaml

Usage (from the repo root):
    python -m src.data.make_grid_split --base seq20val_pos
"""

import argparse
import ast
import csv
import random
import shutil
from pathlib import Path

import cv2

from src.data.make_crop_split import clip_box, clip_to_frame, crop_labels
from src.paths import IMG_H, IMG_W, TRAIN_CSV, YOLO_DATA, split_dir


def grid_starts(length, tile, n):
    """n evenly spaced tile starts covering [0, length], first at 0 and last at length - tile."""
    if n == 1:
        return [(length - tile) // 2]
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]


def visible_fractions(boxes, x0, y0, t):
    """Fraction of each box (x, y, w, h) inside the tile, for boxes that overlap it."""
    out = []
    for b in boxes:
        c = clip_box(b, x0, y0, t, t)
        if c is not None:
            out.append(c[2] * c[3] / (b[2] * b[3]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="seq20val_pos", help="existing full-frame split to take sequences and val from")
    ap.add_argument("--tile", type=int, default=493, help="square tile side in source pixels")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=1408, help="training imgsz the tiles are meant for (sets the label size filter)")
    ap.add_argument("--min-visible", type=float, default=0.5, help="drop tiles with a box less visible than this (but > 0)")
    ap.add_argument("--empty-frac", type=float, default=0.25, help="empty tiles as a fraction of all tiles")
    ap.add_argument("--val-fast-step", type=int, default=1, help="per-epoch val uses every Nth val frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="default: grid<tile>_<cols>x<rows>_<base>")
    args = ap.parse_args()

    t = args.tile
    name = args.name or f"grid{t}_{args.cols}x{args.rows}_{args.base}"
    train_scale = args.imgsz / t
    xs, ys = grid_starts(IMG_W, t, args.cols), grid_starts(IMG_H, t, args.rows)
    cells = [(ci, ri, x0, y0) for ri, y0 in enumerate(ys) for ci, x0 in enumerate(xs)]
    rng = random.Random(args.seed)

    base_dir = split_dir(args.base)
    with open(base_dir / "split.csv") as f:
        base_rows = list(csv.DictReader(f))
    val_seqs = {(r["video_id"], r["sequence"]) for r in base_rows if r["split"] == "val"}
    base_train = {r["image_id"] for r in base_rows if r["split"] == "train"}

    frames = []  # every frame of the train sequences: (image_id, video_id, video_frame, boxes)
    with open(TRAIN_CSV) as f:
        for r in csv.DictReader(f):
            if (r["video_id"], r["sequence"]) in val_seqs:
                continue
            boxes = [clip_to_frame(b) for b in ast.literal_eval(r["annotations"])]
            boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]
            frames.append((r["image_id"], r["video_id"], r["video_frame"], boxes))

    pos, empty_pool, n_sliver = [], [], 0
    for fr in frames:
        for ci, ri, x0, y0 in cells:
            vis = visible_fractions(fr[3], x0, y0, t)
            if not vis:
                empty_pool.append((fr, ci, ri, x0, y0, "empty"))
            elif min(vis) < args.min_visible:
                n_sliver += fr[0] in base_train
            elif fr[0] in base_train:
                pos.append((fr, ci, ri, x0, y0, "pos"))
    n_empty = round(len(pos) * args.empty_frac / (1 - args.empty_frac))
    if n_empty > len(empty_pool):
        raise SystemExit(f"need {n_empty} empty tiles, only {len(empty_pool)} available")
    plan = pos + rng.sample(empty_pool, n_empty)
    print(f"grid x={xs} y={ys} tile {t}: {len(pos)} positive, {n_empty} empty (of {len(empty_pool)}), "
          f"{n_sliver} positive-frame tiles dropped for slivers")

    crop_root = YOLO_DATA / "crops" / name
    shutil.rmtree(crop_root, ignore_errors=True)
    out_split = split_dir(name)
    shutil.rmtree(out_split, ignore_errors=True)
    out_split.mkdir(parents=True)

    plan.sort(key=lambda p: (p[0][1], int(p[0][2])))  # by frame, so each frame is decoded once
    rows, paths, cache_key, img = [], [], None, None
    for (iid, vid, vframe, boxes), ci, ri, x0, y0, kind in plan:
        if cache_key != iid:
            img = cv2.imread(str(YOLO_DATA / "images" / f"video_{vid}" / f"{vframe}.jpg"))
            cache_key = iid
        rel = Path(f"video_{vid}") / f"{vframe}_c{ci}r{ri}.png"
        ip, lp = crop_root / "images" / rel, crop_root / "labels" / rel.with_suffix(".txt")
        ip.parent.mkdir(parents=True, exist_ok=True)
        lp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ip), img[y0 : y0 + t, x0 : x0 + t], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        lines = crop_labels(boxes, x0, y0, t, t, train_scale) if kind == "pos" else []
        lp.write_text("\n".join(lines) + ("\n" if lines else ""))
        paths.append(ip)
        rows.append({"tile": str(rel), "image_id": iid, "col": ci, "row": ri, "x0": x0, "y0": y0, "size": t,
                     "kind": kind, "n_boxes": len(lines)})
        if len(rows) % 500 == 0:
            print(f"\r  wrote {len(rows)}/{len(plan)} tiles", end="", flush=True)
    print(f"\r  wrote {len(rows)}/{len(plan)} tiles")

    rng.shuffle(paths)  # Ultralytics shuffles anyway; keeps --smoke's fraction=0.1 representative
    (out_split / "train.txt").write_text("".join(f"{p}\n" for p in paths))
    with open(out_split / "tiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shutil.copy(base_dir / "split.csv", out_split / "split.csv")
    val = (base_dir / "val.txt").read_text().splitlines()
    (out_split / "val.txt").write_text("".join(f"{p}\n" for p in val))
    (out_split / "val_fast.txt").write_text("".join(f"{p}\n" for p in val[:: args.val_fast_step]))

    (YOLO_DATA / f"{name}.yaml").write_text(
        f"# Generated by src/data/make_grid_split.py --base {args.base} --tile {t} --cols {args.cols} --rows {args.rows} "
        f"--imgsz {args.imgsz} --min-visible {args.min_visible} --empty-frac {args.empty_frac} "
        f"--val-fast-step {args.val_fast_step} --seed {args.seed}\n"
        f"# train: {t}x{t} grid tiles at x={xs} y={ys} (upscaled {train_scale:.4f}x at imgsz {args.imgsz}); "
        f"val: full frames of {args.base}\n"
        f"path: {YOLO_DATA}\n"
        f"train: splits/{name}/train.txt\n"
        f"val: splits/{name}/val_fast.txt\n"
        "names:\n  0: cots\n"
    )


if __name__ == "__main__":
    main()
