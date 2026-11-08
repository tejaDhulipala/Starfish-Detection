"""Convert the Kaggle Great Barrier Reef annotations to YOLO format and write one train/val split.

Layout (under data/yolo/):
    images -> ../train_images          symlink, no copy of the 15 GB of frames
    labels/video_N/<frame>.txt         one file per frame, empty for frames without boxes
    splits/<name>/train.txt, val.txt   absolute image paths, one per line
    splits/<name>/split.csv            which frame went where
    <name>.yaml                        dataset config passed to Ultralytics

Split rule: val = every frame of --val-video. Train = frames with boxes from the
other videos, plus round(neg_frac * n_positive) randomly sampled empty frames.

Labels are deterministic, so they are rewritten on every run. Only the split being
written is replaced; other splits under splits/ are left alone.

Standard library only. Usage:
    python data_scripts/make_yolo_split.py --val-video 0
    python data_scripts/make_yolo_split.py --val-video 0 --neg-frac 0.2
"""

import argparse
import ast
import csv
import random
import shutil
import struct
from pathlib import Path

IMG_W, IMG_H = 1280, 720
PROJECT_DIR = Path(__file__).resolve().parent.parent


def jpeg_size(path):
    """Return (width, height) from the first baseline/progressive SOF marker."""
    with open(path, "rb") as f:
        head = f.read(1 << 16)
    for marker in (b"\xff\xc0", b"\xff\xc2"):
        i = head.find(marker)
        if i != -1:
            h, w = struct.unpack(">HH", head[i + 5 : i + 9])
            return w, h
    raise ValueError(f"no SOF marker in first 64 KB of {path}")


def to_yolo(box):
    """Kaggle box (top-left x, y, width, height in px) -> clipped, normalized (cx, cy, w, h)."""
    x1 = max(0, box["x"])
    y1 = max(0, box["y"])
    x2 = min(IMG_W, box["x"] + box["width"])
    y2 = min(IMG_H, box["y"] + box["height"])
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2 / IMG_W, (y1 + y2) / 2 / IMG_H, (x2 - x1) / IMG_W, (y2 - y1) / IMG_H)


def label_path_for(img_path):
    """Same rule as ultralytics.data.utils.img2label_paths: last /images/ -> /labels/, ext -> .txt."""
    s = str(img_path)
    return Path("/labels/".join(s.rsplit("/images/", 1)).rsplit(".", 1)[0] + ".txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    ap.add_argument("--val-video", type=int, default=0)
    ap.add_argument("--neg-frac", type=float, default=0.0, help="empty train frames as a fraction of positive train frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="split name (default: v{val}val_pos[_neg{pct}])")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    yolo_dir = data_dir / "yolo"
    name = args.name or f"v{args.val_video}val_pos" + (f"_neg{round(args.neg_frac * 100)}" if args.neg_frac else "")

    # Read annotations
    frames = []
    with open(data_dir / "train.csv") as f:
        for r in csv.DictReader(f):
            boxes = ast.literal_eval(r["annotations"])
            frames.append({
                "image_id": r["image_id"],
                "video_id": int(r["video_id"]),
                "sequence": r["sequence"],
                "video_frame": r["video_frame"],
                "boxes": boxes,
            })
    videos = sorted({fr["video_id"] for fr in frames})
    assert args.val_video in videos, f"--val-video {args.val_video} not in {videos}"

    # Images symlink (relative, so the project folder can move)
    yolo_dir.mkdir(exist_ok=True)
    images_link = yolo_dir / "images"
    if not images_link.is_symlink():
        assert not images_link.exists(), f"{images_link} exists and is not a symlink"
        images_link.symlink_to("../train_images")
    for fr in frames:
        fr["img"] = images_link / f"video_{fr['video_id']}" / f"{fr['video_frame']}.jpg"

    # Labels for every frame. Stale Ultralytics caches are removed since label contents may change.
    labels_dir = yolo_dir / "labels"
    for cache in labels_dir.glob("*.cache"):
        cache.unlink()
    n_clipped = n_dropped = 0
    for fr in frames:
        lines = []
        for b in fr["boxes"]:
            n_clipped += (b["x"] < 0 or b["y"] < 0 or b["x"] + b["width"] > IMG_W or b["y"] + b["height"] > IMG_H)
            yb = to_yolo(b)
            if yb is None:
                n_dropped += 1
                continue
            lines.append("0 " + " ".join(f"{v:.6f}" for v in yb))
        fr["n_boxes"] = len(lines)
        lp = label_path_for(fr["img"])
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("\n".join(lines) + ("\n" if lines else ""))

    # Split
    val = [fr for fr in frames if fr["video_id"] == args.val_video]
    rest = [fr for fr in frames if fr["video_id"] != args.val_video]
    train_pos = [fr for fr in rest if fr["n_boxes"] > 0]
    train_neg_pool = [fr for fr in rest if fr["n_boxes"] == 0]
    n_neg = min(round(args.neg_frac * len(train_pos)), len(train_neg_pool))
    train_neg = random.Random(args.seed).sample(train_neg_pool, n_neg)
    train = sorted(train_pos + train_neg, key=lambda fr: (fr["video_id"], int(fr["video_frame"])))

    split_dir = yolo_dir / "splits" / name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)
    for fname, subset in (("train.txt", train), ("val.txt", val)):
        (split_dir / fname).write_text("".join(f"{fr['img']}\n" for fr in subset))
    with open(split_dir / "split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "video_id", "sequence", "video_frame", "n_boxes", "split"])
        for split, subset in (("train", train), ("val", val)):
            for fr in subset:
                w.writerow([fr["image_id"], fr["video_id"], fr["sequence"], fr["video_frame"], fr["n_boxes"], split])
    (yolo_dir / f"{name}.yaml").write_text(
        f"# Generated by data_scripts/make_yolo_split.py --val-video {args.val_video} "
        f"--neg-frac {args.neg_frac} --seed {args.seed}\n"
        f"path: {yolo_dir}\n"
        f"train: splits/{name}/train.txt\n"
        f"val: splits/{name}/val.txt\n"
        f"names:\n  0: cots\n"
    )

    # Checks
    train_ids = {fr["image_id"] for fr in train}
    assert not train_ids & {fr["image_id"] for fr in val}, "train/val overlap"
    for subset in (train, val):
        written = 0
        for fr in subset:
            assert fr["img"].is_file(), f"missing image {fr['img']}"
            assert jpeg_size(fr["img"]) == (IMG_W, IMG_H), f"unexpected size {fr['img']}"
            lines = label_path_for(fr["img"]).read_text().split()
            assert len(lines) % 5 == 0
            vals = [float(v) for v in lines]
            assert all(0 <= v <= 1 for i, v in enumerate(vals) if i % 5), f"coord out of [0, 1] in {fr['img']}"
            written += len(lines) // 5
        assert written == sum(fr["n_boxes"] for fr in subset), "label files disagree with train.csv"

    def summary(subset):
        n_pos = sum(fr["n_boxes"] > 0 for fr in subset)
        return f"{len(subset)} frames ({n_pos} with boxes, {len(subset) - n_pos} empty), {sum(fr['n_boxes'] for fr in subset)} boxes"

    print(f"labels: {len(frames)} files in {labels_dir}  (boxes clipped: {n_clipped}, dropped: {n_dropped})")
    print(f"split '{name}': val = video {args.val_video}, train = videos {[v for v in videos if v != args.val_video]}")
    print(f"  train: {summary(train)}")
    print(f"  val:   {summary(val)}")
    print(f"  yaml:  {yolo_dir / f'{name}.yaml'}")
    print("all checks passed")


if __name__ == "__main__":
    main()
