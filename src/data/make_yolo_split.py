"""Convert the Kaggle Great Barrier Reef annotations to YOLO format and write one train/val split.

Layout (under data/yolo/):
    images -> ../train_images          symlink, no copy of the 15 GB of frames
    labels/video_N/<frame>.txt         one file per frame, empty for frames without boxes
    splits/<name>/train.txt, val.txt   absolute image paths, one per line
    splits/<name>/val_fast.txt         every --val-fast-step-th val frame, used for per-epoch val
    splits/<name>/split.csv            which frame went where
    <name>.yaml                        dataset config passed to Ultralytics

Split rule: val = every frame of --val-video, or (with --val-seq-frac F) every frame of
whole sequences chosen per video to hold about F of that video's frames and of its
frames with boxes (exhaustive search, so deterministic; see pick_val_sequences). Train = frames with boxes
from everything else, plus round(neg_frac * n_positive) randomly sampled empty frames.

The dataset yaml points Ultralytics' val at val_fast.txt: validating on all of
val every epoch would cost more than the training epoch. Adjacent frames are
near-duplicates, so a strided subset loses little. Full-val F2 is computed
separately by src.eval.evaluate, which reads val.txt.

Labels are deterministic, so they are rewritten on every run. Only the split being
written is replaced; other splits under splits/ are left alone.

Standard library only. Usage (from the repo root):
    python -m src.data.make_yolo_split --val-video 0
    python -m src.data.make_yolo_split --val-video 0 --neg-frac 0.2
    python -m src.data.make_yolo_split --val-seq-frac 0.2
"""

import argparse
import ast
import csv
import itertools
import random
import shutil
import struct
from pathlib import Path

IMG_W, IMG_H = 1280, 720
PROJECT_DIR = Path(__file__).resolve().parents[2]


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


def pick_val_sequences(frames, frac):
    """Per video, the subset of its sequences closest to frac of the video's frames and of
    its frames with boxes, with at least one sequence with boxes on each side.

    Box counts are not matched on purpose: a few dense sequences (e.g. 2/22643 holds 96%
    of video 2's boxes) would dominate val, and micro-averaged F2 would measure one scene.
    """
    stats = {}
    for fr in frames:
        st = stats.setdefault((fr["video_id"], fr["sequence"]), [0, 0])
        st[0] += 1
        st[1] += fr["n_boxes"] > 0
    chosen = set()
    for video in sorted({v for v, _ in stats}):
        keys = sorted(k for k in stats if k[0] == video)
        totals = [sum(stats[k][i] for k in keys) for i in range(2)]
        best, best_score = None, float("inf")
        for mask in itertools.product((0, 1), repeat=len(keys)):
            val = [k for k, m in zip(keys, mask) if m]
            train = [k for k, m in zip(keys, mask) if not m]
            if not any(stats[k][1] for k in val) or not any(stats[k][1] for k in train):
                continue
            score = sum(abs(sum(stats[k][i] for k in val) / totals[i] - frac) for i in range(2))
            if score < best_score:
                best, best_score = val, score
        assert best is not None, f"video {video}: no subset with positives on both sides"
        chosen.update(best)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--val-video", type=int, default=None, help="hold out this whole video (default: 0)")
    which.add_argument("--val-seq-frac", type=float, default=None, help="hold out whole sequences, about this fraction")
    ap.add_argument("--neg-frac", type=float, default=0.0, help="empty train frames as a fraction of positive train frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fast-step", type=int, default=5, help="val_fast keeps every Nth val frame")
    ap.add_argument("--name", default=None, help="split name (default: v{video}val_pos or seq{pct}val_pos, plus _neg{pct})")
    args = ap.parse_args()
    if args.val_video is None and args.val_seq_frac is None:
        args.val_video = 0

    data_dir = args.data_dir.resolve()
    yolo_dir = data_dir / "yolo"
    held_out = f"v{args.val_video}" if args.val_seq_frac is None else f"seq{round(args.val_seq_frac * 100)}"
    name = args.name or f"{held_out}val_pos" + (f"_neg{round(args.neg_frac * 100)}" if args.neg_frac else "")

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
    assert args.val_video is None or args.val_video in videos, f"--val-video {args.val_video} not in {videos}"

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
    if args.val_seq_frac is None:
        in_val = lambda fr: fr["video_id"] == args.val_video
        val_desc = f"video {args.val_video}"
    else:
        val_seqs = pick_val_sequences(frames, args.val_seq_frac)
        in_val = lambda fr: (fr["video_id"], fr["sequence"]) in val_seqs
        val_desc = "sequences " + ", ".join(f"{v}/{s}" for v, s in sorted(val_seqs))
    val = [fr for fr in frames if in_val(fr)]
    rest = [fr for fr in frames if not in_val(fr)]
    train_pos = [fr for fr in rest if fr["n_boxes"] > 0]
    train_neg_pool = [fr for fr in rest if fr["n_boxes"] == 0]
    n_neg = min(round(args.neg_frac * len(train_pos)), len(train_neg_pool))
    train_neg = random.Random(args.seed).sample(train_neg_pool, n_neg)
    train = sorted(train_pos + train_neg, key=lambda fr: (fr["video_id"], int(fr["video_frame"])))
    val_fast = val[:: args.val_fast_step]

    split_dir = yolo_dir / "splits" / name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)
    for fname, subset in (("train.txt", train), ("val.txt", val), ("val_fast.txt", val_fast)):
        (split_dir / fname).write_text("".join(f"{fr['img']}\n" for fr in subset))
    with open(split_dir / "split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "video_id", "sequence", "video_frame", "n_boxes", "split"])
        for split, subset in (("train", train), ("val", val)):
            for fr in subset:
                w.writerow([fr["image_id"], fr["video_id"], fr["sequence"], fr["video_frame"], fr["n_boxes"], split])
    (yolo_dir / f"{name}.yaml").write_text(
        f"# Generated by src/data/make_yolo_split.py "
        + (f"--val-video {args.val_video} " if args.val_seq_frac is None else f"--val-seq-frac {args.val_seq_frac} ")
        + f"--neg-frac {args.neg_frac} --seed {args.seed} --val-fast-step {args.val_fast_step}\n"
        f"# val: {val_desc}\n"
        f"# val is the strided val_fast list; full val is splits/{name}/val.txt\n"
        f"path: {yolo_dir}\n"
        f"train: splits/{name}/train.txt\n"
        f"val: splits/{name}/val_fast.txt\n"
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
    print(f"split '{name}': val = {val_desc}")
    print(f"  train: {summary(train)}")
    print(f"  val:   {summary(val)}")
    print(f"  val_fast (every {args.val_fast_step}th val frame): {summary(val_fast)}")
    print(f"  yaml:  {yolo_dir / f'{name}.yaml'}")
    print("all checks passed")


if __name__ == "__main__":
    main()
