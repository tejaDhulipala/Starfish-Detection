"""Tiled-inference F2 for a crop-trained run: does scoring tiles (the training condition) beat full frames?

    python -m src.eval.tiled_eval --run model_dev/c05_yolo26s_crops285_seq20val_pos/seed0

Each val frame is cut into an overlapping grid of tiles the size of the training crops
(crops.csv of the run's split), each tile is predicted at the training imgsz (so it is upscaled
exactly as in training), and boxes are mapped back to frame pixels. Raw tile predictions are
cached in <run>/eval/val_preds_<w>_tiles.csv; merge variants are then scored from the cache:

  nms          all tile boxes, class-agnostic NMS across tiles
  edge+nms     first drop boxes touching an interior tile edge (a neighbouring tile overlaps
               that edge, so the starfish should appear whole there), then NMS

Overlap: with the default 4x4 grid and 449x253 tiles it is 172 x 97 px, so any starfish up to
about that size is fully inside at least one tile. Diagnostic only: index.csv is not updated.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from src.eval.evaluate import CONF_GRID, val_frames
from src.eval.f2 import f2_sweep
from src.paths import IMG_H, IMG_W, ROOT, split_dir

EDGE_PX = 2  # a box within this many frame pixels of an interior tile edge "touches" it


def tile_origins(cw, ch, nx, ny):
    xs = np.round(np.linspace(0, IMG_W - cw, nx)).astype(int)
    ys = np.round(np.linspace(0, IMG_H - ch, ny)).astype(int)
    return [(int(x), int(y)) for y in ys for x in xs]


def predict_tiles(weights, frames, cw, ch, origins, imgsz, out_csv, tiles_per_call):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out_csv.parent.mkdir(exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "conf", "x", "y", "w", "h", "edge"])
        for n, (iid, path, _) in enumerate(frames):
            img = cv2.imread(str(path))  # BGR, as Ultralytics expects
            tiles = [np.ascontiguousarray(img[y0 : y0 + ch, x0 : x0 + cw]) for x0, y0 in origins]
            results = []
            for i in range(0, len(tiles), tiles_per_call):
                results += model.predict(tiles[i : i + tiles_per_call], imgsz=imgsz, conf=CONF_GRID[0], verbose=False)
            for (x0, y0), r in zip(origins, results):
                xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float64)
                for (x1, y1, x2, y2), c in zip(xyxy, r.boxes.conf.cpu().numpy()):
                    # interior edges only: a tile edge on the frame border has no neighbour
                    edge = (
                        (x0 > 0 and x1 <= EDGE_PX)
                        or (y0 > 0 and y1 <= EDGE_PX)
                        or (x0 + cw < IMG_W and x2 >= cw - EDGE_PX)
                        or (y0 + ch < IMG_H and y2 >= ch - EDGE_PX)
                    )
                    w.writerow([iid, f"{c:.4f}", f"{x0 + x1:.1f}", f"{y0 + y1:.1f}", f"{x2 - x1:.1f}", f"{y2 - y1:.1f}", int(edge)])
            print(f"\r  predicted {n + 1}/{len(frames)} frames", end="", flush=True)
    print()


def load_tile_preds(path):
    preds = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            preds.setdefault(r["image_id"], []).append([float(r[k]) for k in ("conf", "x", "y", "w", "h", "edge")])
    return {k: np.array(v) for k, v in preds.items()}


def merge(p, drop_edge, iou):
    """(N, 6) conf x y w h edge -> (M, 5) conf x y w h, rounded like a submission."""
    if drop_edge:
        p = p[p[:, 5] == 0]
    if len(p) == 0:
        return np.zeros((0, 5))
    boxes = torch.tensor(np.column_stack([p[:, 1], p[:, 2], p[:, 1] + p[:, 3], p[:, 2] + p[:, 4]]))
    keep = nms(boxes, torch.tensor(p[:, 0]), iou).numpy()
    out = p[keep, :5].copy()
    out[:, 1:] = np.round(out[:, 1:])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--weights", default="best")
    ap.add_argument("--grid", type=int, nargs=2, default=[4, 4], metavar=("NX", "NY"))
    ap.add_argument("--nms-iou", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--tiles-per-call", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run = args.run.resolve()
    cfg = json.loads((run / "meta.json").read_text())["config"]
    with open(split_dir(cfg["split"]) / "crops.csv") as f:
        r = next(csv.DictReader(f))
    cw, ch = int(r["w"]), int(r["h"])
    imgsz = cfg["train"]["imgsz"]
    origins = tile_origins(cw, ch, *args.grid)
    frames = val_frames(cfg["split"])
    xs, ys = sorted({o[0] for o in origins}), sorted({o[1] for o in origins})
    print(f"{run.relative_to(ROOT)}: {len(frames)} val frames, {len(origins)} tiles of {cw}x{ch} at imgsz {imgsz}, "
          f"overlap {cw - (xs[1] - xs[0])}x{ch - (ys[1] - ys[0])} px")

    tag = f"{args.weights}_tiles{args.grid[0]}x{args.grid[1]}"
    cache = run / "eval" / f"val_preds_{tag}.csv"
    if args.force or not cache.exists():
        predict_tiles(run / "weights" / f"{args.weights}.pt", frames, cw, ch, origins, imgsz, cache, args.tiles_per_call)
    preds = load_tile_preds(cache)

    results = {}
    for drop_edge in (False, True):
        for iou in args.nms_iou:
            name = f"{'edge+' if drop_edge else ''}nms{iou}"
            fr = [(merge(preds[iid], drop_edge, iou) if iid in preds else np.zeros((0, 5)), g) for iid, _, g in frames]
            sweep = f2_sweep(fr, CONF_GRID)
            b = int(np.argmax(sweep["f2"]))
            results[name] = {"f2": float(sweep["f2"][b]), "conf": float(CONF_GRID[b])}
            print(f"  {name:14s} F2 {sweep['f2'][b]:.4f} at conf {CONF_GRID[b]:.2f}")
    (run / "eval" / f"{tag}_f2.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
