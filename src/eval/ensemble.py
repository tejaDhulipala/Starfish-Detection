"""Weighted box fusion (WBF) of several models' boxes for one frame, plus cached-prediction helpers.

    python -m src.eval.ensemble c04 c06 c07 c08      # full-val F2 of the fused cached predictions

The same fusion that gave c04+c06+c07+c08 F2 0.745 (first written as a scratch script in the c06-c08
session): boxes from all models are sorted by confidence and clustered greedily; a box joins the
cluster whose current fused box it overlaps at IoU >= thr, else starts a new cluster. Fused box =
confidence-weighted mean of the cluster's boxes; fused confidence = sum of the cluster's confidences
divided by the number of models (so a box only one model found is diluted). Each model's boxes have
already been through NMS at IoU 0.5 (Ultralytics predict), and no NMS follows the fusion.
"""

import sys

import numpy as np

from src.eval.f2 import iou_matrix
from src.paths import ROOT

Z = np.zeros((0, 5))


def wbf(preds_list, thr=0.4):
    """preds_list: one (N, 5) conf x y w h array per model. Returns fused (M, 5), conf descending."""
    rows = [p for p in preds_list if len(p)]
    if not rows:
        return Z
    allp = np.concatenate(rows, 0)
    allp = allp[np.argsort(-allp[:, 0])]
    clusters, fused = [], []
    for p in allp:
        if fused:
            ious = iou_matrix(p[None, 1:], np.array(fused)[:, 1:])[0]
            j = int(np.argmax(ious))
            if ious[j] >= thr:
                clusters[j].append(p)
                c = np.array(clusters[j])
                w = c[:, 0:1]
                fused[j] = np.concatenate([[c[:, 0].sum()], (c[:, 1:] * w).sum(0) / w.sum()])
                continue
        clusters.append([p])
        fused.append(p.copy())
    out = np.array([[f[0] / len(preds_list), *f[1:]] for f in fused])
    return out[np.argsort(-out[:, 0])]


def config_dir(short):
    """'c08' -> model_dev/c08_... (the one folder starting with that prefix)."""
    hits = sorted(ROOT.glob(f"model_dev/{short}_*"))
    if len(hits) != 1:
        raise ValueError(f"{short}: expected one model_dev folder, found {hits}")
    return hits[0]


def load_cached(short, run="seed0", weights="best"):
    from src.eval.evaluate import load_preds

    return load_preds(config_dir(short) / run / "eval" / f"val_preds_{weights}.csv")


def fused_val_preds(models, thr=0.4):
    """{image_id: fused (M, 5)} over the full val set, from each model's cached val predictions."""
    from src.eval.evaluate import val_frames

    P = [load_cached(m) for m in models]
    frames = val_frames("seq20val_pos")  # every c02+ split (crops, grids) validates on these frames
    return {iid: wbf([p.get(iid, Z) for p in P], thr) for iid, _, _ in frames}, frames


def main():
    from src.eval.evaluate import CONF_GRID
    from src.eval.f2 import f2_sweep

    models = sys.argv[1:]
    fused, frames = fused_val_preds(models)
    sw = f2_sweep([(fused.get(i, Z), g) for i, _, g in frames], CONF_GRID)
    b = int(np.argmax(sw["f2"]))
    tp, fp, fn = (int(sw[k][4, b]) for k in ("tp", "fp", "fn"))
    print(f"{'+'.join(models)}: F2 {sw['f2'][b]:.4f} at conf {CONF_GRID[b]:.2f}  (IoU 0.5: TP {tp} FP {fp} FN {fn})")


if __name__ == "__main__":
    main()
