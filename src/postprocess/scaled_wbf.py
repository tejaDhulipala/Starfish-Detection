"""Threshold-multiple confidence scaling before WBF, on c04+c06+c07+c08 cached val predictions.

Variants (same WBF as the other session's wbf_lib: greedy clusters at IoU >= 0.4 against the
fused box, conf = sum / n_models):
  base   raw scores
  a      s / c*_m for score and coordinate weights
  b      s / c*_m for the fused score; raw s weights the coordinates
Reports full-val best F2 (cutoffs tuned on full val) and a split-half check: sequences are split in two,
per-model c*_m and the ensemble cutoff are tuned on one half and scored on the other (3 seeds x 2 directions).

    python -m src.postprocess.scaled_wbf
"""

import csv
import time

import numpy as np

from src.eval.evaluate import CONF_GRID, load_preds, val_frames  # noqa: E402
from src.eval.f2 import f2_sweep, iou_matrix  # noqa: E402
from src.paths import split_dir  # noqa: E402

from src.paths import ROOT  # noqa: E402

R = str(ROOT / "model_dev/{}/seed0/eval/val_preds_best.csv")
MODELS = {
    "c04": "c04_yolo26s_1920_seq20val_pos_aug",
    "c06": "c06_c05ft_grid493_3x2_seq20val_pos_aug",
    "c07": "c07_c05ft_grid493_3x2_e50_seq20val_pos_aug",
    "c08": "c08_yolo26m_grid493_3x2_seq20val_pos_aug",
}
P = {m: load_preds(R.format(d)) for m, d in MODELS.items()}
frames = val_frames("seq20val_pos")
seq = {r["image_id"]: r["sequence"] for r in csv.DictReader(open(split_dir("seq20val_pos") / "split.csv"))}
Z = np.zeros((0, 5))
FUSED_GRID = np.round(np.arange(0.01, 4.0, 0.01), 2)  # scaled fused scores can exceed 1


def wbf(preds, wts_coord, n_models, thr=0.4):
    """preds: list of (N,5) score x y w h; wts_coord: matching list of (N,) coordinate weights."""
    rows = [np.column_stack([p, w]) for p, w in zip(preds, wts_coord) if len(p)]
    if not rows:
        return Z
    allp = np.concatenate(rows)
    allp = allp[np.argsort(-allp[:, 0], kind="stable")]
    clusters, fused = [], []
    for p in allp:
        if fused:
            ious = iou_matrix(p[None, 1:5], np.array(fused)[:, 1:5])[0]
            j = int(np.argmax(ious))
            if ious[j] >= thr:
                clusters[j].append(p)
                c = np.array(clusters[j])
                w = c[:, 5:6]
                fused[j] = np.concatenate([[c[:, 0].sum()], (c[:, 1:5] * w).sum(0) / w.sum()])
                continue
        clusters.append([p])
        fused.append(p[:5].copy())
    out = np.array([[f[0] / n_models, *f[1:5]] for f in fused])
    return out[np.argsort(-out[:, 0], kind="stable")]


def fuse(ids, variant, cstar):
    out = {}
    for i in ids:
        raw = [P[m].get(i, Z) for m in MODELS]
        if variant == "base":
            sc, wc = raw, [p[:, 0] for p in raw]
        else:
            sc = [np.column_stack([p[:, 0] / cstar[m], p[:, 1:]]) if len(p) else Z for m, p in zip(MODELS, raw)]
            wc = [s[:, 0] for s in sc] if variant == "a" else [p[:, 0] for p in raw]
        out[i] = wbf(sc, wc, len(MODELS))
    return out


def best(preds, sub, grid):
    sw = f2_sweep([(preds.get(i, Z), g) for i, _, g in sub], grid)
    b = int(np.argmax(sw["f2"]))
    return float(sw["f2"][b]), float(grid[b]), sw


def at(preds, sub, grid, c):
    sw = f2_sweep([(preds.get(i, Z), g) for i, _, g in sub], grid)
    return float(sw["f2"][int(np.argmin(np.abs(grid - c)))])


t0 = time.time()
ids = [i for i, _, _ in frames]
cstar_full = {m: best(P[m], frames, CONF_GRID)[1] for m in MODELS}
print("per-model c* (full val):", cstar_full, flush=True)
for v in ("base", "a", "b"):
    fz = fuse(ids, v, cstar_full)
    f, c, sw = best(fz, frames, FUSED_GRID)
    b = int(np.argmax(sw["f2"]))
    tp, fp, fn = (int(sw[k][4, b]) for k in ("tp", "fp", "fn"))
    print(f"full val  {v:4s}  F2 {f:.4f}  cutoff {c:.2f}  TP {tp} FP {fp} FN {fn}  ({time.time() - t0:.0f}s)", flush=True)

seqs = sorted({seq[i] for i in ids})
res = {v: [] for v in ("base", "a", "b")}
for s in range(3):
    rng = np.random.default_rng(s)
    half = set(rng.permutation(seqs)[: len(seqs) // 2])
    A = [f for f in frames if seq[f[0]] in half]
    B = [f for f in frames if seq[f[0]] not in half]
    for tune, test in ((A, B), (B, A)):
        cst = {m: best(P[m], tune, CONF_GRID)[1] for m in MODELS}
        for v in ("base", "a", "b"):
            fz = fuse(ids, v, cst)
            c = best(fz, tune, FUSED_GRID)[1]
            res[v].append(at(fz, test, FUSED_GRID, c))
    print(f"seed {s}: " + "  ".join(f"{v} {np.mean(res[v][-2:]):.4f}" for v in res) + f"  ({time.time() - t0:.0f}s)", flush=True)

print("split-half held-out F2 (mean over 6, sd):")
for v in res:
    d = np.array(res[v]) - np.array(res["base"])
    print(f"  {v:4s}  {np.mean(res[v]):.4f} ± {np.std(res[v]):.4f}   vs base: {np.mean(d):+.4f} (wins {int((d > 0).sum())}/6)")
