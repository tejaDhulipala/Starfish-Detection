"""Soft-max WBF on cached val predictions: fused conf = max(sum / n_models, alpha * max_conf in cluster).
alpha = 0 is the current WBF; alpha = 1 lets one confident model carry a box alone. Boxes are unchanged.
Full-val best F2 per alpha, plus split-half check (cutoff tuned on one half of the sequences, scored
on the other; 3 seeds x 2 directions) against alpha = 0.

    python -m src.postprocess.softmax_wbf
"""
import csv
import numpy as np
from src.eval.ensemble import load_cached, Z
from src.eval.evaluate import CONF_GRID, val_frames
from src.eval.f2 import f2_sweep, iou_matrix
from src.paths import split_dir

def wbf_soft(preds_list, alpha, thr=0.4):
    rows = [p for p in preds_list if len(p)]
    if not rows: return Z
    allp = np.concatenate(rows); allp = allp[np.argsort(-allp[:, 0])]
    clusters, fused = [], []
    for p in allp:
        if fused:
            ious = iou_matrix(p[None, 1:], np.array(fused)[:, 1:])[0]; j = int(np.argmax(ious))
            if ious[j] >= thr:
                clusters[j].append(p); c = np.array(clusters[j]); w = c[:, 0:1]
                fused[j] = np.concatenate([[c[:, 0].sum()], (c[:, 1:] * w).sum(0) / w.sum()]); continue
        clusters.append([p]); fused.append(p.copy())
    n = len(preds_list)
    out = np.array([[max(f[0] / n, alpha * np.array(c)[:, 0].max()), *f[1:]] for f, c in zip(fused, clusters)])
    return out[np.argsort(-out[:, 0])]

frames = val_frames("seq20val_pos")
seq = {r["image_id"]: r["sequence"] for r in csv.DictReader(open(split_dir("seq20val_pos") / "split.csv"))}
seqs = sorted({seq[i] for i, _, _ in frames})
def best(fz, sub):
    sw = f2_sweep([(fz.get(i, Z), g) for i, _, g in sub], CONF_GRID); b = int(np.argmax(sw["f2"]))
    return float(sw["f2"][b]), float(CONF_GRID[b]), sw, b
def at(fz, sub, c):
    sw = f2_sweep([(fz.get(i, Z), g) for i, _, g in sub], CONF_GRID); return float(sw["f2"][int(np.argmin(np.abs(CONF_GRID - c)))])

ALPHAS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
for models in (["c04", "c06", "c07", "c08"], ["c04", "c06", "c08"]):
    P = [load_cached(m) for m in models]
    print(f"\n== {'+'.join(models)}")
    print(f"  {'alpha':>5s} {'F2':>7s} {'cutoff':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s}   split-half vs alpha 0 (mean, wins/6)")
    halves = []
    for s in range(3):
        h = set(np.random.default_rng(s).permutation(seqs)[: len(seqs) // 2])
        A = [f for f in frames if seq[f[0]] in h]; B = [f for f in frames if seq[f[0]] not in h]
        halves += [(A, B), (B, A)]
    base = None
    for a in ALPHAS:
        fz = {i: wbf_soft([p.get(i, Z) for p in P], a) for i, _, _ in frames}
        f, c, sw, b = best(fz, frames)
        ho = np.array([at(fz, te, best(fz, tu)[1]) for tu, te in halves])
        if base is None: base = ho
        d = ho - base
        print(f"  {a:5.1f} {f:7.4f} {c:6.2f} {int(sw['tp'][4,b]):4d} {int(sw['fp'][4,b]):4d} {int(sw['fn'][4,b]):4d}   held-out {ho.mean():.4f}  {d.mean():+.4f}  {int((d > 0).sum())}/6", flush=True)
