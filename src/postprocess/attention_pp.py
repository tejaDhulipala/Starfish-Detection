"""1st place's attention-area post-processing on cached val predictions (WBF ensembles).

Frame N's fused boxes with conf >= T are its attention areas. In frames N+1..N+W (same video,
consecutive frame numbers; the Kaggle API gives no sequence ids), every box with conf > 0.01 that
overlaps an attention area (IoU > 0) gets score += c * IoU, where c is the attention box's conf
(variant A) or the box's own conf (variant B). Within one earlier frame the largest boost counts;
boosts from different earlier frames add up. Boosts always use the original (unboosted) scores. Only boxes whose own conf is
already >= L get boosted (L = 0.01 is 1st place's rule).

    python -m src.postprocess.attention_pp
Reports full-val best F2 and a split-half check (cutoff tuned on one half of the sequences, scored on
the other; 3 seeds x 2 directions) against no post-processing."""
import csv
import numpy as np
from src.eval.ensemble import fused_val_preds, Z
from src.eval.f2 import f2_sweep, iou_matrix
from src.paths import split_dir

GRID = np.round(np.arange(0.01, 2.0, 0.01), 2)
meta = {r["image_id"]: r for r in csv.DictReader(open(split_dir("seq20val_pos") / "split.csv"))}

def attention(fused, T, W, variant, L=0.01):
    out = {}
    for iid, p in fused.items():
        v, f = meta[iid]["video_id"], int(meta[iid]["video_frame"])
        p = p[p[:, 0] > 0.01].copy() if len(p) else Z
        if len(p):
            boost = np.zeros(len(p))
            for k in range(1, W + 1):
                q = fused.get(f"{v}-{f - k}")
                if q is None or not len(q): continue
                a = q[q[:, 0] >= T]
                if not len(a): continue
                io = iou_matrix(p[:, 1:], a[:, 1:])  # (boxes, attention)
                c = a[None, :, 0] if variant == "A" else p[:, 0:1]
                boost += np.where(p[:, 0] >= L, (c * io).max(1), 0.0)  # only boxes already >= L get boosted
            p[:, 0] += boost
        out[iid] = p
    return out

def best(fz, sub):
    sw = f2_sweep([(fz.get(i, Z), g) for i, _, g in sub], GRID); b = int(np.argmax(sw["f2"]))
    return float(sw["f2"][b]), float(GRID[b]), sw, b
def at(fz, sub, c):
    sw = f2_sweep([(fz.get(i, Z), g) for i, _, g in sub], GRID); return float(sw["f2"][int(np.argmin(np.abs(GRID - c)))])

for models in (["c04", "c06", "c07", "c08"], ["c04", "c06", "c08"]):
    fused, frames = fused_val_preds(models)
    seqs = sorted({meta[i]["sequence"] for i, _, _ in frames})
    halves = []
    for s in range(3):
        h = set(np.random.default_rng(s).permutation(seqs)[: len(seqs) // 2])
        A = [x for x in frames if meta[x[0]]["sequence"] in h]; B = [x for x in frames if meta[x[0]]["sequence"] not in h]
        halves += [(A, B), (B, A)]
    print(f"\n== {'+'.join(models)}")
    print(f"  {'setting':22s} {'F2':>7s} {'cutoff':>6s} {'TP':>4s} {'FP':>4s} {'FN':>4s}   held-out   vs none  wins")
    base = None
    settings = [("none", None, None, None, None)] \
             + [(f"{v} T={T} W=3 L=0.01", T, 3, v, 0.01) for v in ("A", "B") for T in (0.3, 0.4, 0.5, 0.6, 0.7)] \
             + [(f"A T=0.5 W={W} L=0.01", 0.5, W, "A", 0.01) for W in (1, 2, 5)] + [(f"{v} T={T} L={L}", T, 3, v, L) for v in ("A", "B") for T in (0.4, 0.5, 0.7) for L in (0.05, 0.1, 0.15, 0.2)]
    for name, T, W, v, L in settings:
        fz = fused if T is None else attention(fused, T, W, v, L)
        f, c, sw, b = best(fz, frames)
        ho = np.array([at(fz, te, best(fz, tu)[1]) for tu, te in halves])
        if base is None: base = ho
        d = ho - base
        print(f"  {name:22s} {f:7.4f} {c:6.2f} {int(sw['tp'][4,b]):4d} {int(sw['fp'][4,b]):4d} {int(sw['fn'][4,b]):4d}   {ho.mean():.4f}  {d.mean():+.4f}  {int((d > 0).sum())}/6", flush=True)
