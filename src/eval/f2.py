"""Competition metric: F2 averaged over IoU thresholds 0.30, 0.35, ..., 0.80.

Rules (from the Kaggle overview page, see CLAUDE.md):
- Per IoU threshold, TP/FP/FN are pooled over all frames (micro-average), then
  F2 = 5 TP / (5 TP + 4 FN + FP). The final score is the mean over the 11 thresholds.
- Within a frame, predictions are matched greedily in descending confidence order.
  Each prediction takes an unmatched ground-truth box with IoU >= threshold.
  Unmatched predictions are FP, unmatched ground truth is FN.
- Two choices the Kaggle page leaves open: it says "greater than the threshold" in
  one place and "at least" in its example (>= is used here), and it does not say
  which ground-truth box a prediction takes when several qualify (the first in
  annotation order is used here).

Boxes are (x, y, w, h) in pixels with (x, y) the top-left corner. A frame's
predictions are an (N, 5) array of (conf, x, y, w, h); its ground truth is (M, 4).

Sweeping the confidence cutoff is cheap because greedy matching in confidence
order is prefix-consistent: dropping every prediction below a cutoff c does not
change how the predictions above c were matched. So each frame is matched once
with all predictions, and TP(c) / FP(c) are counts of matched / unmatched
predictions with conf >= c.
"""

import numpy as np

IOU_THRESHOLDS = np.array([round(0.30 + 0.05 * i, 2) for i in range(11)])


def iou_matrix(a, b):
    """IoU between every box in a (N, 4) and b (M, 4), both (x, y, w, h). Returns (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_frame(preds, gts, iou_thr):
    """Greedy matching for one frame at one IoU threshold.

    Returns (conf, is_tp): confidences sorted descending and, aligned with them,
    whether each prediction matched a ground-truth box.
    """
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 5)
    order = np.argsort(-preds[:, 0], kind="stable")
    preds = preds[order]
    is_tp = np.zeros(len(preds), dtype=bool)
    if len(preds) and len(gts):
        ious = iou_matrix(preds[:, 1:], gts)
        free = np.ones(ious.shape[1], dtype=bool)
        for i in range(len(preds)):
            hits = np.flatnonzero(free & (ious[i] >= iou_thr))
            if len(hits):
                free[hits[0]] = False
                is_tp[i] = True
    return preds[:, 0], is_tp


def _f2(tp, fp, fn):
    tp, fp, fn = (np.asarray(v, dtype=np.float64) for v in (tp, fp, fn))
    den = 5 * tp + 4 * fn + fp
    return np.divide(5 * tp, den, out=np.ones_like(den), where=den > 0)  # nothing to find, nothing predicted: perfect


def f2_sweep(frames, conf_grid):
    """F2 for every confidence cutoff in conf_grid.

    frames: iterable of (preds (N, 5), gts (M, 4)).
    Returns a dict of arrays: conf_grid (C,), f2 (C,) averaged over IoU thresholds,
    and per-threshold f2, tp, fp, fn, each (11, C).
    """
    frames = list(frames)
    conf_grid = np.asarray(conf_grid, dtype=np.float64)
    n_gt = sum(len(np.asarray(g).reshape(-1, 4)) for _, g in frames)
    out = {k: np.zeros((len(IOU_THRESHOLDS), len(conf_grid))) for k in ("tp", "fp", "fn")}
    for t, thr in enumerate(IOU_THRESHOLDS):
        matched = [match_frame(p, g, thr) for p, g in frames]
        conf = np.concatenate([c for c, _ in matched]) if matched else np.zeros(0)
        tp_mask = np.concatenate([m for _, m in matched]) if matched else np.zeros(0, dtype=bool)
        tp_conf, fp_conf = np.sort(conf[tp_mask]), np.sort(conf[~tp_mask])
        out["tp"][t] = len(tp_conf) - np.searchsorted(tp_conf, conf_grid, side="left")  # count with conf >= cutoff
        out["fp"][t] = len(fp_conf) - np.searchsorted(fp_conf, conf_grid, side="left")
        out["fn"][t] = n_gt - out["tp"][t]
    out["f2_per_iou"] = _f2(out["tp"], out["fp"], out["fn"])
    out["f2"] = out["f2_per_iou"].mean(axis=0)
    out["conf_grid"] = conf_grid
    return out


def f2_score(frames, conf_thr=0.0):
    """Competition score for one confidence cutoff."""
    return float(f2_sweep(frames, [conf_thr])["f2"][0])
