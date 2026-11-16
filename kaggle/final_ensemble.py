"""Final ensemble for the Kaggle notebook (Python 3.7 compatible): several passes per model, with
rotations and per-pass input sizes, from one dynamic-shape ONNX session per model.

A pass is (group, model, k, long_side): rotate the frame by np.rot90(frame, k) (counter-clockwise
quarter turns), letterbox to long_side (the short side padded up to a multiple of 32, as Ultralytics
does), run, map the boxes back to the unrotated frame. Passes in the same group are WBF-fused first
into one vote (c17's 3 sizes); then all groups are WBF-fused with conf = sum / n_groups. Mirrors the
val scoring in the scratchpad's c17var.py (src.eval.ensemble.wbf, IoU 0.4).

Models are pinned to GPUs; each GPU runs its passes sequentially in its own thread, the GPUs in parallel.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cots_ensemble import Z, wbf
from cots_onnx import OnnxDetector

W0, H0 = 1280, 720


def input_shape(long_side, k):
    """(H, W) network input for a 1280x720 frame rotated by k quarter turns, at this long side."""
    short = int(math.ceil(round(H0 * long_side / W0) / 32.0) * 32)
    return (short, long_side) if k % 2 == 0 else (long_side, short)


def unrotate(d, k):
    """d: (N, 5) conf x y w h in np.rot90(frame, k). Returns boxes in the original 1280x720 frame."""
    if k % 4 == 0 or not len(d):
        return d
    c, x, y, w, h = d.T
    if k == 1:
        X, Y, Wd, Ht = W0 - (y + h), x, h, w
    elif k == 2:
        X, Y, Wd, Ht = W0 - (x + w), H0 - (y + h), w, h
    else:
        X, Y, Wd, Ht = y, H0 - (x + w), h, w
    return np.column_stack([c, X, Y, Wd, Ht])


class FinalEnsemble:
    def __init__(self, model_paths, model_gpu, passes, thr=0.4, conf=0.01):
        """model_paths: {name: onnx}; model_gpu: {name: gpu id or None for CPU};
        passes: list of (group, name, k, long_side)."""
        self.dets = {m: OnnxDetector(p, conf=conf, device_id=model_gpu[m]) for m, p in model_paths.items()}
        self.passes, self.thr = list(passes), thr
        self.groups = []  # group names in first-seen order
        for g, _, _, _ in self.passes:
            if g not in self.groups:
                self.groups.append(g)
        by_gpu = {}
        for i, (_, m, _, _) in enumerate(self.passes):
            by_gpu.setdefault(model_gpu[m], []).append(i)
        self.by_gpu = list(by_gpu.values())
        self.pool = ThreadPoolExecutor(max_workers=len(self.by_gpu))

    def run_pass(self, bgr, i):
        _, m, k, L = self.passes[i]
        img = np.ascontiguousarray(np.rot90(bgr, k)) if k % 4 else bgr
        x, meta = self.dets[m].preprocess(img, shape=input_shape(L, k))
        return unrotate(self.dets[m].run(x, meta, img.shape[:2]), k)

    def predict_passes(self, frame_rgb):
        bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
        out = [None] * len(self.passes)
        for res in self.pool.map(lambda idx: [(i, self.run_pass(bgr, i)) for i in idx], self.by_gpu):
            for i, p in res:
                out[i] = p
        return out

    def fuse(self, per_pass):
        votes = []
        for g in self.groups:
            ps = [p for (gg, _, _, _), p in zip(self.passes, per_pass) if gg == g]
            votes.append(ps[0] if len(ps) == 1 else wbf(ps, self.thr))
        return wbf(votes, self.thr) if len(votes) > 1 else (votes[0] if votes else Z)

    def __call__(self, frame_rgb):
        return self.fuse(self.predict_passes(frame_rgb))
