"""WBF ensemble of several OnnxDetectors, for the Kaggle notebook (Python 3.7 compatible).

Mirrors src/eval/ensemble.py exactly: each model's boxes (already NMS'd at IoU 0.5, conf > 0.01)
are pooled, sorted by confidence and clustered greedily (join the cluster whose current fused box
has IoU >= thr, else start a new one); fused box = confidence-weighted mean, fused conf = sum of
the cluster's confidences / number of models. No NMS after fusion.

Speed: models with the same input size share one preprocessed tensor, and models are spread over
the available GPUs and run in parallel threads (onnxruntime releases the GIL in session.run).
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cots_onnx import OnnxDetector

Z = np.zeros((0, 5))


def iou_matrix(a, b):
    """IoU between every box in a (N, 4) and b (M, 4), both (x, y, w, h). Same as src/eval/f2.py."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


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
    out = np.array([[f[0] / len(preds_list)] + list(f[1:]) for f in fused])
    return out[np.argsort(-out[:, 0])]


class Ensemble:
    def __init__(self, paths, devices, thr=0.4, conf=0.01):
        """paths: ONNX files; devices: one GPU id per model (models on one GPU run sequentially)."""
        self.dets = [OnnxDetector(p, conf=conf, device_id=d) for p, d in zip(paths, devices)]
        self.devices, self.thr = list(devices), thr
        self.groups = {}  # gpu -> model indices
        for i, d in enumerate(devices):
            self.groups.setdefault(d, []).append(i)
        self.pool = ThreadPoolExecutor(max_workers=len(self.groups))

    def predict_each(self, frame_rgb):
        """Per-model (N, 5) detections, in the order of paths."""
        bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
        hw = bgr.shape[:2]
        pre = {}
        for d in self.dets:  # one preprocess per input size
            k = (d.in_h, d.in_w)
            if k not in pre:
                pre[k] = d.preprocess(bgr)

        def run_group(idx):
            return [(i, self.dets[i].run(*pre[(self.dets[i].in_h, self.dets[i].in_w)], hw)) for i in idx]

        out = [None] * len(self.dets)
        for res in self.pool.map(run_group, self.groups.values()):
            for i, p in res:
                out[i] = p
        return out

    def __call__(self, frame_rgb):
        return wbf(self.predict_each(frame_rgb), self.thr)
