"""Standalone ONNX detector: frame in, (conf, x, y, w, h) out. No Ultralytics needed.

Exists because the competition API (greatbarrierreef) only imports on Kaggle's pinned 2021 image
(Python 3.7, torch 1.9), which cannot install Ultralytics 8.4. The network is exported to ONNX
locally (src.inference.export_onnx); this file replaces the Python around it and mirrors
Ultralytics 8.4.208 exactly:

  pre   LetterBox(auto=True): resize by min(H/h, W/w) with cv2.INTER_LINEAR, pad to the model's
        fixed input with value 114, centred (top = round(dh/2 - 0.1)); BGR -> RGB, /255, CHW
  post  output (1, 5, N) = cx, cy, w, h, score in input pixels (sigmoid already applied);
        keep score > conf, xywh -> xyxy, top max_nms by score, NMS at IoU 0.5, keep max_det;
        undo pad and gain, clip to the frame (scale_boxes)

Python 3.7 compatible on purpose (no walrus, no PEP 604 types). Must match src/inference/predict.py:
check with python -m src.eval.evaluate_onnx before submitting.
"""

import cv2
import numpy as np

try:
    import torch
    from torchvision.ops import nms as _tv_nms
except ImportError:  # numpy fallback; same greedy result for distinct scores
    _tv_nms = None


def _nms(boxes, scores, iou):
    if _tv_nms is not None:
        return _tv_nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou).numpy()
    order, keep = np.argsort(-scores, kind="stable"), []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        w = np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
        h = np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None)
        inter = w * h
        order = rest[inter / (area[i] + area[rest] - inter) <= iou]
    return np.array(keep, dtype=np.int64)


class OnnxDetector:
    def __init__(self, onnx_path, conf=0.01, iou=0.5, max_det=300, max_nms=30000, providers=None):
        import onnxruntime as ort

        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.in_h, self.in_w = inp.shape[2], inp.shape[3]
        self.conf, self.iou, self.max_det, self.max_nms = conf, iou, max_det, max_nms

    def preprocess(self, bgr):
        h, w = bgr.shape[:2]
        r = min(self.in_h / h, self.in_w / w)
        nw, nh = max(int(round(w * r)), 1), max(int(round(h * r)), 1)
        dw, dh = (self.in_w - nw) / 2, (self.in_h - nh) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = bgr if (nw, nh) == (w, h) else cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = img[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), (r, nw, nh, left, top)

    def postprocess(self, out, meta, frame_hw):
        r, nw, nh, left, top = meta
        p = out[0].T  # (N, 5)
        p = p[p[:, 4] > self.conf]
        if len(p) == 0:
            return np.zeros((0, 5))
        p = p[np.argsort(-p[:, 4], kind="stable")[: self.max_nms]]
        xyxy = np.empty((len(p), 4), np.float32)
        xyxy[:, 0] = p[:, 0] - p[:, 2] / 2
        xyxy[:, 1] = p[:, 1] - p[:, 3] / 2
        xyxy[:, 2] = p[:, 0] + p[:, 2] / 2
        xyxy[:, 3] = p[:, 1] + p[:, 3] / 2
        keep = _nms(xyxy, p[:, 4].astype(np.float32), self.iou)[: self.max_det]
        b, s = xyxy[keep].astype(np.float64), p[keep, 4].astype(np.float64)
        h, w = frame_hw
        b[:, [0, 2]] = np.clip((b[:, [0, 2]] - left) / (nw / w), 0, w)
        b[:, [1, 3]] = np.clip((b[:, [1, 3]] - top) / (nh / h), 0, h)
        dets = np.column_stack([s, b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])
        return dets[np.argsort(-s, kind="stable")]

    def __call__(self, frame_rgb):
        """frame_rgb: HxWx3 uint8 RGB (as the Kaggle API delivers). Returns (N, 5) conf, x, y, w, h."""
        bgr = np.ascontiguousarray(frame_rgb[..., ::-1])
        x, meta = self.preprocess(bgr)
        out = self.sess.run(None, {self.input_name: x})[0]
        return self.postprocess(out, meta, bgr.shape[:2])


def to_annotation_string(dets, conf_thr):
    """(N, 5) detections -> Kaggle submission string 'conf x y w h conf x y w h ...'."""
    keep = dets[dets[:, 0] >= conf_thr]
    return " ".join("{:.4f} {} {} {} {}".format(c, int(round(x)), int(round(y)), int(round(w)), int(round(h))) for c, x, y, w, h in keep)
