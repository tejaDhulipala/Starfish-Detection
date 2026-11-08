"""Frame in, detections out. The same code runs locally on val frames and in the Kaggle notebook.

Channel order: the Kaggle API's pixel_array is RGB (channel means of data/example_test.npy
are lowest in channel 0, as expected for red underwater; matches train JPGs loaded as RGB).
Ultralytics treats numpy input as BGR, so frames are flipped before predict.
"""

import numpy as np


class Detector:
    def __init__(self, weights, imgsz=1280, conf=0.01, device=None, max_det=300):
        from ultralytics import YOLO  # imported here so f2/eval code does not need ultralytics

        self.model = YOLO(str(weights))
        self.kwargs = dict(imgsz=imgsz, conf=conf, device=device, max_det=max_det, verbose=False)

    def __call__(self, frames_rgb):
        """frames_rgb: list of HxWx3 uint8 RGB arrays. Returns one (N, 5) array of
        (conf, x, y, w, h) per frame, pixels, (x, y) top-left, sorted by conf descending."""
        results = self.model.predict([np.ascontiguousarray(f[..., ::-1]) for f in frames_rgb], **self.kwargs)
        out = []
        for r in results:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float64)
            conf = r.boxes.conf.cpu().numpy().astype(np.float64)
            dets = np.column_stack([conf, xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])
            out.append(dets[np.argsort(-conf, kind="stable")])
        return out


def to_annotation_string(dets, conf_thr):
    """(N, 5) detections -> Kaggle submission string 'conf x y w h conf x y w h ...'."""
    keep = dets[dets[:, 0] >= conf_thr]
    return " ".join(f"{c:.4f} {round(x)} {round(y)} {round(w)} {round(h)}" for c, x, y, w, h in keep)
