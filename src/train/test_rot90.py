"""RandomRot90 moves boxes exactly with the pixels: python -m pytest src/train"""

import random

import numpy as np
from ultralytics.utils.instance import Instances

from src.train.rot90 import RandomRot90


def _labels(img, xyxy):
    return {"img": img.copy(), "instances": Instances(np.array(xyxy, dtype=np.float32), bbox_format="xyxy", normalized=False)}


def test_boxes_follow_pixels():
    h, w = 60, 100
    img = np.zeros((h, w, 3), np.uint8)
    img[10:20, 30:50] = 255  # box x 30..50, y 10..20
    for seed in range(8):
        random.seed(seed)
        out = RandomRot90(p=1.0)(_labels(img, [[30, 10, 50, 20]]))
        assert out["img"].shape[:2] == (w, h)
        x1, y1, x2, y2 = out["instances"].bboxes[0].round().astype(int)
        ys, xs = np.nonzero(out["img"][..., 0])
        assert (x1, y1, x2, y2) == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def test_normalized_xywh_roundtrip():
    random.seed(0)
    inst = Instances(np.array([[0.4, 0.25, 0.2, 0.1]], np.float32), bbox_format="xywh", normalized=True)
    out = RandomRot90(p=1.0)({"img": np.zeros((60, 100, 3), np.uint8), "instances": inst})
    assert out["instances"].normalized and out["instances"]._bboxes.format == "xywh"
    assert np.all((out["instances"].bboxes >= 0) & (out["instances"].bboxes <= 1))


def test_p0_is_identity():
    img = np.random.default_rng(0).integers(0, 255, (60, 100, 3), np.uint8)
    out = RandomRot90(p=0.0)(_labels(img, [[1, 2, 3, 4]]))
    assert np.array_equal(out["img"], img)


def test_dataset_class_pickles():
    import pickle

    from src.train.rot90 import Rot90YOLODataset

    ds = Rot90YOLODataset.__new__(Rot90YOLODataset)
    ds.rot90_p = 0.5
    assert pickle.loads(pickle.dumps(ds)).rot90_p == 0.5
