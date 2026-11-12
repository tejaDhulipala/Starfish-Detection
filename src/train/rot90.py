"""Random 90-degree rotation for Ultralytics detection training.

Ultralytics has no exact 90-degree rotation (`degrees` rotates by any angle and re-boxes the
rotated corners, which loosens boxes). RandomRot90 turns the image a quarter turn (clockwise
or counter-clockwise, equally likely) with probability p, and moves the boxes exactly.
With flipud 0.5 and fliplr 0.5 also on, p = 0.5 makes all 8 orientations equally likely.

with_rot90(trainer_cls, p) returns a trainer whose training dataset runs RandomRot90 last,
just before Ultralytics' Format step. It survives close_mosaic, which rebuilds the transforms, and pickles for forkserver workers.
"""

import random

import numpy as np
from ultralytics.data.dataset import YOLODataset


class RandomRot90:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, labels):
        if random.random() >= self.p:
            return labels
        img = labels["img"]
        h, w = img.shape[:2]
        inst = labels.pop("instances")
        fmt, was_norm = inst._bboxes.format, inst.normalized
        inst.convert_bbox(format="xyxy")
        inst.denormalize(w, h)
        b = inst.bboxes
        if random.random() < 0.5:  # counter-clockwise (np.rot90 k=1): (x, y) -> (y, w - x)
            k, nb = 1, np.stack([b[:, 1], w - b[:, 2], b[:, 3], w - b[:, 0]], axis=1)
        else:  # clockwise (k=3): (x, y) -> (h - y, x)
            k, nb = 3, np.stack([h - b[:, 3], b[:, 0], h - b[:, 1], b[:, 2]], axis=1)
        inst.update(bboxes=nb.astype(b.dtype, copy=False))
        if was_norm:
            inst.normalize(h, w)  # new width is the old height
        inst.convert_bbox(format=fmt)
        labels["img"] = np.ascontiguousarray(np.rot90(img, k))
        labels["instances"] = inst
        return labels


class Rot90YOLODataset(YOLODataset):
    """YOLODataset whose training transforms end with RandomRot90 (module-level so workers can unpickle it)."""

    rot90_p = 0.0

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment and self.rot90_p > 0:
            transforms.insert(-1, RandomRot90(self.rot90_p))  # before Format
        return transforms


def with_rot90(base, p):
    class Rot90Trainer(base):
        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == "train" and p > 0:
                if type(ds) is not YOLODataset:
                    raise TypeError(f"rot90 expects a YOLODataset, got {type(ds).__name__}")
                ds.__class__ = Rot90YOLODataset
                ds.rot90_p = p
                ds.transforms = ds.build_transforms(hyp=self.args)
            return ds

    Rot90Trainer.__name__ = f"{base.__name__}Rot90"
    return Rot90Trainer
