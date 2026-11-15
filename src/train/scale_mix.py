"""Mixture-of-uniforms zoom distribution for Ultralytics' RandomPerspective.

Ultralytics samples the zoom factor uniformly: s ~ U(scale[0], scale[1]). scale_mix replaces that
with a mixture, e.g. [[0.7, 0.8, 1.2], [0.3, 1.2, 1.8]] = with probability 0.7 draw from U(0.8, 1.2)
(near native size, where most val starfish are), else from U(1.2, 1.8) (zoomed-in tail for the
large close-ups). Each component is [weight, lo, hi]; weights must sum to 1.

How: RandomPerspective._compute_affine_matrix draws s = random.uniform(scale[0], scale[1]).
MixScalePerspective draws s from the mixture first, sets scale = (s, s) for that one call
(uniform(s, s) == s), and restores it, so the rest of Ultralytics' affine code is untouched.

with_scale_mix(trainer_cls, mix) returns a trainer whose training dataset swaps every
RandomPerspective in its transforms (including those nested in Mosaic / MixUp pre_transforms)
for MixScalePerspective. Like rot90 it survives close_mosaic (which rebuilds the transforms),
composes with with_rot90 (apply scale_mix outermost), and pickles for forkserver workers.
"""

import random

from ultralytics.data.augment import RandomPerspective
from ultralytics.data.dataset import YOLODataset

from src.train.rot90 import Rot90YOLODataset


def check_mix(mix):
    mix = [tuple(float(v) for v in c) for c in mix]
    if any(len(c) != 3 or c[0] < 0 or c[1] > c[2] or c[1] <= 0 for c in mix):
        raise ValueError(f"scale_mix components must be [weight, lo, hi] with 0 < lo <= hi: {mix}")
    if abs(sum(c[0] for c in mix) - 1) > 1e-6:
        raise ValueError(f"scale_mix weights must sum to 1: {mix}")
    return mix


def sample_scale(mix):
    r, acc = random.random(), 0.0
    for w, lo, hi in mix:
        acc += w
        if r < acc:
            return random.uniform(lo, hi)
    return random.uniform(mix[-1][1], mix[-1][2])


class MixScalePerspective(RandomPerspective):
    scale_mix = None

    def _compute_affine_matrix(self, img, size):
        s = sample_scale(self.scale_mix)
        orig = self.scale
        self.scale = (s, s)
        try:
            return super()._compute_affine_matrix(img, size)
        finally:
            self.scale = orig


def patch_transforms(obj, mix, seen=None):
    """Swap every RandomPerspective reachable from obj (Compose.transforms, .pre_transform) to MixScalePerspective."""
    seen = set() if seen is None else seen
    if obj is None or id(obj) in seen:
        return 0
    seen.add(id(obj))
    n = 0
    if isinstance(obj, RandomPerspective):
        obj.__class__ = MixScalePerspective
        obj.scale_mix = mix
        n += 1
    for child in getattr(obj, "transforms", None) or []:
        n += patch_transforms(child, mix, seen)
    n += patch_transforms(getattr(obj, "pre_transform", None), mix, seen)
    return n


class ScaleMixYOLODataset(Rot90YOLODataset):
    """Rot90YOLODataset (rot90 only applies if rot90_p > 0) whose RandomPerspective samples from scale_mix."""

    scale_mix = None

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment and self.scale_mix:
            n = patch_transforms(transforms, self.scale_mix)
            if n == 0:
                raise RuntimeError("scale_mix: no RandomPerspective found in the training transforms")
            print(f"scale_mix: {n} RandomPerspective patched to {self.scale_mix}")
        return transforms


def with_scale_mix(base, mix):
    mix = check_mix(mix)

    class ScaleMixTrainer(base):
        def build_dataset(self, img_path, mode="train", batch=None):
            ds = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                if type(ds) not in (YOLODataset, Rot90YOLODataset):
                    raise TypeError(f"scale_mix expects a YOLODataset, got {type(ds).__name__}")
                ds.__class__ = ScaleMixYOLODataset
                ds.scale_mix = mix
                ds.transforms = ds.build_transforms(hyp=self.args)
            return ds

    ScaleMixTrainer.__name__ = f"{base.__name__}ScaleMix"
    return ScaleMixTrainer
