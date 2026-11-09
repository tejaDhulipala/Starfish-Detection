"""Make the competition F2 Ultralytics' per-epoch validation metric.

F2DetectionValidator runs Ultralytics' normal validation (P, R, mAP stay in results.csv
for reference) and additionally computes F2 with a confidence sweep. It sets
fitness = F2, so best.pt and early stopping (patience) follow F2 instead of mAP50-95.

This F2 is on the per-epoch val list (val_fast), with unrounded boxes and the clipped
YOLO labels as ground truth. It tracks the real score closely but is not identical:
the reported number comes from src.eval.evaluate on the full val set.
"""

from copy import copy

import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import LOGGER, ops

from src.eval.f2 import f2_sweep

CONF_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def _ltwh(boxes_xyxy):
    b = boxes_xyxy.cpu().numpy().astype(np.float64).reshape(-1, 4)
    return np.column_stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])


class F2DetectionValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self.f2_frames = []  # (preds (N, 5) conf x y w h, gts (M, 4) x y w h), original-image pixels

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            gt = ops.scale_boxes(pbatch["imgsz"], pbatch["bboxes"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
            pb = self.scale_preds(pred, pbatch)["bboxes"]
            conf = pred["conf"].cpu().numpy().astype(np.float64)
            self.f2_frames.append((np.column_stack([conf, _ltwh(pb)]), _ltwh(gt)))

    def get_stats(self):
        stats = super().get_stats()
        sweep = f2_sweep(self.f2_frames, CONF_GRID)
        b = int(np.argmax(sweep["f2"]))
        self.f2, self.f2_conf = float(sweep["f2"][b]), float(CONF_GRID[b])
        stats["metrics/F2"] = self.f2
        stats["metrics/F2_conf"] = self.f2_conf
        stats["fitness"] = self.f2  # drives best.pt and early stopping
        return stats

    def print_results(self):
        super().print_results()
        LOGGER.info(f"{'F2':>22}{self.f2:>11.4g}  at conf {self.f2_conf:.2f}")


class F2DetectionTrainer(DetectionTrainer):
    def get_validator(self):
        return F2DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)
