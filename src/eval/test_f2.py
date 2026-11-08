"""Hand-made cases for src/eval/f2.py. Run: python -m pytest src/eval"""

import numpy as np
import pytest

from src.eval.f2 import IOU_THRESHOLDS, f2_score, f2_sweep, iou_matrix, match_frame

GT = [[0, 0, 10, 10]]


def test_thresholds():
    assert len(IOU_THRESHOLDS) == 11
    assert IOU_THRESHOLDS[0] == 0.30 and IOU_THRESHOLDS[-1] == 0.80


def test_iou():
    assert iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == pytest.approx(1.0)
    assert iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 20]])[0, 0] == pytest.approx(0.5)
    assert iou_matrix([[0, 0, 10, 10]], [[20, 20, 5, 5]])[0, 0] == 0.0


def test_perfect():
    assert f2_score([([[0.9, 0, 0, 10, 10]], GT)]) == pytest.approx(1.0)


def test_no_predictions():
    assert f2_score([(np.zeros((0, 5)), GT)]) == 0.0


def test_empty_frame_no_predictions_is_perfect():
    assert f2_score([(np.zeros((0, 5)), np.zeros((0, 4)))]) == 1.0


def test_iou_exactly_half_counts_at_thresholds_up_to_half():
    # IoU = 0.5: TP at 0.30..0.50 (5 thresholds, F2 = 1), FP + FN at 0.55..0.80 (6 thresholds, F2 = 0)
    assert f2_score([([[0.9, 0, 0, 10, 20]], GT)]) == pytest.approx(5 / 11)


def test_false_positive_on_empty_frame():
    # Pooled: TP = 1, FP = 1, FN = 0 -> 5 / 6 at every threshold
    frames = [([[0.9, 0, 0, 10, 10]], GT), ([[0.9, 50, 50, 10, 10]], np.zeros((0, 4)))]
    assert f2_score(frames) == pytest.approx(5 / 6)


def test_micro_not_macro_average():
    # Frame A perfect, frame B misses 3 boxes. Pooled: TP = 1, FN = 3 -> 5 / 17.
    # Averaging per frame would give 0.5 instead.
    frames = [([[0.9, 0, 0, 10, 10]], GT), (np.zeros((0, 5)), [[0, 0, 10, 10], [20, 0, 10, 10], [40, 0, 10, 10]])]
    assert f2_score(frames) == pytest.approx(5 / 17)


def test_duplicate_prediction_is_fp():
    # Second box on the same starfish has nothing left to match
    frames = [([[0.9, 0, 0, 10, 10], [0.8, 0, 0, 10, 10]], GT)]
    assert f2_score(frames) == pytest.approx(5 / 6)


def test_confidence_cutoff_drops_low_predictions():
    frames = [([[0.9, 0, 0, 10, 10], [0.1, 50, 50, 10, 10]], GT)]
    assert f2_score(frames, conf_thr=0.0) == pytest.approx(5 / 6)
    assert f2_score(frames, conf_thr=0.5) == pytest.approx(1.0)


def test_higher_confidence_matches_first():
    # Low-conf box is listed first but the high-conf one gets the match
    conf, is_tp = match_frame([[0.2, 0, 0, 10, 10], [0.9, 0, 0, 10, 10]], GT, 0.5)
    assert list(conf) == [0.9, 0.2] and list(is_tp) == [True, False]


def test_first_unmatched_gt_rule():
    # P1 overlaps GT0 (IoU 0.43) and GT1 (IoU 0.82); P2 overlaps only GT1 (0.82; GT0 0.25).
    # "First unmatched GT with IoU >= thr": at 0.30 P1 takes GT0 and P2 takes GT1 -> 2 TP.
    # (A best-IoU rule would give P1 -> GT1, P2 -> FP, GT0 -> FN.)
    preds = [[0.9, 4, 0, 10, 10], [0.8, 6, 0, 10, 10]]
    gts = [[0, 0, 10, 10], [5, 0, 10, 10]]
    assert list(match_frame(preds, gts, 0.30)[1]) == [True, True]
    # At 0.45 GT0 is out of reach for P1, so it takes GT1 and P2 is left over.
    assert list(match_frame(preds, gts, 0.45)[1]) == [True, False]


def test_sweep_matches_rematching_at_each_cutoff():
    # Prefix consistency: one matching pass gives the same counts as re-matching after filtering.
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(50):
        n_gt = rng.integers(0, 4)
        gts = np.column_stack([rng.uniform(0, 100, (n_gt, 2)), rng.uniform(5, 20, (n_gt, 2))])
        jitter = gts + rng.normal(0, 3, gts.shape)
        n_fp = rng.integers(0, 3)
        fps = np.column_stack([rng.uniform(0, 100, (n_fp, 2)), rng.uniform(5, 20, (n_fp, 2))])
        boxes = np.vstack([jitter, fps])
        preds = np.column_stack([rng.uniform(0, 1, len(boxes)), boxes])
        frames.append((preds, gts))
    grid = np.linspace(0, 1, 21)
    swept = f2_sweep(frames, grid)["f2"]
    for c, s in zip(grid, swept):
        filtered = [(p[p[:, 0] >= c], g) for p, g in frames]
        assert s == pytest.approx(f2_score(filtered, 0.0))
