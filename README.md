# Starfish Detection

## Summary
 - Objective: Crown-of-thorns starfish (COTS) detection on the Kaggle `tensorflow-great-barrier-reef` data.
 - Used a ten-model ensemble of YOLO26s and YOLO26m, trained at 2.85x and 1.5x, with test-time augmentations to achieve a 0.700 private dataset F2 score. This would've been the 27th on the leaderboard if it were submitted before the deadline. The full descriptio of the final submission is available here: https://docs.google.com/document/d/1IlJIUWlzXzE9oK6iSRH5iznNSovMsfwjFqIORJHocGs/edit?tab=t.sjikmkx1ik3v
 - Trained dozens of model and did tens of experiments, all documented in detail here: https://docs.google.com/document/d/1IlJIUWlzXzE9oK6iSRH5iznNSovMsfwjFqIORJHocGs/edit?tab=t.kqet3g5pm434
<img width="1464" height="99" alt="{CC33339B-B6B3-45DD-BFA3-119A82AD22F0}" src="https://github.com/user-attachments/assets/28220054-7ac4-4d97-a049-8d2e70e01e0d" />


## Repo Organization

src/                     all code, run as modules from the repo root (python -m src....)
  paths.py               project paths
  index.py               model_dev/index.csv helpers
  data/                  make_yolo_split.py, make_crop_split.py, make_grid_split.py (overlapping tiles),
                         make_addval_split.py (train+val splits), check_labels.ipynb
  train/                 train.py, f2_validator.py, rot90.py, scale_mix.py, test_rot90.py
  eval/                  f2.py (competition metric), test_f2.py, evaluate.py, evaluate_onnx.py,
                         ensemble.py (WBF), tiled_eval.py, fail_analysis.py, training_curves.py
  inference/             predict.py (frame in, detections out), export_onnx.py, export_onnx_dyn.py (FP16, any input size)
  postprocess/           attention_pp.py, scaled_wbf.py, softmax_wbf.py (post-processing experiments)
  rescore/               make_candidates.py, make_train_crops.py, train_cls.py (rescoring classifier experiments)
model_dev/
  index.csv              one row per config, with its best F2
  cNN_<desc>/
    config.yaml          frozen once it has a run
    seed<N>/             run output (gitignored): meta.json, results.csv, weights/, eval/
  rescore/rNN_<desc>/    rescoring classifier configs and results
kaggle/                  README.md (each submission and its score), cots_onnx.py (Ultralytics-free ONNX detector),
                         cots_ensemble.py, final_ensemble.py (multi-model WBF)
  final/, final7/, ...   one folder per submitted notebook: submission.py + kernel-metadata.json
scripts/                 train_and_eval.sh
pretrained/              downloaded pretrained weights (gitignored)
data/                    Kaggle data and generated YOLO labels/splits (gitignored)
