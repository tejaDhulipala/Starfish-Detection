# Starfish Detection

## Summary
 - Objective: Crown-of-thorns starfish (COTS) detection on the Kaggle `tensorflow-great-barrier-reef` data.
 - Results: Used a ten-model ensemble of YOLO26s and YOLO26m, trained at 2.85x and 1.5x, with test-time augmentations to achieve a 0.700 private dataset F2 score. This would've been the 27th on the leaderboard if it were submitted before the deadline. The full descriptio of the final submission is available here: [final submission description](https://docs.google.com/document/d/1IlJIUWlzXzE9oK6iSRH5iznNSovMsfwjFqIORJHocGs/edit?tab=t.sjikmkx1ik3v)
 - Trained dozens of model and did tens of experiments, all documented in detail here: [detailed experiment log](https://docs.google.com/document/d/1IlJIUWlzXzE9oK6iSRH5iznNSovMsfwjFqIORJHocGs/edit?tab=t.kqet3g5pm434)
<img width="1464" height="99" alt="{CC33339B-B6B3-45DD-BFA3-119A82AD22F0}" src="https://github.com/user-attachments/assets/28220054-7ac4-4d97-a049-8d2e70e01e0d" />

## Usage

### How to run a new training run

1. Make a new config folder by copying the closest existing one, and change
   what you are testing. Configs are frozen once they have a run, so every
   change gets a new folder.

   ```bash
   mkdir model_dev/c27_<short_desc>
   cp model_dev/c08_yolo26m_grid493_3x2_seq20val_pos_aug/config.yaml model_dev/c27_<short_desc>/
   ```

   `model` is the starting weights (a pretrained name like `yolo26m.pt`, or a
   path to a `best.pt`), `split` is the dataset, and everything under `train:`
   is passed to Ultralytics.

2. Smoke test (1 epoch on 10% of train) to check memory and speed:

   ```bash
   python -m src.train.train --config model_dev/c27_<short_desc>/config.yaml --smoke
   ```

3. Train, then score the best weights on the full validation set:

   ```bash
   python -m src.train.train --config model_dev/c27_<short_desc>/config.yaml --seed 0
   python -m src.eval.evaluate --run model_dev/c27_<short_desc>/seed0 --weights best
   ```

   Or do both with `scripts/train_and_eval.sh model_dev/c27_<short_desc>`.
   Per-epoch F2 is in `seed0/results.csv`; the full-val F2 and best
   confidence cutoff are in `seed0/eval/best_f2.json` and `model_dev/index.csv`.
   An interrupted run continues with `--resume`.

### How to submit a model to Kaggle

The competition notebook runs on Kaggle's 2021 image (Python 3.7, no
internet), which can't run Ultralytics, so models are submitted as ONNX files
run by `kaggle/cots_onnx.py`.

1. Export the weights to ONNX (FP16, any input size):

   ```bash
   python -m src.inference.export_onnx_dyn --weights model_dev/<run>/seed0/weights/best.pt --out kds/<name>.onnx --fp16
   ```

2. Build a Kaggle dataset folder (`kds/` above) with the ONNX files,
   `kaggle/cots_onnx.py`, `kaggle/cots_ensemble.py`, `kaggle/final_ensemble.py`,
   the `onnxruntime_gpu-1.11.0` (cp37) and `flatbuffers-2.0` wheels, and a
   `dataset-metadata.json`. Then upload it:

   ```bash
   kaggle datasets create -p kds            # first time
   kaggle datasets version -p kds -m "msg"  # later updates
   ```

3. Copy the closest notebook folder (for example `kaggle/final/`). In
   `kernel-metadata.json`, set a new `id` and point `dataset_sources` at your
   dataset. In `submission.py`, set the models, input sizes and `CONF_THR`
   (the confidence cutoff tuned on validation). Then push it:

   ```bash
   kaggle kernels push -p kaggle/<folder>
   kaggle kernels status <username>/<kernel-name>
   ```

   The run on Kaggle's sample frames prints a per-frame timing benchmark.
   Check it fits the 9 hour limit (about 13,000 test frames) before submitting.

4. Submit the notebook version to the competition:

   ```bash
   kaggle competitions submit -c tensorflow-great-barrier-reef -k <username>/<kernel-name> -v <version> -f submission.csv -m "description"
   ```

   Scoring takes as long as the notebook runs on the hidden test set (hours).
   Check progress with `kaggle competitions submissions -c tensorflow-great-barrier-reef`.

## Repo Organization
```
src/                   all code, run as modules (python -m src....)
  paths.py             project paths
  index.py             model_dev/index.csv helpers
  data/                train/val splits, crops, overlapping tiles
  train/               train.py, F2 validator, rot90 and zoom augmentation
  eval/                F2 metric, evaluation, WBF ensembling, failure analysis
  inference/           predict.py, ONNX export
  postprocess/         post-processing experiments
  rescore/             rescoring classifier experiments
model_dev/
  index.csv            one row per config, with its best F2
  cNN_<desc>/
    config.yaml        frozen once it has a run
    seed<N>/           run output (gitignored): results.csv, weights/, eval/
  rescore/             rescoring classifier configs and results
kaggle/

  README.md            each submission and its score
  cots_onnx.py         Ultralytics-free ONNX detector
  final_ensemble.py    multi-model, multi-size WBF ensemble
  final/, ...          one folder per submitted notebook
scripts/               train_and_eval.sh
pretrained/            downloaded pretrained weights (gitignored)
data/                  Kaggle data and YOLO labels/splits (gitignored)
```
