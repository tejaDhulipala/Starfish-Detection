# Starfish Detection

Crown-of-thorns starfish (COTS) detection on the Kaggle `tensorflow-great-barrier-reef` data, aimed at onboard inference for underwater camera traps.

## Layout

```
src/                 all code, run as modules from the repo root (python -m src....)
  paths.py           project paths
  index.py           model_dev/index.csv helpers
  data/              make_yolo_split.py, make_crop_split.py, check_labels.ipynb
  train/             train.py
  eval/              f2.py (competition metric), test_f2.py, evaluate.py, evaluate_onnx.py
  inference/         predict.py (frame in, detections out), export_onnx.py
model_dev/
  index.csv          one row per config, with its best F2
  cNN_<desc>/
    config.yaml      frozen once it has a run
    seed<N>/         run output (gitignored): meta.json, results.csv, weights/, eval/
kaggle/              submission.py + kernel-metadata.json, cots_onnx.py (Ultralytics-free ONNX detector)
pretrained/          downloaded pretrained weights (gitignored)
data/                Kaggle data and generated YOLO labels/splits (gitignored)
```

## Workflow

Run everything from the repo root with the venv's Python.

```bash
# 1. Labels and split (writes data/yolo/)
python -m src.data.make_yolo_split --val-video 0

# 2. Metric tests
python -m pytest src/eval

# 3. Train (smoke test first: 1 epoch on 10% of train)
python -m src.train.train --config model_dev/c01_yolo26s_1280_v0val_pos/config.yaml --smoke
python -m src.train.train --config model_dev/c01_yolo26s_1280_v0val_pos/config.yaml --seed 0

# 4. Full-val F2 with a confidence sweep
python -m src.eval.evaluate --run model_dev/c01_yolo26s_1280_v0val_pos/seed0
```

## Conventions

- One config per experiment. To change anything, copy the folder to a new `cNN_...` name.
- Runs of the same config differ only by seed.
- Per-epoch validation during training uses the split's `val_fast` list (a strided subset of val, set by `--val-fast-step`). `src.eval.evaluate` scores the full val set and gives the reported F2.
- `fitness: f2` in a config makes per-epoch validation compute F2 and use it for `best.pt` and early stopping (`src/train/f2_validator.py`). Ultralytics' mAP columns stay in `results.csv` for reference.
- Splits: `--val-video N` holds out a whole video; `--val-seq-frac 0.2` holds out whole sequences, chosen per video.
- Kaggle: the competition API only imports on Kaggle's pinned 2021 image (Python 3.7, torch 1.9), which cannot run Ultralytics 8.4. So the model is exported to ONNX (`src.inference.export_onnx`), run by `kaggle/cots_onnx.py` with onnxruntime-gpu 1.11, and checked against Ultralytics on val with `src.eval.evaluate_onnx` before submitting. The private dataset `tejadhu/starfish-c04-onnx` holds the ONNX file, `cots_onnx.py` and the wheels.
- Crop splits (`src.data.make_crop_split --base <split> --scale S`): train on crops of (1280/S) x (720/S) px at `imgsz: 1280, rect: true`, which upscales them S times; set `val_imgsz` in the config (e.g. 3648 for S = 2.85) so per-epoch val and `src.eval.evaluate` run on full frames at the same scale.
