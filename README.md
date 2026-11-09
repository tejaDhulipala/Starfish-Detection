# Starfish Detection

Crown-of-thorns starfish (COTS) detection on the Kaggle `tensorflow-great-barrier-reef` data, aimed at onboard inference for underwater camera traps.

## Layout

```
src/                 all code, run as modules from the repo root (python -m src....)
  paths.py           project paths
  index.py           model_dev/index.csv helpers
  data/              make_yolo_split.py, check_labels.ipynb
  train/             train.py
  eval/              f2.py (competition metric), test_f2.py, evaluate.py
  inference/         predict.py (frame in, detections out; shared with Kaggle)
model_dev/
  index.csv          one row per config, with its best F2
  cNN_<desc>/
    config.yaml      frozen once it has a run
    seed<N>/         run output (gitignored): meta.json, results.csv, weights/, eval/
kaggle/              submission notebook
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
