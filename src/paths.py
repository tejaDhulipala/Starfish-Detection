"""Project paths in one place. Everything is absolute, so code works from any cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRAIN_CSV = DATA / "train.csv"
YOLO_DATA = DATA / "yolo"
MODEL_DEV = ROOT / "model_dev"
PRETRAINED = ROOT / "pretrained"

IMG_W, IMG_H = 1280, 720


def split_yaml(split):
    return YOLO_DATA / f"{split}.yaml"


def split_dir(split):
    return YOLO_DATA / "splits" / split
