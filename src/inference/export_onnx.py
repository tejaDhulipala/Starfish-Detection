"""Export a run's weights to ONNX for the Kaggle notebook (kaggle/cots_onnx.py runs it).

    python -m src.inference.export_onnx --run model_dev/c04_yolo26s_1920_seq20val_pos_aug/seed0

Fixed input of the padded val/predict shape: imgsz W -> (round(720 * W/1280) up to a multiple of 32, W),
e.g. 1920 -> 1088x1920. Opset 12 so the 2021 Kaggle image's onnxruntime can load it. nms=None keeps
the raw one-to-many head output (1, 5, N), which is what src.inference.predict scored (NMS applied after).
"""

import argparse
import json
import math
from pathlib import Path

from src.paths import IMG_H, IMG_W


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--weights", default="best")
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = json.loads((args.run / "meta.json").read_text())["config"]
    w = cfg.get("val_imgsz") or cfg["train"]["imgsz"]
    h = math.ceil(round(IMG_H * w / IMG_W) / 32) * 32
    out = YOLO(str(args.run / "weights" / f"{args.weights}.pt")).export(
        format="onnx", imgsz=(h, w), opset=args.opset, simplify=True, dynamic=False, half=False, batch=1, nms=None
    )
    print(f"{out}  (input 1x3x{h}x{w})")


if __name__ == "__main__":
    main()
