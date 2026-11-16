"""Export weights to a dynamic-shape ONNX (any H, W multiple of 32), optionally converted to FP16.

For the final Kaggle ensemble, where one model runs at several sizes and rotations (portrait inputs
for 90/270 degrees): one session per model instead of one per shape keeps GPU memory down on the
T4s, and FP16 uses their tensor cores. FP32 exports on the CPU, FP16 on the GPU. Opset 12 for the 2021 Kaggle image's onnxruntime 1.11.

    python -m src.inference.export_onnx_dyn --weights model_dev/<run>/seed0/weights/best.pt --out x.onnx [--fp16]
"""

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    import tempfile

    from ultralytics import YOLO

    work = Path(tempfile.mkdtemp())  # export writes next to the .pt; keep it away from the run folder's own best.onnx
    pt = work / args.weights.name
    shutil.copy(args.weights, pt)
    # FP16 needs a CUDA export (Ultralytics casts the model with .half() and traces it; tracing at imgsz 640
    # takes about 1 GB, so it can share the GPU with a training run). onnxconverter-common's post-hoc
    # float16 conversion breaks the Resize nodes (fp16 scales), so it is not used.
    tmp = YOLO(str(pt)).export(format="onnx", imgsz=640, opset=args.opset, simplify=True, dynamic=True,
                               half=args.fp16, batch=1, nms=None, device=0 if args.fp16 else "cpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp, args.out)
    shutil.rmtree(work)
    print(args.out, "fp16" if args.fp16 else "fp32")


if __name__ == "__main__":
    main()
