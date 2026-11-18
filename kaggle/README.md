# Kaggle submissions

Each folder is one submitted notebook. Scores are public / private.

| Folder | What was submitted | Public | Private |
|---|---|---|---|
| `submission.py` (top level) | A single YOLO26s run on full frames upscaled to 1920 px | 0.484 | 0.552 |
| `r2/` | The same YOLO26s run at 3840 px, twice its training size | 0.437 | 0.345 |
| `c04ms/` | The same YOLO26s run at three sizes (1920, 2112, 2304 px) and merged | 0.511 | 0.565 |
| `ens4/` | Three models merged: the 1920 px YOLO26s plus a YOLO26s and a YOLO26m trained on 2.85x upscaled tiles | 0.584 | 0.676 |
| `final/` | **Best submission.** Five models (three YOLO26s and two YOLO26m), all fine-tuned on train plus validation data, run at several sizes and rotations for 8 merged votes | 0.581 | **0.700** |
| `final7/` | The same as `final/` with one rotation dropped to run faster | 0.580 | 0.698 |

Shared code: `cots_onnx.py` runs one ONNX model, `cots_ensemble.py` and `final_ensemble.py` merge several with weighted boxes fusion.
