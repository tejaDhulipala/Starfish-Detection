# Autonomous run status (feat/cls-rescore-4x)

Updated by Claude while the user is away. Newest first.

## Queue (tmux session `starfish`)
1. c09 training (window `c09`): c06 fine-tuned at 4x (352 px tiles, imgsz 1408), 20 epochs, ~13 min/epoch (~4.3 h)
2. then window `chain`: c09 full-val eval -> classifier with extra train crops (224, 128) -> c10 training (~3.3 h) -> c10 eval

## Results so far
- Classifier rescoring, val candidates only (3 sequence folds, OOF):
  AUC at IoU>0.5: detector 0.970 vs classifier 0.843 (224), 0.838 (128), 0.737 (320).
  Best rescored F2 0.737 (mean of det and cls, 224) vs 0.744 detector-only. Not helping: too little data
  (~600 positive crops per fold). Next: add 25k crops from train frames (jittered GT + background).
- c09 smoke (1 epoch, 10% of tiles): F2 0.476 at 4x (c06 was 0.697 at 2.85x; expected dip after the scale change).
  Peak GPU memory during 5120 val is ~15 GB of 16, so nothing else can share the GPU during c09/c10.
