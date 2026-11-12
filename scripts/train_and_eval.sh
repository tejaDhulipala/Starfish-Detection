#!/usr/bin/env bash
# Train one config/seed, then score it on the full val set. Usage: scripts/train_and_eval.sh <config dir> [seed]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-../Starfish-Detection/.venv/bin/python}
CFG=$1; SEED=${2:-0}
"$PY" -m src.train.train --config "$CFG/config.yaml" --seed "$SEED"
"$PY" -m src.eval.evaluate --run "$CFG/seed$SEED" --weights best
echo "DONE $CFG seed$SEED"
