#!/usr/bin/env bash
# The full CIFAR-10 study: the baseline and four ablation runs (8 epochs each),
# then two 30-epoch candidates for the final model (src.evaluate picks one on
# validation accuracy). About 2.5 hours on a laptop CPU, a few minutes on a GPU.
set -e
PY=${PYTHON:-python}
for exp in baseline full no-batchnorm no-augment no-dropout; do
  $PY -m src.train --experiment $exp --epochs 8 "$@"
done
$PY -m src.train --experiment full --epochs 30 "$@"
$PY -m src.train --experiment no-dropout --epochs 30 "$@"
