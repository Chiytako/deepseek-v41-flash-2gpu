#!/bin/bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
for L in 20 3 37; do
  uv run --no-project --with numpy python -I $DSV41/tools/r3_fold_measure.py --layer $L --cov-stride 2 --out $DSV41/work/r3fold/layer$L.json > $DSV41/work/r3fold/layer$L.log 2>&1
done
