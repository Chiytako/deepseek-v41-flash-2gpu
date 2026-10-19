#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
# v1 = L0: 1017 原本から Viterbi (exllamav3 1.5.1) で再符号化 + scale refit (H=I)。v0 を比較基準 (ref) に shard 単位で検査。
# 使い方: requant_v1.sh test <gpu> <unit> <limit> | gpu <gpu> <first_layer> <last_layer>
set -uo pipefail
MODELS=$MODELS
SRC=$MODELS/DeepSeek-V4.1-Flash-1017; REF=$MODELS/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw
OUT=$MODELS/dsv41-v1-work; mkdir -p $OUT/logs
IMAGE=${IMAGE:-dsv41-quant:151}; RECIPE=$DSV41/build/qrecipe
run() { local g=$1; shift
  sudo docker run --rm --name dsv41-requant-g$g --gpus "\"device=$g\"" --ipc host --shm-size 16g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v $MODELS:$MODELS -v $RECIPE:/recipe --entrypoint python3 $IMAGE -u /recipe/tools/requant_full.py encode \
    --src $SRC --ref $REF --out $OUT --tile-chunk ${TILE_CHUNK:-4608} "$@"; }
case $1 in
  test) run $2 --unit $(printf '%05d' $3) --limit $4 --check-first ${CHECK:-2} ;;
  gpu) g=$2; for L in $(seq $3 $4); do u=$(printf '%05d' $((L+3)))
         [ -f $OUT/units/$u.json ] && { echo "skip $u"; continue; }
         run $g --unit $u > $OUT/logs/unit-$u.log 2>&1; echo "unit $u rc=$? $(date)" >> $OUT/logs/g$g.log; done ;;
esac
