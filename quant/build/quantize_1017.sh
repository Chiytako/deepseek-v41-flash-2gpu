#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
# 1017 の routed experts を EXL3 K2 (MCG, 無校正, greedy beam 16) にする。diffbot recipe/quantize/quantize.sh を当機向けに。
# 使い方: quantize_1017.sh link | gpu <0|1> <first_layer> <last_layer> | finish
set -uo pipefail
MODELS=$MODELS
SRC=${SRC:-DeepSeek-V4.1-Flash-1017}; DST=${DST:-DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw}
IMAGE=dsv41-quant:5be8; RECIPE=$DSV41/build/qrecipe
BEAM=${BEAM:-16}; BATCH=${BATCH:-8}; LOGS=$DSV41/build/quant-logs; mkdir -p $LOGS
opts=(--rm --ipc host --shm-size 16g --user "$(id -u):$(id -g)" -e HOME=/tmp -v $MODELS:/models -v $RECIPE:/recipe --entrypoint python3)
tool=(-u /recipe/tools/quantize_experts_exl3_sm120.py --src /models/$SRC --dst /models/$DST)
shard() { printf 'model-%05d-of-00048.safetensors' $(( $1 + 3 )); }
case $1 in
  link) sudo docker run "${opts[@]}" $IMAGE "${tool[@]}" --link-only ;;
  dry) sudo docker run "${opts[@]}" $IMAGE "${tool[@]}" --dry-run ;;
  gpu) g=$2; f=""; for i in $(seq $3 $4); do f+="$(shard $i) "; done
       sudo docker run --name dsv41-quant-g$g --gpus "\"device=$g\"" "${opts[@]}" $IMAGE "${tool[@]}" \
         --allow-partial --batch $BATCH --greedy --beam $BEAM --device cuda:0 --only-files $f > $LOGS/quant-g$g-$3-$4.log 2>&1 ;;
  finish) sudo docker run "${opts[@]}" $IMAGE "${tool[@]}" --link-only && \
          sudo docker run "${opts[@]}" $IMAGE -u /recipe/tools/rebuild_index.py --dst /models/$DST
          ls $MODELS/$DST/model-*-of-00048.safetensors | wc -l ;;
esac
