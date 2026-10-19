#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
# v1 パック = v0 の全ファイルを hardlink し、expert shard (00003〜00042) を v1 の出力 (dsv41-v1-work/shards) に置き換える。
set -euo pipefail
M=$MODELS
V0=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw; W=$M/dsv41-v1-work; V1=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1
n=$(ls $W/shards/model-*.safetensors | wc -l); [ "$n" = 40 ] || { echo "v1 shards: $n/40"; exit 1; }
mkdir -p $V1; cd $V0
find . -type d | while read -r d; do mkdir -p "$V1/$d"; done
find . -type f | while read -r f; do ln -f "$V0/$f" "$V1/$f"; done
for s in $W/shards/model-*.safetensors; do ln -f "$s" "$V1/$(basename $s)"; done
echo "assembled $V1"; $DSV41/build/stage_hf.sh $V1 local/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1 1017e3c200000000000000000000000000000012
