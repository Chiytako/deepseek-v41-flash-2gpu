#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
# EXL3 パックを ds41rt が読む HF キャッシュ形式に hardlink で並べる。
set -euo pipefail
PACK=${1:-$MODELS/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw}
ID=${2:-local/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw}
REV=${3:-1017e3c200000000000000000000000000000002}
HF=$DSV41/hf
root=$HF/hub/models--${ID//\//--}
snap=$root/snapshots/$REV
mkdir -p "$snap" "$root/refs"
echo -n "$REV" > "$root/refs/main"
cd "$PACK"
find . -type d | while read -r d; do mkdir -p "$snap/$d"; done
find . -type f | while read -r f; do ln -f "$PACK/$f" "$snap/$f"; done
echo "staged $(find "$snap" -name 'model-*.safetensors' | wc -l) shards at $snap"
jq -c .quantization_config "$snap/config.json"
