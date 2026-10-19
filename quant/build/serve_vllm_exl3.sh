#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
# DeepSeek-V4.1-Flash-1017 EXL3 2.0bpw (pack v0) を本家 vLLM + vllm-exl3 (diffbot sm_120 レシピ) で TP2 起動する。
# ds41rt (口 8000) との A/B 用。口 8001、モデル名 deepseek-v4.1-flash。
#
# 構成:
#   image   : dsv41-flash-exl3-sm120-prestage (= diffbot の dsv41-flash-exl3-sm120 + nktlabs Engram prestage +
#             diffbot sitecustomize + FlashInfer sm120 prefill パッチ + xmoe を焼き込んだ派生イメージ)
#   Engram  : NVMe 上の shard 47/48 を preadv (DSV41_ENGRAM_DISK=1)。行の取得を prepare_inputs に移し
#             (DSV41_ENGRAM_PRESTAGE=1)、decode は CUDA graph のまま。pinned は staging の計 96MiB だけ。
#   RAM     : 実 RAM 121GiB。container は --memory=$MEM_LIMIT (既定 112g、page cache 込み) で、--memory-swap も同値 (swap 0)。
#             Engram の「熱い行」はページキャッシュ (この上限の内側) が持つ。
#   KV      : GPU 上に固定 8GiB/GPU (--kv-cache-memory)。CPU への KV offload / swap は使わない。
#
# 使い方:
#   DRY_RUN=1 ./serve_vllm_exl3.sh          # docker コマンドを表示し、GPU 無し container で引数を検証するだけ
#   ./serve_vllm_exl3.sh                    # 本番起動 (前景。Ctrl-C で container を消す)
#   PARITY=1 ./serve_vllm_exl3.sh           # 初回推奨: eager + prestage parity 検査 (DSV41_ENGRAM_PRESTAGE_CHECK=1)
#   ENGRAM_PRESTAGE=0 EAGER=1 ./serve_vllm_exl3.sh   # 比較用: diffbot/sfxnz の disk 経路 (eager)
#
# 主なつまみ (環境変数): PORT MAX_NUM_SEQS MAX_MODEL_LEN KV_MEM UTIL SPEC(dspark|none) THINKING(default|off)
#   EAGER XMOE_EXT(空=stock exl3_moe) MEM_LIMIT ENGRAM_THREADS API_KEY(既定 なし=認証無し) FORCE=1(GPU 使用中でも起動)
set -u
PACK=${PACK:-$MODELS/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw}
IMAGE=${IMAGE:-dsv41-flash-exl3-sm120-prestage}
NAME=${NAME:-dsv41-vllm-exl3}
PORT=${PORT:-8001}
SERVED=${SERVED:-deepseek-v4.1-flash}
CACHE_DIR=${CACHE_DIR:-$DSV41/build/vllm_exl3_cache}   # FlashInfer JIT / torch.compile / Triton
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-524288}    # V4.1 の KV は小さい: 8GiB で約 3.8M token (diffbot)
MAX_BATCHED=${MAX_BATCHED:-4096}          # chunked prefill の 1 step 上限 (prestage staging もこれで決まる)
KV_MEM=${KV_MEM:-8589934592}              # 8 GiB / GPU
UTIL=${UTIL:-0.92}
EAGER=${EAGER:-0}
SPEC=${SPEC:-dspark}
NSPEC=${NSPEC:-5}
THINKING=${THINKING:-default}
INDEXER_KV=${INDEXER_KV-mxfp4}            # sm_120 の 32-token 圧縮 page には MXFP4 indexer cache が要る
LINEAR_BACKEND=${LINEAR_BACKEND-marlin}   # dense MXFP8: decode は Marlin W8A16 が速い
NO_CUSTOM_AR=${NO_CUSTOM_AR-1}            # custom all-reduce は graph capture で落ちる (diffbot)
XMOE_EXT=${XMOE_EXT-xmoe}                 # image 内 /opt/xmoe/build/xmoe。空 = stock exllamav3 exl3_moe
FI_ARCH=${FI_ARCH:-12.0f}
ENGRAM_DISK=${ENGRAM_DISK:-1}
ENGRAM_PRESTAGE=${ENGRAM_PRESTAGE:-1}
ENGRAM_THREADS=${ENGRAM_THREADS:-32}
PARITY=${PARITY:-0}
MEM_LIMIT=${MEM_LIMIT:-112g}              # container の RAM 上限 (page cache 込み)。swap は使わせない
SHM=${SHM:-16g}
API_KEY=${API_KEY:-}
DRY_RUN=${DRY_RUN:-0}
FORCE=${FORCE:-0}
# MAX_NUM_SEQS x (1 + NSPEC) まで capture する (16 x 6 = 96)
if [[ -z ${CAPTURE_SIZES:-} ]]; then
  if [[ $SPEC == dspark ]]; then top=$(( MAX_NUM_SEQS * (NSPEC + 1) )); else top=$MAX_NUM_SEQS; fi
  CAPTURE_SIZES=""
  for s in 1 2 4 6 8 12 16 24 32 48 64 80 96 112 128 160 192; do (( s <= top )) && CAPTURE_SIZES+="$s "; done
  [[ " $CAPTURE_SIZES" == *" $top "* ]] || CAPTURE_SIZES+="$top"
fi

if [[ $PARITY == 1 ]]; then EAGER=1; ENGRAM_DISK=1; ENGRAM_PRESTAGE=1; fi
[[ $ENGRAM_PRESTAGE == 1 && $ENGRAM_DISK != 1 ]] && { echo "ENGRAM_PRESTAGE=1 は ENGRAM_DISK=1 が必要"; exit 1; }
if [[ $ENGRAM_DISK != 1 ]]; then
  echo "ENGRAM_DISK=0 は Engram 表 (~190GiB) を pinned host RAM に置く経路。RAM 121GiB の当機では起動できない。"; exit 1
fi
if [[ $ENGRAM_PRESTAGE != 1 && $EAGER != 1 ]]; then
  echo "prestage 無しの disk 経路は forward 内で preadv するので CUDA graph 不可。EAGER=1 を付けること。"; exit 1
fi
[[ -f $PACK/config.json ]] || { echo "PACK=$PACK に config.json が無い"; exit 1; }

docker_() { sudo docker "$@"; }
docker_ image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE が無い (r4_vllm_ab.md の手順で作る)"; exit 1; }
mkdir -p "$CACHE_DIR"

extra=()
if [[ $EAGER == 1 ]]; then
  extra+=(--enforce-eager)
else
  # prefill は eager のまま (FULL_DECODE_ONLY)。#56771 は FULL_AND_PIECEWISE で prefill を graph に入れたときの不具合。
  extra+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}' --cudagraph-capture-sizes $CAPTURE_SIZES)
fi
[[ -n $NO_CUSTOM_AR ]] && extra+=(--disable-custom-all-reduce)
[[ -n $INDEXER_KV ]] && extra+=(--attention-config "{\"indexer_kv_dtype\":\"$INDEXER_KV\"}")
[[ -n $LINEAR_BACKEND ]] && extra+=(--linear-backend "$LINEAR_BACKEND")
[[ $THINKING == off ]] && extra+=(--default-chat-template-kwargs '{"thinking":false,"reasoning_effort":"low"}')
[[ $SPEC == dspark ]] && extra+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$NSPEC,\"draft_sample_method\":\"probabilistic\"}")
[[ -n $API_KEY ]] && extra+=(--api-key "$API_KEY")

denv=(
  -e VLLM_PLUGINS=vllm_exl3
  -e DSV41_ENGRAM_DISK=$ENGRAM_DISK -e DSV41_ENGRAM_DISK_THREADS=$ENGRAM_THREADS
  -e DSV41_ENGRAM_PRESTAGE=$ENGRAM_PRESTAGE -e DSV41_ENGRAM_PRESTAGE_CHECK=$PARITY
  -e XMOE_EXT=$XMOE_EXT -e XMOE_DIR=/opt/xmoe/build
  -e TORCH_CUDA_ARCH_LIST=12.0a -e FLASHINFER_CUDA_ARCH_LIST=$FI_ARCH -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
)
# 追加の通し: VLLM_EXL3_* / NCCL_PROTO / NCCL_ALGO / DENSE_HYBRID_M / XMOE_* (diffbot と同じ)
while IFS= read -r kv; do [[ -n $kv ]] && denv+=(-e "$kv"); done < <(env | grep -E "^(VLLM_EXL3_[A-Z0-9_]+|NCCL_PROTO|NCCL_ALGO|DENSE_HYBRID_M|XMOE_[A-Z_]+)=" | grep -vE "^XMOE_(EXT|DIR)=")

PACK_DIR=$(dirname "$PACK")
run=(docker run --rm --name "$NAME" --gpus all --network host --ipc host --shm-size "$SHM"
  --memory "$MEM_LIMIT" --memory-swap "$MEM_LIMIT"
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  -v "$PACK_DIR":"$PACK_DIR":ro -v "$CACHE_DIR":/root/.cache
  "${denv[@]}"
  --entrypoint vllm "$IMAGE")
args=(serve "$PACK"
  --tensor-parallel-size 2 --host 0.0.0.0 --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN" --kv-cache-dtype fp8 --kv-cache-memory "$KV_MEM"
  --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED"
  --kernel-config '{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}'
  --block-size 64 --quantization exl3 --language-model-only
  --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
  --reasoning-parser deepseek_v41
  --served-model-name "$SERVED" --trust-remote-code "${extra[@]}" ${EXTRA_ARGS:-})

printf '%q ' sudo "${run[@]}" "${args[@]}"; echo

if [[ $DRY_RUN == 1 ]]; then
  # GPU 無しの container で: (1) 引数を vLLM の parser に通す (2) パッチ済みモジュールと xmoe の import (3) Engram disk reader で 1017 の行を読む
  docker_ run --rm --network none -e LD_LIBRARY_PATH=/usr/local/cuda/compat -v "$PACK_DIR":"$PACK_DIR":ro -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_PRESTAGE=1 \
    -e XMOE_EXT=$XMOE_EXT -e XMOE_DIR=/opt/xmoe/build -e VLLM_PLUGINS=vllm_exl3 \
    ${DRYRUN_PY:+-v $DRYRUN_PY:/opt/dsv41-ab/dryrun.py:ro} --entrypoint python3 "$IMAGE" /opt/dsv41-ab/dryrun.py "$PACK" -- "${args[@]:1}"
  exit $?
fi

# ---- preflight (本番のみ) ----
if [[ $FORCE != 1 ]]; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  (( used > 4000 )) && { echo "GPU が使用中 (最大 ${used} MiB)。ds41rt / 評価ジョブを止めてから。FORCE=1 で無視"; exit 1; }
  ss -ltn | grep -q ":$PORT " && { echo "口 $PORT が使用中"; exit 1; }
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  (( avail < 60 )) && { echo "MemAvailable ${avail}GiB < 60GiB。他の常駐を止めてから。FORCE=1 で無視"; exit 1; }
fi
cleanup() { docker_ rm -f "$NAME" >/dev/null 2>&1; }
trap 'cleanup; exit 130' INT TERM
sudo "${run[@]}" "${args[@]}"
rc=$?; cleanup; exit $rc
