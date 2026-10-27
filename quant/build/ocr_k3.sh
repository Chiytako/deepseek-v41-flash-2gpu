#!/usr/bin/env bash
# 日本語 OCR で使われる routed expert を K3 (3bpw) に上げた abl パックを作る。
#   window   ds41rt を止める → capture → select → quant → abl で再起動 (GPU を使う区間はここだけ)
#   capture  元の重みで OCR 較正 35 問を teacher forcing し、専門家の選択回数を数える (GPU 0/1 並列)
#   select   層ごとに順位付け → work/vl/ocr_k3_select.json
#   quant    各層の上位 PER_LAYER_Q 個 (既定 40) を原本から K3 で符号化 → $K3 (GPU 0: 層 0-19、GPU 1: 層 20-39)
#   assemble abl に上位 PER_LAYER 個 (既定 32) を組み込んだ snapshot を作る (CPU、sudo)
# 環境変数: DS41RT_CONFIG (window の最後に再起動する設定、既定 local-abl-t1.config)
set -uo pipefail
D=${DSV41:-$PWD}; M=${MODELS:-$D/models}  # publish: workspace root and model dir
ORIG=$M/DeepSeek-V4.1-Flash-1017; T=$D/work/teacher; LOG=$T/ocr_k3.log
HUB=$D/hf/hub; BASE_REPO=models--local--DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1-l2e-abl
BASE_REV=1017e3c2000000000000000000000000000000ab; BASE=$HUB/$BASE_REPO/snapshots/$BASE_REV
NEW_REPO=models--local--DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1-l2e-abl-ocrk3; NEW_REV=1017e3c20000000000000000000000000000ab03
K3=$M/qad/k3-ocr; SEL=$D/work/vl/ocr_k3_select.json
PER_LAYER_Q=${PER_LAYER_Q:-40}; PER_LAYER=${PER_LAYER:-32}
DS41RT_CONFIG=${DS41RT_CONFIG:-local-abl-t1.config}
dock() { local n=$1 g=$2; shift 2
  sudo docker run --rm --name dsv41-ocrk3-$n --gpus "\"device=$g\"" --network none --ipc host --shm-size 16g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -v $M/qad:/qad -v $HUB:/hub:ro \
    -v $D/ref:/ref:ro -v $ORIG:/model:ro -v $D/harness:/h:ro -v $D/tools:/tools:ro -v $D/build/qrecipe:/recipe:ro \
    -v $D/work/vl:/vl:ro -v $T:/t --entrypoint bash dsv41-quant:151 -c "$*" ; }
step() { echo "== $(date '+%m-%d %H:%M:%S') $1" | tee -a $LOG; }
gpu_free() {
  local used i
  for i in $(seq 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    ((used < 4000)) && return 0
    sleep 2
  done
  echo "GPU が使用中 (${used} MiB)" >&2; return 1
}

capture() {
  gpu_free || return 1
  awk 'NR % 2 == 1' $D/work/vl/vl_ocr_ja_calib.jsonl > $T/ocrcal_p0.jsonl
  awk 'NR % 2 == 0' $D/work/vl/vl_ocr_ja_calib.jsonl > $T/ocrcal_p1.jsonl
  step "capture start"
  for g in 0 1; do
    dock cap$g $g "cd /h && python3 stream_ref_vl.py --vl-in /t/ocrcal_p$g.jsonl --image-root /vl --teacher-out /t/ocrcal_p$g.pt \
      --max-seq-len 8192 --expert-budget-gb 60" > $T/ocrcal_p$g.log 2>&1 &
  done
  wait
  [[ -s $T/ocrcal_p0.pt && -s $T/ocrcal_p1.pt ]] || { step "capture failed"; return 1; }
  step "capture done"
}

select_() {
  dock sel 0 "python3 /tools/vl/select_k3.py /t/ocrcal_p0.pt /t/ocrcal_p1.pt --out /t/ocr_k3_select.json" | tee -a $LOG
  cp $T/ocr_k3_select.json $SEL
}

quant() {
  gpu_free || return 1
  mkdir -p $K3
  step "quant start (K3, ${PER_LAYER_Q}/layer)"
  local g r
  for g in 0 1; do
    r=$([[ $g == 0 ]] && echo 0-19 || echo 20-39)
    dock q$g $g "python3 /tools/requant_k3.py --src /model --base /hub/$BASE_REPO/snapshots/$BASE_REV --select /t/ocr_k3_select.json \
      --per-layer $PER_LAYER_Q --layers $r --out /qad/k3-ocr" > $T/ocr_k3_quant_g$g.log 2>&1 &
  done
  wait
  local n; n=$(ls $K3/layer_*.json 2>/dev/null | wc -l)
  step "quant done: $n / 40 layers"
  ((n == 40))
}

assemble() {
  step "assemble (${PER_LAYER}/layer)"
  sudo mkdir -p $HUB/$NEW_REPO/snapshots $HUB/$NEW_REPO/refs
  sudo python3 $D/tools/assemble_k3.py --base $BASE --k3 $K3 --per-layer $PER_LAYER --out $HUB/$NEW_REPO/snapshots/$NEW_REV | tee -a $LOG
  echo -n $NEW_REV | sudo tee $HUB/$NEW_REPO/refs/main > /dev/null
}

case ${1:-} in
  window)
    step "stop ds41rt"
    (cd $D/ext/ds41rt && sudo ./stop.sh --config "$DS41RT_CONFIG") >> $LOG 2>&1
    capture && select_ && quant
    step "restart ds41rt ($DS41RT_CONFIG)"  # 失敗しても戻す
    (cd $D/ext/ds41rt && sudo HF_HOME=$D/hf ./run-local.sh --config "$DS41RT_CONFIG" --restart) 2>&1 | tee -a $LOG ;;
  capture) capture ;;
  select) select_ ;;
  quant) quant ;;
  assemble) assemble ;;
  *) sed -n 2,8p "$0"; exit 2 ;;
esac
