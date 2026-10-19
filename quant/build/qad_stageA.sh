#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
# QAD_PLAN §1.1〜1.3 (段 A)。GPU0: jakl32 教師 → v1 採点。GPU1: check-decomp → capture smoke → verify → check-teacher → 試し学習。
D=$DSV41; M=$MODELS
ORIG=$M/DeepSeek-V4.1-Flash-1017; V1=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1
Q=$M/qad; mkdir -p $Q/cap $Q/l2 $Q/packs $D/work/qad; LOG=$D/work/qad/stageA.log
qrun() { local n=$1 g=$2; shift 2
  sudo docker run --rm --name dsv41-qad-$n --gpus "\"device=$g\"" --network none --ipc host --shm-size 16g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v $D/ref:/ref:ro -v $ORIG:/model:ro -v $D/harness:/h:ro -v $D/tools:/tools:ro \
    -v $D/work/teacher:/t -v $D/work/calib:/c:ro -v $M:$M \
    --entrypoint bash dsv41-quant:151 -c "cd /h && $*" ; }
qcpu() { sudo docker run --rm --network none -e CUDA_VISIBLE_DEVICES= \
    -v $D/ref:/ref:ro -v $D/harness:/h:ro -v $D/tools:/tools:ro -v $D/work/teacher:/t -v $D/work/calib:/c:ro \
    -v $ORIG:/model:ro -v $M:$M --entrypoint bash dsv41-quant:151 -c "cd /h && $*" ; }
step() { echo "== $(date +%H:%M:%S) $1" >> $LOG; }
(
 step "g0 jakl32 teacher start"
 qrun jaklT 0 "python3 stream_ref.py --teacher-in /c/jakl32.jsonl --teacher-out /t/jakl32_teacher.pt --max-seq-len 8192 --expert-budget-gb 20 > /t/teacher_jakl32.log 2>&1"
 step "g0 jakl32 teacher rc=$?"
 qrun v1jakl 0 "python3 stream_ref.py --teacher-in /c/jakl32.jsonl --teacher-ref /t/jakl32_teacher.pt --teacher-out /t/jakl32_v1.pt --exl3-pack $V1 --max-seq-len 8192 --expert-budget-gb 20 > /t/kl_v1_jakl32.log 2>&1"
 step "g0 v1 jakl32 rc=$?"; grep "^\[student\] ALL" $D/work/teacher/kl_v1_jakl32.log >> $LOG
) &
(
 step "g1 check-decomp"
 qrun dec 1 "python3 qad_l2.py check-decomp --pack $V1 --layer 0 --experts 0,7,200 && python3 qad_l2.py check-decomp --pack $V1 --layer 39 --experts 0,7,200" >> $LOG 2>&1
 step "g1 check-decomp rc=$?"
 qrun capS 1 "python3 capture_moe.py run --input /c/calib64k.jsonl --exclude /t/agent24.jsonl --exclude /t/ja16.jsonl --exclude /c/jakl32.jsonl --out $Q/cap/smoke --preset 64k --rows-per-layer 8192 --limit-seqs 3 --exl3-pack $V1" > $D/work/qad/cap_smoke.log 2>&1
 step "g1 capture smoke rc=$?"
 qcpu "python3 capture_moe.py verify $Q/cap/smoke && python3 capture_moe.py info $Q/cap/smoke" >> $LOG 2>&1
 step "g1 verify rc=$?"
 qrun tch 1 "python3 qad_l2.py check-teacher --orig-ckpt /model --capture $Q/cap/smoke --layer 0 --experts 0,7 && python3 qad_l2.py check-teacher --orig-ckpt /model --capture $Q/cap/smoke --layer 39 --experts 0,7" >> $LOG 2>&1
 step "g1 check-teacher rc=$?"
 qrun l2S 1 "python3 qad_l2.py train --capture $Q/cap/smoke --layers 0,39 --orig-ckpt /model --pack $V1 --out $Q/l2/smoke --min-tokens 64 --verbose > $Q/l2/smoke.log 2>&1"
 step "g1 train smoke rc=$?"
 for f in $Q/l2/smoke/qad_l2_L*.json; do jq -c '{layer, accepted, rejected, cold, seconds, layer_val}' $f >> $LOG 2>&1; done
) &
wait; step "stageA done"
