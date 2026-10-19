#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
D=$DSV41; M=$MODELS
ORIG=$M/DeepSeek-V4.1-Flash-1017; V1=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1; V0=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw
Q=$M/qad; P64=$Q/packs/v1-l2p64student; LOG=$D/work/qad/stageV.log
qrun() { local n=$1 g=$2; shift 2
  sudo docker run --rm --name dsv41-qad-$n --gpus "\"device=$g\"" --network none --ipc host --shm-size 16g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v $D/ref:/ref:ro -v $ORIG:/model:ro -v $D/harness:/h:ro -v $D/tools:/tools:ro -v $D/work/vl:/vl:ro \
    -v $D/work/teacher:/t -v $D/work/calib:/c:ro -v $M:$M --entrypoint bash dsv41-quant:151 -c "cd /h && $*" ; }
qcpu() { sudo docker run --rm --network none -e CUDA_VISIBLE_DEVICES= -v $D/ref:/ref:ro -v $D/harness:/h:ro -v $D/tools:/tools:ro \
    -v $D/work/vl:/vl:ro -v $D/work/teacher:/t -v $D/work/calib:/c:ro -v $ORIG:/model:ro -v $M:$M --entrypoint bash dsv41-quant:151 -c "cd /h && $*" ; }
step() { echo "== $(date +%H:%M:%S) $1" >> $LOG; }
SR="python3 stream_ref_vl.py --max-seq-len 8192 --expert-budget-gb 20"
(
 qrun vlTja 0 "$SR --vl-in /vl/vl_kl_ja.jsonl --teacher-out /t/vlkl_ja_teacher.pt > /t/teacher_vlkl_ja.log 2>&1"; step "g0 teacher ja rc=$?"
 for v in v1:$V1 l2p64:$P64 v0:$V0; do n=${v%%:*}; p=${v#*:}
   qrun vlSja$n 0 "$SR --vl-in /vl/vl_kl_ja.jsonl --teacher-ref /t/vlkl_ja_teacher.pt --teacher-out /t/vlkl_ja_$n.pt --exl3-pack $p > /t/kl_${n}_vlkl_ja.log 2>&1"
   step "g0 student ja $n rc=$? $(grep '^\[student\] ALL' $D/work/teacher/kl_${n}_vlkl_ja.log)"; done
) &
(
 qrun vlS 1 "$SR --vl-in /vl/vl_smoke.jsonl --teacher-out /t/vl_smoke_teacher.pt > /t/teacher_vl_smoke.log 2>&1 && $SR --vl-in /vl/vl_smoke.jsonl --teacher-ref /t/vl_smoke_teacher.pt --teacher-out /t/vl_smoke_self.pt > /t/kl_vl_smoke_self.log 2>&1"
 step "g1 smoke rc=$?"; grep -hE '^\[(route|teacher|student|load|mem)\]' $D/work/teacher/teacher_vl_smoke.log $D/work/teacher/kl_vl_smoke_self.log >> $LOG
 qrun vlC 1 "$SR --vl-in /vl/vl_smoke_ctrl.jsonl --teacher-out /t/vl_smoke_ctrl.pt > /t/teacher_vl_smoke_ctrl.log 2>&1"
 step "g1 ctrl rc=$?"; grep -hE '^\[(route|teacher)\]' $D/work/teacher/teacher_vl_smoke_ctrl.log >> $LOG
 qrun capV 1 "python3 capture_moe.py run --input /vl/vl_calib.jsonl --img-boost 2 --out $Q/cap/smoke_vl --preset 64k --rows-per-layer 8192 --limit-seqs 3 --exl3-pack $V1 > $Q/cap/smoke_vl.log 2>&1"
 step "g1 capture vl rc=$?"; qcpu "python3 capture_moe.py verify $Q/cap/smoke_vl && python3 capture_moe.py info $Q/cap/smoke_vl" >> $LOG 2>&1; step "g1 verify vl rc=$?"
 qrun vlTen 1 "$SR --vl-in /vl/vl_kl_en.jsonl --teacher-out /t/vlkl_en_teacher.pt > /t/teacher_vlkl_en.log 2>&1"; step "g1 teacher en rc=$?"
 for v in v1:$V1 l2p64:$P64 v0:$V0; do n=${v%%:*}; p=${v#*:}
   qrun vlSen$n 1 "$SR --vl-in /vl/vl_kl_en.jsonl --teacher-ref /t/vlkl_en_teacher.pt --teacher-out /t/vlkl_en_$n.pt --exl3-pack $p > /t/kl_${n}_vlkl_en.log 2>&1"
   step "g1 student en $n rc=$? $(grep '^\[student\] ALL' $D/work/teacher/kl_${n}_vlkl_en.log)"; done
) &
wait
for pair in "v0 v1" "v1 l2p64"; do set -- $pair
  qcpu "python3 /tools/kl_compare.py --set vlja /t/vlkl_ja_$1.pt /t/vlkl_ja_$2.pt --set vlen /t/vlkl_en_$1.pt /t/vlkl_en_$2.pt --jsonl vlja=/vl/vl_kl_ja.jsonl --jsonl vlen=/vl/vl_kl_en.jsonl --ja-sets vlja --boot 2000" > $D/work/qad/vl_compare_$1_$2.txt 2>&1
done
qcpu "python3 /tools/zh_teacher_mass.py --teacher /t/vlkl_ja_teacher.pt --student v1=/t/vlkl_ja_v1.pt --student l2p64=/t/vlkl_ja_l2p64.pt --examples 0" > $D/work/qad/vl_zh.txt 2>&1
step "stageV done"
