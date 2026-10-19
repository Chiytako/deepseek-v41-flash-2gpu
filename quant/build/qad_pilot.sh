#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
# QAD_PLAN §1.5 pilot + §4.1 評価。学生 stream (GPU0) と教師 stream (GPU1) を並べる。
D=$DSV41; M=$MODELS
ORIG=$M/DeepSeek-V4.1-Flash-1017; V1=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1
Q=$M/qad; LOG=$D/work/qad/pilot.log
source <(sed -n '/^qrun()/,/^step()/p' $D/build/qad_stageA.sh)
EX="--exclude /t/agent24.jsonl --exclude /t/ja16.jsonl --exclude /c/jakl32.jsonl"
lane() { local s=$1 g=$2 extra=$3
  step "$s capture start"
  qrun cap$s $g "python3 capture_moe.py run --input /c/calib64k.jsonl $EX --out $Q/cap/p64_$s --preset 64k $extra > $Q/cap/p64_$s.log 2>&1"
  step "$s capture rc=$?"
  qrun l2$s $g "python3 qad_l2.py train --capture $Q/cap/p64_$s --layers 0-39 --orig-ckpt /model --pack $V1 --out $Q/l2/p64_$s > $Q/l2/p64_$s.log 2>&1"
  step "$s train rc=$?"
  qcpu "python3 qad_l2.py apply --pack $V1 --diff $Q/l2/p64_$s --out $Q/packs/v1-l2p64$s --mode overlay" >> $LOG 2>&1
  step "$s apply rc=$?"
  for t in agent24 ja16 jakl32; do
    src=/t/$t.jsonl; [ $t = jakl32 ] && src=/c/jakl32.jsonl
    qrun ev$s$t $g "python3 stream_ref.py --teacher-in $src --teacher-ref /t/${t}_teacher.pt --teacher-out /t/${t}_l2p64$s.pt --exl3-pack $Q/packs/v1-l2p64$s --max-seq-len 8192 --expert-budget-gb 20 > /t/kl_l2p64${s}_$t.log 2>&1"
    step "$s eval $t rc=$? $(grep '^\[student\] ALL' $D/work/teacher/kl_l2p64${s}_$t.log)"
  done
}
lane student 0 "--exl3-pack $V1" &
lane teacher 1 "" &
wait
T=/t
for s in student teacher; do
  qcpu "python3 /tools/kl_compare.py --set agent24 $T/agent24_v1.pt $T/agent24_l2p64$s.pt --set ja16 $T/ja16_v1.pt $T/ja16_l2p64$s.pt --set jakl32 $T/jakl32_v1.pt $T/jakl32_l2p64$s.pt --boot 2000" > $D/work/qad/pilot_compare_$s.txt 2>&1
done
step "pilot done"
