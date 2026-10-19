#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
D=$DSV41; M=$MODELS
ORIG=$M/DeepSeek-V4.1-Flash-1017; V1=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1
Q=$M/qad; LOG=$D/work/qad/main.log
source <(sed -n '/^qrun()/,/^step()/p' $D/build/stageV.sh)
EX="--exclude /t/agent24.jsonl --exclude /t/ja16.jsonl --exclude /c/jakl32.jsonl --exclude /vl/vl_kl.jsonl"
step "capture start"
for g in 0 1; do qrun cap$g $g "python3 capture_moe.py run --input /c/calib256k.jsonl --input /vl/vl_calib.jsonl --img-boost 2 $EX --out $Q/cap/main_p$g --part $g/2 --preset 192k --exl3-pack $V1 > $Q/cap/main_p$g.log 2>&1" & done; wait
step "capture done"; qcpu "python3 capture_moe.py verify $Q/cap/main_p0 && python3 capture_moe.py verify $Q/cap/main_p1" >> $LOG 2>&1; step "verify rc=$?"
qrun l2a 0 "python3 qad_l2.py train --capture $Q/cap/main_p0 --capture $Q/cap/main_p1 --layers 0-19 --img-weight 1.5 --img-guard-tol 0 --orig-ckpt /model --pack $V1 --out $Q/l2/main_e > $Q/l2/main_e_0.log 2>&1" &
qrun l2b 1 "python3 qad_l2.py train --capture $Q/cap/main_p0 --capture $Q/cap/main_p1 --layers 20-39 --img-weight 1.5 --img-guard-tol 0 --orig-ckpt /model --pack $V1 --out $Q/l2/main_e > $Q/l2/main_e_1.log 2>&1" &
wait; step "train done $(ls $Q/l2/main_e/qad_l2_L*.json 2>/dev/null | wc -l) layers"
qcpu "python3 qad_l2.py apply --pack $V1 --diff $Q/l2/main_e --out $Q/packs/v1-l2e --mode overlay" >> $LOG 2>&1; step "apply rc=$?"
P=$Q/packs/v1-l2e; SR="python3 stream_ref.py --max-seq-len 8192 --expert-budget-gb 20"; SV="python3 stream_ref_vl.py --max-seq-len 8192 --expert-budget-gb 20"
(
 for t in jakl32 agent24; do src=/t/$t.jsonl; [ $t = jakl32 ] && src=/c/jakl32.jsonl
   qrun e0$t 0 "$SR --teacher-in $src --teacher-ref /t/${t}_teacher.pt --teacher-out /t/${t}_l2e.pt --exl3-pack $P > /t/kl_l2e_$t.log 2>&1"; step "eval $t $(grep '^\[student\] ALL' $D/work/teacher/kl_l2e_$t.log)"; done
) &
(
 qrun e1ja16 1 "$SR --teacher-in /t/ja16.jsonl --teacher-ref /t/ja16_teacher.pt --teacher-out /t/ja16_l2e.pt --exl3-pack $P > /t/kl_l2e_ja16.log 2>&1"; step "eval ja16 $(grep '^\[student\] ALL' $D/work/teacher/kl_l2e_ja16.log)"
 for s in ja en; do qrun e1vl$s 1 "$SV --vl-in /vl/vl_kl_$s.jsonl --teacher-ref /t/vlkl_${s}_teacher.pt --teacher-out /t/vlkl_${s}_l2e.pt --exl3-pack $P > /t/kl_l2e_vlkl_$s.log 2>&1"; step "eval vl$s $(grep '^\[student\] ALL' $D/work/teacher/kl_l2e_vlkl_$s.log)"; done
) &
wait
for base in v1 l2p64student; do
  qcpu "python3 /tools/kl_compare.py --set agent24 /t/agent24_$base.pt /t/agent24_l2e.pt --set ja16 /t/ja16_$base.pt /t/ja16_l2e.pt --set jakl32 /t/jakl32_$base.pt /t/jakl32_l2e.pt --boot 2000" > $D/work/qad/main_compare_$base.txt 2>&1; done
qcpu "python3 /tools/kl_compare.py --set vlja /t/vlkl_ja_v1.pt /t/vlkl_ja_l2e.pt --set vlen /t/vlkl_en_v1.pt /t/vlkl_en_l2e.pt --ja-sets vlja --boot 2000" > $D/work/qad/main_compare_vl.txt 2>&1
for s in ja16 agent24 jakl32; do qcpu "python3 /tools/zh_teacher_mass.py --teacher /t/${s}_teacher.pt --student v1=/t/${s}_v1.pt --student l2e=/t/${s}_l2e.pt --examples 0" 2>&1 | grep -E "answer_text  m1" | sed "s/^/$s /" >> $D/work/qad/main_zh.txt; done
step "main done"
