#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
L=$DSV41/build/quant-logs/chain.log
until [ "$(cat $MODELS/dsv41-v1-work/logs/g*.log 2>/dev/null | grep -c 'rc=')" -ge 40 ]; do sleep 30; done
while sudo docker ps --format '{{.Names}}' | grep -q dsv41-requant; do sleep 10; done
echo "v1 quant done $(date) fails=$(cat $MODELS/dsv41-v1-work/logs/g*.log | grep -vc 'rc=0')" >> $L
$DSV41/build/assemble_v1.sh >> $L 2>&1 || { echo "assemble failed" >> $L; exit 1; }
P=$MODELS/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1
for spec in "0 agent24" "1 ja16"; do set -- $spec; g=$1; n=$2
  sudo docker rm -f dsv41-kl-$n >/dev/null 2>&1
  sudo docker run -d --name dsv41-kl-$n --gpus "\"device=$g\"" --network none --ipc host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -v $DSV41/ref:/ref:ro -v $MODELS/DeepSeek-V4.1-Flash-1017:/model:ro -v $P:/pack:ro -v $DSV41/harness:/h -v $DSV41/work/teacher:/t --entrypoint bash dsv41-quant:5be8 -c "cd /h && python3 stream_ref.py --teacher-in /t/$n.jsonl --teacher-out /t/${n}_v1.pt --teacher-ref /t/${n}_teacher.pt --exl3-pack /pack --max-seq-len 8192 --expert-budget-gb 24 > /t/kl_v1_$n.log 2>&1"
done
echo "v1 KL started $(date)" >> $L
while sudo docker ps --format '{{.Names}}' | grep -q dsv41-kl-; do sleep 30; done
echo "v1 KL done $(date)" >> $L
grep -h "^\[student\] ALL" $DSV41/work/teacher/kl_v1_*.log >> $L
