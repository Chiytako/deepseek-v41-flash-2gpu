#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
sleep 30
while sudo docker ps --format '{{.Names}}' | grep -qx dsv41-quant-g1; do sleep 20; done
echo "g1 quant done $(date)" >> $DSV41/build/quant-logs/chain.log
sudo docker rm -f dsv41-teacher-ja >/dev/null 2>&1
sudo docker run --name dsv41-teacher-ja --gpus '"device=1"' --network none --ipc host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $DSV41/ref:/ref:ro -v $MODELS/DeepSeek-V4.1-Flash-1017:/model:ro -v $DSV41/harness:/h -v $DSV41/work/teacher:/t \
  --entrypoint bash ghcr.io/local-inference-lab/vllm:release-10152026 -c 'cd /h && python3 stream_ref.py --teacher-in /t/ja16.jsonl --teacher-out /t/ja16_teacher.pt --max-seq-len 8192 --expert-budget-gb 20 > /t/teacher_ja.log 2>&1'
echo "ja teacher done $(date)" >> $DSV41/build/quant-logs/chain.log
