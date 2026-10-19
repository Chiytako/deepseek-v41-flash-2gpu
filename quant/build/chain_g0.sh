#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
while [ "$(sudo docker inspect -f '{{.State.Running}}' dsv41-teacher 2>/dev/null)" = "true" ]; do sleep 20; done
echo "teacher done $(date)" >> $DSV41/build/quant-logs/chain.log
$DSV41/build/quantize_1017.sh gpu 0 0 19
echo "g0 quant done $(date) rc=$?" >> $DSV41/build/quant-logs/chain.log
