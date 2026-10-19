#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
while sudo docker ps --format '{{.Names}}' | grep -qx dsv41-kl-agent24; do sleep 20; done
$DSV41/build/requant_v1.sh gpu 0 0 19
