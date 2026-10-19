#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
# dsv41-flash-exl3-sm120 (diffbot recipe rev 28b7ab71 の docker/build.sh で作る) から、当機用の派生イメージ
# dsv41-flash-exl3-sm120-prestage を作る。BuildKit は使わず docker run + docker commit (diffbot の NOBUILDKIT=1 と同じ方式)。
#
# 焼き込むもの (diffbot serve.sh / nktlabs serve-prestage.sh が bind mount していたものを image 側に固定する):
#   - nktlabs prestage の 4 ファイル -> vllm/models/deepseek_v4_1/{common/engram.py,common/engram_disk.py,
#                                         nvidia/model.py,nvidia/model_state.py}
#   - diffbot patches/vllm/sitecustomize.py -> /usr/lib/python3.12/sitecustomize.py (XMOE hook, MXFP4 indexer の sm_120 解禁 等)
#   - diffbot patches/flashinfer/sparse_mla_sm120_prefill.cu (32-token extra page) と、prebuilt AOT の
#     jit_cache/sparse_mla_sm120 を空にする (= serve.sh の empty-aot mount と同じ。初回起動で JIT build される)
#   - kernels/build/xmoe (kernels/build.sh の成果物) -> /opt/xmoe/build/xmoe
#   - /opt/dsv41-ab/dryrun.py (GPU 無しの起動前検査)
set -euo pipefail
RECIPE=${RECIPE:-$DSV41/ext/diffbot_exl3_2bpw_recipe/recipe}
PRESTAGE=${PRESTAGE:-$DSV41/ext/nktlabs_prestage/patches/prestage}
OVERLAY=${OVERLAY:-$DSV41/build/vllm_exl3_overlay}
BASE=${BASE:-dsv41-flash-exl3-sm120}
TAG=${TAG:-dsv41-flash-exl3-sm120-prestage}
[[ -d $RECIPE/kernels/build/xmoe ]] || { echo "xmoe 未 build: IMAGE=$BASE $RECIPE/kernels/build.sh"; exit 1; }
C=dsv41-prestage-build; sudo docker rm -f $C >/dev/null 2>&1 || true
sudo docker run --name $C --network none --entrypoint bash \
  -v "$PRESTAGE":/mnt/prestage:ro -v "$RECIPE":/mnt/recipe:ro -v "$OVERLAY":/mnt/overlay:ro "$BASE" -lc '
set -euxo pipefail
pkg() { python3 -c "import importlib.util as u,sys; s=u.find_spec(sys.argv[1]); print(\"PKG=\"+(s.submodule_search_locations[0] if s else \"\"))" "$1" 2>/dev/null | sed -n "s/^PKG=//p" | tail -n1; }
V=$(pkg vllm)
M=$V/models/deepseek_v4_1
FI=$(pkg flashinfer)
FIJ=$(pkg flashinfer_jit_cache)
[ -n "$V" ] && [ -n "$FI" ] && [ -n "$FIJ" ]
mkdir -p /opt/dsv41-ab/orig
cp $M/common/engram.py $M/common/engram_disk.py $M/nvidia/model.py $M/nvidia/model_state.py /opt/dsv41-ab/orig/
cp /mnt/prestage/engram.py /mnt/prestage/engram_disk.py $M/common/
cp /mnt/prestage/model.py /mnt/prestage/model_state.py $M/nvidia/
cp /mnt/recipe/patches/vllm/sitecustomize.py /usr/lib/python3.12/sitecustomize.py
cp $FI/data/csrc/sparse_mla_sm120_prefill.cu /opt/dsv41-ab/orig/sparse_mla_sm120_prefill.cu.orig
cp /mnt/recipe/patches/flashinfer/sparse_mla_sm120_prefill.cu $FI/data/csrc/sparse_mla_sm120_prefill.cu
if [ -d $FIJ/jit_cache/sparse_mla_sm120 ]; then rm -rf $FIJ/jit_cache/sparse_mla_sm120; fi
mkdir -p $FIJ/jit_cache/sparse_mla_sm120
mkdir -p /opt/xmoe/build && cp -r /mnt/recipe/kernels/build/xmoe /opt/xmoe/build/xmoe
cp /mnt/overlay/dryrun.py /opt/dsv41-ab/dryrun.py
find $M -name __pycache__ -prune -exec rm -rf {} +
python3 -m py_compile $M/common/engram.py $M/common/engram_disk.py $M/nvidia/model.py $M/nvidia/model_state.py
echo "recipe=diffbot@28b7ab71 prestage=nktlabs@main built=$(date -Is)" > /opt/dsv41-ab/VERSION'
sudo docker commit \
  --change 'ENV VLLM_PLUGINS=vllm_exl3' --change 'ENV DSV41_ENGRAM_DISK=1' --change 'ENV DSV41_ENGRAM_PRESTAGE=1' \
  --change 'ENV XMOE_EXT=xmoe' --change 'ENV XMOE_DIR=/opt/xmoe/build' \
  --change 'ENTRYPOINT ["vllm","serve"]' --change 'CMD []' $C "$TAG"
sudo docker rm $C >/dev/null
sudo docker image ls "$TAG"
