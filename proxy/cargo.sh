#!/usr/bin/env bash
# ホストに cargo が無いので、rust:1.90-bookworm コンテナで cargo を実行する補助。
# 例: ./cargo.sh test --release ; ./cargo.sh build --release
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec sudo docker run --rm --network host --user "$(id -u):$(id -g)" \
  -e CARGO_HOME=/cargo-home -e CARGO_TARGET_DIR=/src/target \
  -v "$here/.cargo-home:/cargo-home" -v "$here/ds41gw:/src" -v "$here/vendor:/vendor:ro" -w /src \
  ${CARGO_DOCKER_ARGS:-} rust:1.90-bookworm cargo "$@"
