#!/usr/bin/env bash
# ds41gw (ds41rt 前段の API 変換プロキシ) のビルドと起動。
#   ./run_proxy.sh            ビルドして (再) 起動する
#   ./run_proxy.sh build      イメージだけ作る
#   ./run_proxy.sh start      既存イメージで (再) 起動する
#   ./run_proxy.sh stop       止めて消す
#   ./run_proxy.sh test       cargo test (rust コンテナで実行。ds41rt は使わない)
#   ./run_proxy.sh logs       docker logs -f
# 環境変数:
#   DS41GW_PORT      待ち受けの口 (既定 30001)。空白区切りで複数可 (例 "30001 30000")。config の listen を上書きする
#   DS41GW_UPSTREAM  上流 (既定 http://localhost:8000)
#   DS41GW_LISTEN_IPS 待ち受ける IP (既定 "127.0.0.1"。LAN / Tailscale に出すならその IP を足す)
#   DS41GW_CONFIG    設定ファイル (既定 ./config/ds41gw.toml)
#   DS41GW_TOKENIZER tokenizer.json (必須。ds41rt が読んでいる snapshot のもの)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image="ds41gw:0.1.0"
name="ds41gw"
ports="${DS41GW_PORT:-30001}"
port="${ports%% *}"   # ヘルスチェックに使う口 (先頭)
upstream="${DS41GW_UPSTREAM:-http://localhost:8000}"
ips="${DS41GW_LISTEN_IPS:-127.0.0.1}"
config="${DS41GW_CONFIG:-$here/config/ds41gw.toml}"
tokenizer="${DS41GW_TOKENIZER:-}"

build() {
  sudo docker build -t "$image" -f "$here/docker/Dockerfile" "$here"
}

start() {
  if [[ ! -f "$tokenizer" ]]; then
    echo "DS41GW_TOKENIZER に tokenizer.json のパスを指定する (例: \$HF_HOME/hub/models--…/snapshots/<rev>/tokenizer.json)" >&2
    return 2
  fi
  mkdir -p "$here/state" "$here/logs/req"
  chmod 700 "$here/state" "$here/logs" "$here/logs/req"
  # 口と上流は環境変数で上書きした写しを使う (元の config は変えない)
  local rendered="$here/state/ds41gw.rendered.toml"
  local listen=""
  for p in $ports; do for ip in $ips; do listen+="\"$ip:$p\", "; done; done
  listen="[${listen%, }]"
  python3 - "$config" "$rendered" "$listen" "$upstream" <<'PY'
import re, sys
src, dst, listen, upstream = sys.argv[1:5]
text = open(src).read()
text = re.sub(r'(?m)^listen\s*=.*$', 'listen = ' + listen, text, count=1)
text = re.sub(r'(?m)^url\s*=.*$', 'url = "' + upstream + '"', text, count=1)
open(dst, "w").write(text)
PY
  chmod 600 "$rendered"
  if sudo docker inspect "$name" >/dev/null 2>&1; then
    sudo docker rm -f "$name" >/dev/null
  fi
  sudo docker run -d --name "$name" --network host --restart unless-stopped --init \
    --user "$(id -u):$(id -g)" --stop-timeout 30 \
    --health-cmd "/usr/local/bin/ds41gw --probe http://127.0.0.1:$port/health" --health-interval 30s \
    -v "$rendered:/etc/ds41gw/ds41gw.toml:ro" \
    -v "$here/state:/var/lib/ds41gw" \
    -v "$here/logs:/var/log/ds41gw" \
    -v "$tokenizer:/models/tokenizer.json:ro" \
    "$image" --config /etc/ds41gw/ds41gw.toml
  for _ in $(seq 1 50); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "ds41gw is up on ${listen} (upstream $upstream)"
      curl -sS "http://127.0.0.1:$port/health/ready"; echo
      return 0
    fi
    sleep 0.2
  done
  echo "ds41gw did not become healthy; see: sudo docker logs $name" >&2
  return 1
}

case "${1:-all}" in
  all) build; start ;;
  build) build ;;
  start) start ;;
  stop) sudo docker rm -f "$name" ;;
  logs) sudo docker logs -f "$name" ;;
  test) "$here/cargo.sh" test --release ;;
  *) echo "usage: $0 [all|build|start|stop|logs|test]" >&2; exit 2 ;;
esac
