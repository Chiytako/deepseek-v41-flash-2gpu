#!/usr/bin/env python3
# replay_openai.py (GLM 時代の要求ログの再生器、非公開) の写しに、出力上限 --max-tokens を足したもの (r7_proxy_spec.md §16.4 の 2)。
# 記録の max_tokens (32000 など) をそのまま送ると稼働中の ds41rt を長く占有するので、上限を付けて再生する。
# 使い方: python3 tests/replay_capped.py <reqlog または ds41gw の要求ログ> --url http://127.0.0.1:30001 --conc 4 --max-tokens 64 --limit 40
# ds41gw 自身の要求ログ (logs/req/YYYYMMDD.jsonl) も、`obj` が Chat 本文なのでそのまま読める (§16.4 の 4)。
import argparse, importlib.util, json, sys

# replay_openai.py (GLM 時代の再生器) は公開していない。各自で用意し、REPLAY_OPENAI で場所を渡す
import os
spec = importlib.util.spec_from_file_location("replay_openai", os.environ.get("REPLAY_OPENAI", "replay_openai.py"))
ro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ro)

orig_run_one = ro.run_one


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-bytes", type=int, default=0, help="これより大きい要求は飛ばす (0 で無制限)")
    a, rest = ap.parse_known_args()
    cap = a.max_tokens

    def load(path, limit):
        # replay_openai.load と同じ読み方。ただし obj が無い行 (検査で断った要求) は飛ばす
        reqs = []
        for line in open(path, errors="replace"):
            i = line.find("{")
            if i < 0 or '"request.received.openai"' not in line:
                continue
            try:
                d = json.loads(line[i:])
            except Exception:
                continue
            if not isinstance(d.get("obj"), dict):
                continue
            o = {k: v for k, v in d["obj"].items() if k in ro.KEEP and v is not None}
            if not o.get("messages"):
                continue
            if a.max_bytes and len(json.dumps(o)) > a.max_bytes:
                continue
            reqs.append(o)
            if limit and len(reqs) >= limit:
                break
        return reqs

    def run_one(url, body, timeout):
        body = dict(body)
        for k in ("max_tokens", "max_completion_tokens"):
            if k in body:
                body[k] = min(int(body[k]), cap)
        body.setdefault("max_tokens", cap)
        return orig_run_one(url, body, timeout)

    ro.load = load
    ro.run_one = run_one
    # ds41rt は稼働中なので、同時数は 4 本までに抑える
    if "--conc" in rest:
        i = rest.index("--conc")
        rest[i + 1] = str(min(int(rest[i + 1]), 4))
    else:
        rest += ["--conc", "4"]
    sys.argv = [sys.argv[0]] + rest
    ro.main()


if __name__ == "__main__":
    main()
