#!/usr/bin/env bash
: "${DSV41:=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}" "${MODELS:=$DSV41/models}"  # publish: host paths -> env
D=$DSV41; R=$D/results/l2e; mkdir -p $R; LOG=$R/eval.log; PY=$D/tools/.venv/bin/python
step() { echo "== $(date +%H:%M:%S) $1" >> $LOG; }
cd $D/ext/ds41rt
sudo HF_HOME=$D/hf ./stop.sh --config local-v1.config >/dev/null 2>&1
sudo HF_HOME=$D/hf ./run-local.sh --config local-l2e.config > $R/launch.log 2>&1
timeout 300 bash -c 'until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done' || { step "launch failed"; exit 1; }
step "ds41rt l2e up"
python3 - >> $LOG 2>&1 <<'PY'
import json,urllib.request
def chat(msgs,mt=300,**kw):
    b={"model":"deepseek-ai/DeepSeek-V4.1-Flash","messages":msgs,"max_tokens":mt,"temperature":0,**kw}
    return json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/chat/completions",json.dumps(b).encode(),{"Content-Type":"application/json"}),timeout=600))
r=chat([{"role":"user","content":"日本の首都はどこですか？一言で答えてください。"}],60); print("Q1:",r["choices"][0]["message"]["content"])
tools=[{"type":"function","function":{"name":"get_weather","description":"指定した都市の今日の天気を返す","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
r=chat([{"role":"user","content":"東京と大阪の今日の天気を調べて比較してください。"}],600,tools=tools); m=r["choices"][0]
print("TOOL:",m["finish_reason"],json.dumps(m["message"].get("tool_calls"),ensure_ascii=False))
PY
step "smoke done"
$PY scripts/bench-ds41-release-decode.py --base-url http://localhost:8000 --tokenizer $MODELS/DeepSeek-V4.1-Flash-1017/tokenizer.json --label l2e-greedy --repeats 1 --nonce-seed 7 --output $R/release_greedy.json > $R/release.log 2>&1
python3 -c "import json;d=json.load(open('$R/release_greedy.json'));print('weighted',d['repeat_summaries'][0]['weighted_observed_decode_tokens_per_second']);[print(c['case'],round(c['median_observed_decode_tokens_per_second'],1)) for c in d['case_summaries']]" >> $LOG 2>&1
step "release bench done"
python3 $D/build/bench_conc.py 1 4 8 16 > $R/bench_conc.txt 2>&1; cat $R/bench_conc.txt >> $LOG; step "bench_conc done"
cd $D; $PY tools/vl_eval.py --out work/vl/results/l2e.jsonl --compare work/vl/results/v1.jsonl > $R/vl_eval.txt 2>&1; tail -40 $R/vl_eval.txt >> $LOG; step "vl_eval done"
$PY tools/zh_gen_eval.py gen --url http://localhost:8000 --model deepseek-ai/DeepSeek-V4.1-Flash --label l2e --prompts work/zh/promptset/prompts.jsonl --out work/zh/gen/l2e_t1.jsonl --temp 1.0 --top-p 1.0 --seeds 1 --max-tokens 1024 --concurrency 16 > $R/zh_gen.log 2>&1
$PY tools/zh_gen_eval.py score work/zh/gen/v1_t1.jsonl work/zh/gen/l2e_t1.jsonl --prompts work/zh/promptset/prompts.jsonl > $R/zh_score.txt 2>&1; tail -25 $R/zh_score.txt >> $LOG
step "all done"
