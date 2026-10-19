# エンジン A/B: ds41rt vs vLLM (+vllm-exl3 +xmoe +Engram disk prestage)  2026-10-18

同じ重み: v1 パック (EXL3 K2 MCG、Viterbi+refit)。2x RTX PRO 6000 Max-Q、RAM 128GB、Engram は NVMe。片方ずつ起動。
decode: build/bench_conc.py (日本語/英語の説明・コード 8 種、T=0.6 top_p 0.95、max_tokens 1024、思考込み)。2 回実行。
prefill: build/bench_prefill.py (日本語+英語の技術文書、max_tokens 1、毎回 salt で prefix cache を外す)。

| 指標 | ds41rt v15 (0-spark) | vLLM 0909 + vllm-exl3 + xmoe + prestage | 差 |
|---|---|---|---|
| c1 合計 tok/s | 101〜103 (1 本 103〜112) | 88〜105 (1 本 99〜113) | ほぼ同等 |
| c4 合計 | 203〜222 | 158〜182 | ds41rt +25% |
| c8 合計 | 290〜300 | 225〜241 | ds41rt +26% |
| c16 合計 | **403〜412** (1 本 31) | 279〜284 (1 本 22) | **ds41rt +45%** |
| prefill 8K | 4,348 tok/s | 1,597 | ds41rt 2.7 倍 |
| prefill 36K | 4,340〜4,848 | 1,798 | 2.5 倍 |
| prefill 67K | 4,661〜4,697 | 2,060 | 2.3 倍 |
| 起動時間 | 50 s | ~10 分 (初回 JIT 込み) | |
| API | /v1/chat/completions のみ (tools・画像・JSON Schema 可) | /v1/responses, /v1/completions, /v1/messages, logprobs, n, penalty | vLLM が広い |

結論: 性能はすべての指標で ds41rt が上 (c16 +45%、prefill 2.3〜2.7 倍)。vLLM の利点は API の広さ。
LIL vLLM / SGLang は 2 GPU では 2bit 専門家を読めない (r2_lil_lowbit.md, web_engines.md) ため比較対象外。
参考: GLM-5.3-Flash (LIL vLLM, 並列 16) は c16 384 / c1 224。

## ds41rt 付属ベンチ (scripts/bench-ds41-release-decode.py、greedy、9 カテゴリ、v1、当機 Max-Q 300W)
weighted 139.0 tok/s (公表 EXL3 2x6000 0-spark 400W: 146.2)、code 195.5 (公表 225.2)、code-reasoning 172.4、math 193.9、structured-json 160.4、
topic 107.8、multilingual 107.6、hello 95.5、fable 90.6。測定中の電力 平均 169W (上限 300W)、SM 1,961MHz → 電力律速ではない。
bench_conc.py の c1 ≈100 は T0.6・思考込み・散文多めの重い負荷のため。
