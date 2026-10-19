# deepseek-recipe-patched (vendored)

deepseek-recipe / -core / -encoding 0.1.0 (MIT, DeepSeek) に、reasoning effort の数値 (1〜100) 対応を足したもの。
`ext/ds41rt/third_party/deepseek-recipe-patched/` (ds41rt 改修版 v15-effort が使うもの) の写しで、
ds41gw とプロンプトの描画を揃えるために置く (`render-effort` は不要なので写していない)。

- core: `ReasoningEffort::Budget(u8)`、`score()` / `level()`、`RequestedReasoningEffort` (整数・数値文字列・段階名の解析)
- recipe: Chat / Responses / Anthropic の effort 型が整数と数値文字列も受ける
- recipe: 要求の名前 `medium` / `xhigh` は `Budget(62)` / `Budget(88)` に写す (`ReasoningEffort::MEDIUM_SCORE` / `XHIGH_SCORE`)。
  **公式と違う独自の対応**である。公式の encoding が名前で定めるのは low 50・high 75・max 100 の 3 つだけで、
  公式の deepseek-recipe 0.1.0 は medium / xhigh を high と同じ 75 に写す (そのままでは medium・high・xhigh が同じプロンプトになる)。
  minimal / low 50、high 75、max / ultra 100、none / off / disabled (思考なし)、数値 1〜100 は変えていない。未指定は 75 のまま
- encoding: V4.1 は `Reasoning Effort: N` をそのまま描く。V4 は段階に丸める (`level()`: ≤61 low、62〜88 high、他 max。
  medium 62 と xhigh 88 が公式どおり high になるよう、中間 62.5 / 87.5 の丸めより high 側を 1 つずつ広げてある)

ds41gw は `ds41gw/Cargo.toml` の `[patch.crates-io]` で使う。更新するときは ds41rt 側の写しをそのまま上書きし、`./cargo.sh update -p deepseek-recipe` で `Cargo.lock` を直す。
