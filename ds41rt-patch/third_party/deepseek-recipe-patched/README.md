# deepseek-recipe-patched

deepseek-recipe / -core / -encoding 0.1.0 (MIT、DeepSeek) に、reasoning effort の数値 (1〜100) 対応を足したもの。
ds41rt 改修版 (`ds41rt-coordinator:v15-effort`) は `rust/Cargo.toml` の `[patch.crates-io]` でこれを使う。
API プロキシ ds41gw も同じ写しを `vendor/deepseek-recipe-patched/` に置き、count_tokens の描画を ds41rt と揃えている
(片方を変えたらもう片方にも同じ変更を入れる)。

- core: `ReasoningEffort::Budget(u8)`、`score()` / `level()`、`RequestedReasoningEffort` (整数・数値文字列・段階名の解析)、
  `MEDIUM_SCORE` (62) / `XHIGH_SCORE` (88)
- recipe: Chat / Responses / Anthropic の effort 型が整数と数値文字列も受ける
- encoding: V4.1 は `Reasoning Effort: N` をそのまま描く。V4 は段階に丸める
- `render-effort`: 1 つの Chat 要求の V4.1 プロンプトを描いて見せる確認用の小さな道具
  (`cargo run --manifest-path third_party/deepseek-recipe-patched/Cargo.toml -p render-effort -- '"medium"' 62 '"xhigh"' 88`)

## effort の写し方

| 要求の値 | V4.1 のプロンプト | 公式 (deepseek-recipe 0.1.0) |
|---|---|---|
| `none` / `off` / `disabled` | 思考なし | `none` のみ |
| `minimal` / `low` | 50 | 50 |
| **`medium`** | **62 (独自)** | 75 |
| `high`、未指定 | 75 | 75 |
| **`xhigh`** | **88 (独自)** | 75 |
| `max` (Anthropic は `ultra` も) | 100 | 100 |
| 整数 1〜100、数値文字列 `"60"` | そのまま | 受けない |

**`medium`→62 と `xhigh`→88 は公式と違う独自の対応である。**
DeepSeek の公式の encoding が段階の名前で定めるのは low (50)・high (75)・max (100) の 3 つだけで、
公式の deepseek-recipe は OpenAI / Anthropic の API の名前を受けるために `medium` と `xhigh` を high と同じ 75 に写す。
そのままでは `medium`・`high`・`xhigh` が同じプロンプトになるので、V4.1 の 1〜100 の尺度で
`medium` を low と high の間 (62)、`xhigh` を high と max の間 (88) に置いた。`medium` と `62`、`xhigh` と `88` は同じプロンプトになる。

V4 (数値の尺度が無いモデル) の `level()` は ≤61 low、62〜88 high、89 以上 max に丸める。
中間 (62.5 / 87.5) での丸めより high 側を 1 つずつ広げてあり、`medium` と `xhigh` は公式どおり high になる。
