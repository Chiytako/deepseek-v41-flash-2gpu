# r7: ds41rt 前段の API 変換プロキシ (ds41gw) 仕様

作成: 2026-10-19。版: 仕様 v0.1 (実装前)。
対象: `localhost:8000` で稼働中の ds41rt v15 (DeepSeek-V4.1-Flash-1017 EXL3 2bit QAD、`ghcr.io/tpurtell/ds41rt-coordinator:v15`)。
成果物の置き場: `proxy/` (この仕様の段階では評価スクリプト `r7_eval/` だけがある)。

根拠となる調査:

| 略号 | ファイル | 内容 |
|---|---|---|
| [C] | `research/r7_clients.md` | GLM 時代の実クライアントの要求 (reqlog 31,913 件、アクセスログ) |
| [A] | `research/r7_ds41rt_api.md` | ds41rt の本番経路 (`native_v41.rs` + deepseek-recipe 0.1.0) の API 仕様 (約 80 ケースの実地試験) |
| [O] | `research/r7_options.md` | 実装手段 4 案の比較 (LiteLLM 実測、recipe、vLLM、自作) |

パスの略記:
- `recipe/` = `research/tmp/r7/crates/deepseek-recipe-0.1.0/src/`
- `api/` = `ext/ds41rt/rust/crates/ds41rt-api/src/`
- `client/` = [O] §1.2 で読んだ OSS のコーディングエージェントの実装 (公開版では名前を伏せた。ログ上は [C] §3.1 の型 B に当たる)
- `snap/` = `hf/hub/models--local--DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1-l2e/snapshots/1017e3c2000000000000000000000000000000e2/` (ds41rt が `--snapshot` で読んでいるもの。`/proc/844728/cmdline`)

凡例: **[必須]** 初版に入れる / **[任意]** 設定で有効化、または第 2 版 / **[要決定]** ユーザーが決める事項 (既定値の提案つき)。

---

## 0. 要約

1. **自作する。** Rust (axum + tokio + reqwest) の単一バイナリ `ds41gw` とし、ds41rt と同じ `deepseek-recipe =0.1.0` を部品として使う ([O] §0、§5)。LiteLLM 1.106.0 は Responses のイベント順序が壊れており、同時 16 本で TTFT p50 が +23〜46 ms 遅れるので使わない ([O] §3.1)。
2. **上流は ds41rt の `/v1/chat/completions` だけを使い、常に `stream:true` で呼ぶ。** 非ストリームの要求はプロキシで集約する。これで切断の伝搬、途中失敗の検出、reasoning の採取を 1 本の経路で扱える。
3. **提供する口**: `/v1/chat/completions` (正規化して素通し)、`/v1/responses` (POST / GET / DELETE / input_items、状態保持あり)、`/v1/messages` と `/v1/messages/count_tokens` (Anthropic)、`/v1/completions` (chat に包む擬似実装)、`/v1/models`、`/health`、`/health/ready`、`/metrics`。
4. **実クライアントが ds41rt で落ちる原因をすべて吸収する** ([C] §0、[A] §10): model の別名、temperature 省略時の greedy、`tool_reference`、developer ロール、`chat_template_kwargs` の思考指示、`max_completion_tokens`、`stream_options` の付け過ぎ、`tool_choice:"none"` での DSML 漏れ、ツールスキーマの型、429。
5. **同時実行**: プロキシが上流への同時数を 16 に絞り、超えた分はプロキシ内の FIFO で待たせる。ds41rt の 429 は、クライアントに返す前にプロキシで再試行する。
6. **口は既定で 30001。** GLM 時代の `:30000` に切り替えるかはユーザー判断 (§15.3)。

ユーザーが決める事項 (提案つき) は §18.1 にまとめた。

---

## 1. 前提 (調査で確定した事実)

### 1.1 クライアント側 ([C])

- エンドポイント: chat 33,113 件、Responses 2,910 件、`/v1/models` 3,141 件。`/v1/completions`、`/v1/messages`、API キーの使用は 0 件 ([C] §2.1)。
- chat の 27.1% (8,664 件) と Responses の全件が temperature を送らない ([C] §3.2、§4.1)。
- 非ストリームの主力は型 A の自動コーディング (21,063 件、同時 12〜16 本、tools 付き 98.8%) ([C] §3.1、§5)。それ以外はほぼ 100% ストリーム。
- tools 付きは 91.8%、`strict` は 0 件。画像は `data:` URL だけで、最大 7 枚/要求 ([C] §3.2、§3.3)。
- 型 B がツールを遅延ロードした後の要求 346 件が、tool メッセージに `[{"type":"tool_reference","name":…}]` を含む ([C] §3.3)。
- Responses (型 F、98 件) は全件ストリーム、状態なしで毎ターン全履歴を送り、reasoning 項目を送り返す。並列関数呼び出しがある。`max_output_tokens` は 128000 と推定 ([C] §4.1)。
- 型 B の Responses 実装は `stream:true` 固定、`store:false`、`include:["reasoning.encrypted_content"]`、`reasoning.summary:"auto"`、`prompt_cache_key` にセッション ID を送る。終端イベントが無いストリームはエラーとして扱う。メッセージ id は 64 文字以内を要求する (いずれもクライアントのコードで確認)。

### 1.2 ds41rt 側 ([A])

- model は `deepseek-ai/DeepSeek-V4.1-Flash` と完全一致でないと 400 (`api/native_v41.rs:268-270`)。
- temperature を省くと greedy (`api/native_v41.rs:172`)。`max_tokens` を省くと 393,216 で、KV をプロンプト長 + max_tokens で予約する ([A] §3、§9.2)。
- 思考を制御できるのは `thinking.type` と `reasoning_effort` (小文字の none/minimal/low/medium/high/xhigh/max) だけ。プロンプト上の値は 50/75/100 の 3 段階 ([A] §5)。
- `developer` ロール、`tool_reference`、`input_text` 型のパーツ、トップレベルの `thinking:true`、`n≠1`、`stream:false` + `stream_options`、`parameters` が `"type":"object"` でないツールは 400 ([A] §3、§4、§6)。
- tool 履歴は、assistant の tool_calls の直後に N 件が連続していなければ 400 ([A] §4.2)。
- `tool_choice:"none"` か tools 省略で履歴に tool_calls があると、DSML の生テキストが content に漏れる ([A] §6.2)。
- SSE: usage は最終チャンクに同居し、`include_usage` に関係なく付く。途中失敗ではエラーイベントも `[DONE]` も無く切断する ([A] §7.2)。
- 同時実行: 実行枠 16、バッファ 16、待機者 16 (25 秒)。超えると 429 + `Retry-After: 1`。待ち行列とプレフィルの間は何も流れない。切断すると生成は即座に止まる ([A] §9)。
- 画像は user と **tool** のメッセージで使える (`recipe/protocol/openai/chat_completion/request/convert.rs:422` で tool の content を `into_content` で受ける)。上限 16 枚 ([A] §4.1)。
  - [C] §0 の P1「tool ロールに画像は不可」は誤りである。`into_text` (`convert.rs:289`) が使われるのは system と assistant だけである (`convert.rs:344,367`)。
- 推奨サンプリングは `temperature 1.0`、`top_p 1.0`、`max_tokens ≥ 256K` (`snap/README.md:77-84`)。

---

## 2. 実装手段の決定

### 2.1 決定

**(b2+d): Rust の自作プロキシに deepseek-recipe =0.1.0 を組み込む。** 名前は `ds41gw`。

| 使うもの | 用途 | 根拠 |
|---|---|---|
| `deepseek_recipe::protocol::openai::responses::request::ResponsesRequest` + `ProtocolRequest::convert` | Responses 要求 → `Conversation` | `recipe/request/mod.rs:58-80` |
| `deepseek_recipe::protocol::anthropic::messages::request::MessagesRequest` + `convert` | Messages 要求 → `Conversation` | 同上 |
| `ResponsesChunkGenerator` / `MessagesChunkGenerator` (`ChunkGenerator` trait) | `OutputChunk` → 各 API の SSE イベント | `recipe/stream/mod.rs:42-74`、`recipe/protocol/openai/responses/response/chunk_generator.rs:284-412` |
| `AppendDelta<ResponsesStreamEvent> for ResponsesResponse` | 非ストリーム応答の組み立て | `recipe/protocol/openai/responses/response/accumulator.rs` |
| `deepseek-recipe-encoding` の `render_conversation` | 逆写像の単体試験、`count_tokens` | `deepseek-recipe-encoding-0.1.0/src/v4/mod.rs:215` |
| `tokenizers` crate + `snap/tokenizer.json` | `count_tokens`、`reasoning_tokens` の計数、文脈長の事前検査 | — |

自作する部分: Chat の正規化、`Conversation` → Chat の逆写像、ds41rt の差分 → `OutputChunk` の写像、応答オブジェクトの補完、状態保存 (previous_response_id)、reasoning の再注入と encrypted_content、JSON Schema と strict の受け渡し、待ち行列、ログ、認証、計測。見込み 1,200〜1,800 行 ([O] §3.4)。

### 2.2 理由

1. **前段と後段の変換規則が揃う。** ds41rt 自身が同じ recipe =0.1.0 で Chat を変換している (`api/../Cargo.toml:9-11`)。Responses / Messages の意味論 (developer の合成、namespace の平坦化、apply_patch、reasoning の結合) を DeepSeek 公式の実装に任せられる。
2. **イベント順序が構造上正しい。** recipe の生成器は項目を常に 1 つだけ開き、`sequence_number` を単調に増やす (`chunk_generator.rs:61-65,67-93,95-179`)。LiteLLM で見つかった欠落・逆行・入れ子は起きない ([O] §3.1、§3.2)。
3. **性能。** 自作の追加遅延は Python 試作でも TTFT +0.8 ms だった ([O] §3.4)。Rust ではそれ以下の見込み。LiteLLM は TTFT p50 +23〜46 ms、p95 +190〜410 ms ([O] §3.1)。
4. **ds41rt を触らない。** ds41rt 本体に route を足す案 (b1) は再ビルドと再起動が要る。長期的な upstream PR 候補とする ([O] §3.2)。

### 2.3 採らない案

| 案 | 理由 |
|---|---|
| LiteLLM 1.106.0 | Responses の `sequence_number` の欠落と逆行、`reasoning_summary_part.added` の欠落、項目の入れ子。`/v1/messages` の本文が空になる経路がある。`previous_response_id` は Postgres が無いと黙って無視される (`session_handler.py:297`)。573 MB / 243 パッケージ ([O] §3.1) |
| vLLM の Responses 移植 | EngineClient と密結合。保存は自らメモリリークと注記 (`serving.py:153-180`) ([O] §3.3) |
| HF responses.js | reasoning と状態保持が未実装 ([O] §3.3) |
| Python (aiohttp) 自作 | 性能は足りるが、recipe の Python バインディングは解析済みチャンクを受け取れない。イベント状態機械を全部自作することになる ([O] §3.2、§3.4)。**第二候補**として残す |

### 2.4 依存の固定

- `deepseek-recipe`、`deepseek-recipe-core`、`deepseek-recipe-encoding` はすべて `=0.1.0`。`Cargo.lock` の checksum が ds41rt の `Cargo.lock` と一致することを CI (ビルドスクリプト) で検査する ([A] §1)。
- ds41rt が recipe を上げたら、ds41gw も同じ版に上げる (§18.2)。

---

## 3. 全体構成

```
クライアント (コーディングエージェント各種 (Chat / Responses) / OpenAI SDK / Anthropic SDK)
   │  :30001 (既定)。:30000 への切替はユーザー判断 (§15.3)
   ▼
ds41gw (Rust 単一バイナリ、docker --network host)
   │  ① 解析・認証・ログ ② API 別の変換 ③ 待ち行列 (上流同時 16) ④ 上流は常に stream:true
   ▼
ds41rt :8000 /v1/chat/completions (変更しない)
```

### 3.1 要求の処理段階 (全 API 共通)

1. 受信。本文の上限は 96 MiB (ds41rt と同じ。`api/native_v41/images.rs:18`)。`Content-Type` が無くても JSON として解析する (ds41rt の 415 を避ける。[A] §2)。
2. 認証 (§12.1)。要求 id `rid` を発行し、`x-request-id` ヘッダで返す。
3. JSON の解析。構文誤りは各 API 形式の 400 (§4.6)。
4. API 別の検査と変換 → ds41rt に送る Chat 本文 (`upstream_body`) を作る。
5. 要求ログを書く (§13)。
6. 待ち行列に入り、上流の枠を得る (§11)。
7. ds41rt に `stream:true` で送る。上流の各チャンクを解析し、API 別の出力器に渡す (§5.4、§6.8、§7.4)。
8. 終了ログを書く。

### 3.2 ds41rt の SSE → `OutputChunk` の写像 (Responses / Messages / completions 共通)

| ds41rt のチャンク | `OutputChunk` | 備考 |
|---|---|---|
| 先頭 (`delta.role`) | `Start{system_fingerprint, usage: 0}` | ds41rt は usage を最後にしか出さない。prompt の usage は `Finish` 時に差し替える (§6.8.3) |
| `delta.reasoning_content` (非空) | `Reasoning{content}` | |
| `delta.content` (非空) | `Raw{content}` | DSML フィルタ (§5.3.9) を通した後 |
| `delta.tool_calls[i]` で `id` あり | `ToolCallBegin` → `ToolCall{tool_name: 逆写像した名前, arguments}` | ds41rt は index ごとに逐次に流す ([A] §7.2) ので、recipe の「1 つずつ開く」形と合う |
| `delta.tool_calls[i].function.arguments` のみ | `ToolArgumentsDelta{content}` | |
| `finish_reason` + `usage` | `Finish{reason, stop_sequence, usage}` | `stop`→`Stop`、`length`→`Length`、`tool_calls`→`ToolCalls`、`content_filter`→`ContentFilter`。proxy 側で stop を照合した場合は `StopSequence` (§7.3) |
| `[DONE]` 無しの EOF / 読み取りエラー | (写像しない) | 途中失敗として §11.5 で処理 |

---

## 4. 共通の方針

### 4.1 model 名 [必須]

- **受理**: 設定 `models.alias_mode`。
  - `any` (既定): どの名前でも ds41rt の正規名 `deepseek-ai/DeepSeek-V4.1-Flash` に置き換える。GLM 時代に `<model>` 98 件、`x` 20 件、`default` 1 件が来ていたため ([C] §3.2)。
  - `listed`: `models.aliases` にある名前だけ受理し、他は 404 `model_not_found`。
- **既定の別名一覧** (`/v1/models` にも載せる): `deepseek-ai/DeepSeek-V4.1-Flash` (正規名)、`DeepSeek-V4.1-Flash`、`deepseek-v4.1-flash`、`GLM-5.3-Flash` (GLM 時代の 99.6%)。
- **応答の `model`**: 既定ではクライアントが送った名前をそのまま返す (`models.echo_requested=true`)。false なら正規名。

### 4.2 サンプリングの既定 [必須] [要決定]

- temperature を省いた要求に、既定値を入れる。ds41rt は省略で greedy になる (`api/native_v41.rs:172`) が、GLM/vLLM 時代のクライアントはサーバー既定の 1.0 で動いていた ([A] §10)。
- **提案する既定**: `temperature 1.0`、`top_p 1.0`。モデルの推奨値である (`snap/README.md:77-82`)。
- 明示された値はそのまま使う (例: 型 A の 0.6 / 0.95 / top_k 20)。`top_p` だけ省いた場合は 1.0 を入れる。`top_k`、`min_p`、`seed` は入れない。
- 設定 `sampling.default = "recommended" | "greedy" | {temperature, top_p}`。
- 代償: サンプリングは greedy より約 5% 遅い ([O] §5.1 の 4、`r2_ds41rt.md` §6)。

### 4.3 思考 (thinking) と effort [必須]

**方針: 要求の指定を優先し、指定が無ければ thinking on・effort high とする。** GLM 時代の `EFFORT` 既定 high (`$GLM53/../run_glm53_vllm.sh:18,92`、`$GLM53/../run_glm53.sh:199`) を踏襲する。ds41rt の既定 (on / 75) とも一致する ([A] §5)。

#### 4.3.1 ds41rt に送る値

proxy は最終的に、ds41rt の `thinking:{"type":"enabled"|"disabled"}` と `reasoning_effort` (小文字の none/low/high/max) だけを送る。effort の対応 (設定 `thinking.effort_map` で変更可):

| 要求の値 | ds41rt に送る値 | プロンプト上の N | 備考 |
|---|---|---|---|
| `none` | `thinking:disabled` | — | |
| `minimal`、`low` | `low` | 50 | ds41rt でも同じ ([A] §5)。thinking は止まらない |
| `medium`、`high`、省略 | `high` | 75 | |
| `xhigh` | **`max`** | 100 | ds41rt 自身は xhigh を 75 にする ([A] §5)。OpenAI の xhigh は high より上なので max に寄せる |
| `max` | `max` | 100 | |
| 大文字 (`HIGH`) | 小文字にして上の表 | | ds41rt は大文字を 400 にする |
| 数値 `x` | `x ≤ 0` → none、`0 < x < 1` は `x×100` とみなす、`≤ 50` → low、`≤ 75` → high、それ以上 → max | | ds41rt は数値を 400 にする ([A] §0 の 2) |
| 上記以外の文字列 | 400 `invalid_reasoning_effort` | | |

#### 4.3.2 Chat での指定の優先順位

上から順に、最初に見つかったもので thinking の on/off を決める。

1. トップレベル `thinking` が object (`{"type":"enabled"|"adaptive"|"disabled", "budget_tokens"?}`) → そのまま。`adaptive` は enabled として送る。
2. トップレベル `thinking` が bool → `{type: enabled|disabled}` に正規化 (ds41rt は bool で 400。[C] §3.4)。
3. `reasoning_effort` (トップレベル) が `none` → disabled。
4. `chat_template_kwargs.enable_thinking`、`chat_template_kwargs.thinking`、トップレベルの `enable_thinking` (bool) → false なら disabled、true なら enabled。ds41rt はこれらを無視する ([A] §5)。
5. どれも無ければ `thinking.default` (既定 enabled)。

effort は、`reasoning_effort` (トップレベル) → `chat_template_kwargs.reasoning_effort` → `thinking.budget_tokens` (§4.3.3) → `thinking.default_effort` (既定 high) の順に決める。

`chat_template_kwargs`、`enable_thinking` は上流に送る前に取り除く。

#### 4.3.3 budget_tokens の写像

ds41rt は `budget_tokens` を無視する ([A] §5)。effort に写す (Anthropic の `thinking.budget_tokens` と Chat の `thinking.budget_tokens` 共通、設定可):

| budget_tokens | effort |
|---|---|
| `< 1024` | low (Anthropic の最小は 1024) |
| `1024 〜 8191` | low |
| `8192 〜 32767` | high |
| `≥ 32768` | max |

思考の長さを実際に制限する手段は `max_tokens` だけである。budget を超えたら打ち切る機能は入れない。

#### 4.3.4 clear_thinking [任意、既定 on]

- `chat_template_kwargs.clear_thinking:true` が来たら、履歴のすべての assistant から `reasoning_content` を削る (§5.3.7 の再注入もしない)。
- GLM の「過去の思考を消す」指示に相当する。ds41rt はすべての turn の思考を再描画する ([A] §4.3)。型 E (短い要約) 2,070 件が送っていた ([C] §3.1)。
- 既定値 (送られない場合) は false (GLM の vLLM 期の既定 `clear_thinking:false`。`run_glm53_vllm.sh:92`)。

#### 4.3.5 小さな max_tokens と thinking [任意、既定 off] [要決定]

- 例: タイトル生成 (`max_output_tokens 64`、effort high) や型 E の短い要約 (`max_tokens 120`、`enable_thinking:true`)。thinking on のままだと思考だけで上限に達し、`content:""`、`finish_reason:"length"` になりうる ([A] §5)。
- 設定 `thinking.auto_off_below_max_tokens = 0` (無効)。正の値 N にすると、`max_tokens < N` かつ要求が thinking を明示していない場合に disabled にする。
- 型 E は `enable_thinking:true` を明示しているので、この機能では救えない。実地で空応答になるか確かめてから決める (§16.3 の T-A9)。

### 4.4 出力トークン上限 [必須] [要決定]

- Chat: `max_tokens` が無く `max_completion_tokens` があれば、`max_tokens` に移す (ds41rt は `max_completion_tokens` を無視する。[A] §3)。両方あれば `max_tokens` を使い、`max_completion_tokens` は捨てる。
- Responses: `max_output_tokens` → `max_tokens`。Messages: `max_tokens` (必須)。
- 両方とも無い場合: `limits.default_max_tokens` を入れる。**提案する既定は 262,144** (`snap/README.md:83` の「≥ 256K」)。
  - ds41rt の既定 393,216 のままだと、1 本あたりプロンプト長 + 393,216 の KV を予約する ([A] §9.2)。
  - 262,144 なら、16 本の平均プロンプトが 50 万トークンでも約 1,220 万トークンで、プール約 1,470 万に収まる。
- `16.0` のような整数値の小数は整数に直す。0 や負数は ds41rt と同じく 400。

### 4.5 ID の形式

| 対象 | 形式 | 長さ |
|---|---|---|
| Responses の response id | `resp_` + 32 桁 hex | 37 |
| Responses の item id (recipe が派生) | `rs_{resp}_{i}`、`msg_{resp}_{i}`、`fc_{resp}_{i}` | 最大 約 45 (型 B の 64 文字制限内) |
| Responses の call_id (recipe) | `call_{resp}_{n}` | 約 44 |
| Messages の message id | `msg_` + 32 桁 hex。tool_use id は `toolu_{msg}_{n}` (recipe) | |
| Chat | ds41rt の `chatcmpl-<uuid>` と `call_chatcmpl-<uuid>_<i>` をそのまま返す ([A] §7.1) | |

### 4.6 エラーの形 [必須]

ds41rt のエラーは 2 種類の JSON と text/plain が混在する ([A] §8)。proxy は API ごとに 1 形式に統一する。

- OpenAI 形式 (`/v1/chat/completions`、`/v1/responses`、`/v1/completions`、`/v1/models`):
  `{"error":{"message":"…","type":"…","param":"…"|null,"code":"…"|null}}`
- Anthropic 形式 (`/v1/messages*`):
  `{"type":"error","error":{"type":"…","message":"…"}}`

| 状況 | HTTP | OpenAI `type` / `code` | Anthropic `type` |
|---|---|---|---|
| JSON の構文誤り、proxy の検査で不正 | 400 | `invalid_request_error` / 個別 (例 `unsupported_tool`) | `invalid_request_error` |
| ds41rt の 400 (`native_v41_error`、text/plain の 400、strict 検査) | 400 | `invalid_request_error` / `upstream_rejected`。`param` があれば引き継ぐ | `invalid_request_error` |
| 認証失敗 | 401 | `authentication_error` / `invalid_api_key` | `authentication_error` |
| `previous_response_id` が見つからない | 400 | `invalid_request_error` / `previous_response_not_found`、`param:"previous_response_id"` | — |
| 未対応の機能 (logprobs、background 等) | 400 | `invalid_request_error` / `unsupported_parameter`、`param` に項目名 | `invalid_request_error` |
| 文脈長の超過 (§11.6) | 400 | `invalid_request_error` / `context_length_exceeded` | `invalid_request_error` |
| 本文が 96 MiB 超 | 413 | `invalid_request_error` / `request_too_large` | `request_too_large` |
| 待ち行列の時間切れ・満杯 (§11.2) | 429 + `Retry-After` | `rate_limit_error` / `queue_timeout` | **529** `overloaded_error` |
| ds41rt に接続できない、503 | 503 | `server_error` / `upstream_unavailable` | `overloaded_error` (529) |
| ds41rt の 500、途中失敗 (非ストリーム) | 502 | `server_error` / `upstream_error` | `api_error` (500) |
| 未定義のパス | 404 | `invalid_request_error` / `not_found` | `not_found_error` |

- `message` には、ds41rt の文言を `upstream: …` として含める (切り詰め済み 512 文字。`api/error.rs:70-79`)。

### 4.7 未対応の機能の扱い

- 設定 `compat.unsupported = "reject" | "ignore"`。既定は `reject`。実クライアントの使用は 0 件である ([C] §3.2)。
- `reject` で 400 を返すもの: `logprobs:true`、`top_logprobs>0`、Responses の `include` に `message.output_text.logprobs`、`background:true`、`conversation`、`prompt` (再利用プロンプト)、組み込みツール (§6.4)。
- 黙って無視するもの (警告ヘッダ `x-ds41gw-warning: ignored=<項目,…>` を付ける): `frequency_penalty`/`presence_penalty`/`repetition_penalty` の 0 以外 (ds41rt は範囲だけ検査して無視する。[A] §3)、`logit_bias`、`service_tier`、`text.verbosity`、`truncation`、`prompt_cache_retention`。
- 記録だけするもの: `user`、`metadata`、`safety_identifier`、`prompt_cache_key` (§13)。

---

## 5. `/v1/chat/completions`

### 5.1 位置づけ

正規化して素通しする。ds41rt の機能 (tools / strict / response_format / 画像 / 思考 / top_k / min_p / seed / stop) はそのまま使える。

### 5.2 要求のフィールド別の処理

| フィールド | 処理 | 根拠 |
|---|---|---|
| `model` | §4.1 | [A] §3 |
| `messages` | §5.3 | |
| `stream` | 上流には常に true。クライアントへの形は元の値に従う | §3 |
| `stream_options` | `stream:false` なら削除 (ds41rt は 400)。`include_usage` は §5.4 の出力形に使う | `convert.rs:498-502` |
| `max_tokens` / `max_completion_tokens` | §4.4 | |
| `temperature` / `top_p` | §4.2 | |
| `top_k` / `min_p` / `seed` / `stop` | 素通し。`stop` が 17 個以上なら proxy が 400 を返す | [A] §3 |
| `n` | §5.5 | |
| `logprobs` / `top_logprobs` | true / 正の値なら 400 (§4.7)。false / 0 / null は削除して送る | |
| penalty 系 | 範囲 [-2,2] の外は 400。中は素通し (ds41rt が無視)、0 以外なら警告ヘッダ | `convert.rs:106-115` |
| `thinking` / `reasoning_effort` / `chat_template_kwargs` / `enable_thinking` | §4.3 | |
| `tools` | §5.3.8 | |
| `tool_choice` | 素通し。ただし `"none"` と履歴の組み合わせは §5.3.9 | |
| `parallel_tool_calls` | 素通し | |
| `response_format` | 素通し (text / json_object / json_schema / regex) | [A] §6.3 |
| `tool_decoding_assistance` | 素通し (ds41rt の拡張) | |
| `user` / `metadata` / `store` | 記録して削除 | |
| その他の未知のフィールド | 素通し (ds41rt は無視する) | [A] §3 |

### 5.3 messages の正規化 (この順で適用)

1. **ロール**: `developer` → `system` [必須] ([A] §4)。`function` ロール (旧 OpenAI 形式) → `tool` (`name` を tool_call と照合)。
2. **null の content**: user と tool で null → `""`。assistant で content と tool_calls がどちらも null (か `tool_calls:[]`) → `content:""` にし、`tool_calls` を削除する ([A] §4、§4.2)。
3. **パーツの型**:
   - `input_text` / `output_text` → `text`。
   - `tool_reference` (`{"type":"tool_reference","name":X}`) → `{"type":"text","text":"Tool loaded: X"}` [必須]。型 B の 346 要求が 400 になる原因 ([C] §3.3)。
   - `image_url` が文字列 (`{"type":"image_url","image_url":"data:…"}`) → object 形 `{"url":…}`。
   - `input_image` (`image_url` か `file_data`) → `image_url`。
   - 上記以外の未知の型 → `{"type":"text","text":<JSON 文字列>}` に変換し、警告ヘッダを付ける。
   - system と assistant の画像は、ds41rt が 400 にする。proxy は `[image omitted]` のテキストに置き換える。
4. **assistant の思考の別名**: `reasoning` (vLLM 形式) や `reasoning_text` → `reasoning_content` ([A] §4)。
5. **tool_calls の整形**: `function.arguments` が object なら JSON 文字列にする (ds41rt は 400。[A] §4.2)。`id` が無ければ `call_ds41gw_<rid>_<i>` を振り、続く tool メッセージの対応が取れない場合は順番で対応させる。
6. **ツール履歴の補修** [必須、設定 `compat.repair_tool_history=true`]。ds41rt の厳格な規則 ([A] §4.2) を満たすように並べ直す。
   - assistant の tool_calls の後ろ、次の assistant までにある tool メッセージを集め、呼び出し順に直後へ並べる。間の user メッセージは、その後ろに移す。
   - 結果が無い呼び出しには、`{"role":"tool","tool_call_id":…,"content":"[no result: the tool call was interrupted]"}` を補う。
   - 対応する呼び出しが無い tool メッセージは、user メッセージ `[tool result <id>]\n<content>` にする。
   - 同じ id の重複は最初の 1 件を残す。
   - 補修したら警告ヘッダ `x-ds41gw-warning: repaired_tool_history` を付け、ログに残す。
7. **reasoning の再注入** [必須、§10.1]: thinking on のとき、`reasoning_content` が無い (か空の) assistant に、保存しておいた思考を補う。clear_thinking のときはしない。
8. **ツール定義の正規化** [必須] ([A] §6.1):
   - `type` が無ければ `"function"` を補う。
   - 平坦形 (`{"type":"function","name":…,"parameters":…}`、Responses 形式) → `{"type":"function","function":{…}}`。
   - `parameters` が `{}`、`type` が無い、`type` が配列 (`["object"]`、`["object","null"]`) → `{"type":"object", ...元の properties/required}`。`properties` も無ければ `{"type":"object","properties":{}}`。
   - `parameters` が object 以外のスキーマ (例 `{"type":"string"}`) → 400 `invalid_tool_schema`。
   - **名前の写像**: `^[A-Za-z0-9_-]{1,128}$` に合わない名前 (例 `a.b`、Responses の namespace の `ns::name`) → 不正な文字を `_` にし、衝突したら `_2` などを付け、128 文字に切る。要求ごとの対応表を持ち、履歴の tool_calls の名前も同じ規則で写し、出力の名前は元に戻す。
9. **`tool_choice:"none"` と DSML の漏れ** [必須] ([A] §6.2、GLM 時代 187/1191 件):
   - 条件: `tool_choice:"none"`、または `tools` が無い・空で、履歴に tool_calls か tool メッセージがある。
   - 既定の対策 (`compat.tool_none_strategy = "flatten"`): 履歴のツールのやり取りを平文にする。
     - assistant の tool_calls → assistant の content に `[tool call] <name>(<arguments>)` を追記し、tool_calls を削除する。
     - tool メッセージ → user メッセージ `[tool result] <content>` (画像はそのまま user の画像)。
     - `tools` と `tool_choice` は送らない。
     - 代償: この要求だけプロンプトが変わるので、prefix キャッシュが外れる。
   - 併用する安全策: 出力の content に `<｜DSML｜` が現れたら、そこから DSML ブロックの終わりまでを除き、ログに `dsml_leak` を記録する (`compat.strip_dsml=true`)。ストリームでは `<｜` で始まる可能性のある末尾を最大 16 文字だけ保留して判定する。
   - 他の選択肢 (`keep_tools`): tools を残して `tool_choice:"auto"` で送り、出力に tool_calls が出たら捨てる。応答本文が空になりうるので既定にしない。

### 5.4 応答の形

#### 5.4.1 ストリーム

ds41rt のチャンクを 1 つずつ解析し、次だけを書き換えて送る。

- `model` → §4.1 の名前。
- tool の名前 → §5.3.8 の逆写像。
- content → DSML フィルタ (§5.3.9)。
- **最終チャンクと usage**:
  - `stream_options.include_usage:true` のとき: OpenAI の形に合わせて、`finish_reason` を持つチャンク (`usage:null`) と、`choices:[]` に usage だけを載せたチャンクに分ける [任意、既定 on]。ds41rt は 1 つのチャンクに同居させる ([A] §7.2)。
  - 指定が無いとき: ds41rt と同じく最終チャンクに usage を残す (無害なため)。設定で削除もできる。
- **usage の補完**: `completion_tokens_details.reasoning_tokens` を加える。proxy が `reasoning_content` の全文を `snap/tokenizer.json` で数えた値である。ds41rt は出さない ([A] §7.1)。dSpark では 1 チャンクに複数トークンが入りうるので、チャンク数では数えない。
- `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` は残す。
- 終端は `data: [DONE]`。
- keep-alive: 上流から 15 秒間何も来なければ、SSE のコメント行 `: keep-alive\n\n` を送る (§11.4)。
- 途中失敗: §11.5。

#### 5.4.2 非ストリーム

上流のストリームを集約して `chat.completion` を作る。

- `id`、`created`、`system_fingerprint` は ds41rt の値。
- `message.content`: 本文の連結。ツールだけの応答では ds41rt と同じく `""` (`compat.null_content_for_tool_calls=true` で null にできる)。
- `message.reasoning_content`: thinking on のときだけ付ける (ds41rt と同じ)。
- `message.tool_calls`: index ごとに連結した arguments。
- `finish_reason`、`usage` は §5.4.1 と同じ規則。`logprobs:null`。
- 集約中に上流が途中で切れたら、§11.5 の規則で再試行するか 502 を返す。

### 5.5 `n > 1` [任意、既定 on]

- 実クライアントの使用は 0 件 ([C] §3.2)。ds41rt は n≠1 を 400 にする ([A] §3)。
- proxy は n 本の要求に分けて送る。上限は `limits.max_n=8` で、超えたら 400。n 本はそれぞれ上流の枠を 1 つずつ使う。
- `seed` があれば i 本目に `seed+i` を使う。temperature が 0 (greedy) のときは、同じ出力になる旨を警告ヘッダで返す。
- 非ストリーム: `choices[i].index=i` にまとめる。usage は prompt を 1 本分、completion を合計にする。
- ストリーム: 各本のチャンクを `choices[0].index` を i に書き換えて交互に流す。`[DONE]` は全本の終了後に 1 回送る。

---

## 6. `/v1/responses`

### 6.1 処理の流れ

1. 生の JSON を前処理する (§6.2〜§6.6)。recipe が扱わない項目 (`previous_response_id`、`text.format.json_schema`、`tools[].strict`、reasoning の summary / encrypted_content、組み込みツールの拒否、名前の写像) をここで処理する。
2. 前処理した本文を `ResponsesRequest` に deserialize し、`convert(ConversionOptions::new().with_default_thinking_mode(true))` で `ConversationRequest` に変換する。
3. **逆写像**: `Conversation` の `InputMessage` (System / User / Assistant / Tool / LatestReminder。`deepseek-recipe-core/src/messages.rs`) を Chat の messages に戻す。
   - User と Tool の content にある画像プレースホルダ `<｜image｜>` (`core/src/multimodal/mod.rs:15`) で文字列を分割し、`image_sources` を順に `image_url` パーツとして挟む。
   - Assistant の `reasoning_content`、`tool_calls` はそのまま。
   - LatestReminder は ds41rt の `latest_reminder` ロールで送る ([A] §4)。
   - tools は `Conversation` のツール定義から Chat の tools を作る (`strict` は recipe が保持する。`recipe/protocol/openai/responses/request/tools.rs:227`)。
   - thinking / effort は `ConversationRequest` の解決結果を §4.3.1 の表で写す。
4. 以降は Chat と同じく上流へ送る (§5.3 の 6〜9 も適用)。
5. 出力は `ResponsesChunkGenerator::new(resp_id, model).with_custom_tool_names(req.custom_tool_names())` に `OutputChunk` (§3.2) を流し、イベントを後処理して送る (§6.8)。

### 6.2 要求のトップレベルのフィールド

| フィールド | 対応 | 処理 |
|---|---|---|
| `model` | ✔ | §4.1 |
| `input` (文字列 / 項目の配列) | ✔ | §6.3 |
| `instructions` | ✔ | recipe が system にする。`previous_response_id` で連鎖する場合も、OpenAI と同じく前の instructions は引き継がない |
| `tools` | ✔ | §6.4 |
| `tool_choice` | ✔ | §6.4 |
| `parallel_tool_calls` | ✔ | Chat の `parallel_tool_calls` |
| `reasoning.effort` | ✔ | recipe → §4.3.1 の表 |
| `reasoning.summary` / `generate_summary` | ✔ | 出力の reasoning の形を決める (§6.8.2) |
| `text.format` | ✔ | `text` → なし、`json_object` → `response_format:{type:json_object}`、`json_schema` → `response_format:{type:"json_schema","json_schema":{name, schema, strict, description}}`。recipe は json_schema を無視する (`convert.rs:653`) ので、生の本文から取り出して Chat 本文に入れる |
| `text.verbosity` | 無視 | 警告ヘッダ |
| `max_output_tokens` | ✔ | §4.4 |
| `temperature` / `top_p` | ✔ | §4.2 |
| `stream` | ✔ | 非ストリームは集約 (§6.8.4) |
| `store` | ✔ | 既定 true (OpenAI と同じ)。§6.6 |
| `previous_response_id` | ✔ | §6.6 |
| `include` | 一部 | `reasoning.encrypted_content` → §10.2。`message.output_text.logprobs` → 400。その他 (組み込みツール関連) は無視 |
| `metadata` | ✔ | 応答にそのまま返し、保存する。最大 16 組 |
| `max_tool_calls` | ✔ | §11.7 の上限として使う |
| `truncation` | 無視 | `auto` でも切り詰めない。文脈長の超過は 400 (§11.6) |
| `prompt_cache_key` / `prompt_cache_retention` / `safety_identifier` / `user` | 記録 | 型 B はセッション ID を `prompt_cache_key` で送る。ログのセッション解析に使う |
| `service_tier` | 無視 | 応答では `"default"` |
| `top_logprobs` | 0 以外は 400 | |
| `background` | true なら 400 | 実例 0 件 ([C] §4.3 の 6) |
| `conversation` / `prompt` | 400 | |

### 6.3 input の項目型

| 項目 | 処理 |
|---|---|
| 文字列の `input` | user メッセージ 1 件 |
| `message` (role: user / assistant / system / developer。`type` 省略も可) | recipe が変換する。developer と system は system になる (`recipe/…/responses/request/schema.rs:58-80`) |
| content: `input_text` / `output_text` / `refusal` | テキスト |
| content: `input_image` (`image_url` が data: か http(s)、`detail`) | 画像。`file_id` は 400 |
| content: `input_file` | 400 `unsupported_parameter` (ds41rt は画像の data: だけ受ける。[A] §4.1) |
| `function_call` (`call_id`、`name`、`arguments`、`namespace`) | assistant の tool_calls。連続する function_call は recipe が 1 つの assistant にまとめる (並列呼び出し) |
| `function_call_output` (`output` が文字列、または `input_text` / `input_image` の配列) | tool メッセージ。画像は tool メッセージの `image_url` パーツにする。ds41rt は tool の画像を受ける (§1.2) |
| `reasoning` | 次の assistant の `reasoning_content`。本文は §6.5 の順で探す |
| `custom_tool_call` / `custom_tool_call_output` | recipe が処理する (apply_patch のみ) |
| `item_reference` (`id`) | 保存済みの項目から引く。無ければ 400 |
| `web_search_call`、`file_search_call`、`computer_call` 等 | recipe は無視する (`schema.rs:66-68`)。proxy も無視し、警告ヘッダを付ける |

画像の合計が 16 枚を超える場合: 古い画像から順に `[image omitted]` のテキストに置き換え、新しい 16 枚を残す (`limits.image_overflow = "drop_oldest" | "reject"`、既定 drop_oldest)。

### 6.4 tools と tool_choice

- `function` (平坦形 `{type, name, description, parameters, strict}`): Chat の function ツールにする。`strict` は Chat の `function.strict` に入れる。ds41rt は文法と完了後の検証で strict を強制する ([A] §6.1)。`parameters` は §5.3.8 で正規化する。
- `namespace`: recipe が `ns::name` に平坦化する。この名前は ds41rt の名前規則に合わないので、§5.3.8 の写像で `ns__name` にして送り、出力は `ns::name` に戻してから生成器に渡す (生成器が `::` で namespace を分ける。`chunk_generator.rs:339-342`)。
- `custom` (`apply_patch`): recipe が function ツールに変換する。出力は `custom_tool_call` 項目になる。
- **組み込みツールは非対応で、400 を返す**: `web_search` / `web_search_preview` / `web_search_2025_08_26`、`file_search`、`code_interpreter`、`computer_use_preview`、`image_generation`、`mcp`、`local_shell`、`shell`。`code`: `unsupported_tool`、`param: tools[i].type`。
  - recipe の既定は web_search を黙って捨て、他の未知の型も無視する (`recipe/request/options.rs`、`schema.rs:240-256`)。proxy は先に生の本文で検査する。
  - `compat.ignore_builtin_tools = ["web_search"]` のように列挙した型だけ、捨てて続行できる (既定は空)。
- `tool_choice`: `auto` / `none` / `required` / `{type:"function",name}` / `{type:"custom",name}` は recipe に任せる。`{type:"allowed_tools", mode, tools}` は、tools をその一覧に絞ってから mode (`auto` / `required`) にする。組み込みツールの指定は 400。
- `tool_choice:"none"` と function_call の履歴: Chat と同じ対策 (§5.3.9)。

### 6.5 reasoning 項目の本文の探し方 [必須]

recipe は `content[].text` しか読まない (`recipe/…/responses/request/convert.rs:376-407`)。proxy は前処理で、次の順に本文を探して `content:[{"type":"reasoning_text","text":…}]` を埋める。

1. `content[].text` (型 B はこれを含めて送り返す)。
2. `summary[].text` を `\n\n` で連結したもの。
3. `encrypted_content` を §10.2 で復号したもの。SGLang の `sglang-reasoning-v1.` 形式 (base64(zlib)、[C] §4.1) も復号する。GLM 時代のセッションを引き継ぐためである。
4. `id` が proxy の発行した `rs_…` で、保存 (§6.6) か再注入ストア (§10.1) にあれば、その本文。
5. どれも無ければ空 (ds41rt は `<think></think>` と描画する。[A] §4.3)。

### 6.6 状態保持 (store / previous_response_id) [必須]

実クライアントの使用は 0 件だが ([C] §4.1)、OpenAI Agents SDK などに備えて初版から入れる。黙って無視することはしない ([O] §5.2)。

- **保存する条件**: `store` が true か省略。`store:false` は保存しない (型 B)。
- **保存する内容**: response id ごとに、(a) 応答オブジェクト全体、(b) この要求の `input` を展開した項目 (item_reference を解決済み)、(c) `previous_response_id`、(d) `instructions`、`tools`、`text`、`reasoning` などの要求パラメータ、(e) 作成時刻。項目には proxy が id を振る (`msg_…`、`fc_…` など)。
- **previous_response_id の解決**: 連鎖を親までたどり、`[親の input 項目, 親の output 項目, …, 今回の input]` を作る。最大の深さは 1,000。
  - 見つからない、期限切れ、`store:false` で保存していない → 400 `previous_response_not_found`。
  - `instructions` は引き継がない (OpenAI と同じ)。tools は今回の要求のものを使う。
- **保存先**:
  - プロセス内の LRU: 既定 2,000 件 / 1 GiB。
  - sqlite (WAL): `proxy/state/responses.sqlite`。本文は zstd で圧縮する。
  - 期限: 30 日 (OpenAI の保存期間に合わせる)。容量の上限は 20 GiB で、超えたら古い順に消す。1 時間ごとに掃除する。
  - `state.backend = "memory" | "sqlite"`、既定 sqlite。
- **API**:
  - `GET /v1/responses/{id}`: 保存した応答オブジェクト。`?stream=true` は 400 (background が無いので不要)。`include[]` は無視する。
  - `DELETE /v1/responses/{id}`: `{"id":…,"object":"response","deleted":true}`。無ければ 404。
  - `GET /v1/responses/{id}/input_items?limit=&order=&after=`: `{"object":"list","data":[…],"first_id","last_id","has_more"}`。
  - `POST /v1/responses/{id}/cancel`: background が無いので 400。
- 秘匿: 保存する内容は要求ログと同じ扱い (§13.3)。

### 6.7 応答オブジェクトの補完

recipe の `ResponsesResponse` は id、object、created_at、model、status、completed_at、error、incomplete_details、output、usage しか持たない (`recipe/…/responses/response/schema.rs:13-25`)。OpenAI SDK の `Response` 型に合わせて、proxy は response を含むすべてのイベント (created / in_progress / completed / incomplete / failed) と非ストリーム応答に次を加える。

`instructions`、`metadata`、`parallel_tool_calls`、`previous_response_id`、`reasoning {effort, summary}`、`store`、`temperature`、`top_p`、`text {format}`、`tool_choice`、`tools` (要求で受けた形のまま)、`truncation` (`disabled`)、`max_output_tokens`、`service_tier:"default"`、`user`、`background:false`。

### 6.8 ストリーミングのイベント列

#### 6.8.1 順序 (reasoning → 本文 → 並列ツール 2 本の例)

recipe の生成器が出す順序に、proxy が補完と後処理を加える。`sequence_number` は proxy が送出の直前に 0 から振り直す (挿入や置換をしても単調になるようにするため)。

```
response.created            (status in_progress, output [])
response.in_progress
response.output_item.added  (oi=0, reasoning, id rs_…_0, content [] / summary [])
response.content_part.added (oi=0, ci=0, part {type:reasoning_text, text:""})
response.reasoning_text.delta  × n
response.reasoning_text.done   (全文)
response.content_part.done
response.output_item.done   (oi=0, status completed)
response.output_item.added  (oi=1, message, role assistant, id msg_…_1)
response.content_part.added (oi=1, ci=0, part {type:output_text, text:"", annotations:[]})
response.output_text.delta  × n
response.output_text.done
response.content_part.done
response.output_item.done   (oi=1。tool call が続く場合 phase は commentary)
response.output_item.added  (oi=2, function_call, id fc_…_2, call_id call_…_0, name, arguments "")
response.function_call_arguments.delta × n
response.function_call_arguments.done  (全引数)
response.output_item.done   (oi=2)
response.output_item.added  (oi=3, function_call, call_id call_…_1)
response.function_call_arguments.delta × n
response.function_call_arguments.done
response.output_item.done   (oi=3)
response.completed          (status completed, usage, output 全体)
```

- 項目は常に 1 つだけ開いており、added と done が対になる (`chunk_generator.rs:67-179`)。
- `max_tokens` に達した場合: 開いている項目を status incomplete で閉じ、`response.incomplete` (`incomplete_details.reason:"max_output_tokens"`) で終わる (`chunk_generator.rs:375-406`)。
- SSE の書式: `event: <type>\ndata: <json>\n\n`。最後に `[DONE]` は送らない (OpenAI と同じ)。
- keep-alive: 15 秒間イベントが無ければ `: keep-alive\n\n` (§11.4)。

#### 6.8.2 reasoning の出力形式

`reasoning.output_mode` (既定 `auto`) で決める。

| 条件 | ストリームのイベント | output_item.done の reasoning 項目 |
|---|---|---|
| `auto` で、要求に `reasoning.summary` が無い | recipe のまま (`content_part.added` + `reasoning_text.delta/done` + `content_part.done`) | `content:[{reasoning_text}]`、`summary:[]` |
| `auto` で、要求に `reasoning.summary` がある (型 B は `"auto"` を送る) | `reasoning_text` 系を次に置き換える: `response.reasoning_summary_part.added` (summary_index 0、part `{type:summary_text,text:""}`) → `response.reasoning_summary_text.delta` × n → `response.reasoning_summary_text.done` → `response.reasoning_summary_part.done` | `summary:[{type:summary_text,text}]` と `content:[{reasoning_text}]` の両方 |
| `text` / `summary` | 常に前者 / 常に後者 | 同上 |

- 同じ本文を reasoning_text と summary_text の両方の delta で流すことはしない。型 B は両方の delta を思考として読むので、二重になる。
- `include` に `reasoning.encrypted_content` があれば、done の reasoning 項目に `encrypted_content` を加える (§10.2)。設定 `reasoning.always_encrypted=false` を true にすると、指定が無くても付ける。

#### 6.8.3 usage の差し替え

- ds41rt は usage を最終チャンクにしか出さない。`Start` には 0 を渡し、`Finish` の後で completed / incomplete イベントの `response.usage` を ds41rt の値で上書きする。
  - `input_tokens` = `prompt_tokens`、`input_tokens_details.cached_tokens` = `prompt_cache_hit_tokens`、`output_tokens` = `completion_tokens`、`output_tokens_details.reasoning_tokens` = proxy が数えた値 (§5.4.1)、`total_tokens`。
- recipe の usage は reasoning_tokens が常に 0 である (`schema.rs:139-160`)。

#### 6.8.4 非ストリーム

イベント列を `ResponsesResponse` に `AppendDelta` で集約し (`accumulator.rs`)、§6.7 の補完をして返す。HTTP 200、`object:"response"`。`max_output_tokens` に達した場合も 200 で `status:"incomplete"`。

#### 6.8.5 失敗

- 上流の途中失敗 (§11.5) では、開いている項目は閉じずに `response.failed` (`response.status:"failed"`、`response.error:{code:"server_error",message}`) を送って終える。型 B は `response.failed` をエラーとして扱う。
- 待ち行列の時間切れなど、ヘッダを送る前の失敗は HTTP エラー (§4.6)。ヘッダを送った後 (§11.4 の早期ヘッダ) の場合は、`response.created` の後に `response.failed` を送る。

---

## 7. `/v1/messages` (Anthropic)

実クライアントの使用は 0 件 ([C] §2.1)。Claude Code や Anthropic SDK を使うクライアントに備えて入れる。

### 7.1 要求

recipe の `MessagesRequest` (`recipe/protocol/anthropic/messages/request/schema.rs:11-23`) で変換し、§6.1 と同じ逆写像で Chat にする。

| フィールド | 処理 |
|---|---|
| `model` | §4.1。`claude-*` も `alias_mode:any` なら受理する |
| `system` (文字列 / text ブロックの配列) | system。`cache_control` は無視する |
| `messages[].content` (文字列 / ブロックの配列) | 下表 |
| `max_tokens` | 必須 (Anthropic と同じく、無ければ 400) |
| `temperature` / `top_p` | 素通し (省略時は §4.2) |
| `top_k` | recipe に無いので、生の本文から取り出して Chat の `top_k` に入れる |
| `stop_sequences` | proxy 側で照合する (§7.3)。ds41rt には送らない |
| `stream` | ✔ |
| `tools` | `{name, description, input_schema}` → Chat の function ツール。`input_schema` は §5.3.8 で正規化する。`type` が `bash_*`、`text_editor_*`、`computer_*`、`web_search_*`、`code_execution_*` などのサーバー / クライアント組み込みツールは 400 |
| `tool_choice` | `auto` / `any` (→ required) / `tool` (→ 名前指定) / `none`。`disable_parallel_tool_use:true` → `parallel_tool_calls:false` |
| `thinking` | `{type:"enabled", budget_tokens}` → enabled + §4.3.3 の effort。`{type:"disabled"}` → disabled。`{type:"adaptive"}` → enabled |
| `output_config.effort` | low / medium / high / max → §4.3.1 の表 |
| `metadata.user_id` | 記録 |
| `anthropic-version` / `anthropic-beta` ヘッダ | 受理して無視する |

content ブロック:

| ブロック | 処理 |
|---|---|
| `text` | テキスト。`citations` は無視 |
| `image` (`source.type`: `base64` / `url`) | `image_url` (`data:<media_type>;base64,…` か URL)。`file` 型は 400 |
| `document` | 400 (PDF 等は非対応) |
| `tool_use` (assistant) | tool_calls。`input` は JSON 文字列にする |
| `tool_result` (user) | tool メッセージ。content は文字列か text / image ブロックの配列。`is_error:true` は本文の先頭に `[error] ` を付ける |
| `thinking` (`thinking`、`signature`) | 直後の assistant の `reasoning_content`。signature は検証しない (§7.4) |
| `redacted_thinking` | 捨てる |
| `server_tool_use` / `web_search_tool_result` | recipe の既定どおり 400 (`options.rs` の `messages_web_search: Reject`) |

### 7.2 応答 (非ストリーム)

```json
{"id":"msg_…","type":"message","role":"assistant","model":"<要求名>",
 "content":[{"type":"thinking","thinking":"…","signature":"…"},
            {"type":"text","text":"…"},
            {"type":"tool_use","id":"toolu_msg_…_0","name":"…","input":{…}}],
 "stop_reason":"tool_use","stop_sequence":null,
 "usage":{"input_tokens":…,"output_tokens":…,"cache_read_input_tokens":…,"cache_creation_input_tokens":0}}
```

`stop_reason` の写像 (`recipe/protocol/anthropic/messages/response/schema.rs:81-91`):

| ds41rt / proxy | `stop_reason` |
|---|---|
| `stop` (EOS) | `end_turn` |
| proxy が stop_sequences に一致 | `stop_sequence` (`stop_sequence` に一致した文字列) |
| `length` | `max_tokens` |
| `tool_calls` | `tool_use` |
| `content_filter` | `refusal` |

### 7.3 stop_sequences

- ds41rt に `stop` を送ると、一致した文字列が出力から除かれ、`finish_reason:"stop"` になる ([A] §3)。EOS と区別できず、`stop_sequence` を返せない。
- そのため Messages では proxy が本文 (text の差分) を照合する。最長の stop 文字列の長さ − 1 文字だけを保留し、一致したらそこで切り、上流の接続を閉じて生成を止める ([A] §9.4)。思考とツール呼び出しは照合しない (ds41rt と同じ)。
- 上限 16 個 (ds41rt と同じ)。

### 7.4 ストリーミングのイベント列

recipe の `MessagesChunkGenerator` (`chunk_generator.rs:99-207`) に従う。thinking on で、本文とツール 1 本の例:

```
event: message_start          {"type":"message_start","message":{…,"content":[],"usage":{"input_tokens":N,"output_tokens":0,…}}}
event: content_block_start    {"index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}
event: ping
event: content_block_delta    {"index":0,"delta":{"type":"thinking_delta","thinking":"…"}}   × n
event: content_block_delta    {"index":0,"delta":{"type":"signature_delta","signature":"…"}}
event: content_block_stop     {"index":0}
event: content_block_start    {"index":1,"content_block":{"type":"text","text":""}}
event: content_block_delta    {"index":1,"delta":{"type":"text_delta","text":"…"}}   × n
event: content_block_stop     {"index":1}
event: content_block_start    {"index":2,"content_block":{"type":"tool_use","id":"toolu_…","name":"…","input":{}}}
event: content_block_delta    {"index":2,"delta":{"type":"input_json_delta","partial_json":"…"}}   × n
event: content_block_stop     {"index":2}
event: message_delta          {"delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":N,"output_tokens":M,"cache_read_input_tokens":K}}
event: message_stop
```

- block は逐次に開いて閉じる (前の block の stop の後に次の start)。
- **thinking の signature は非空**: `ds41gw-sig-v1.` + HMAC-SHA256 (§10.2 と同じ鍵) の先頭 16 バイトの base64url。recipe の既定 (応答 id。`chunk_generator.rs:34-38`) を `with_signature` で置き換える。入力の signature は検証しない (改ざんされても思考の本文を使うだけである)。
- **usage**: `message_start` の `input_tokens` は、proxy が tokenizer で数えた見積もり (§11.6 の計数を流用) を入れる。`message_delta.usage` には ds41rt の最終値 (`input_tokens`、`output_tokens`、`cache_read_input_tokens`) を入れる。
- keep-alive: 15 秒間イベントが無ければ `event: ping\ndata: {"type":"ping"}\n\n`。
- 途中失敗: `event: error\ndata: {"type":"error","error":{"type":"api_error","message":"…"}}` を送って閉じる。待ち行列の時間切れは HTTP 529 `overloaded_error`。

### 7.5 `/v1/messages/count_tokens`

- 要求は `/v1/messages` と同じ形 (max_tokens は不要)。§7.1 の変換で `Conversation` を作り、`render_conversation` で描画した文字列を `snap/tokenizer.json` で数える。
- 応答: `{"input_tokens": N}`。ds41rt は呼ばない。
- 画像は 1 枚あたりの固定値 (設定 `tokens.image_estimate`、既定 1,024) で概算する。

---

## 8. `/v1/completions` [任意、既定 on]

- 実クライアントの使用は 0 件 ([C] §2.1)。クライアント側の中継が通す口なので用意する ([C] §2.3)。
- **ds41rt に生のプロンプトを流す経路は無い** ([A] §2: `/v1/completions` は 404、route は `/v1/chat/completions` だけ)。そこで chat に包む擬似実装とする。
  - `prompt` (文字列) → user メッセージ 1 件。配列 (複数プロンプト) は §5.5 と同じく本数に分ける。トークン id の配列は 400。
  - 思考は既定で disabled (`completions.thinking=false`)。生の続きを期待するクライアントのためである。
  - `max_tokens` の既定は 16 (OpenAI の legacy と同じ)。`temperature`、`top_p`、`stop`、`seed`、`n` は Chat と同じ。
  - `echo`、`suffix`、`logprobs`、`best_of>1` は 400。
- 応答: `{"id":"cmpl-…","object":"text_completion","created","model","choices":[{"index","text","logprobs":null,"finish_reason"}],"usage"}`。ストリームは同じ形のチャンクと `[DONE]`。
- 注意: 内部ではチャットテンプレートが掛かるので、厳密な補完 (FIM や文の途中からの続き) にはならない。応答ヘッダ `x-ds41gw-warning: completions_emulated_via_chat` を付ける。

---

## 9. `/v1/models`、ヘルスチェック、その他の口

### 9.1 `/v1/models`

OpenAI と Anthropic の両 SDK が読める形にする (フィールドの和集合)。

```json
{"object":"list",
 "data":[{"id":"GLM-5.3-Flash","object":"model","type":"model","created":1792339106,"created_at":"2026-10-19T00:00:00Z",
          "owned_by":"ds41gw","display_name":"DeepSeek-V4.1-Flash (alias GLM-5.3-Flash)",
          "root":"deepseek-ai/DeepSeek-V4.1-Flash","max_context_tokens":1048576,"max_output_tokens":393216}, …],
 "has_more":false,"first_id":"…","last_id":"…"}
```

- 正規名と §4.1 の別名を全部載せる。`max_context_tokens` / `max_output_tokens` は起動時と 60 秒ごとに ds41rt の `/v1/models` から取る ([A] §2)。
- `GET /v1/models/{id}`: 1 件。無ければ 404 (`alias_mode:any` でも一覧にある名前だけ)。

### 9.2 ヘルスチェック

| パス | 内容 |
|---|---|
| `GET /health` | proxy が動いていれば 200 `{"status":"ok"}` (liveness)。docker の HEALTHCHECK に使う |
| `GET /health/ready` | ds41rt の `/health` (2 秒で時間切れ) が 200 で、待ち行列が満杯でなければ 200。それ以外は 503 と理由 (`upstream_down`、`queue_full`) |
| `GET /metrics` | Prometheus 形式 (§13.4) |
| `GET /v1/proxy/stats` | proxy の統計 JSON と、ds41rt の `/v1/stats` (`http_queue_*`、`totals`。[A] §2) の写し |

### 9.3 その他

- `GET /get_model_info`: 実装しない (404)。vLLM 期に 404 を 2,009 回返しても支障が無かった ([C] §2.1)。
- `OPTIONS *`: CORS が有効なときだけ応答する (§12.2)。
- その他の未定義のパス: 404 (§4.6)。

---

## 10. reasoning の保存・再注入と encrypted_content

### 10.1 再注入ストア [必須、既定 on]

- **目的**: 履歴に思考を返さないクライアントに、生成時の思考を補う。型 A は 61/61 件で返していない ([O] §1.1)。補うと prefix キャッシュの命中が直前のターンの分だけ増え (1,515 → 1,569 tok、[O] §2)、モデルが生成時と同じ文脈を見る。
- **保存**: 上流のストリームから、応答ごとに思考の全文を取り、次のキーで保存する。
  - tool_calls がある応答: 各 tool_call id (Chat は ds41rt の `call_chatcmpl-…`、Responses は recipe の `call_resp_…`、Messages は `toolu_…`)。
  - tool_calls が無い応答: 本文 (content) の前後の空白を除いた SHA-256。
- **注入**: thinking on の要求で、assistant に `reasoning_content` が無い (か空の) とき、最初の tool_call id か本文のハッシュで引いて補う。Responses で reasoning 項目が欠けた function_call も同じ。
- **注入しない**: clear_thinking (§4.3.4)、thinking off、`reasoning.reinject=false`。
- **容量**: プロセス内の LRU。既定 512 MiB / 24 時間。再起動で消える (消えても応答は正しく、キャッシュ命中が減るだけである)。
- 命中率を計測する (§13.4)。

### 10.2 自前の encrypted_content [必須]

- **形式**: `ds41gw1.<payload>.<mac>`。
  - `payload` = base64url(zstd(JSON `{"v":1,"t":<思考本文>,"ts":<unix 秒>}`))。
  - `mac` = base64url(HMAC-SHA256(key, payload) の先頭 16 バイト)。
  - 鍵は `state/secret.key` (32 バイト、初回起動時に生成、0600)。設定で指定もできる。
- 暗号化はしない (思考はクライアントに平文でも返しているので、秘匿の必要が無い)。改ざんの検出だけを行う。
- **復号**: mac が合わなければ空として扱い、ログに `bad_encrypted_content` を残す (400 にはしない。鍵を作り直した後の古いセッションで会話を止めないため)。
- `sglang-reasoning-v1.` 形式も復号する (§6.5)。
- 大きさ: 思考 数十 KB で、zstd 後 base64 で数 KB〜十数 KB。

---

## 11. 同時実行、待ち、タイムアウト、キャンセル

### 11.1 上流の枠

- ds41rt の実行枠は 16 ([A] §9.1)。proxy は上流への同時要求を `upstream.max_concurrency = 16` に制限する (tokio の Semaphore)。
- 1 要求が使う枠は 1 (n>1 では n。§5.5)。ストリームが終わるか切断で解放する。
- 16 に絞るので、通常は ds41rt の待ち行列 (バッファ 16、待機者 16、25 秒) を使わない。ds41rt の 429 は、proxy を通らない直接のアクセス (ds41rt は `0.0.0.0:8000` で待ち受けている。`ss -ltnp`) と KV の保留 ([A] §9.1) でだけ起こる見込みである。

### 11.2 proxy の待ち行列

- FIFO。設定は `queue.max_waiting = 256`、`queue.timeout = 600s`。
  - 待機者が 256 を超えたら、即座に 429 (`Retry-After: 5`)。Messages は 529。
  - 600 秒待っても枠が空かなければ 429 (`Retry-After` は直近の平均待ち時間から計算)。
- 優先度は持たない (初版)。API キーごとの上限 (§12.1) は、全体の枠の手前で数える。
- 型 A の同時 12〜16 本の非ストリーム要求は、すべて待ち行列で吸収できる ([C] §5)。

### 11.3 上流の 429 と接続失敗の再試行

- ds41rt が 429 を返した場合 (応答のバイトを 1 つも受け取っていない): `Retry-After` (1 秒) を下限に、0.5 → 1 → 2 → 4 秒 (上限 4 秒) の間隔で再試行する。合計の待ちは `queue.timeout` の残りまで。この間は枠を返し、待ち行列の先頭に戻す。
- 接続の拒否・リセット (ds41rt の再起動中など): 同じく再試行するが、合計 30 秒まで。それを超えたら 503。
- ds41rt の 503 (`worker queue is closed`): 再試行せず 503。

### 11.4 タイムアウトと keep-alive

- ds41rt にはサーバー側のタイムアウトも heartbeat も無い ([A] §9.4)。待ち行列とプレフィルの間は何も流れない ([A] §7.2)。新しい要求のプレフィル中は他の要求のデコードも止まる (`scheduler.rs:242-243`)。
- proxy の時間制限:

| 項目 | 既定 | 内容 |
|---|---|---|
| `upstream.connect_timeout` | 2 秒 | |
| `upstream.header_timeout` | なし | ds41rt は受け入れまでヘッダを返さない ([A] §7.2)。待ちは ds41rt の 25 秒に任せる |
| `upstream.idle_timeout` | 600 秒 | 上流から 1 バイトも来ない時間の上限。超えたら途中失敗 (§11.5) として扱い、接続を切る |
| `upstream.total_timeout` | なし | 長い思考 (数十万トークン) を許す |
| `client.keepalive_interval` | 15 秒 | ストリームで、クライアントに何も送らない時間がこれを超えたら keep-alive を送る (Chat / Responses は SSE コメント、Messages は ping) |

- **早期ヘッダ**: ストリーム要求が proxy の待ち行列で `client.early_headers_after = 10 秒` を超えたら、200 と SSE のヘッダを先に送り、keep-alive を始める。その後に上流がエラーを返したら、ストリーム内のエラー (§11.5 の形) で伝える。クライアントの read timeout に掛からないようにするためである。
- 非ストリーム: 完了までバイトが届かない ([A] §9.4)。設定 `client.nonstream_whitespace_keepalive = false` を true にすると、15 秒ごとに空白 1 文字を本文の先頭に送る (JSON として有効)。ただし HTTP の状態は 200 に確定するので、エラーも 200 + エラー JSON になる。既定は off。

### 11.5 途中失敗の検出と伝搬 [必須]

- ds41rt は生成の途中で失敗すると、エラーイベントも `[DONE]` も送らずに接続を切る ([A] §7.2)。
- 検出: `finish_reason` を持つチャンクの前に EOF か読み取りエラーが来たら、途中失敗とする。`finish_reason` の後に `[DONE]` が無いだけなら正常終了として扱う。
- クライアントへの伝え方:

| API | ストリーム | 非ストリーム |
|---|---|---|
| Chat | `data: {"error":{"message":"upstream stream terminated unexpectedly","type":"server_error","code":"upstream_disconnected"}}\n\n` を送り、`[DONE]` を送らずに閉じる | 再試行 (下記) の後、502 |
| Responses | `response.failed` (§6.8.5) | 502 |
| Messages | `event: error` (`api_error`) | 500 `api_error` |
| completions | Chat と同じ | 502 |

- 非ストリームでは、クライアントにまだ何も送っていないので、`upstream.retry_midstream_nonstream = 1` 回だけ最初から再試行する。ストリームでは再試行しない (出力が二重になるため)。

### 11.6 文脈長の事前検査 [任意、既定 on]

- ds41rt はプロンプトが 1,048,576 トークン以上だと 400 ではなく 500 を返す ([A] §8)。
- proxy は本文の文字数が 300 万字を超える要求だけ、`render_conversation` + tokenizer で数え、`max_context_tokens` (§9.1) 以上なら 400 `context_length_exceeded` を返す。それ以下は数えない (遅延を増やさないため)。

### 11.7 ツール呼び出しの上限とループ遮断 [任意]

- ds41rt には 1 応答あたりのツール呼び出し数の上限が無い ([A] §6.2)。GLM 時代は `GLM_MAX_TOOL_CALLS=8` で抑えていた (`$GLM53/patches/protocol.py:2366-2374`)。
- `limits.max_tool_calls_per_response = 32` (0 で無制限)。Responses の `max_tool_calls` があればその値。上限に達したら、それ以降の tool_calls を捨て、上流を切って `finish_reason:"tool_calls"` で閉じる。ログに `tool_call_cap` を残す。
- 反復の検出 (同じ n-gram が続く) による遮断は第 2 版とする。

### 11.8 キャンセル [必須]

- クライアントが切断したら、上流の reqwest のストリームを drop して TCP を閉じる。ds41rt は各トークンの前に切断を検査し、生成を止める ([A] §9.4、実測で 1 秒後に停止)。
- 待ち行列にいる間の切断: 待ちから外し、上流には送らない。
- 非ストリームでもクライアントの切断を検知する。axum のハンドラの future が drop されることを試験で確かめる (§16.2 の T-M6)。検知できない場合は、接続の読み取り側を監視するタスクを足す。
- n>1 では全本を止める。

---

## 12. 認証と CORS

### 12.1 API キー [任意、既定 off]

- GLM 時代はキーを使っていない ([C] §2.1)。クライアント側の中継は上流に `Authorization` を渡さない ([C] §2.3)。
- 設定 `auth.keys = [{name, sha256, max_concurrency?}]`。空なら認証しない。
- 受け取るヘッダ: `Authorization: Bearer <key>` と `x-api-key: <key>` (Anthropic)。照合は SHA-256 の定数時間比較。
- `auth.allow_without_key = ["127.0.0.1/32", "172.17.0.0/16"]` の送信元はキーなしで通す (中継やローカルのコンテナのため)。
- `max_concurrency` を持つキーは、そのキーの同時数をこの値までにする (§11.2)。
- ログにはキーの `name` だけを書く (§13.3)。

### 12.2 CORS [任意、既定 off]

- 設定 `cors.allow_origins = []` (空で無効)。値があれば、該当する Origin にだけ `Access-Control-Allow-Origin`、`-Allow-Headers: authorization, content-type, x-api-key, anthropic-version, anthropic-beta, openai-beta`、`-Allow-Methods: GET, POST, DELETE, OPTIONS`、`-Expose-Headers: x-request-id, x-ds41gw-warning, retry-after` を返す。`OPTIONS` の preflight は 204。

---

## 13. ログと計測

### 13.1 要求ログ [必須]

- 置き場: `proxy/logs/req/YYYYMMDD.jsonl` (UTC の日付で切り替え)。前日分は zstd で圧縮する。
- **replay_openai.py で再生できる形**にする。replay は `"request.received.openai"` を含む行の最初の `{` から JSON を読み、`obj` を Chat 本文として使う (`$GLM53/replay_openai.py:9-24`)。

受信の行 (1 要求 1 行):

```json
{"event":"request.received.openai","ts":"2026-10-19T03:12:45.123Z","rid":"gw_…",
 "api":"responses","path":"/v1/responses","method":"POST",
 "client_ip":"100.x.y.z","user_agent":"…","key":"<キー名>|null",
 "stream":true,"model_requested":"GLM-5.3-Flash",
 "prompt_cache_key":"…","session_hint":"…",
 "obj":{ …ds41rt に送った Chat 本文 (正規化後)… },
 "orig":{ …受け取った元の本文… },
 "normalized":["model_alias","temperature_default","developer_to_system","tool_reference_to_text", …]}
```

終了の行:

```json
{"event":"request.finished","ts":"…","rid":"gw_…","status":200,"upstream_status":200,
 "queue_ms":12,"ttft_ms":840,"dur_ms":15230,"upstream_retries":0,
 "finish_reason":"tool_calls","usage":{…},"reasoning_tokens":512,
 "tool_calls":["bash","read"],"reinjected":3,"warnings":[…],"error":null,
 "client_disconnected":false}
```

- `orig` は Responses / Messages の生 JSON を残すためのものである。今回の調査で欠けていた情報 ([C] §1(c)、(d)) を補う。
- 設定 `log.bodies = "full" | "meta"`。`meta` では `obj` と `orig` を書かない (再生はできなくなる)。既定 `full`。
- 出力の本文は既定で書かない (`log.outputs = false`)。true にすると、終了の行に content / reasoning / tool_calls の全文を加える。

### 13.2 容量

- GLM 時代の reqlog は 9 日で 19 GB だった ([C] §1)。
- 画像の `data:` URL は、既定でハッシュと大きさ (`{"sha256":…,"bytes":…,"mime":…}`) に置き換える (`log.images = "hash" | "full"`)。再生には full が要る。
- 保存期間 14 日 (`log.retention_days`)、合計の上限 50 GiB (`log.max_total_gb`)。超えたら古い日から消す。

### 13.3 秘匿の扱い

- 要求ログと状態保存には、プロンプト全文 (ソースコード、ファイルの中身、bash の出力) が入る。秘密情報が混ざりうる前提で扱う。
- ディレクトリ `logs/` と `state/` は 0700、ファイルは 0600 (コンテナは uid 1000 で動かす)。
- `Authorization`、`x-api-key`、`Cookie` ヘッダは書かない。キーは名前だけを書く。
- ログを外部の監視に送るのは計測値 (§13.4) だけで、本文は送らない。
- 調査資料に引用するときは、[C] §4.2 と同じく伏せ字にする。

### 13.4 計測 (`/metrics`)

- `ds41gw_requests_total{api,stream,status}`、`ds41gw_queue_waiting`、`ds41gw_queue_wait_seconds` (ヒストグラム)、`ds41gw_upstream_inflight`、`ds41gw_upstream_retries_total{reason}`、`ds41gw_midstream_failures_total`、`ds41gw_client_disconnects_total`。
- `ds41gw_ttft_seconds{api}`、`ds41gw_overhead_seconds{api}` (proxy 内で費やした時間: 受信から上流送信まで + 上流の最初のバイトからクライアントへの最初のバイトまで)。
- `ds41gw_reinject_total{result=hit|miss}`、`ds41gw_normalizations_total{kind}`、`ds41gw_dsml_leak_total`、`ds41gw_tool_call_cap_total`。
- ds41rt の `/v1/stats` (`http_queue_waits`、`http_queue_rejects`、`http_queue_len`、`totals`) を 10 秒ごとに取り、`ds41rt_*` として出す。

---

## 14. 設定ファイル

`proxy/config/ds41gw.toml` (例。既定値を書いたもの):

```toml
[server]
listen = ["127.0.0.1:30001", "<HOST_IP>:30001"]   # 外向きのアドレス。GLM 時代は外向きのアドレスと 127.0.0.1 だけに出していた
max_body_mb = 96

[upstream]
url = "http://127.0.0.1:8000"
model = "deepseek-ai/DeepSeek-V4.1-Flash"
max_concurrency = 16
connect_timeout_s = 2
idle_timeout_s = 600
retry_midstream_nonstream = 1

[queue]
max_waiting = 256
timeout_s = 600

[client]
keepalive_interval_s = 15
early_headers_after_s = 10
nonstream_whitespace_keepalive = false

[models]
alias_mode = "any"
aliases = ["DeepSeek-V4.1-Flash", "deepseek-v4.1-flash", "GLM-5.3-Flash"]
echo_requested = true

[sampling]
default = "recommended"          # temperature 1.0 / top_p 1.0。"greedy" で ds41rt の素の挙動  [要決定]

[thinking]
default = "enabled"
default_effort = "high"
effort_map = { none = "none", minimal = "low", low = "low", medium = "high", high = "high", xhigh = "max", max = "max" }
auto_off_below_max_tokens = 0

[limits]
default_max_tokens = 262144      # [要決定]
max_n = 8
max_tool_calls_per_response = 32
image_overflow = "drop_oldest"

[compat]
unsupported = "reject"
repair_tool_history = true
tool_none_strategy = "flatten"
strip_dsml = true
null_content_for_tool_calls = false
ignore_builtin_tools = []

[reasoning]
output_mode = "auto"
always_encrypted = false
reinject = true
reinject_store_mb = 512
reinject_ttl_h = 24

[state]
backend = "sqlite"
path = "/var/lib/ds41gw/responses.sqlite"
memory_items = 2000
ttl_days = 30
max_gb = 20

[tokens]
tokenizer = "/models/tokenizer.json"
image_estimate = 1024

[log]
dir = "/var/log/ds41gw"
bodies = "full"
outputs = false
images = "hash"
retention_days = 14
max_total_gb = 50

[auth]
keys = []
allow_without_key = ["127.0.0.1/32", "172.17.0.0/16"]

[cors]
allow_origins = []
```

---

## 15. 起動と配備

### 15.1 置き場

```
proxy/
  ds41gw/            Cargo プロジェクト (src/, tests/, Cargo.toml, Cargo.lock)
  docker/Dockerfile
  config/ds41gw.toml
  run_ds41gw.sh      ビルドと起動
  state/             responses.sqlite、secret.key (0700)
  logs/req/          要求ログ (0700)
  tests/             SDK 試験 (Python)、再生の補助スクリプト
  r7_eval/           既存の評価スクリプト
```

src の構成: `main.rs`、`config.rs`、`upstream.rs` (クライアント・待ち行列・再試行)、`sse.rs` (上流の解析、`OutputChunk` への写像)、`chat/{normalize,stream,aggregate,n}.rs`、`responses/{request,reverse,events,store,api}.rs`、`messages/{request,events,count_tokens}.rs`、`completions.rs`、`models.rs`、`reasoning/{reinject,encrypted}.rs`、`names.rs` (ツール名の写像)、`dsml.rs`、`log.rs`、`metrics.rs`、`auth.rs`。

### 15.2 docker で動かす

- ホストに cargo が無い。recipe は edition 2024 なので、Rust 1.85 以上が要る ([O] §3.4)。多段ビルドにする。
  - ビルド: `rust:1.86-bookworm` (要 pull)。`cargo build --release --locked`。
  - 実行: `debian:bookworm-slim` + `ca-certificates`。
- 起動 (`run_ds41gw.sh` の中身の要点):

```sh
sudo docker build -t ds41gw:0.1.0 -f proxy/docker/Dockerfile proxy
sudo docker run -d --name ds41gw --network host --restart unless-stopped --init \
  --user 1000:1000 --stop-timeout 30 \
  -v $DSV41/proxy/config:/etc/ds41gw:ro \
  -v $DSV41/proxy/state:/var/lib/ds41gw \
  -v $DSV41/proxy/logs:/var/log/ds41gw \
  -v $HF_HOME/hub/models--local--DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1-l2e/snapshots/1017e3c2000000000000000000000000000000e2/tokenizer.json:/models/tokenizer.json:ro \
  ds41gw:0.1.0 --config /etc/ds41gw/ds41gw.toml
```

- `--network host` にする理由: ds41rt と同じく host ネットワークで動いている (`docker inspect`: NetworkMode host)。ポート転送を挟まないので、送信元 IP がログに残る (GLM 時代は全件 `172.17.0.1` で、判別に使えなかった。[C] §1(d))。
- 再起動の方針: `--restart unless-stopped`。proxy は状態を sqlite に置くので、再起動しても previous_response_id は続く (再注入ストアは消える)。
  - ds41rt のコンテナは `RestartPolicy: no` である (`docker inspect`)。ds41rt が落ちても proxy は動き続け、`/health/ready` が 503、要求は 503 になる。
- Tailscale の IP に bind するので、tailscaled が上がる前に起動すると bind に失敗して落ちる。restart policy で再試行されるので、そのままでよい。
- 停止: `SIGTERM` を受けたら新しい要求を断り (503)、実行中のストリームの終了を最大 30 秒待つ。
- ログの確認: `sudo docker logs ds41gw` (起動・エラー)、要求ログは §13.1。

systemd で動かす案も考えられるが、cargo がホストに無く、ds41rt も docker なので docker に揃える。

### 15.3 口の切り替え [要決定]

- 初版は `:30001` で待ち受け、試験は `:30001` で行う。
- GLM 時代のクライアントは `:30000` を向いている (GLM の起動スクリプトと中継の上流 URL。[C] §2.3)。`:30000` は今は空いている (`ss -ltnp`)。
- 切り替えの方法は 2 つ。どちらを採るかはユーザーが決める。
  1. 設定の `listen` に `127.0.0.1:30000` と外向きのアドレスの `:30000` を足して再起動する (30001 と併存できる)。
  2. クライアント側の base URL を `:30001` に変える。
- ds41rt の `0.0.0.0:8000` は、proxy を経由しない直接のアクセスを許している。ds41rt の次の計画的な再起動の時に `127.0.0.1:8000` に絞るかどうかも、ユーザー判断とする (ds41rt は今は止めない)。

---

## 16. テスト計画

制約: ds41rt を止めない。ds41rt への試験は同時 4 本以内、`max_tokens` は小さく (原則 ≤ 256、思考ありは effort low か ≤ 512)。GPU を使う別作業はしない。高い同時数の試験は mock で行う。

### 16.1 単体試験 (`cargo test`、ds41rt 不要)

| ID | 内容 | 合格条件 |
|---|---|---|
| T-U1 | Chat の正規化 (§5.3) の表駆動試験。[C] §3.3 の `tool_reference` 実例、developer、null content、arguments の object、`["object"]` スキーマ、名前 `a.b`、`stream:false`+`stream_options`、`thinking:true`、`chat_template_kwargs.enable_thinking:false`、`reasoning_effort:"HIGH"`/`3`/`"xhigh"`、`max_completion_tokens` | 期待する上流本文と一致 |
| T-U2 | ツール履歴の補修 (欠けた結果、間の user、孤立した tool、重複 id) | 出力が ds41rt の規則 ([A] §4.2) を満たす (recipe の `ChatCompletionRequest::convert` が Ok を返す) |
| T-U3 | **逆写像の無損失性**: Responses / Messages 要求 → `Conversation` (A) → Chat → recipe の Chat 変換 → `Conversation` (B) として、`render_conversation(A) == render_conversation(B)`。[C] §4.2 の例 1〜3 の形、並列関数呼び出し、reasoning、画像、developer + instructions、namespace、apply_patch | 描画したプロンプトが一致 ([O] §6 のリスク) |
| T-U4 | イベント順序の検査器: `sequence_number` の単調増加、開いている項目が常に 1 つ以下、added/done の対、content_part の対、終端イベントがちょうど 1 つ | recipe の出力と proxy の後処理 (summary への置換、usage の差し替え) の両方で通る |
| T-U5 | encrypted_content: 往復、改ざん (空として扱う)、`sglang-reasoning-v1.` の復号 | |
| T-U6 | 状態保存: 3 段の連鎖、store:false の親 (400)、期限切れ (400)、DELETE 後の参照 (400)、input_items のページ送り | |
| T-U7 | DSML フィルタ: 差分の境界で `<｜DSML｜` が割れる場合 | 漏れない、通常の文字は遅延 16 文字以内 |
| T-U8 | stop_sequences (Messages): 境界をまたぐ一致 | `stop_reason:"stop_sequence"` と正しい切り位置 |

### 16.2 mock 上流での結合試験 (ds41rt 不要)

`proxy/r7_eval/mock.py` を拡張し、次の動作を選べるようにする: 並列 16 を超えたら 429、生成途中の切断、プレフィル相当の無音 (30 秒)、並列ツール、`length` での打ち切り、tool_calls だけの応答。

| ID | 内容 | 合格条件 |
|---|---|---|
| T-M1 | 同時 32 本の非ストリーム + tools (mock の上限 16) | クライアントへの 429 が 0 件、全件 200 |
| T-M2 | 同時 16 本のストリームで追加遅延 (`bench.py`) | TTFT p50 +5 ms 以内、差分間隔 p99 の悪化 5 ms 以内 ([O] §3.4 の目標) |
| T-M3 | 途中切断 | Chat はエラーイベント (`[DONE]` 無し)、Responses は `response.failed`、Messages は `event: error`。非ストリームは 1 回再試行後に 502 |
| T-M4 | 30 秒の無音 | 15 秒ごとに keep-alive、早期ヘッダが 10 秒で出る |
| T-M5 | ストリームのクライアント切断 | mock が 100 ms 以内に切断を観測する |
| T-M6 | 非ストリームのクライアント切断 | 同上 (§11.8) |
| T-M7 | 上流の 429 が 3 回続いた後に成功 | クライアントは 200、`upstream_retries=3` がログに残る |
| T-M8 | ds41rt 停止の模擬 (接続拒否) | 30 秒再試行の後 503、`/health/ready` が 503 |

### 16.3 実機 (ds41rt) での SDK 試験

`proxy/tests/` に pytest で置く。SDK の版は固定する (`openai`、`anthropic` の、試験時点の最新版を `requirements.txt` に記録する)。逐次実行か同時 4 本以内、`max_tokens` は小さく。

OpenAI SDK (Chat):

| ID | 内容 |
|---|---|
| T-C1 | 非ストリーム / ストリーム、`model="GLM-5.3-Flash"` (別名)、temperature 省略 |
| T-C2 | ツール往復 2 ターン (並列 2 本)。2 ターン目で reasoning を返さない場合に再注入が効く (`prompt_cache_hit_tokens` が増える。[O] §2 の `cache_rt.py` と同じ測り方) |
| T-C3 | 画像 (data: URL、2 枚) と、tool メッセージの画像 |
| T-C4 | `response_format` json_schema (strict) |
| T-C5 | `tool_choice:"none"` + ツール履歴で DSML が漏れない |
| T-C6 | `stream_options.include_usage` で `choices:[]` の usage チャンクが来る。`reasoning_tokens` が入る |
| T-C7 | `n=2`、`logprobs:true` (400)、`chat_template_kwargs.enable_thinking:false` で reasoning_content が出ない |

OpenAI SDK (Responses):

| ID | 内容 |
|---|---|
| T-R1 | `client.responses.create` 非ストリーム、`instructions` + 文字列 input |
| T-R2 | `client.responses.stream()` で `get_final_response()` が組み立てられ、T-U4 の検査器を通る |
| T-R3 | 関数呼び出しの往復 (並列 2 本)。`include=["reasoning.encrypted_content"]`、`store=False` で、返した reasoning 項目 (content を削って encrypted_content だけにした場合も) で思考が引き継がれる |
| T-R4 | `previous_response_id` で 3 ターン (名前を覚えているか)。GET、input_items、DELETE 後の参照が 400 |
| T-R5 | `text.format` json_schema、`reasoning.effort` none / low / xhigh、`reasoning.summary:"auto"` で summary イベント |
| T-R6 | 組み込みツール (`web_search`) が 400、`background:true` が 400 |
| T-R7 | `input_image`、`function_call_output` の画像 |

Anthropic SDK:

| ID | 内容 |
|---|---|
| T-A1 | `messages.create` 非ストリーム、system あり |
| T-A2 | `messages.stream()` (thinking enabled、budget_tokens 2048) で `get_final_message()` が組み立てられる。thinking の signature が非空 |
| T-A3 | tool_use → tool_result の往復 (並列 2 本、`is_error` あり) |
| T-A4 | 画像 (base64) |
| T-A5 | `stop_sequences` で `stop_reason:"stop_sequence"` |
| T-A6 | `count_tokens` の値と、実際の `input_tokens` の差が 5% 以内 |

その他:

| ID | 内容 |
|---|---|
| T-X1 | `/v1/completions` の非ストリーム / ストリーム |
| T-X2 | `/v1/models` を OpenAI SDK と Anthropic SDK の両方で列挙できる |
| T-X3 | 適合試験: Open Responses の `compliance-tests.ts` (`research/tmp/r7/openresponses_compliance-tests.ts`) を node のコンテナで実行する ([O] §3.3) |
| T-A9 | 型 E (短い要約) の形 (`max_tokens 120`、`enable_thinking:true`、effort low) とタイトル生成の形 (`max_output_tokens 64`、effort high) で空応答になるか (§4.3.5 の判断材料) |

### 16.4 GLM 要求ログの再生

1. **互換性の総当たり**: reqlog の chat から、クライアントの型別 ([C] §3.1 の A、B、B + tool_reference、B の補助、C 新旧、D、E) に各 30 件を無作為に選ぶ。proxy 経由で `max_tokens=1` (思考ありは 1 トークンで止まる)、非ストリーム、逐次で送る。合格条件: 全件 200 ([C] §3.4 では tool_reference の 10 件が 400 だった)。80KB 超の要求 (型 D の長い履歴) も含める。
2. **ストリームの再生**: `replay_openai.py` を `--url http://localhost:30001 --conc 4` で動かす。ただし replay は記録の `max_tokens` (32000 など) をそのまま送るので、上限を付ける引数 `--max-tokens 64` を追加した写しを `proxy/tests/replay_capped.py` に置いて使う。合格条件: 全件が `finish_reason` を返し、途中失敗 0 件。
3. **Responses**: [C] §4.2 の例 1〜3 の形 (伏せ字を合成テキストに置き換えたもの) を送り、200 とイベント順序 (T-U4) を確かめる。
4. **proxy 自身の要求ログからの再生**: §13.1 の形が replay で読めることを確かめる (`obj` が Chat 本文)。

### 16.5 実クライアント

実際のコーディングエージェント (OpenAI Responses、OpenAI Chat、Anthropic Messages (`ANTHROPIC_BASE_URL=http://localhost:30001`) の各クライアント) で、ツールを 2〜3 回呼ぶ短い作業を 1 回ずつ行う。ツールの遅延ロード (`tool_reference`) を 1 回使う。

### 16.6 受け入れ条件

- 16.1〜16.4 がすべて合格。
- T-M2 の追加遅延の目標を満たす。
- 実機試験の間、ds41rt の `/v1/stats` の `http_queue_rejects` が増えない。

---

## 17. 実装の順序

| 段 | 内容 | 終わりの条件 |
|---|---|---|
| M1 | 骨組み: 設定、上流クライアント、待ち行列、ログ、`/health`、`/v1/models`、Chat の正規化と素通し (ストリーム / 非ストリーム集約) | T-U1、T-U2、T-M1、T-M3、T-M5、T-C1〜C7、16.4 の 1・2 |
| M2 | Responses (状態なし): 前処理、recipe の変換、逆写像、イベント生成、reasoning の補完、encrypted_content、再注入 | T-U3〜U5、T-R1〜R3、R5〜R7、16.4 の 3、実クライアント |
| M3 | Responses の状態保持 (sqlite、GET / DELETE / input_items) | T-U6、T-R4 |
| M4 | Messages と count_tokens、completions | T-U8、T-A1〜A6、T-X1 |
| M5 | 認証、CORS、`/metrics`、ツール呼び出しの上限、適合試験、口の切り替え (ユーザー判断の後) | T-X2、T-X3、16.5 |

---

## 18. 未確定事項とリスク

### 18.1 ユーザーが決める事項

| 事項 | 提案 | 根拠 |
|---|---|---|
| temperature を省いたときの既定 | `temperature 1.0 / top_p 1.0` を入れる | モデルの推奨値 (`snap/README.md:77-82`)。GLM 時代のクライアントはサーバー既定の 1.0 で動いていた。greedy より約 5% 遅い |
| 出力上限の既定 (`max_tokens` が無い場合) | 262,144 | `snap/README.md:83`。ds41rt の既定 393,216 より KV の予約が小さい |
| `xhigh` の扱い | max (100) | ds41rt 自身は 75。OpenAI の意味に寄せる |
| 待ち受けの口 | 初版 30001。`:30000` への切り替えは試験の後 | §15.3 |
| ds41rt の `0.0.0.0:8000` | 次の計画的な再起動で `127.0.0.1` に絞る | proxy を迂回したアクセスが 429 と KV の保留を起こす |
| 小さな max_tokens での thinking 自動 off | 既定 off。T-A9 の結果で決める | §4.3.5 |
| 要求ログの本文を残すか | `full` (14 日、50 GiB 上限) | §13 |

### 18.2 リスク

- **recipe 0.1.0 が若い。** commit は 2 つで、API が変わる恐れがある ([O] §6)。ds41rt と同じ `=0.1.0` に固定し、ds41rt の版上げに合わせて追従する。
- **逆写像の無損失性は未検証である。** developer の合成、apply_patch の説明文、LatestReminder、画像プレースホルダの分割で、ds41rt が描画するプロンプトが recipe 単体の描画とずれる可能性がある。T-U3 で確かめる ([O] §6)。
- **Responses の実際の要求は推定である。** 生 JSON は残っておらず、`include`・`store`・`text.format`・`truncation`・`tool_choice`・`strict` の実値は分からない ([C] §6)。proxy は未知の値を記録し、運用開始後に `orig` のログで確かめる。
- **プレフィル中の全停止。** 新しい要求のプレフィル中は、両 lane のデコードが止まる (`scheduler.rs:242-243`)。長い文脈のエージェントが 16 本並ぶと、全ストリームが数十秒止まりうる ([A] §9.3)。proxy では keep-alive でクライアントの時間切れを防ぐことしかできない。プレフィルの長さを見て受け入れを遅らせる制御は、効果を測ってから検討する。
- **tool_calls の途中で max_tokens に達する。** finish_reason が length になり、不完全な arguments が返りうる ([A] §7.1)。Chat はそのまま返す。Responses は項目を incomplete で閉じる。
- **`tool_choice:"none"` の平文化はプロンプトを変える。** その要求だけ prefix キャッシュが外れる。DSML フィルタだけで足りるかは、実地の頻度を見て判断する。
- **非ストリームを上流のストリームで集約するので、ds41rt の非ストリーム固有の挙動と差が出る。** 生成後の strict 検証の失敗は、非ストリームでは 500 だが、ストリームでは切断になる ([A] §7.2)。proxy はどちらも途中失敗として扱う ([O] §6)。
- **429 と 25 秒待機の挙動は実地で確かめていない** ([A] §11)。proxy は同時 16 に絞るので、通常は起きない想定である。
- **reasoning_effort の minimal と low は thinking を止めない。** 効果は 50/75/100 の 3 段階しかない ([A] §5)。
- **思考の本文をログと状態保存に残す。** 秘匿の扱いは §13.3 による。
- **[C] §0 の「tool ロールに画像は不可」は誤りだった。** recipe のコードでは tool の画像を受け付ける (§1.2)。function_call_output の画像は、user に移さず tool メッセージに入れる。T-R7 で実地でも確かめる。
