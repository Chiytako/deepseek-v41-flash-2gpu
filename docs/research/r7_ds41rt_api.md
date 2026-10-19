# r7: ds41rt の API 仕様 (稼働中の v15 serve-native を、コードと実地の両方で確定)

作成日: 2026-10-19 / 対象: `localhost:8000` で稼働中の ds41rt (container `ds41rt-coordinator`、image `ghcr.io/tpurtell/ds41rt-coordinator:v15`)
凡例: 【実測】はこの調査で実際に要求を投げて確かめた結果、【コード】はソースから読み取った挙動、【推測】は間接的な根拠による判断。

パスの略記:
- `api/` = `ext/ds41rt/rust/crates/ds41rt-api/src/`
- `daemon/` = `ext/ds41rt/rust/crates/ds41rt-daemon/src/`
- `core/` = `ext/ds41rt/rust/crates/ds41rt-core/src/`
- `recipe/` = `research/tmp/r7/crates/deepseek-recipe-0.1.0/src/`
- `enc/` = `research/tmp/r7/crates/deepseek-recipe-encoding-0.1.0/src/`

検証用の資材:
- 実地試験スクリプト: `research/tmp/r7/probe/probe.py` (約 80 ケース。すべて逐次実行し、max_tokens は 8〜400)
- GLM 時代の要求ログの集計: `research/tmp/r7/probe/reqkeys.py`

---

## 0. 結論 (proxy 設計に効く点)

1. **実際に要求を処理しているのは `api/native_v41.rs` の `chat()` と、外部 crate `deepseek-recipe 0.1.0` の変換器である。** `api/request.rs` と `api/openai.rs` は legacy の `real-ds4-full` 経路のもので、本番では使われない。
   - r2 などで `request.rs` を根拠に書いた挙動は、本番とは違う。たとえば `max_completion_tokens`、`enable_thinking`、`developer` ロール、`top_k ≤ 64` の制約がそうである。
   - 要求の JSON は、まず `serde_json::Value` として受け取る。そこから `top_k`、`min_p`、`seed` などを直接読み、残りを recipe の `ChatCompletionRequest` に deserialize する (`api/native_v41.rs:202-255`)。
2. **未知のフィールドは無視される (400 にはならない)。** ただし、既知のフィールドで型や値が違うと 400 になる。【実測】
   - 無視されて実害が出るものは次の 4 つ。
     - `max_completion_tokens`: 無視されるので、既定の 393,216 で生成される。
     - `chat_template_kwargs`: 無視されるので、`enable_thinking:false` が効かない。
     - `enable_thinking`: 無視される。
     - `reasoning_effort` を数値や大文字で渡す場合: 無視ではなく 400 になる。
3. **思考 (thinking) を制御できるのは `thinking.type` と `reasoning_effort` の 2 つだけである。**
   - 何も指定しなければ thinking は有効で、effort は high (プロンプト上の "Reasoning Effort: 75") になる。
   - effort の値と扱いは次のとおり。
     - `low` と `minimal` は 50 になる。`minimal` でも thinking は止まらない。
     - `medium`、`high`、`xhigh` はすべて 75 になる。
     - `max` は 100 になる。
     - `none` は thinking を無効にする。
4. **`developer` ロールは 400 になる。** 受け付けるロールは `system`、`user`、`assistant`、`tool`、`latest_reminder` の 5 つ。【実測】
5. **`tool_choice:"none"` を指定するか tools を省くと、ツール呼び出しの解析が止まる。** その状態で履歴に tool_calls があると、モデルが DSML の生テキストを `content` に出してしまう。【実測】
   - GLM 時代の要求の 16% (187/1191) が、tools 付きの `tool_choice:"none"` だった (§10)。proxy での対策が必須である。
6. **ツール定義の `parameters` は `"type":"object"` (文字列) でなければ 400 になる。** `{}` や `["object"]` も 400 になる。`parameters` を省いた場合は通る。名前は `[A-Za-z0-9_-]{1,128}` に限られる。
7. **ストリームの末尾の形は OpenAI と少し違う。**
   - usage は `finish_reason` を持つ最終チャンクに同居する。`choices:[]` だけの usage 専用チャンクは無い。
   - `stream_options.include_usage` に関係なく、usage は常に付く。true のときは、途中のチャンクに `"usage":null` が入る。
   - 末尾は `data: [DONE]` で終わる。
   - 生成の途中で失敗すると、**エラーイベントを送らずに接続を切る**。この場合 `[DONE]` も来ない。
8. **usage に `completion_tokens_details.reasoning_tokens` は無い。** 思考のトークンは `completion_tokens` に含まれる。代わりに DeepSeek 形式の `prompt_cache_hit_tokens` と `prompt_cache_miss_tokens` が付く。
9. **同時実行と待ち行列の構成は次のとおり。**
   - 実行枠 16、バッファ 16、待機者 16 (待機の上限は 25 秒)。これを超えた要求は、即座に 429 (`Retry-After: 1`) になる。
   - 待ち行列やプレフィルの間は、HTTP の応答ヘッダも SSE のバイトも何も送られない。
   - 新しい要求のプレフィル中は、他の要求のデコードが止まる【コード】。
   - サーバー側には全体のタイムアウトが無い。クライアントが接続を切ると、生成も止まる【実測】。

---

## 1. 稼働中の版と、読むべきコード

| 項目 | 値 | 根拠 |
|---|---|---|
| image | `ds41rt-coordinator:v15`、`DS41RT_ENGINE_COMMIT=bd06bec42e…` | `sudo docker inspect ds41rt-coordinator` |
| clone の HEAD | `3067d06`。`bd06bec..HEAD` の差分は README、docs、`*.config` だけで、Rust は同一 | `git diff --stat bd06bec HEAD` |
| 起動引数 | `serve-native … --rtx-gpus 2 --listen 0.0.0.0:8000 --prefill-batch-tokens 2048 --concurrency 16 --prefix-cache-entries 8 --max-context-tokens 1048576 --max-output-tokens 393216 --http-queue-depth 16 --http-queue-wait-ms 25000 --rtx-expert-layers 40 --host-cache-bytes 4GiB --dspark` | `/proc/844728/cmdline` |
| HTTP router | `native_v41::router_with_console` | `daemon/v41_native_serve.rs:102` |
| 要求の変換 | `deepseek-recipe =0.1.0`、`deepseek-recipe-encoding =0.1.0`、`deepseek-recipe-core =0.1.0` (crates.io、MIT、repo `deepseek-ai/deepseek-recipe`) | `ds41rt-api/Cargo.toml:9-11` |
| recipe の取得 | crates.io から取得し、sha256 が `Cargo.lock` の checksum と一致 (`e8cc7a0c…`、`7addeead…`、`709b39e0…`) | `research/tmp/r7/crates/` |

- recipe には `/v1/responses` (`recipe/protocol/openai/responses/`) と Anthropic `/v1/messages` (`recipe/protocol/anthropic/messages/`) の変換器も入っている。
  - ds41rt はこれらを route に登録していない。
  - proxy を Rust で書けば、同じ crate を再利用できる【推測】。

---

## 2. エンドポイント【実測】

| パス | メソッド | 内容 |
|---|---|---|
| `/v1/chat/completions` | POST | 本体。GET を送ると 405 (`allow: POST`) |
| `/v1/models` | GET | `{"object":"list","data":[{"id":"deepseek-ai/DeepSeek-V4.1-Flash","object":"model","owned_by":"deepseek-ai","max_context_tokens":1048576,"max_output_tokens":393216}]}` |
| `/health` | GET | worker の queue が閉じていなければ 200、閉じていれば 503 (`api/native_v41.rs:120-126`) |
| `/v1/stats` | GET | 統計の JSON。`totals`、`host_cache`、`dspark_policy`、`http_queue_waits`、`http_queue_wait_ms_sum`、`http_queue_rejects`、`http_queue_len` (`api/native_v41.rs:108-115`) |
| `/`、`/v1/console`、`/v1/console/snapshot` | GET | ライブコンソール (HTML、WebSocket、JSON) (`api/native_v41.rs:94-98`) |
| `/v1/responses`、`/v1/completions`、`/v1/messages` | POST | **404 (本文は空)** |

- 認証は無い。どのヘッダも見ていない。
- 要求本文の上限は 96MiB (`DefaultBodyLimit`。`api/native_v41/images.rs:18`)。
- `content-type: application/json` が無い要求は、415 になる。本文は text/plain で `Expected request with \`Content-Type: application/json\``。

---

## 3. 要求のトップレベルのフィールド

recipe のスキーマ定義は `recipe/protocol/openai/chat_completion/request/schema.rs:12-48` にある。冒頭のコメントにあるとおり、未知のフィールドは無視される。

| フィールド | 型と値域 | 挙動 | 根拠 / 実測 |
|---|---|---|---|
| `model` | string、必須 | `deepseek-ai/DeepSeek-V4.1-Flash` と完全に一致しなければ 400 (`model must be …`)。`-1017` を付けたものも 400 | `api/native_v41.rs:268-270`【実測】 |
| `messages` | array、必須、1 件以上 | 空なら 400 (`Empty input messages`)。詳細は §4 | `recipe/…/convert.rs:331-333` |
| `stream` | bool か null | 省略は false | |
| `stream_options.include_usage` | bool | `stream` が true でないのに付けると **400** (`stream_options should be set along with stream = true`)。未知のキー (例 `continuous_usage_stats`) は無視 | `convert.rs:498-502`【実測】 |
| `max_tokens` | u32 | 0 は 400。負数と小数 (`16.0`) は serde で 400。上限は `min(393216, 1048576 − prompt)` に切り詰める (エラーにはならない)。**省略すると 393,216** | `api/native_v41/limits.rs:42-58`、`daemon/v41_native_serve/scheduler/admission.rs:19`【実測: 10^9 を送っても 200】 |
| `max_completion_tokens` | — | **無視される**。5 を送っても 100 トークン生成した | 【実測】 |
| `temperature` | number、0〜2 | **省略、null、0 のときは greedy (argmax)**。`< 1e-5` も greedy。2 を超えると 400 (`invalid target sampling parameter: temperature`) | `api/native_v41.rs:172`、`core/target_sampling.rs:49,115-117` |
| `top_p` | number、(0,1] | 省略すると 1.0 (無効)。greedy のときは使われない | `api/native_v41.rs:173` |
| `top_k` | integer | `0` と `-1` は無効。正の値はそのまま使う (legacy の上限 64 は本番には無い)。`-2` 以下は 400。`1` は greedy | `api/native_v41.rs:175-189`【実測】 |
| `min_p` | number、0〜1 | 本番は対応している (legacy では拒否していた) | `api/native_v41.rs:174` |
| `seed` | i64 | 負の値も受け付ける (`as u64`)。文字列は 400。省略すると時刻から生成する | `api/native_v41.rs:190-198,249-251`【実測】 |
| `stop` | string か string[] (最大 16) | 17 個以上は 400。照合するのは**回答本文 (と JSON 出力) だけ**で、思考とツール呼び出しの markup は対象外。一致した文字列は出力から除かれ、`finish_reason:"stop"` になる | `convert.rs:29,490-497`、`recipe/stream/state_machine.rs:43-46`、`processor.rs:132-135,151-152`【実測】 |
| `n` | u8 | 1 以外は 400 (`Invalid n value …`) | `convert.rs:101-105`【実測】 |
| `frequency_penalty`、`presence_penalty` | number、[-2,2] | 範囲だけ検査し、**その後は無視される** (範囲外は 400) | `convert.rs:106-115`【実測】 |
| `repetition_penalty`、`logprobs`、`top_logprobs`、`user`、`metadata`、`store` など | — | 無視される。応答の `logprobs` は常に null | 【実測】 |
| `response_format` | `{"type":"text"\|"json_object"\|"json_schema"\|"regex"}` | §6.3 | |
| `tools`、`tool_choice`、`parallel_tool_calls` | — | §6 | |
| `tool_decoding_assistance` | bool (ds41rt の拡張) | 既定は true。bool 以外は 400 | `api/native_v41.rs:203-207` |
| `thinking` | `{"type":"enabled"\|"adaptive"\|"disabled"}` | §5。それ以外の値 (例 `auto`) は 400。`budget_tokens` は無視 | `schema.rs:205-220`【実測】 |
| `reasoning_effort` | `none\|minimal\|low\|medium\|high\|xhigh\|max` (小文字の文字列だけ) | §5。`HIGH`、`ultra`、数値は 400 | `schema.rs:225-235`【実測】 |
| `enable_thinking`、`chat_template_kwargs` | — | **無視される** | 【実測】 |
| `min_tokens`、`ignore_eos` | — | 本番では無視される (legacy の `request.rs` 専用) | `api/openai.rs:39-46` は legacy |

### 3.1 型エラーの文言

serde の deserialize エラーは、そのまま 400 の `message` になる。512 文字を超える部分は切り捨てる (`api/error.rs:70-79`)。

文言の例:
- `missing field \`model\``
- `invalid type: floating point \`16.0\`, expected u32`
- `unknown variant \`developer\`, expected one of …`
- `data did not match any variant of untagged enum ChatCompletionRequestContent`

---

## 4. messages の形

ロールの enum は `#[serde(tag="role")]` で定義されている (`schema.rs:51-71`)。

| ロール | 必須の項目 | 備考 |
|---|---|---|
| `system` | `content` (string か text ブロックの配列) | 画像を含むと 400 (`Image in system message is unsupported`)。空文字の system は捨てる (`convert.rs:343-347`)。途中に置いた system もそのまま描画する (`enc/v4/dsv41.rs:48-50`) |
| `user` | `content` (string か ブロックの配列) | **content が null だと 400**。`name` は無視される |
| `assistant` | `content` か `tool_calls` のどちらか | 両方とも null だと 400 (`convert.rs:361-365`)。`reasoning_content` (string) を受け付ける。`reasoning` や `reasoning_text` という別名は**無い** (legacy にだけある) |
| `tool` | `content` (string か ブロックの配列)、`tool_call_id` | **content が null だと 400**。必ず直前の assistant の tool_calls の直後に、件数どおり並べる (§4.2) |
| `latest_reminder` | `content` (string) | DeepSeek V4 固有のロール |
| `developer` | — | **400** |

### 4.1 content のブロック (`schema.rs:101-118`)

- 使えるブロックは `{"type":"text","text":…}`、`{"type":"image_url","image_url":{"url":…,"detail":…}}`、`{"type":"file","file_data":"data:…"}` の 3 種類。
  - `file_id` は 400 になる。
  - それ以外の type (例 `input_text`) は 400。
- 複数の text ブロックは `"\n\n"` で連結する (`convert.rs:283`)。
- 画像は user と tool で使える。assistant と system に置くと 400。
  - 枚数の上限は 16 (`api/native_v41.rs:299-301`)。
  - 1 枚の上限は 32MiB、合計の上限は 64MiB (`api/native_v41/images.rs:16-17`)。
  - http(s) の URL はサーバー側で取得する。接続のタイムアウトは 10 秒、1 枚あたり 30 秒、全体で 60 秒。リダイレクトは 3 回まで。認証情報を含む URL は拒否する (`images.rs:25-70`)。
- 入力の文字列に `<｜image｜>` などの画像用特殊トークンが含まれると、400 になる (`recipe/protocol/validation.rs:13-18`)。

### 4.2 ツール履歴の厳格さ (`convert.rs:371-453`)

- assistant の `tool_calls` の要素で必須なのは、`id` と `function.name`、`function.arguments` (**文字列**) である。
  - `type` は無くてもよい。履歴に付いている `index` は無視される。
  - arguments を object で渡すと 400 (`invalid type: map, expected a string`)【実測】。
- `tool_calls: []` は 400。
- **tool_calls を持つ assistant の直後に、tool メッセージがちょうど N 件並んでいなければならない。**
  - 間に user を挟むと 400。件数が足りなくても 400。
  - 存在しない id、重複した id も 400 になる。
  - 先行する tool_calls が無い tool メッセージも 400。
  - tool 結果の順番は、呼び出しの順番に並べ替えてから描画する (`enc/v4/mod.rs:380-384`)。
- 連続する user メッセージは 1 つに連結される (`enc/v4/mod.rs:354-372`)。

### 4.3 履歴にある思考の描画

- **thinking が有効なときは、すべての履歴の assistant について `<think>{reasoning_content}</think>{content}` を再描画する** (`enc/v4/mod.rs:184-197`)。
  - reasoning_content が無いときは、`<think></think>` と描画する。
  - 最新の turn より前の思考も削らない。
- thinking が無効なときは、`</think>{content}` と描画し、reasoning_content は捨てる。
- 最後の assistant メッセージを途中まで書いて続きを生成させる (prefill continuation) ことはできない。最後の assistant は EOS で閉じられ、その後に新しい assistant turn が始まる【実測: `asst_prefill_last`】。

---

## 5. 思考 (thinking) の制御

決定の手順は次のとおり (`recipe/…/convert.rs:81-98`、既定値は `api/native_v41.rs:259-261`)。

1. `thinking.type` があればそれに従う。`enabled` と `adaptive` は on、`disabled` は off。
2. 無ければ `reasoning_effort` を見る。`none` なら off、それ以外の値なら on。
3. どちらも無ければ on になる (ds41rt は `with_default_thinking_mode(true)` を指定している)。
4. on のときの effort: `reasoning_effort` で指定すればその値になり、省略すれば High になる。

effort は、プロンプトの先頭に置く system 文 `Reasoning Effort: N (range 1-100, …)` の N に変換される (`enc/v4/dsv41.rs:6-15,68-77`)。

| 要求値 | 内部の値 | N | 実測 |
|---|---|---|---|
| 省略 / `high` / `medium` / `xhigh` / `thinking.enabled` | High (Xhigh も 75 として描画) | 75 | thinking on、prompt 43 tok (無効時は 17 tok) |
| `low` / `minimal` | Low | 50 | thinking on |
| `max` | Max | 100 | — |
| `none` | off | 文を出さない | prompt 17 tok、reasoning_content も無い |
| `thinking:{"type":"disabled"}` に `reasoning_effort:"max"` を併記 | off (thinking が優先) | — | `api/native_v41.rs:555` のテスト |

- off のときは、プロンプトの末尾が `<｜Assistant｜></think>` になる。on のときは `<think>` で終わる (`api/native_v41.rs:561-565`)。
- **`enable_thinking`、`chat_template_kwargs.enable_thinking`、`chat_template_kwargs.thinking` は効かない。** 3 つとも thinking on のまま、reasoning_content が出た【実測】。
- **`thinking.budget_tokens` は効かない** (`schema.rs:210-211`)。思考の長さを制限する手段は `max_tokens` しか無い。
  - 思考の途中で `max_tokens` に達すると、`content:""`、`finish_reason:"length"` になり、reasoning_content だけが返る【実測】。
- 環境変数 `DS41RT_REAL_FULL_ENABLE_THINKING` は legacy 専用で、本番には効かない (`api/request.rs:678`)。

---

## 6. ツール

### 6.1 定義 (`convert.rs:155-215`、`recipe/protocol/validation.rs:88-115`)

- `type` は必須で、値は `"function"` だけを受け付ける。省略すると 400 (`missing field \`type\``)【実測】。
- `function.name` は `^[A-Za-z0-9_-]{1,128}$` でなければならない。
  - `.` を含む名前は 400【実測】。
  - `mcp__search__searxng_web_search` のような名前は通る。
  - 同じ名前が重複すると 400。
- `function.parameters` は省略でき、その場合は `{}` として描画する【実測: 200】。
  - **指定する場合は `"type":"object"` (文字列で厳密に一致) が必須**。かつ jsonschema としてコンパイルできなければならない。
  - `{}` や `{"type":["object"]}` は 400【実測】。
  - `properties` が無いだけなら通る。
- `function.strict` が true のときは、OpenAI の strict 規則を適用する。
  - strict の検査 (全プロパティを required に入れる、`additionalProperties:false`) は `response_format` に対してだけ事前に行う (`api/native_v41.rs:214-237`)。
  - ツールの strict は、文法制約と完了後の jsonschema 検証で強制する (`api/native_v41/tools.rs:43-50,65-82`)。
- 件数の上限やスキーマの大きさの上限は、本番の経路には見当たらない。上限 128 件、1MiB は legacy の `request.rs` にだけある。

### 6.2 tool_choice

| 値 | 挙動 |
|---|---|
| 省略 / `"auto"` | tools があれば解析を有効にする。既定の `tool_decoding_assistance:true` では、`<｜DSML｜ calls>` の文法 (triggered_tags) を常に掛ける (`api/native_v41/tools.rs:55-61`、`constraints.rs:35-37`) |
| `"required"` / `{"type":"function","function":{"name":…}}` | ds41rt が要求を `auto` に書き換えてから recipe に渡す。そのため **thinking on でも通る**【実測: `tool_required_think` は 200】。recipe 単体では 400 になる (`convert.rs:209-213`)。思考が終わった後に、文法で必ず呼び出しを出させる (`tools.rs:6-28`、`constraints.rs:34`)。名前を指定した場合は、そのツールだけを残す |
| `"none"` | **tools 定義を捨てる** (`convert.rs:164-171`)。この状態で履歴に tool_calls があると、モデルは DSML の生テキストを `content` に出し、`finish_reason:"stop"` になる。tools を省いた場合も同じになる【実測: `tool_choice_none_with_hist`、`tool_hist_no_tools_decl`】 |

- `parallel_tool_calls:false` を指定すると、文法で 1 件に制限する (`stop_after_first`)【実測: 1 件だけ返った】。省略すると true になる。
- ツール呼び出しの件数に上限は無い。抑える手段は `max_tokens` だけである。GLM 時代のパッチにあった呼び出し回数の上限に当たるものは無い。

### 6.3 response_format (`api/native_v41/constraints.rs:8-49`)

| type | 挙動 |
|---|---|
| `text` | 制約なし |
| `json_object` | **プロンプトのどこかに "json" (大文字小文字は問わない) が無いと 400** (`convert.rs:468-473`)。文法で object に制約し、system に Response Format の指示を加える【実測】 |
| `json_schema` | `json_schema.schema` を XGrammar で制約し、完了後に jsonschema で検証する。`strict:true` のときは事前に strict 規則を検査し、違反すると 400 (エラーの形は §8 の別形式)【実測】 |
| `regex` | `{"type":"regex","regex":"…"}` (ds41rt の拡張) |

- thinking on の場合、制約は `</think>` (トークン id 128822) の後にだけ掛かる。思考の部分は自由に生成される (`constraints.rs:41-45`)。

---

## 7. 応答の形

### 7.1 非ストリーム【実測】

```json
{"id":"chatcmpl-<uuid4>","object":"chat.completion","created":1792339106,
 "model":"deepseek-ai/DeepSeek-V4.1-Flash","system_fingerprint":"ds41rt-native-fp4-kv-dspark",
 "choices":[{"index":0,"message":{"role":"assistant","content":"…","reasoning_content":"…",
   "tool_calls":[{"id":"call_chatcmpl-<uuid4>_0","type":"function",
                  "function":{"name":"get_weather","arguments":"{\"city\": \"Tokyo\"}"}}]},
   "logprobs":null,"finish_reason":"tool_calls"}],
 "usage":{"prompt_tokens":281,"completion_tokens":76,"total_tokens":357,
   "prompt_tokens_details":{"cached_tokens":0},
   "prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":281}}
```

- `id` の形式は `chatcmpl-` に uuid v4 を付けたもの (`api/native_v41.rs:292`)。
- **tool_call の id は `call_{応答 id}_{index}`** (例 `call_chatcmpl-3829…_0`、約 52 文字) (`recipe/…/chunk_generator.rs:133`)。
  - 応答をまたいでも一意である。
  - 9 文字の英数字といった制約は満たさない。
- `content` の値は場合によって変わる。
  - thinking on でツール呼び出しだけの応答では、`""` になる。
  - thinking off でツール呼び出しだけの応答でも、`""` になる。
  - null にはならない。応答を組み立てる関数が空文字を追記するためである (`recipe/…/response/schema.rs:286-294`、`chunk_generator.rs:155-160`)。
- `reasoning_content` は thinking on のときだけ付く (`skip_serializing_if`)。
- arguments は `{"city": "Tokyo"}` のように、コロンの後に空白を入れた JSON 文字列になる。
- `finish_reason` の値は次のとおり (`recipe/stream/processor.rs:151-163`)。
  - `stop`: EOS で終わった場合と、stop 文字列に一致した場合。
  - `length`: max_tokens に達した場合。
  - `tool_calls`: EOS で終わり、かつ tool_calls がある場合。
  - `content_filter`: 定義はあるが、実際には出ない。
  - **tool_calls の途中で max_tokens に達した場合は `length` になる。** 不完全な tool_calls が残る可能性がある【コード】。
- usage は次のとおり (`recipe/…/response/schema.rs:181-219`)。
  - `completion_tokens` は、思考、markup、stop 文字列を含むチャンクの全トークンを合計した値。
  - **`completion_tokens_details` は出力されない** (常に None)。そのため reasoning_tokens も無い。
  - `cached_tokens` と `prompt_cache_hit_tokens` は prefix cache が当たったトークン数で、両者は同じ値。
- 独自フィールドの `metrics` は、本番の応答には無い (legacy の `openai.rs:228` だけにある)。

### 7.2 SSE (`api/native_v41.rs:389-416`)【実測】

- ヘッダは `content-type: text/event-stream` と `cache-control: no-cache`。
- 各イベントは `data: {json}\n\n` の形。コメント行、keep-alive、`event:` 行は無い。最後は `data: [DONE]\n\n` で終わる。
- チャンクは `object:"chat.completion.chunk"`、`choices` の要素は 1 個 (index 0)、`logprobs:null` を持つ。

チャンクの並び (thinking on でツールを呼ぶ場合。`tool_st_think`):

```
{"delta":{"role":"assistant","content":null,"reasoning_content":""}}         ← 先頭 (思考 off なら {"role":"assistant","content":""})
{"delta":{"content":null,"reasoning_content":"We"}}                          ← 思考はトークン単位
…
{"delta":{"content":"5","reasoning_content":null}}                           ← 本文 (思考 on のときは reasoning_content:null が付く)
{"delta":{"tool_calls":[{"index":0,"id":"call_chatcmpl-…_0","type":"function","function":{"name":"get_weather","arguments":""}}]}}
{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]}}         ← 引数はトークン単位の差分
{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\""}}]}}
…
{"delta":{"tool_calls":[{"index":1,"id":"call_chatcmpl-…_1",…,"arguments":""}}]}}
…
{"delta":{"content":"","reasoning_content":null},"finish_reason":"tool_calls","usage":{…}}   ← 最終チャンク (usage を同居)
[DONE]
```

- ツールの開始チャンクは `id`、`type`、`name`、`arguments:""` を持つ。その後は `index` と `function.arguments` だけの差分が続く (`chunk_generator.rs:123-153`)。
  - 1 つの呼び出しの差分が終わってから、次の index に進む。index が混ざることは無い。
- usage は**常に最終チャンクに付く**。`include_usage:true` のときは、それより前のチャンクに `"usage":null` が入る (`chunk_generator.rs:59-65`)。
- 思考が無い場合、ツール呼び出しの前に本文の差分が出ることがある (例 "I'll check the weather …")。
- **応答ヘッダが送られるのは、worker が要求を受け入れて `Ready` を送った後である** (`api/native_v41.rs:345-350`、`daemon/v41_native_serve/scheduler.rs:351-355`)。
  - 待ち行列にいる間は、ヘッダも返らない。
  - 受け入れからプレフィルが終わるまでの間は、先頭のチャンクの後に何も流れない。
- 生成の途中で失敗した場合 (worker のエラー、required のツールを出さなかった、strict のスキーマ違反、DSML の不正) は、body stream にエラーを返して接続を切る。
  - `[DONE]` も error イベントも送らない (`api/native_v41.rs:377-405`、`api/native_v41/tools.rs:257-261` のテスト)。
- 非ストリームで同じ失敗が起きた場合は、500 になる (`api/native_v41.rs:424-435`)。

---

## 8. エラー

エラーの形式は 2 種類あり、一部のケースはそもそも JSON で返らない。

| 発生源 | HTTP | 本文 |
|---|---|---|
| 変換や検証の失敗のほとんど (serde、recipe、model、max_tokens、sampling、画像) | 400 | `{"error":{"message":"…","type":"native_v41_error"}}` (`param` と `code` は無い。`api/native_v41.rs:127-140`) |
| `response_format` の strict の検査 | 400 | `{"error":{"message":"…","type":"invalid_request_error","param":"response_format.json_schema.schema","code":"invalid_request"}}` (`api/error.rs:45-55,81-98`) |
| JSON の構文誤り | 400 | **text/plain**: `Failed to parse the request body as JSON: …` (axum の既定の rejection) |
| Content-Type が無い | 415 | **text/plain** |
| 待ち行列が満杯、または 25 秒待っても空かない | 429 | `{"error":{"message":"request queue is full or its wait budget expired","type":"native_v41_error"}}` と `Retry-After: 1` (`api/native_v41.rs:302-310`) |
| worker が停止している | 503 | `worker queue is closed` |
| worker の失敗、生成後の検証の失敗、**プロンプトが 1,048,576 トークン以上** | 500 | 同じ形式 (`native_v41_error`)。トークン長の超過は `NativeFailure::Worker` 扱いになるため 400 ではなく 500 になる (`daemon/v41_native_serve/scheduler.rs:262-271`、`limits.rs:48-58`)【コード】 |
| 文法 (constraint) のコンパイル失敗 | 400 | `NativeFailure::BadRequest` (`daemon/v41_native_serve/constraints.rs:29`) |
| 未定義のパス | 404 | 本文は空 |

- `message` は 512 文字で切り詰めて、`… (truncated)` を付ける (`api/error.rs:70-79`)。

---

## 9. 同時実行、待ち行列、タイムアウト、トークン上限

### 9.1 受け入れの 3 段階 (`api/native_v41/admission.rs:25-44`、`daemon/v41_native_serve.rs:73`、`daemon/v41_native_serve/scheduler.rs:244-262`)

1. **実行枠は `--concurrency 16`** (CLI の範囲は 1..=16。`daemon/cli.rs:652-654`)。枠は 2 つの lane で共有し、lane の容量は 8 である。
2. **バッファは `mpsc::channel(16)`** (`--http-queue-depth 16`)。worker は実行枠が空いたときにだけ、ここから要求を `try_recv` する。チャネルに入った要求はタイムアウトせずに待つ。
3. **待機者は semaphore で 16 まで** (待機者の数はバッファの深さと同じ)。各待機者は、最大 `--http-queue-wait-ms 25000` の間、チャネルに空きが出るのを待つ。
   - 時間切れになると 429 を返す。
   - 17 人目の待機者は、即座に 429 になる。
- 合計すると、同時に 48 本までは受け付ける。そのうち実行されるのは 16 本だけで、33 本目以降は最大 25 秒で 429 になりうる。
- KV のトークン予算が足りないときは、`Pending` として 1 本を保留する (`scheduler.rs:305-313`)。
- `/v1/stats` の `http_queue_waits`、`http_queue_rejects`、`http_queue_len` で監視できる。

### 9.2 KV の予約【コード + 起動ログ】

- 受け入れのときに、**プロンプト長と `max_tokens` (切り詰め後) の合計**を KV のページとして予約できるか検査する (`scheduler/admission.rs:47-51`、`scheduler.rs:292-298`)。
- 起動ログの `source_pages=[28704,28704,28704,57408]` に `PAGE_ROWS=256` と圧縮率 (2,2,2,1) を掛けると、KV プール全体はおよそ **1,470 万トークン**になる (`daemon/v41_compressor.rs:25-30`、`v41_compressor/source_cache.rs:30`)【計算】。
- `max_tokens` を省くと、1 本あたりプロンプト長 + 393,216 を予約する。
  - 16 本の平均プロンプトが 20 万トークンなら、約 950 万トークンで収まる。
  - 平均が 50 万トークン級になると、満杯になって受け入れが保留される。
- proxy で `max_tokens` を明示する (GLM 時代の 32k や 131k に合わせる) ほうが安全である【推測】。

### 9.3 プレフィルとデコードの関係

- 新しい要求のプレフィルは、受け入れのループの中で同期的に実行する。その間、両 lane は他の要求のデコードを進めない。
  - コード中のコメントに "Prefill currently owns both lanes; mixed prefill/decode interleaving is a subsequent policy" とある (`scheduler.rs:242-243,368-371`)。
  - 長いプロンプトが入るたびに、実行中の全ストリームが数秒から数十秒止まる【推測: 4〜6k tok/s とすると 20 万トークンで 30〜50 秒】。
- prefix cache の保持は 8 件 (`--prefix-cache-entries 8`) と、ホストキャッシュ 4GiB。

### 9.4 タイムアウト

- サーバーには、要求全体のタイムアウトも、アイドルのタイムアウトも、SSE の heartbeat も無い。
- 時間の制限があるのは、待機の 25 秒と画像取得 (§4.1) だけである。
- **クライアントが接続を切ると、生成をすぐに止める。** 各トークンの出力の前に `events.is_closed()` を検査している (`scheduler.rs:91`)。
  - 実測では、1 秒で接続を切ったところ、出力は 98 トークンで止まった。admitted と retired の数も一致した。
- 長い非ストリームの要求では、応答が完了するまでバイトが届かない。そのため、クライアント側の read timeout の設定に注意が要る。

### 9.5 実測の目安 (thinking off、200 トークン上限、1〜60 を数える)

- 1 本のときは 120 トークンを 0.50 秒で生成した (約 240 tok/s。dSpark 込み)。
- 4 本を並列に流すと、各 1.03 秒で全員が同時に完了した。

---

## 10. GLM 時代のクライアント要求との差

集計の方法:
- `reqkeys.py` を使い、SGLang の reqlog の `request.received.openai` を数えた (5 ファイル、1,191 件)。
- この obj は SGLang が既定値を補った後の形なので、0 や false の項目は実際には送られていない可能性がある。
- 全 3.2 万件の event 種別は `request.received.openai` だけで、`/v1/responses` の記録は無い。
- vLLM のログの直近 40 ファイルでも、`POST /v1/chat/completions` が 15,880 件あっただけである。

| GLM 時代の要求 | 件数 | ds41rt での結果 | proxy での対処 |
|---|---|---|---|
| `chat_template_kwargs.enable_thinking=false` | 3 | 無視されて thinking on になる | `thinking:{"type":"disabled"}` に変換する |
| `chat_template_kwargs.enable_thinking=true` / `thinking=true` / `clear_thinking` | 1,093 | 無視される (既定で on なので結果は同じ) | `clear_thinking:true` は GLM の「過去の思考を消す」指示。ds41rt は履歴の reasoning_content をすべて再描画するので、必要なら proxy で消す |
| `reasoning_effort` が low、medium、high、max、minimal | 382 | 受け付ける。medium は high と同じ。minimal は low と同じで、thinking は on のまま | 数値や大文字が来る場合は正規化する |
| `max_completion_tokens: 131072` | 83 | **無視されて 393,216 になる** | `max_tokens` に移す |
| `max_tokens` が 32000、131072、120 など | 1,101 | そのまま使う | — |
| `tool_choice:"none"` (tools と履歴あり) | 187 | **DSML が content に漏れる** | ツール履歴を平文に変換するか、tools を残したまま auto にして proxy 側で tool_calls を捨てるか、出力の DSML を除去する |
| `stream:false` | 873 (73%) | 対応している。完了するまでバイトが届かない | クライアントのタイムアウトを確認する |
| `stream_options:{include_usage:true, continuous_usage_stats:false}` | 318 | 受け付ける (未知のキーは無視) | `stream:false` に stream_options を付けた要求は 400 になるので、proxy で落とす |
| `temperature 0.6 / top_p 0.95 / top_k 20` | 691 | 受け付ける | — |
| 温度を省略 | 331 | **greedy になる** (GLM/vLLM では既定の 1.0 だった) | 必要なら既定値を補う |
| `model:"GLM-5.3-Flash"` | 全件 | **400** | 書き換える |
| `developer` ロール | 0 | 400 | 念のため system に変換する |
| ツールの `parameters` (全件 `type:"object"`、properties の無いものが 192 件) | 12,307 | 受け付ける | `type` の無いスキーマが来たら補う |
| 画像 (`image_url`) | 50 | 受け付ける | — |
| tool の content | 全件 string | 受け付ける | null は `""` に変換する |
| assistant の content が null で tool_calls あり | 5,604 | 受け付ける | — |

---

## 11. 未確認事項

- 429 と 25 秒待機の実地確認。同時 4 本の制約があるため、満杯の状態を作れなかった。
- 1M トークン超のプロンプトで本当に 500 が返るかどうか (コード上の判断)。
- 画像の実地での投入。r2 と r6 で確認済みのため、今回は試していない。
- `tool_decoding_assistance:false` を指定したときの、ツール呼び出しの崩れ方。
- 生成途中で切断した SSE を、各クライアント (コーディングエージェントや OpenAI SDK) がどう見せるか。
