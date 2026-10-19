# r7: GLM 時代の実要求を ds41gw 経由で ds41rt に再生する (2026-10-19)

GLM-5.3-Flash 時代の実クライアント (コーディングエージェント型、複数のクライアント) の要求 96 件を、ds41gw (`127.0.0.1:30001`) 経由で稼働中の ds41rt (`localhost:8000`) に再生した。
見たのは、エラー率、応答の形 (SSE の順序・ツール呼び出しの JSON・名前・id・usage)、追加遅延 (ds41gw 経由と ds41rt 直の TTFT の差) である。
ds41rt には触れず、同時 4 本以内・出力 256 トークン以内で送った。試験中、ds41rt の `/v1/stats` の `http_queue_rejects` は 0 のままだった。

結論:

- **エラー率 0/96** (修正の前後とも)。途中失敗、DSML の漏れ、上流の再試行、待ち行列の待ちはすべて 0 だった。
- **ツール呼び出しは 58 本。arguments が JSON の object として読めなかったものは、すべて出力上限 (256) で途中で切れた呼び出しだった。**
  切れていない呼び出しで読めないものは 0 本。名前の未定義・id の重複・空の id も 0 だった。
- **バグを 1 件見つけて直した** (§4.1)。Responses で `include:["reasoning.encrypted_content"]` を付けると、
  `response.output_item.done` の reasoning 項目と `response.completed` の `output` で `encrypted_content` が食い違うことがあった。
- **追加遅延は TTFT の中央値で +3.5 ms** (逐次、接頭辞キャッシュは温めた状態、59 組)。本文の大きさに比例して伸び、
  60KB 未満では +1.7 ms、150〜400KB では +4.3 ms だった。組の 1 割強に +20〜40 ms の外れ値があり、原因は ds41rt が TCP_NODELAY を付けていないことである (§5.3)。
- **同じ理由で、keep-alive のクライアントは ds41rt 直より ds41gw 経由のほうが TTFT が約 40 ms 短い** (§5.3)。
- 短い要約の補助要求 (`max_tokens 120`、effort low) は 10 回中 8 回が本文が空 (`finish_reason:"length"`) だった。§18.1 の 6 (小さな max_tokens で思考を自動で切るか) の判断材料になる (§6)。

## 1. 道具と成果物

| ファイル | 内容 |
|---|---|
| `proxy/tests/build_replay_corpus.py` | reqlog から要求集を作る。Chat は `request.received.openai` の `obj` を使う。Responses は描画済みプロンプトから逆変換して復元する (§2.2) |
| `proxy/tests/replay_corpus.jsonl` | 要求集 96 件 (Chat 66、Responses 30) |
| `proxy/tests/replay_real.py` | `run`: ds41gw 経由で再生して形を検査する。`pairs`: ds41rt 直と ds41gw 経由の TTFT を組で比べる |
| `proxy/tests/replay_variants.py` | 実要求を少し変えて別の経路を通す (ストリームの入れ替え、encrypted_content と summary、Responses の 2 ターン往復) |
| `tests/results_replay_real.jsonl` | 修正後の再生 (96 件)。修正前は `tests/results_replay_real_before_fix.jsonl` |
| `tests/results_replay_variants.jsonl` | 変形の再生 (修正後、28 件) |
| `tests/results_replay_pairs_c1.jsonl` | TTFT の組 (逐次、D P P D / P D D P、max_tokens 8) |
| `tests/results_replay_pairs_c4.jsonl` | TTFT の組 (同時 4、max_tokens 256、1 回ずつ。§5.2 のとおり遅延の比較には使えない) |

再実行の手順:

```sh
cd proxy
python3 tests/build_replay_corpus.py                                  # 要求集を作り直す (seed 19、決定的)
python3 tests/replay_real.py run --out tests/results_replay_real.jsonl  # 同時 4、出力 256
python3 tests/replay_real.py pairs --in tests/results_replay_real.jsonl --out tests/results_replay_pairs_c1.jsonl --conc 1 --abba --max-tokens 8
python3 tests/replay_variants.py --out tests/results_replay_variants.jsonl
```

## 2. 要求集

### 2.1 Chat (66 件、reqlog の生 JSON)

`build_replay_corpus.py` の `CHAT_PLAN` で、ファイルとクライアントの型を指定して無作為に選んだ (seed 19)。
クライアントの型の判定は `tests/replay_compat.py` の `label()` を使う (規則は利用者が与える。r7_clients.md §3.1 の型 A〜E)。送る項目は `replay_openai.py` の `KEEP` に `stream`・`stream_options`・`response_format` を足したもので、
SGLang が足した既定値 (`return_*` など) は落とした。model は記録どおり (`GLM-5.3-Flash`) にした。

| クライアントの型 (r7_clients.md §3.1) | 件数 | 形 |
|---|---:|---|
| B | 15 | ストリーム、ツール 4〜13 個。200KB 超を 6 件含む (最大 32 万字・99,999 トークン) |
| B + `tool_reference` | 4 | ds41rt 直では 400 になる形 (r7_clients.md §3.4) |
| B の補助 | 5 | 子エージェントなど |
| C (新版) | 10 | 500KB 超を 3 件含む。画像付き 6 件 (data URL) |
| C (旧版) | 5 | |
| C のタイトル生成 | 2 | `tool_choice:"none"`、ツール履歴なし、max_tokens なし |
| D | 5 | 約 9 万トークンの長い履歴 |
| A | 12 | **非ストリーム**、ツール 6 個、`max_tokens 32000`、temperature 0.6 |
| E (短い要約) | 5 | 非ストリーム、`max_tokens 120`、effort low、`clear_thinking:true`、`tool_choice:"none"` |
| その他 | 3 | ツール 58 個 |

出所は `reqlog-F.log` など 12 ファイル (要求集の `src` に `ファイル:rid` を残した)。
前回の互換性の総当たり (README「GLM 要求ログの再生」) では型 D が抜けていたが、今回は `reqlog-A.log` から 5 件入れた。

### 2.2 Responses (30 件、描画済みプロンプトからの復元)

reqlog に Responses の生 JSON は無い (r7_clients.md §1(c))。そこで、`request.received` (rid が `resp_`) の描画済みプロンプト (GLM-5.3 のチャットテンプレート) を逆変換して要求を作った。
規則は r7_clients.md §4.2 の推定復元と同じである (`build_replay_corpus.py` の `rebuild_responses`)。

- `<|system|>Reasoning Effort: X` → `reasoning.effort` (小文字。主要求は `max`、タイトルは `high`)
- `<|system|>\n# Tools … <tools>` の各行 → 平坦形の function tool (27 個)
- `<|system|>Developer instructions:\n…` → `developer` の message
- `<|user|>` → `user` の message (`input_text`)
- `<|assistant|>` の `<think>…</think>` → `reasoning` 項目。次の 3 形式を順に使い分けた: SGLang の `encrypted_content` (`sglang-reasoning-v1.` + base64(zlib))、`summary`、`content`
- `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` → `function_call` (call_id は振り直した。値はスキーマが string ならそのまま、ほかは JSON として読む)
- `<|observation|>` の `<tool_response>` → `function_call_output` (呼び出しの順に対応させる)
- `max_new_tokens` が 127,998 なら `max_output_tokens` 128000、62 ならタイトル生成として 64 にした

出所は Responses の要求が残る 2 ファイルの 131 件で、長さで層別して 30 件を選んだ (タイトル生成 1 件、入力 2〜155 項目、最大 30 万字・74,305 トークン)。
実データは 100% ストリームだが、非ストリームの集約も見るため 6 件に 1 件 (5 件) を非ストリームにした。

**復元の確認**: 3 件 (resp-009、020、029) で、ds41gw が ds41rt に送った Chat 本文 (要求ログの `obj`) を見た。
reasoning 項目の数と assistant の `reasoning_content` の数・本文が全件一致し (3 形式とも)、function_call と tool メッセージの数も一致した (16/16、32/32、57/57)。

## 3. 検査の内容 (`replay_real.py`)

| 種類 | 検査 |
|---|---|
| Chat ストリーム | 各 data が JSON、`object`・`model` (要求した名前が返るか)、chunk id が 1 つ、`[DONE]` で終わる、`include_usage` なら `choices:[]` の usage チャンク、choices の数 = n、finish_reason が 1 回、`tool_calls` の index・id・name が途中で変わらない、DSML が content に漏れない |
| Chat 非ストリーム | `object`、`model`、role、finish_reason、`tool_calls[].type`、arguments が文字列、usage |
| Responses ストリーム | event 名と `type` の一致、`sequence_number` が 0 から連番、最初が `response.created`、項目は 1 つずつ開閉、delta の `item_id` が開いている項目、`function_call_arguments.done` = delta の連結、終了イベントが 1 つで最後、**終了イベントの `output` と各 `output_item.done` の項目が一致**、`model`、usage |
| Responses 非ストリーム | `object`、`status`、`model`、項目の id、usage |
| ツール呼び出し (共通) | id が空でない・重複しない、名前が要求の tools にある、arguments が JSON の object (上限で切れたものは別に数える)、スキーマの `required` の欠落 (注記だけ) |
| usage (後で集計) | prompt > 0、total = prompt + completion、reasoning_tokens ≤ completion、completion ≤ 上限、`length` なら completion = 上限 |

## 4. 結果

### 4.1 見つけたバグと修正: encrypted_content が done と completed で食い違う

**症状**: `replay_variants.py` の V2 (Responses に `include:["reasoning.encrypted_content"]`・`reasoning.summary:"auto"`、ストリーム) で、
4 件中 2 件 (resp-001、resp-004 の変形) が「`output[0]` と `output_item.done` が違う」で不合格になった。違っていたのは `encrypted_content` だけだった。

**原因**: `Post::process` (`ds41gw/src/responses.rs`) は、`response.output_item.done` のときと `response.completed` / `response.incomplete` のときの 2 回、`fix_item` で `encrypted_content` を作っていた。
`Signer::encrypt` (`ds41gw/src/reasoning.rs:155-161`) は本文に `ts` (秒単位の時刻) を入れるので、思考の項目が閉じてから応答が終わるまでに秒をまたぐと、2 つの blob が別物になる。
どちらも復号すると同じ思考なので往復は壊れないが、同じ項目が 2 通りの値で届く。保存される応答 (GET) は completed 側の値になる。

**修正** (`ds41gw/src/responses.rs` の `response.completed` の分岐): `output_item.done` で送った項目を id で引き、そのまま使う。done で送っていない項目だけ `fix_item` を通す。

**試験**:

- `ds41gw/tests/common/mod.rs` の `check_responses_events` に「終了イベントの `output` と各 `output_item.done` の項目が一致する」を足した。これを使う既存の Responses 試験すべてに効く。
- mock に `#mock:slowtools` (思考の後に 1.1 秒待ってから本文とツール呼び出しを返す) を足した。
- `ds41gw/tests/responses.rs` に `encrypted_content_is_identical_in_done_and_completed` を足した。summary 無し・`auto` の両方で、done・completed・GET の 3 つの値が一致することを見る。
- `./cargo.sh test --release`: 全件合格 (lib 19、chat 20、lossless 5、messages 6、responses 10、review 13。responses と review は並行して作業している別の試験を含む)。
- イメージを作り直して `./run_proxy.sh start` で再起動した後、V2 は 6/6 件 (変形全体では 28/28 件) が合格した。

### 4.2 再生 (96 件、同時 4、出力 256)

| | 修正前 (`results_replay_real_before_fix.jsonl`) | 修正後 (`results_replay_real.jsonl`) |
|---|---|---|
| 合格 | 96/96 | 96/96 |
| HTTP エラー・ストリーム内エラー | 0 | 0 |
| 所要時間 | 469 s | 362 s |
| Chat の finish | length 41、tool_calls 18、stop 7 | length 40、tool_calls 20、stop 6 |
| Responses の status | incomplete 17、completed 13 | incomplete 21、completed 9 |
| ツール呼び出し | 58 本 (読める 40、上限で切れた 18 = Chat 12 + Responses 6) | 58 本 (読める 38、上限で切れた 20 = 12 + 8) |
| 上限で切れていないのに読めない arguments | 0 | 0 |
| 名前の未定義・id の重複・空の id | 0 | 0 |
| `required` の欠落 (モデルの出力) | 1 (bash の `description`) | 0 |
| 途中失敗・DSML の漏れ・上流の再試行・待ち行列の待ち | 0・0・0・0 ms | 0・0・0・0 ms |
| usage の矛盾 (§3 の最終行) | 0 | — |

- ds41gw が付けた正規化 (要求ログの `normalized`): `model_alias` 96、`temperature_default` 79、`chat_template_kwargs` 41、`max_completion_tokens` 10、`clear_thinking` 5、`tool_reference_to_text` 4。
  実データのツール履歴は全件が正しい形だったので、補修 (`repaired_tool_history`) は 0 件だった。
- `model` は全件、要求した名前 (`GLM-5.3-Flash`) で返った (修正後の再生で検査した)。
- 上限で切れた呼び出しは、Chat では `finish_reason:"length"` で途中までの arguments を返し、Responses では `status:"incomplete"`・`incomplete_details.reason:"max_output_tokens"` の応答に途中までの `function_call` 項目が入る。
  どちらも OpenAI と同じ振る舞いで、クライアントは finish / status を見て捨てる必要がある。
- 接頭辞キャッシュ: 同じ会話の連続するターン (resp-020 → 021 → 022 など) は、前のターンのプロンプトから末尾 128 トークン前後を除いてすべて当たった (`prompt_cache_hit_tokens` / `prompt_tokens` が 0.99〜1.0)。
  ds41gw の変換 (Responses → Chat、思考の結合、正規化) でターン間の接頭辞が崩れていないことを示す。

### 4.3 変形 (`replay_variants.py`、28 件、修正後すべて合格)

| 変形 | 件数 | 結果 |
|---|---:|---|
| V1n: 型 B・C・D のストリーム要求を非ストリームに | 8 | 合格 |
| V1s: 型 A の非ストリーム要求をストリーム + `include_usage` に | 6 | 合格 |
| V2: Responses に `include` encrypted_content + `summary:"auto"` (ストリーム 4、非ストリーム 2) | 6 | 修正前 4/6 (§4.1)、修正後 6/6 |
| V3: Responses の 2 ターン往復 (ds41gw の出力を履歴に足し、ツール結果を付けて次を送る。reasoning は encrypted_content だけ / 出力のまま) | 8 | 7 組で 1 ターン目の思考が 2 ターン目の上流本文の `reasoning_content` に一字一句戻った。1 件は 256 トークンで呼び出しまで届かず飛ばした |

## 5. 追加遅延 (ds41gw 経由 − ds41rt 直)

### 5.1 測り方

- ds41rt 直の本文には、ds41gw が ds41rt に送った Chat 本文 (要求ログの `obj`、`x-request-id` で引く) に `stream:true`・`include_usage` を足したものを使った。どちらの経路も ds41rt では同じ処理になる。
- 画像付きの要求は、要求ログの画像がハッシュに置き換わっていて ds41rt 直に送れないので除いた。非ストリームの要求も除いた (TTFT がクライアントに見えない)。残りは 68 件。
- TTFT は、最初に中身のある delta (`content` / `reasoning_content` / `tool_calls`、Responses は `*.delta`) が届くまでの時間で、要求を送り始めた時刻から測る。
- 各組の前に ds41rt 直で `max_tokens:1` の要求を送って接頭辞キャッシュを温め、D P P D か P D D P (無作為) の順で 4 回送って、ds41gw の平均 − 直の平均を取った。出力は 8 トークンまで、逐次 (同時 1)。
  最初の思考トークンが空文字列になる要求や、思考を飛ばしてツール呼び出しから始まる要求 (ds41rt は呼び出しが確定するまで delta を出さない) では TTFT が取れないので、取れた 59 組で集計した。

### 5.2 結果

| 本文の大きさ | 組 | 差の中央値 (各経路の速いほう同士) |
|---|---:|---|
| 60KB 未満 | 21 | +1.7 ms |
| 60〜150KB | 19 | +3.0 ms |
| 150〜400KB (〜10 万トークン) | 21 | +4.3 ms |

全体 (59 組、平均同士の差): **p50 +3.5 ms**、p10 +0.2 ms、p90 +20.8 ms。Chat 38 組は p50 +3.4 ms、Responses 21 組は p50 +3.5 ms。
TTFT の中央値は ds41gw 47 ms、直 41 ms だった (キャッシュが当たった状態)。

- 差は本文の大きさにほぼ比例する。要求の解析 (`parse_json`)、`orig.clone()`、正規化、要求ログ用の複製と直列化 (`Ctx::log_received` が `prepare_body` で `obj` と `orig` を複製して `to_string` する。`ds41gw/src/core.rs:158-195`、`ds41gw/src/chat.rs:982-995`)、上流向けの直列化が、上流へ送る前に同期で走るためと考える。
  目標の +5 ms (仕様 §16) は中央値では守れているが、30 万字を超えると近づく。要求ログの複製と直列化を書き込みスレッドに移せば (`orig` は prepare の後に使わないので、複製せずに渡せる) 縮められる見込みだが、今回は手を入れていない。
- 同時 4・出力 256 で 1 回ずつ比べた組 (`results_replay_pairs_c4.jsonl`、43 組) は、p50 +7.3 ms だが p10 −201 ms / p90 +548 ms とばらつき、+17〜23 s の組が 4 つあった。
  別の長い要求の前処理 (prefill) に ds41rt の中で待たされた時間で、どちらの経路でも起きていた (直が 23 s 遅れた組もある)。ds41gw の遅延の比較には使えない。

### 5.3 +20〜40 ms の外れ値と keep-alive: ds41rt が TCP_NODELAY を付けていない

逐次の組の 1 割強 (59 組中 10 組が +15 ms 超) で、ds41gw の 2 回のうち 1 回だけが約 40 ms 遅かった (例: chat-034 で 43.7 / 3.6 ms、直は 3.0 / 3.0 ms)。40 ms は Linux の遅延 ACK の最小値である。

- ds41rt は HTTP の口を `axum::serve(listener, …)` で開いており、`.tcp_nodelay(true)` を付けていない (`ext/ds41rt/rust/crates/ds41rt-daemon/src/commands/coordinator.rs:148`)。
  ds41rt 内の他の TCP (`ds41rt-transport`) は `set_nodelay(true)` している。
  このため ds41rt の SSE の小さな書き込みは Nagle で、受け手が前の書き込みの ACK を返すまで待たされる。受け手の ACK は、つなぎっぱなしの接続では最大 40 ms 遅れる。
- ds41rt 直で確かめた (chat-034 の本文、出力 8、20 回): 毎回新しい接続にすると TTFT は 2.8 ms で、40 ms の空きは 0 回だった。
  同じ接続を使い回すと、2 回目以降は**毎回**、最初のチャンク (role だけ) と次のチャンクの間が 40 ms 空き、中身のある最初の delta は 42.5 ms だった。
- ds41gw 経由で、クライアントが接続を使い回す場合 (openai SDK などの既定) の TTFT は 3.0 ms だった。ds41gw は受け口に TCP_NODELAY を付けているので、クライアント側では空きが出ない。
  28,418 トークンの要求 (chat-000、キャッシュ済み、10 回) では、ds41gw 経由 41.6 ms、ds41rt 直 78.7 ms だった。
  **keep-alive のクライアントにとっては、ds41rt 直より ds41gw 経由のほうが TTFT が約 37〜40 ms 短い。**
- ds41gw から ds41rt への接続も使い回し (reqwest の接続プール) なので、ds41gw 経由でもときどき 40 ms 待たされる。これが外れ値である。
  ds41gw 側で受け手の遅延 ACK を止めるには、上流のソケットに読むたび `TCP_QUICKACK` を付ける必要がある。reqwest 0.12 はソケットを触らせないので (`connector_layer` の返り値は不透明な `Conn`)、今回は直していない。
  直すなら ds41rt 側で `axum::serve(…).tcp_nodelay(true)` の 1 行を足すのが確実である (次に ds41rt を再起動する機会に。今回は ds41rt に触れていない)。
- 生成中のチャンクの間隔にも同じ空きが出る。直・新しい接続・出力 200 では、10 回で 4 回、35 ms 以上の空きがあった。

## 6. 判断材料 (仕様 §18.1 の 6: 小さな max_tokens で思考を自動で切るか)

- 型 E の短い要約 (`max_tokens 120`、`reasoning_effort:"low"`、`chat_template_kwargs:{clear_thinking:true, thinking:true, enable_thinking:true}`、非ストリーム) は、修正前 5 件中 3 件、修正後 5 件中 5 件で本文が空 (`finish_reason:"length"`) だった。
  思考だけで 120 トークンを使い切っている (空だった回の思考は 186〜585 字)。本文が返った 2 回も 26 字と 33 字だった。
- 型 F (Responses) のタイトル生成 (`max_output_tokens 64`、effort high) は 2 回とも `status:"incomplete"` で本文が空だった。
- 型 C のタイトル生成 (max_tokens を送らないので今回は 256 に抑えた) は 4 回とも本文を返した (15〜21 字)。
- 型 E は思考ありを明示しているので、§4.3.5 の自動切り替えを入れる場合は、明示された思考指定より小さな max_tokens を優先する規則が要る。

## 7. 触れていないこと・残る不確実性

- Responses の `developer` は、recipe どおり user として描画される (`deepseek-recipe-0.1.0/src/protocol/openai/responses/request/convert.rs:343`)。
  GLM 時代の SGLang は `<|system|>Developer instructions:\n…` と system として描画していた (§2.2 の reqlog)。型 F の 2 万字の指示が user ターンになるので、
  応答の質に効く可能性がある。README の「仕様との差」に挙がっている既知の差で、今回は変えていない。
- 復元した Responses は推定である (§2.2)。`include`・`store`・`text`・`tool_choice`・`parallel_tool_calls` は痕跡が無いので送っていない。
- 出力を 256 に抑えたので、長い出力 (32,000 トークンの型 A など) の途中の振る舞いは見ていない。同時 12〜16 本の負荷も見ていない (ds41rt が稼働中のため同時 4 本まで)。
- 再生中、並行して作業している別の試験が ds41gw を何度か再起動した (`/v1/proxy/stats` の `uptime_s`)。最初の試し打ちの 1 回はその再起動に当たって切れたので、数に入れていない。

## 8. 追記: 最終版での追加遅延の再測定

§5 で「要求ログ用の複製と直列化が、上流へ送る前に同期で走っている」と見ていた点を直した後に、同じ手順で測り直した。

- 直したこと:
  - 受信の行の画像の伏せ字と直列化を、要求ログの書き込みスレッドに移した (`proxy/ds41gw/src/reqlog.rs` の `render`、`proxy/ds41gw/src/core.rs` の `Ctx::log_received`)。
    `orig` は受け取った本文の `Bytes` (参照の複製) から書き込みスレッドで読み直す。
  - Chat の `prepare` に渡す `orig.clone()` をやめた (`proxy/ds41gw/src/chat.rs` の `handle`)。
  - 上流へ送る本文を、`stream:true` にするための深い複製をせずに直列化するようにした (`proxy/ds41gw/src/upstream.rs` の `StreamBody`)。
- 手順: `python3 tests/replay_real.py pairs --in tests/results_replay_real.jsonl --out tests/results_replay_pairs_c1_final.jsonl --conc 1 --abba --max-tokens 8`。
  68 組すべてが両経路とも 200 だった。
- 結果 (TTFT の差、プロキシ経由 − 直):

| | 組 | p50 (前 → 後) | p90 (前 → 後) |
|---|---:|---:|---:|
| Chat | 38 → 36 | +3.4 → +1.6 ms | +12.0 → +11.8 ms |
| Responses | 21 → 20 | +3.5 → +2.9 ms | +22.4 → +22.3 ms |
| 全体 | 59 → 56 | +3.5 → +2.4 ms | +20.8 → +20.9 ms |

- 新しく足した `ds41gw_overhead_seconds` (要求ログでは `overhead_ms`。本文を読み終えてから上流へ送り出すまで、待ち行列を除く) は、
  この再生と SDK 試験を含む要求で Chat p50 0.62 ms / 最大 2.4 ms (113 件)、Responses p50 1.03 ms / 最大 2.9 ms (85 件)、Messages p50 0.17 ms (20 件) だった。
  Responses に残る約 1 ms は、主に recipe の変換と逆写像の分と見ている (内訳は測っていない)。
- p90 の +20 ms 前後の外れ値は変わらない。§5 のとおり ds41rt の TCP_NODELAY (`ds41rt-daemon/src/commands/coordinator.rs:148`) が原因で、プロキシ側では直せない。
