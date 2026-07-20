# 申し送り: 2026-07-20 監査セッション (Claude Opus 4.8 → 次セッション)

事実のみ。判断・優先順位・推奨は含めない。
各項目に **検証状況** を付す。「実測」= 本セッションでコマンドを実行して得た出力。
「サブエージェント報告」= 並列監査エージェントの出力で、私が独立に再検証していないもの。

注意: このファイルは `tmp/` にあり `.gitignore` 対象。git には入っていない。
同内容の要点は `~/.claude/projects/-home-puniu-nc2000-engine/memory/` に4ファイルある
(`eval-engine-divergences` / `bot-blind-spot-map` / `format-regulation-mismatch` / `cx-cpu-offload`)。

---

## 1. セッション開始時の状態

- 前セッションは WSL 破綻により非常停止。`tmp/OFFLOAD-NOTES-from-cleanup.md` に前任者の申し送りあり。
- 作業ツリーに未コミットの変更があった: `crates/bot/src/eval.rs`, `crates/bot/src/mcts.rs`,
  `crates/engine/src/battle/pokemon.rs` (M) と `crates/bot/tests/evasion.rs`,
  `crates/bot/examples/replay_analysis.rs`, `crates/bot/examples/rental_strength.rs`,
  `data/preview-tables-research/`, `data/research-v0/` (未追跡)。
- リポジトリルートに Showdown リプレイHTML 2件（ファイル名・本文に両プレイヤーのハンドル）。

**実測** 開始時 load average 2.19 / メモリ 7940MB中 available 6116MB / nproc 12。

## 2. 本セッションで行った git 操作

**push はしていない**（public repo + Pages のため）。ローカルコミット4件、ツリーは clean。

| commit | 内容 |
|---|---|
| `4a93a0c` | evasion 修正の退避。`Battle::hit_probability` 追加 + eval の threat 項に結合。`EvalWeights::couple_evasion` 追加。`.gitignore` に `*battle-gen2nintendocup2000*.html` と `.playwright-mcp/` を追加（リプレイHTMLはハンドルを含むため除外、commit `4a5b0f9` と同じ理由） |
| `609d3c3` | 修正前の研究成果物の退避（認証テーブル13枚、lineage 12件）+ `rental_strength.rs` |
| `6b0960d` | `crates/bot/examples/damage_conformance.rs` 新規（差分ハーネス、先頭1体版） |
| `e6505df` | 同ハーネスを選出3体×3体に拡張 |

## 3. 実行したテストと結果

**実測**
- `cargo check -j 4 -p nc2000-bot --tests --examples` → 通過
- `cargo test --release -j 4 -p nc2000-bot --test evasion` → 5 passed 0 failed
- `cargo test --release -j 4 -p nc2000-bot` → 全緑
  (bots 10 / evasion 5 / import 1 / preview_tables 2 / teamgen 5)

`preview_tables.rs` は2件・0.19秒で完了する。焼いたテーブルの内容を新evalで検証してはいない。

## 4. 差分ハーネスの実測結果

再現コマンド:
```
cargo run --release -j 4 -p nc2000-bot --example damage_conformance
```

`eval::expected_hit_fraction` と engine の `get_damage_synthetic` を、命中率を除算し
急所・乱数を両側で抑止した上で比較。メタプール34チームの全ペア、選出3体×3体、20222トリプル。

```
move          n      mean     worst
return       432    0.0000    0.0000
explosion   2417    0.4966    0.4499
selfdestruct 430    0.5029    0.4937
fissure      184    0.0000    0.0000
horndrill     87    0.0000    0.0000
（他36技）           0.985〜1.018
cost per call:  eval 318 ns   engine core 1366 ns   ratio 4.3x
```

先頭1体のみの版（2301トリプル）では `return` は検出されなかった。

**ハーネスの適用範囲**: 各ペアの初手局面のみ。boost / 壁 / 天候 / 状態異常はいずれも未設定。
天候倍率・1.1倍タイプ強化アイテム・256スタットロールオーバー・997上限の各乖離は、
この条件では発火しないため**未検証**。

## 5. コード上で確認した事実

**実測（dex を直接読んで確認）**
- `return` の `basePower` は 0、category は Physical。同様に bp=0 なのは
  `frustration, flail, reversal, magnitude, present, counter, mirrorcoat, bide, hiddenpower`。
- `eval.rs:257` は `ms.category == Category::Status || base_power <= 0` で `0.0` を返す。
  `hiddenpower` のみ `eval.rs:220-224` の分岐で type/power が差し替わる。
- 型別 Hidden Power エントリは dex に存在し、category は正しい
  (`hiddenpowerice` = Special, `hiddenpowerbug` = Physical 等、17エントリ)。
  総称 `hiddenpower` のみ Physical / bp=0。
- `moveexec.rs:2664`: `if selfdestruct && def_stat == 1 { defense = floor(defense/2) }`。
- `moveexec.rs:337-348`: `("hiddenpower","onModifyMove")` が実行時に move_type と category を再代入。
- `get_damage` は `&mut self`。PRNG消費点は2箇所:
  `prng.random_chance(1, CRIT_MULT[crit_ratio])`（`will_crit.is_none()` のとき）と
  `prng.random_range(217,256)`（`!no_damage_variance` のとき）。
- `get_damage_synthetic` は `pub`。`replay_analysis.rs:8` が既に使用しており、
  同ファイル 32-45行に「`get_damage` は `onModifyMove` を走らせないので Hidden Power の
  type/category を手で植える」旨のコメントと実装がある。
- `preview.rs:687` は `EvalWeights::default()` を使用。
- `expected_hit_fraction` / `best_hit_fraction` を engine の実ダメージと突き合わせるテストは
  リポジトリに存在しない（`4a93a0c` 以前は参照テスト自体が0件）。
- `PairTable` (`preview.rs:218-238`) のフィールドは team_a/team_b/actions/space_version/
  screen/support/refine/sol/cfg/secs。eval・エージェントの指紋欄は無い。
- `grep` で `crates/bot/src/eval.rs` に volatile / side_condition / weather / field への
  参照は0件（ヒットは doc コメント1行と `ms.disabled` 1行のみ）。

**実測（メタプール 204セット / 34チームを集計）**
| 項目 | 数 |
|---|---|
| Mean Look | 16セット / Perish Song 16セット / 両方持ち 15セット・15チーム |
| Rest | 55セット |
| Explosion + Selfdestruct | 53セット |
| Substitute 6 / Double Team 3 | |
| bp=0 の攻撃技を持つセット | 16（Return 11, Counter 2, Reversal 1, Frustration 1, Present 1） |
| Hidden Power | 44セット（Ice 16, Bug 15, Grass 6, Flying 4, Fighting 3） |

**実測（フォーマット関連）**
- リポジトリ内で `gen2nc2000` を指す箇所: `validate.rs:1`, `essence.rs:14`,
  `battle/mod.rs:386`, `tools/export-learnsets.js`, `tools/build-meta-pool.js`,
  `tools/ps.js:10`, `tools/ps-client.js`（`--format gen2nc2000` 既定値）。
- リプレイログのルール行: Species Clause / Item Clause / Sleep Clause Mod /
  Freeze Clause Mod / **OHKO Clause** / HP Percentage Mod。
  tier 文字列は `gen2nintendocup2000noohkostadium2strict`。
- `data/meta-pool-v0/README.md` の記述:
  「The shipped PS `gen2nc2000` ruleset has **no evasion, OHKO, or Bright Powder bans**
  … so HC7.5 sets with Bright Powder / Double Team / Horn Drill / Fissure are legal verbatim.」
- メタプールで即死技を持つチーム3件（いずれも T1 = Historia Cup 7.5 top-8）:
  `hc75-4th-kg`(Steelix Fissure) / `hc75-top8-tako`(Snorlax Fissure) /
  `hc75-top8-shinobu`(Nidoking Horn Drill, Tauros Horn Drill)。
- `data/learnsets-gen2.json` は `meta.format: gen2nc2000`。**50種**の学習セットが
  即死技を許可（charizard, blastoise, nidoqueen, nidoking, poliwrath, golem, arbok,
  dugtrio, ekans, sandshrew, sandslash, nidorina, nidoranm, nidorino, diglett, …）。
- `crates/engine/src/validate.rs` に ohko / fissure / horndrill / guillotine の文字列は0件。
  実装されているクローズは species / item / nickname。
- engine は即死技を実装している（`dex.rs:518` の `ohko: bool`、
  `moveexec.rs:1868-1898`、`moveexec.rs:2022`）。禁止処理は無い。
- `data/community-rentals-v0/` は 28チーム / 161セット / 43種。
  `ASSESSMENT.md` に「**「一撃無し２０００ルール」**(no-OHKO NC2000) metagame」由来と明記。
  そのうち cban 23 (Tauros Horn Drill, Machamp Fissure) / 24 (Snorlax Fissure) /
  25 (Dugtrio Fissure, Tauros Horn Drill) が即死技を持つ。
- `tools/parse-community-rentals.js:284` に既存の記述:
  `ruleset: 'no-OHKO NC2000 variant (differs from shipped gen2nc2000 which bans neither
  OHKO nor evasion nor Bright Powder)'`

## 6. 本セッション中に撤回した主張

- 私は当初「Hidden Power が誤ったステータス対で採点されている（22セット影響）」と
  断定して報告した。**誤りだった。** dex は型別エントリを持ち category は正しく、
  プールのセットはそちらを使う。ハーネス実測で HP 系は全て比 1.00〜1.01。
  総称 `hiddenpower` 経路（PSプロトコル由来の可能性）は未検証のまま。
- サブエージェントが「Mean Look + Perish Song 16/16セット、15/16チーム」と報告したが、
  実測では両方持ち15セット・**34チーム中15チーム**。

## 7. サブエージェント報告（私が独立検証していないもの）

6面の並列監査を実施。以下は各エージェントの報告であり、上記「実測」欄に無いものは
**私による再検証を経ていない**。

- **volatile/side condition 監査**: engine の condition 登録は51個
  （status 7 / on-mon volatile 34 / side 4 / field 3 / rule 2）。eval の参照は0。
  プール footprint 順のランキングあり。睡眠ターンカウンタは `status_state.DK::Time`
  (`conditions.rs:148-151`、tick は `:160-166` の `onBeforeMove` のみ)、
  Toxic カウンタは `residualdmg` volatile の `DK::Counter`。
- **探索監査**: `greedy_pick` (`mcts.rs:291-329`) は `SearchChoice::Move` のみ走査するため
  両者生存時に自発交代を選べない。全技0点のとき `cs[rng.below(cs.len())]` に落ちる。
  プール204セット中10セットが点数化可能な技を持たない（8/34チーム）。
  `hp_buckets:16` は transposition key のみに影響。同時手番の情報漏洩は無しと結論。
  wasm の shipped agent は argmax-visits（純戦略）。
- **隠れ情報監査**: 実戦経路で bot は相手の真の状態を受け取らない（`ProtocolSearcher` は
  `Dex` のみから構築、`import.rs:1094-1104` で相手ロスターを belief 由来に差し替え）。
  非プール相手では `Belief` が候補0となりフォールバック1体に退化。
  敗戦2局とも両チームが非プールで、`TableSet::side_index` が None を返し、
  595テーブルは参照されていない、と報告。
- **人間コーパス分析**: `nc2000stadium2_spectator_logs.zip` の570戦・12226ターンを集計。
  eval が見られない状態が87.9%の対戦・58.5%のターンに存在。ターン比で Spikes 21.4%、
  睡眠カウンタ20.1%、Substitute 9.2%、混乱8.6%。Explosion は58.1%の対戦で使用。
  Rest が最多使用技（1053回）。Snorlax は97.9%のチームに在籍。
  プール41種に対し人間は151種を使用。**集計は全てエージェントが行い、私は再実行していない。**
- **preview/bake 監査**: `TableSet::lookup` (`preview.rs:450`) は完全一致署名。
  support 選抜の採否境界の BR utility 差は中央値0.006、同統計の標準誤差は約0.065、
  18ペア中3ペアで均衡台が screen top-8 の外、と報告。
- **敗戦検死**: 両局とも 3-0、人間側の損害0。1局目T2でGengarを繰り出しThunder Waveで麻痺、
  T3でZap Cannon（Perish Song / Destiny Bond は eval 0.000）。2局目は Substitute +
  Double Team +6 → Baton Pass → Belly Drum Snorlax の Return で3体。
  evasion 修正（`4a93a0c`、19:39）は両局（12:15 / 14:30）より後であり、**両局は修正前バイナリ**。

## 8. 未解決事項

- **敗戦検死が報告した「説明のつかない着手」**（私による再現未実施）:
  1局目T6 Dynamic Punch (eval 0.053) を Fire Blast (0.365) より選択、
  1局目T12 Hidden Power (0.314) を Earthquake (0.667) より選択。後者は同カテゴリ・同壁・
  同命中帯で威力半分。エージェントの候補仮説は (a) 1000イテレーション下の訪問回数ノイズ、
  (b) `SearchChoice::Move` → `/choose move N` のスロット対応バグ
  （全16選択でスロット2が一度も使われていない）。**いずれも未確認。**
- コミュニティレンタルDB（no-OHKO由来と明記）の3チームが即死技を持つ理由。
- 2局目のログに turn 3 (`t:1784525768`) → turn 4 (`t:1784526843`) で約18分の壁時計ギャップ。
  M15b クライアントの再接続経路との関係は未調査。
- 総称 `hiddenpower` が PS プロトコル経由で届く場合の eval 挙動。
- ハーネス未適用領域（boost / 壁 / 天候 / 状態異常のある局面）。

## 9. 未着手のまま残っている前セッションからの作業

- M11 認証テーブル: 13/32 完了（`data/preview-tables-research/`、pool-05 が5/8、rand-3 が8/8）。
  残り19枚。前任申し送りの実測は 1枚あたり3〜5分（`--threads 11`）。
- M11a 本走: 未 launch。計画は README M11a 節。
- 595ペア再bake: 未着手。ユーザ指示により **bake は無期限延期**。

## 10. 環境

- `~/cx` に CPU オフロード用ハーネス（GCP Spot `c2d-highcpu-*`、us-central1、
  ladder 4/8/16/32/56、`MAX_VCPU=56`）。`cx` は PATH 上。systemd-user timer
  `cx-reconcile.timer` 稼働中。2026-07-20 に end-to-end 検証済みと README に記載。
- 前任申し送り（`tmp/OFFLOAD-NOTES-from-cleanup.md`）が記録するWSL破綻の原因:
  (1) `pgrep -f` の自己マッチによる16.5時間の空転、(2) `--threads 11` と `--threads 10` の
  同時起動（12コアに21スレッド要求）。加えて本セッションで確認した事実として、
  当該機のメモリは7940MB。
- `bake_preview` は `--pairs i:j,...` と `--teams lo-hi` を受け付ける
  (`bake_preview.rs:165-189`)。`pair_seed = cfg.seed ^ ((i as u64) << 32) ^ ((j as u64) << 16)`
  (`bake_preview.rs:211`) で、分割方法に依存しない。既存ファイルはスキップされる。

## 11. 敗戦リプレイの所在

public repo に入れられない（ログ本文に対戦相手のハンドルが含まれる）ため、
**リポジトリ外**に退避してある。git のどの操作からも影響を受けない。

```
~/nc2000-replays/battle-3623.log   3989 chars  (12:15 の対局)
~/nc2000-replays/battle-3629.log   4132 chars  (14:30 の対局)
```

Showdown プロトコル形式。元の HTML も同ディレクトリにあり、リポジトリルートにも
残っている（`.gitignore` の `*battle-gen2nintendocup2000*.html` で除外済み）。
HTML からの再抽出は `re.search(r'class="battle-log-data">(.*?)</script>', s, re.S)`。

## 12. 本セッションで発生させた問題

commit `4a93a0c`（本セッション）で、`crates/bot/examples/replay_analysis.rs:1` と
`crates/bot/tests/evasion.rs:5` のコメントに**対戦相手のハンドルを書き込んだ状態でコミットした**。
これは commit `4a5b0f9`（「external human-play logs (player handles; public repo)」）が
確立した方針に反する。

- 現在のファイルからは除去済み（後述のコミット）。
- **git 履歴には残っている**（`4a93a0c` 以降の全コミット）。
- **未 push のため公開はされていない。**
- 履歴の書き換え（rebase / amend / force-push）は設定で拒否されているため私は実行できない。
  push 前に履歴から消すならユーザによる squash / filter が必要。そのままでよければ
  fix-forward 済みの現状で足りる。

## 13. 本セッションで実行していないこと

- push（public repo のため未実施）
- 人間コーパスの集計の再実行
- 敗戦検死が挙げた「説明のつかない着手」の再現
- 修正前後での対戦強度の実測（arena 等）
- ハーネスの摂動局面への拡張
