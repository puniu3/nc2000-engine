# nc2000-engine

Pokemon Showdown の **`[Gen 2] NC 2000`**（mod: `gen2stadium2`）を Rust に移植し、bot 研究の探索速度を桁上げするプロジェクト。

**準拠先は「PS に現に実装されているもの」**。カートリッジ実機や Stadium 2 実機との乖離は追わない。正しさの定義 = `tools/gen-fixtures.js` が PS から生成したゴールデンフィクスチャとの**ビット一致**（状態 + PRNG シード、全スナップショット点で）。

## 構成

```
tools/            PS(参照実装)から生成する Node スクリプト群（要: PS_ROOT=PSリポジトリ, node build 済み）
  export-dex.js            平坦化した gen2stadium2 Dex を data/ へ書き出す
  gen-prng-vectors.js      PRNG パリティ用ベクタ
  gen-fixtures.js          ゴールデンフィクスチャ生成（live対戦→inputLogリプレイ→スナップショット抽出）
  gen-porting-checklist.js PORTING.md 再生成
data/gen2stadium2.json     参照データ（関数はコールバック名リストに置換済み、psCommit をmetaに記録）
fixtures/prng-vectors.json PRNG ベクタ
fixtures/corpus-v1/        60バトル（puredata 30 + full 30、計2,268ターン/2,585スナップショット）
crates/engine/             エンジン本体（prng 完動 / dex ローダ / state / choice / battle=未実装スタブ）
crates/conformance/        適合性ハーネス（フィクスチャschema・差分レポータ・replayテスト）
PORTING.md                 移植チェックリスト（377コールバック、生成物）
```

## 検証の仕組み（スナップショット契約）

- フィクスチャの `choices` は PS inputLog の正規化済み選択列（例: `team 5, 6, 1` / `move surf` / `switch 2`）。
- スナップショット点 = **ログが伸びた入力行の直後**。各点で `turn / requestState / prngSeed / field / sides(全ポケモンのHP・状態・ランク・PP・volatiles)` + そのターンのログ行を記録。
- `prngSeed` は PS `Gen5RNG.getSeed()` 形式（10進16bitリム4つのカンマ結合）。**シード一致 = 乱数消費順の一致**。結果だけ合っていても消費順がズレれば即検出される。
- `|t:|`（壁時計）は生成時に除去済み。

## ワークフロー

```bash
# 全テスト（現在: PRNGパリティ・dexロード・フィクスチャschema = green）
cargo test
# 本丸のリプレイ適合（エンジンがマイルストーン1に達したら ignored を外す）
cargo test -p conformance --test replay -- --include-ignored
# フィクスチャ再生成（PS更新後など）
node tools/export-dex.js && node tools/gen-porting-checklist.js
node tools/gen-fixtures.js --n 30 --pool puredata --out fixtures/corpus-v1/puredata --seed 100
node tools/gen-fixtures.js --n 30 --pool full     --out fixtures/corpus-v1/full     --seed 200
```

移植の進め方: コールバックを1個移植するたびに `PORTING.md` をチェック → replay テストが緑のまま成長させる。乖離時は `compare::Divergence` が「最初に割れたスナップショット + JSONパス + そのターンのログ」まで自動局所化する。

### マイルストーン

1. **M1 — puredata corpus 緑**: チーム初期化（Gen2 DV/努力値→実数値、`data/mods/gen2/scripts.ts` の式）→ チームプレビュー → 交代 → 純データ技のダメージパイプライン（Gen2 getDamage）→ 残留処理 + 状態異常37種。
2. **M2 — full corpus 緑**: コールバック技88種 + アイテム38種 + Stadium Sleep/Freeze Clause 等のランタイムルール。
3. **M3 — 探索API**: `Battle` は既に flat/Copy 設計。apply/undo か clone ベースの列挙APIを載せ、DUCT/MCTS へ。
4. 以後: exhaustive-runner 型のカバレッジ強制コーパス、友人シナリオfixture、実対戦の予測/実測自動差分。

## 実測ベースライン（この機・WSL）

- PS(TS): 6.5バトル/s・570ターン/s・クローン5.5ms → 木探索は不成立
- 目標: 10⁵–10⁶ターン/s、クローン=サブμs（同種先行例 pkmn/engine は公称 PS比1000x超）

## 移植時の地雷（このリポジトリで実測済みの事実）

1. **sim は渡された特性を Gen 2 でも発動させる**。バリデータ正規形は `ability: 'No Ability'`。空文字のままpack往復すると種族デフォルト特性が充填され、Shed Skin等が発動して対戦結果が変わる（battle-005で実証）。フィクスチャは全て validator 通過済み。
2. **Gen 2 の DV 整合性**: SpA DV = SpD DV、HP DV = Atk/Def/Spe/Spc の最下位ビットから導出。満たさないチームはバリデータが弾く。
3. **チームプレビュー後、PS は `side.pokemon` を選出3匹に切り詰める**。スナップショットのパーティサイズは 6(プレビュー時) → 3(開始後)。
4. リプレイは inputLog の `>player` 行（packedチーム）から構成しないと live と一致しない（生成器は対策済み）。
5. PRNG は Gen5 64bit LCG。`random(n)` は `(next * n) >> 32`（JS float 演算と n < 2^21 で完全一致、debug_assert 済み）。

## 参照

- 参照実装: `PS_ROOT`（デフォルト `~/pokemon-showdown`、`node build` 必須）。データの由来コミットは `data/gen2stadium2.json` の `meta.psCommit`。
- スコープ実測: 技267(コールバック持ち88) / アイテム62(38) / 状態37 / 特性0 / 種族251(Uber5除外で246)、コールバック計377、使用イベントフック76種。
