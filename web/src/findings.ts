// M14b: localized messages for machine-readable findings — the wasm
// Validator's `{severity, code, ...params}` objects (crates/engine/src/
// validate.rs is the code catalogue) plus the PS-export parser's `ps-*`
// codes (ps-import.ts). One template table per locale, keyed by
// `severity:code` — the same code can appear as an error (validation
// verdict) and as a fix (canonicalization note) with different phrasing.
// Unknown codes fall back to the raw code + params, so a validator ahead
// of this table degrades readably instead of crashing.

import { itemName, locale, moveName, speciesName, typeName } from "./i18n";

export interface Finding {
  severity: "error" | "fix";
  code: string;
  mon?: string;
  slot?: number;
  line?: number;
  [k: string]: unknown;
}

type Tmpl = (f: Finding) => string;

const s = (v: unknown) => String(v ?? "?");
const list = (v: unknown) => (Array.isArray(v) ? v.join(", ") : s(v));
/** Move display from a PS id ("bodyslam"): ja name table hit, en keeps the id. */
const mv = (v: unknown) => moveName(s(v));
const it = (v: unknown) => itemName(s(v));
const ty = (v: unknown) => typeName(s(v));

const EN: Record<string, Tmpl> = {
  // ---- validation errors
  "error:json-invalid": (f) => `Not a valid team: ${s(f.detail)}`,
  "error:team-size": (f) =>
    f.min !== undefined
      ? `A team needs at least ${s(f.min)} Pokémon (got ${s(f.size)})`
      : `A team can have at most ${s(f.max)} Pokémon (got ${s(f.size)})`,
  "error:species-clause": () => `Duplicate species (Species Clause)`,
  "error:item-clause": (f) => `Duplicate item “${it(f.item)}” (Item Clause)`,
  "error:nickname-clause": (f) => `Duplicate nickname “${s(f.name)}”`,
  "error:level-sum": (f) =>
    `The 3 lowest levels already sum to ${s(f.sum)} — over the ${s(f.limit)} total-level limit`,
  "error:level-sum-highest": (f) =>
    `The L${s(f.level)} Pokémon can never be picked: any 3 including it exceed the ${s(f.limit)} total-level limit`,
  "error:species-unknown": (f) => `Unknown species “${s(f.species)}” (not in Gen 2)`,
  "error:species-banned": () => `Banned from this format (Uber)`,
  "error:level-min": (f) => `Level ${s(f.level)} is below the minimum (${s(f.min)})`,
  "error:level-max": (f) => `Level ${s(f.level)} is above the maximum (${s(f.max)})`,
  "error:species-underleveled": (f) =>
    `Must be at least level ${s(f.min)} (evolution floor; currently ${s(f.level)})`,
  "error:move-none": () => `No moves`,
  "error:move-count": (f) => `Too many moves (${s(f.count)}; max ${s(f.max)})`,
  "error:move-unknown": (f) => `Unknown move “${mv(f.move)}”`,
  "error:move-duplicate": (f) => `Duplicate move “${mv(f.move)}”`,
  "error:move-illegal": (f) => `Can't learn ${mv(f.move)}`,
  "error:move-level": (f) =>
    `${mv(f.move)} needs at least level ${s(f.min)} (currently ${s(f.level)})`,
  "error:hp-type-conflict": (f) =>
    `Two typed Hidden Powers (${ty(f.a)} / ${ty(f.b)})`,
  "error:hp-type-mismatch": (f) =>
    `Hidden Power ${ty(f.want)}, but these DVs give ${ty(f.derived)}`,
  "error:item-unknown": (f) => `Unknown item “${s(f.item)}” (not in Gen 2)`,
  "error:dv-spc": () =>
    `Sp. Atk and Sp. Def DVs must be equal (Gen 2 has one Special DV)`,
  "error:dv-hp": (f) =>
    `HP DV must be ${s(f.expected)} (it is derived from the other DVs)`,
  "error:dv-gender": (f) =>
    `Gender ${s(f.gender)} contradicts the Attack DV (expected ${s(f.expected)})`,
  "error:dv-shiny": (f) =>
    f.expected === true
      ? `These DVs make it shiny — set Shiny: Yes`
      : `Shiny needs exact DVs (Def/Spe/Spc 10, Atk DV 2–3 mod 4)`,
  "error:unown-forme": (f) =>
    `These DVs give Unown letter ${s(f.letter)} — only forme A is supported`,
  "error:ev-range": (f) => `Stat exp out of range: ${s(f.stat)} = ${s(f.value)}`,
  "error:ev-spc": () => `Sp. Atk and Sp. Def stat exp must be equal`,
  "error:ev-zero": () => `All-zero stat exp is rejected`,
  "error:nickname-length": (f) =>
    `Nickname longer than ${s(f.max)} characters`,
  "error:nickname-species": (f) =>
    `Nickname “${s(f.name)}” impersonates another species`,
  // ---- canonicalization fixes (informational)
  "fix:level-default": (f) => `level set to ${s(f.level)}`,
  "fix:gender-fill": (f) => `gender set to ${s(f.expected)} (from the Attack DV)`,
  "fix:gender-species": (f) => `gender set to ${s(f.expected)} (fixed for this species)`,
  "fix:ability-canonical": () => `ability normalized to “No Ability”`,
  "fix:nature-canonical": () => `nature normalized to Serious`,
  "fix:happiness-range": () => `happiness clamped to 0–255`,
  "fix:evs-missing": () => `stat exp filled at max (255)`,
  "fix:move-duplicate": (f) => `duplicate move removed (${mv(f.move)})`,
  "fix:iv-range": (f) => `IVs clamped to 0–31 (${list(f.stats)})`,
  "fix:unown-forme": () => `DVs rewritten to the Unown-A spread`,
  "fix:hp-type-dvs": (f) => `DVs set to the Hidden Power ${ty(f.type)} spread`,
  "fix:dv-spc": () => `Sp. Def DV mirrored from Sp. Atk`,
  "fix:dv-hp": (f) => `HP DV derived from the others (${s(f.expected)})`,
  "fix:dv-gender": (f) => `gender corrected to ${s(f.expected)} (Attack DV)`,
  "fix:dv-shiny": (f) =>
    f.expected === true ? `shiny set (the DVs are shiny)` : `shiny flag cleared (DVs are not shiny)`,
  "fix:ev-range": (f) => `stat exp clamped to 0–255 (${list(f.stats)})`,
  "fix:ev-spc": () => `Sp. Def stat exp mirrored from Sp. Atk`,
  "fix:ev-zero": () => `all-zero stat exp refilled at max`,
  "fix:nickname-species": () => `nickname reset (matches another species' name)`,
  "fix:nickname-length": () => `nickname truncated to 18 characters`,
  "fix:nickname-clause": () => `duplicate nickname reset`,
  // ---- parser errors (line-anchored)
  "error:ps-empty": () => `Nothing to import — paste a team in the PS export format`,
  "error:ps-multiple-teams": () =>
    `The paste contains more than one team — import one at a time`,
  "error:ps-header-expected": (f) =>
    `Expected a Pokémon header line (“Name @ Item”), got “${s(f.text)}”`,
  "error:ps-line-unknown": (f) => `Unrecognized line: “${s(f.text)}”`,
  "error:ps-stat-chunk": (f) =>
    `Can't read stat entry “${s(f.chunk)}” (expected e.g. “252 Atk”)`,
  "error:ps-stat-unknown": (f) => `Unknown stat name “${s(f.stat)}”`,
  "error:ps-number": (f) => `${s(f.field)} must be a number`,
};

const JA: Record<string, Tmpl> = {
  "error:json-invalid": (f) => `チームとして解釈できません: ${s(f.detail)}`,
  "error:team-size": (f) =>
    f.min !== undefined
      ? `チームには最低${s(f.min)}体必要です(現在${s(f.size)}体)`
      : `チームは最大${s(f.max)}体までです(現在${s(f.size)}体)`,
  "error:species-clause": () => `同じポケモンが複数います(種族条項)`,
  "error:item-clause": (f) => `同じ持ち物が複数あります: ${it(f.item)}(アイテム条項)`,
  "error:nickname-clause": (f) => `ニックネームが重複しています: 「${s(f.name)}」`,
  "error:level-sum": (f) =>
    `レベルの低い3体だけで合計${s(f.sum)} — 合計レベル上限${s(f.limit)}を超えます`,
  "error:level-sum-highest": (f) =>
    `L${s(f.level)}のポケモンは選出不可能です: どの3体に入れても合計上限${s(f.limit)}を超えます`,
  "error:species-unknown": (f) =>
    `不明なポケモン: 「${s(f.species)}」(第2世代に存在しません)`,
  "error:species-banned": () => `このフォーマットでは使用禁止です(Uber)`,
  "error:level-min": (f) => `レベル${s(f.level)}は下限(${s(f.min)})未満です`,
  "error:level-max": (f) => `レベル${s(f.level)}は上限(${s(f.max)})を超えています`,
  "error:species-underleveled": (f) =>
    `レベル${s(f.min)}以上が必要です(進化レベル制限; 現在${s(f.level)})`,
  "error:move-none": () => `わざがありません`,
  "error:move-count": (f) => `わざが多すぎます(${s(f.count)}個; 最大${s(f.max)}個)`,
  "error:move-unknown": (f) => `不明なわざ: 「${mv(f.move)}」`,
  "error:move-duplicate": (f) => `わざが重複しています: ${mv(f.move)}`,
  "error:move-illegal": (f) => `${mv(f.move)}は覚えられません`,
  "error:move-level": (f) =>
    `${mv(f.move)}の習得にはレベル${s(f.min)}以上が必要です(現在${s(f.level)})`,
  "error:hp-type-conflict": (f) =>
    `タイプ指定のめざめるパワーが2つあります(${ty(f.a)} / ${ty(f.b)})`,
  "error:hp-type-mismatch": (f) =>
    `めざめるパワー(${ty(f.want)})ですが、このDVでは${ty(f.derived)}タイプになります`,
  "error:item-unknown": (f) =>
    `不明な持ち物: 「${s(f.item)}」(第2世代に存在しません)`,
  "error:dv-spc": () =>
    `とくこう/とくぼうのDVは同じ値が必要です(第2世代の特殊DVは1つ)`,
  "error:dv-hp": (f) =>
    `HPのDVは${s(f.expected)}である必要があります(他のDVから導出されます)`,
  "error:dv-gender": (f) =>
    `性別${s(f.gender)}はこうげきDVと矛盾します(正しくは${s(f.expected)})`,
  "error:dv-shiny": (f) =>
    f.expected === true
      ? `このDVは色違いです — Shiny: Yes にしてください`
      : `色違いには特定のDVが必要です(防/速/特 10、攻DV 2–3 mod 4)`,
  "error:unown-forme": (f) =>
    `このDVではアンノーンの文字が${s(f.letter)}になります(対応はAのみ)`,
  "error:ev-range": (f) => `努力値が範囲外です: ${s(f.stat)} = ${s(f.value)}`,
  "error:ev-spc": () => `とくこう/とくぼうの努力値は同じ値が必要です`,
  "error:ev-zero": () => `努力値がすべて0のセットは拒否されます`,
  "error:nickname-length": (f) =>
    `ニックネームが${s(f.max)}文字(UTF-16)を超えています`,
  "error:nickname-species": (f) =>
    `ニックネーム「${s(f.name)}」は他のポケモンの名前と紛らわしいため使えません`,
  "fix:level-default": (f) => `レベルを${s(f.level)}に設定`,
  "fix:gender-fill": (f) => `性別を${s(f.expected)}に設定(こうげきDV由来)`,
  "fix:gender-species": (f) => `性別を${s(f.expected)}に設定(この種族は固定)`,
  "fix:ability-canonical": () => `特性を「No Ability」に正規化`,
  "fix:nature-canonical": () => `性格をSerious(まじめ)に正規化`,
  "fix:happiness-range": () => `なつき度を0–255に収めました`,
  "fix:evs-missing": () => `努力値を最大(255)で補完`,
  "fix:move-duplicate": (f) => `重複したわざを削除(${mv(f.move)})`,
  "fix:iv-range": (f) => `個体値を0–31に収めました(${list(f.stats)})`,
  "fix:unown-forme": () => `アンノーンAのDV配分に書き換え`,
  "fix:hp-type-dvs": (f) => `めざめるパワー(${ty(f.type)})のDV配分に設定`,
  "fix:dv-spc": () => `とくぼうDVをとくこうDVに揃えました`,
  "fix:dv-hp": (f) => `HPのDVを他のDVから導出(${s(f.expected)})`,
  "fix:dv-gender": (f) => `性別を${s(f.expected)}に修正(こうげきDV由来)`,
  "fix:dv-shiny": (f) =>
    f.expected === true
      ? `色違いに設定(DVが色違いの値)`
      : `色違いを解除(DVが色違いの値ではありません)`,
  "fix:ev-range": (f) => `努力値を0–255に収めました(${list(f.stats)})`,
  "fix:ev-spc": () => `とくぼう努力値をとくこうに揃えました`,
  "fix:ev-zero": () => `すべて0の努力値を最大で補完`,
  "fix:nickname-species": () => `ニックネームをリセット(他の種族名と衝突)`,
  "fix:nickname-length": () => `ニックネームを18文字に切り詰め`,
  "fix:nickname-clause": () => `重複したニックネームをリセット`,
  "error:ps-empty": () =>
    `読み取れるチームがありません — PSエクスポート形式で貼り付けてください`,
  "error:ps-multiple-teams": () =>
    `複数のチームが含まれています — 1チームずつ取り込んでください`,
  "error:ps-header-expected": (f) =>
    `ポケモンの見出し行(「名前 @ 持ち物」)が必要です: 「${s(f.text)}」`,
  "error:ps-line-unknown": (f) => `解釈できない行です: 「${s(f.text)}」`,
  "error:ps-stat-chunk": (f) =>
    `能力値の項目「${s(f.chunk)}」を解釈できません(例: 252 Atk)`,
  "error:ps-stat-unknown": (f) => `不明な能力名: 「${s(f.stat)}」`,
  "error:ps-number": (f) => `${s(f.field)}は数値が必要です`,
};

/** Localized message body for one finding (no anchor — the caller renders
 * the mon/slot/line chip). */
export function findingText(f: Finding): string {
  const table = locale() === "ja" ? JA : EN;
  const tmpl = table[`${f.severity}:${f.code}`];
  if (tmpl) return tmpl(f);
  // unknown code: degrade readably
  const params = Object.entries(f)
    .filter(([k]) => !["severity", "code", "mon", "slot"].includes(k))
    .map(([k, v]) => `${k}=${String(v)}`)
    .join(", ");
  return params ? `${f.code} (${params})` : f.code;
}

/** Anchor label for a finding: the mon it points at ("#2 Snorlax"), the
 * paste line ("L12"), or null for team-level findings. */
export function findingAnchor(f: Finding): string | null {
  if (f.mon !== undefined && f.slot !== undefined) {
    const id = s(f.mon);
    const nm = speciesName(id);
    // en fallback shows the raw PS id — capitalize it for display
    const disp = nm === id ? id.charAt(0).toUpperCase() + id.slice(1) : nm;
    return `#${Number(f.slot) + 1} ${disp}`;
  }
  if (f.line !== undefined) {
    return locale() === "ja" ? `${s(f.line)}行目` : `line ${s(f.line)}`;
  }
  return null;
}
