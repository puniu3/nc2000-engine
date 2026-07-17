// M13 i18n: a light two-locale string layer — no library.
//
// - Locale: "ja" by default when navigator.language starts with "ja", else
//   "en"; a manual toggle on the title screen persists to localStorage.
// - Dex names (species / moves / items / types) come from
//   data/i18n-ja.json (generated once from PokéAPI by
//   tools/build-i18n-ja.js), fetched at startup and keyed by PS id. Any
//   missing entry — or the table failing to load at all — falls back to
//   the English name the wasm bridge already provides.
// - Small closed sets (statuses, stats, volatile/side conditions, UI
//   strings, narration templates) are hand-authored here and in narrate.ts.

import type { Locale, UIStrings } from "./i18n-strings";
import { STRINGS } from "./i18n-strings";

export type { Locale, UIStrings };

const LS_KEY = "nc2000-locale";

function detect(): Locale {
  try {
    const s = localStorage.getItem(LS_KEY);
    if (s === "en" || s === "ja") return s;
  } catch {
    /* storage unavailable */
  }
  return typeof navigator !== "undefined" &&
    (navigator.language ?? "").toLowerCase().startsWith("ja")
    ? "ja"
    : "en";
}

let current: Locale = detect();
applyDocumentLocale();

function applyDocumentLocale() {
  if (typeof document === "undefined") return;
  document.documentElement.lang = current;
  document.title =
    current === "ja" ? "NC2000 — 金銀バトルデモ" : "NC2000 — Gen 2 battle demo";
}

export function locale(): Locale {
  return current;
}

export function setLocale(l: Locale): void {
  current = l;
  try {
    localStorage.setItem(LS_KEY, l);
  } catch {
    /* storage unavailable */
  }
  applyDocumentLocale();
}

/** Current UI string table. */
export function ui(): UIStrings {
  return STRINGS[current];
}

// ---------------------------------------------------------------- names

interface NameTables {
  species: Record<string, string>;
  moves: Record<string, string>;
  items: Record<string, string>;
  types: Record<string, string>;
}

let jaNames: NameTables | null = null;

/** Load the JP name tables (non-fatal: on failure everything falls back
 * to English). Called once at app startup. */
export async function loadJaNames(fetchJson: () => Promise<unknown>): Promise<void> {
  try {
    const data = (await fetchJson()) as Partial<NameTables> | null;
    if (data && data.species && data.moves && data.items && data.types) {
      jaNames = data as NameTables;
    }
  } catch {
    jaNames = null;
  }
}

/** PS-id normalization, matching sim's toID for the names we receive
 * ("Poké Ball" -> pokeball, "Nidoran-F" -> nidoranf, "Farfetch’d" ->
 * farfetchd). */
export function toId(s: string): string {
  return s
    .normalize("NFD")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

export function speciesName(en: string): string {
  return (current === "ja" && jaNames?.species[toId(en)]) || en;
}

export function moveName(en: string): string {
  return (current === "ja" && jaNames?.moves[toId(en)]) || en;
}

export function itemName(en: string): string {
  return (current === "ja" && jaNames?.items[toId(en)]) || en;
}

/** UI type strings arrive capitalized ("Electric"); the table is keyed by
 * lowercase id. */
export function typeName(en: string): string {
  return (current === "ja" && jaNames?.types[en.toLowerCase()]) || en;
}

// ------------------------------------------------- statuses / stats / conds

const STATUS_JA: Record<string, string> = {
  brn: "やけど",
  par: "まひ",
  slp: "ねむり",
  frz: "こおり",
  psn: "どく",
  tox: "もうどく",
  fnt: "ひんし",
};

/** Status code ("par") -> display noun. English keeps the raw code (the
 * established compact badge form). */
export function statusName(code: string): string {
  if (current === "ja") return STATUS_JA[code] ?? code;
  return code;
}

const STATUS_LONG_EN: Record<string, string> = {
  brn: "burned",
  par: "paralyzed",
  slp: "asleep",
  frz: "frozen",
  psn: "poisoned",
  tox: "badly poisoned",
  fnt: "fainted",
};

/** Screen-reader status wording: the English badge codes ("par") are
 * cryptic when spoken, so expand them; the Japanese nouns are already
 * plain words. */
export function statusLongName(code: string): string {
  if (current === "ja") return STATUS_JA[code] ?? code;
  return STATUS_LONG_EN[code] ?? code;
}

const STAT_LONG_EN: Record<string, string> = {
  atk: "Attack",
  def: "Defense",
  spa: "Sp. Atk",
  spd: "Sp. Def",
  spe: "Speed",
  accuracy: "accuracy",
  evasion: "evasion",
};

const STAT_LONG_JA: Record<string, string> = {
  atk: "こうげき",
  def: "ぼうぎょ",
  spa: "とくこう",
  spd: "とくぼう",
  spe: "すばやさ",
  accuracy: "めいちゅうりつ",
  evasion: "かいひりつ",
};

/** Full stat name for narration ("Attack" / こうげき). */
export function statLongName(code: string): string {
  const table = current === "ja" ? STAT_LONG_JA : STAT_LONG_EN;
  return table[code] ?? code;
}

const BOOST_EN: Record<string, string> = {
  atk: "Atk",
  def: "Def",
  spa: "SpA",
  spd: "SpD",
  spe: "Spe",
  accuracy: "Acc",
  evasion: "Eva",
};

const BOOST_JA: Record<string, string> = {
  atk: "攻",
  def: "防",
  spa: "特攻",
  spd: "特防",
  spe: "速",
  accuracy: "命中",
  evasion: "回避",
};

/** Compact boost-chip label ("Atk" / 攻). */
export function boostLabel(code: string): string {
  const table = current === "ja" ? BOOST_JA : BOOST_EN;
  return table[code] ?? code;
}

// Volatile / side / field condition display names, keyed by PS id. The
// narration path also feeds display names ("Leech Seed") — normalize with
// toId before lookup.
const COND_EN: Record<string, string> = {
  raindance: "Rain",
  sunnyday: "Sun",
  sandstorm: "Sandstorm",
  reflect: "Reflect",
  lightscreen: "Light Screen",
  safeguard: "Safeguard",
  spikes: "Spikes",
  mist: "Mist",
  confusion: "Confused",
  substitute: "Substitute",
  leechseed: "Leech Seed",
  curse: "Curse",
  encore: "Encore",
  attract: "Attract",
  nightmare: "Nightmare",
  partiallytrapped: "Trapped",
  meanlook: "Mean Look",
  focusenergy: "Focus Energy",
  lockedmove: "Locked",
  mustrecharge: "Recharging",
  perishsong: "Perish Song",
  disable: "Disable",
  foresight: "Foresight",
  destinybond: "Destiny Bond",
  perish3: "Perish count 3",
  perish2: "Perish count 2",
  perish1: "Perish count 1",
  perish0: "Perish count 0",
};

const COND_JA: Record<string, string> = {
  raindance: "あめ",
  sunnyday: "にほんばれ",
  sandstorm: "すなあらし",
  reflect: "リフレクター",
  lightscreen: "ひかりのかべ",
  safeguard: "しんぴのまもり",
  spikes: "まきびし",
  mist: "しろいきり",
  confusion: "こんらん",
  substitute: "みがわり",
  leechseed: "やどりぎのタネ",
  curse: "のろい",
  encore: "アンコール",
  attract: "メロメロ",
  nightmare: "あくむ",
  partiallytrapped: "バインド",
  meanlook: "くろいまなざし",
  focusenergy: "きあいだめ",
  lockedmove: "あばれ状態",
  mustrecharge: "はんどう",
  perishsong: "ほろびのうた",
  disable: "かなしばり",
  foresight: "みやぶられた",
  destinybond: "みちづれ",
  perish3: "ほろびのカウント3",
  perish2: "ほろびのカウント2",
  perish1: "ほろびのカウント1",
  perish0: "ほろびのカウント0",
};

/** Condition display name. Accepts a PS id ("leechseed") or a protocol
 * display name ("Leech Seed"); ja falls back through the move table (many
 * volatiles are named after their move), then to the raw input. */
export function condName(raw: string): string {
  const id = toId(raw);
  if (current === "ja") {
    return COND_JA[id] ?? jaNames?.moves[id] ?? COND_EN[id] ?? raw;
  }
  return COND_EN[id] ?? raw;
}
