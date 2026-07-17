// M14b: PS teambuilder export/import text -> team JSON (the array the wasm
// Validator / Battle constructor consume). A faithful mirror of PS's
// sim/teams.ts import path for the export format, with two deliberate
// differences, both surfaced as structured findings instead of silent
// mangling (they localize like validator codes; `line` is 1-based):
//
// - garbage PS silently ignores (an unrecognized property line, an unknown
//   stat name or non-numeric value in an EVs/IVs chunk, a block whose first
//   line is clearly not a mon header) is a line-anchored parse error — a
//   paste that PS itself exports never triggers any of them;
// - a missing `Level:` line stays absent from the JSON (PS's importer
//   force-fills 100, which this format would then reject; absent level =
//   the validator's `level-default` fix -> 55, matching PS's teambuilder
//   default for the format).
//
// Tolerated exactly like PS: blank lines / `---` between mons, `===` team
// headers (captured as a default team name), CRLF, stray indentation,
// `- ` / `~ ` move bullets, `Hidden Power [Ice]` bracket form, ` (M)` /
// ` (F)` gender markers, `Nick (Species)` nicknames, `@ item` with
// "noitem"/empty tails, Trait:/Ability:/Nature/Happiness/Shiny/IVs/EVs
// lines in any order, and the `- Frustration` happiness-0 rule.

export interface ParseFinding {
  severity: "error";
  code: string;
  /** 1-based line number in the pasted text. */
  line: number;
  /** The offending line's text (trimmed). */
  text?: string;
  /** Extra params per code (stat, chunk, field, ...). */
  [k: string]: unknown;
}

export interface ParsedSet {
  name: string;
  species: string;
  item: string;
  ability: string;
  moves: string[];
  nature?: string;
  gender?: string;
  level?: number;
  happiness?: number;
  shiny?: boolean;
  evs?: Record<string, number>;
  ivs?: Record<string, number>;
}

export interface ParseResult {
  sets: ParsedSet[];
  findings: ParseFinding[];
  /** From a `=== [format] Folder/Name ===` header, if present. */
  teamName: string | null;
}

const STAT_IDS: Record<string, string> = {
  hp: "hp",
  atk: "atk",
  attack: "atk",
  def: "def",
  defense: "def",
  spa: "spa",
  spatk: "spa",
  spc: "spa",
  special: "spa",
  spd: "spd",
  spdef: "spd",
  spe: "spe",
  speed: "spe",
};

const toId = (s: string) =>
  s
    .normalize("NFD")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");

/** Property prefixes that can never start a mon block — a block whose first
 * line looks like one of these gets `ps-header-expected` instead of being
 * misread as a species name (PS would misread; the error is clearer). */
const PROPERTY_PREFIXES = [
  "ability:",
  "trait:",
  "level:",
  "shiny:",
  "happiness:",
  "evs:",
  "ivs:",
  "hidden power:",
  "pokeball:",
];

function looksLikeProperty(line: string): boolean {
  const l = line.toLowerCase();
  return (
    line.startsWith("-") ||
    line.startsWith("~") ||
    PROPERTY_PREFIXES.some((p) => l.startsWith(p)) ||
    /^[a-z]+ nature\b/.test(l)
  );
}

function parseStatLine(
  raw: string,
  lineNo: number,
  defaultValue: number,
  findings: ParseFinding[],
): Record<string, number> {
  const out: Record<string, number> = {
    hp: defaultValue,
    atk: defaultValue,
    def: defaultValue,
    spa: defaultValue,
    spd: defaultValue,
    spe: defaultValue,
  };
  for (const chunk of raw.split("/")) {
    const c = chunk.trim();
    if (!c) continue;
    const m = /^(-?\d+)\s+(.+)$/.exec(c);
    if (!m) {
      findings.push({
        severity: "error",
        code: "ps-stat-chunk",
        line: lineNo,
        chunk: c,
      });
      continue;
    }
    const stat = STAT_IDS[toId(m[2])];
    if (!stat) {
      findings.push({
        severity: "error",
        code: "ps-stat-unknown",
        line: lineNo,
        stat: m[2],
      });
      continue;
    }
    out[stat] = parseInt(m[1], 10);
  }
  return out;
}

/** Parse one non-first line of a mon block. Returns false when the line is
 * unrecognized (caller reports it). */
function parseProperty(
  line: string,
  lineNo: number,
  set: ParsedSet,
  findings: ParseFinding[],
): boolean {
  const num = (raw: string, field: string): number | null => {
    const v = parseInt(raw.trim(), 10);
    if (Number.isNaN(v)) {
      findings.push({
        severity: "error",
        code: "ps-number",
        line: lineNo,
        field,
        text: raw.trim(),
      });
      return null;
    }
    return v;
  };

  if (line.startsWith("Trait: ") || line.startsWith("Ability: ")) {
    set.ability = line.slice(line.indexOf(": ") + 2).trim();
    return true;
  }
  if (/^Shiny:\s*/i.test(line)) {
    set.shiny = /^Shiny:\s*yes$/i.test(line.trim());
    return true;
  }
  if (line.startsWith("Level: ")) {
    const v = num(line.slice(7), "level");
    if (v !== null) set.level = v;
    return true;
  }
  if (line.startsWith("Happiness: ")) {
    const v = num(line.slice(11), "happiness");
    if (v !== null) set.happiness = v;
    return true;
  }
  if (line.startsWith("Pokeball: ") || line.startsWith("Tera Type: ")) {
    return true; // recognized, irrelevant to gen 2 — ignore like PS does
  }
  if (line.startsWith("Hidden Power: ")) {
    // newer-export HP-type annotation: remember it so a plain "Hidden
    // Power" move can be typed at the end
    (set as { _hpType?: string })._hpType = line.slice(14).trim();
    return true;
  }
  if (line.startsWith("EVs: ")) {
    set.evs = parseStatLine(line.slice(5), lineNo, 0, findings);
    return true;
  }
  if (line.startsWith("IVs: ")) {
    set.ivs = parseStatLine(line.slice(5), lineNo, 31, findings);
    return true;
  }
  if (/^[A-Za-z]+ [Nn]ature/.test(line)) {
    const idx = line.toLowerCase().indexOf(" nature");
    set.nature = line.slice(0, idx);
    return true;
  }
  if (line.startsWith("-") || line.startsWith("~")) {
    let move = line.slice(line.charAt(1) === " " ? 2 : 1).trim();
    const br = /^Hidden Power \[(.+)\]$/.exec(move);
    if (br) move = `Hidden Power ${br[1]}`;
    if (move === "Frustration" && set.happiness === undefined) {
      set.happiness = 0;
    }
    if (move) set.moves.push(move);
    return true;
  }
  return false;
}

/** First line of a block: `Nickname (Species) (G) @ Item`. */
function parseHeader(line: string, set: ParsedSet): void {
  const parts = line.split(" @ ");
  let head = parts[0];
  if (parts.length > 1) {
    const item = parts[1].trim();
    set.item = toId(item) === "noitem" ? "" : item;
  }
  if (head.endsWith(" (M)")) {
    set.gender = "M";
    head = head.slice(0, -4);
  } else if (head.endsWith(" (F)")) {
    set.gender = "F";
    head = head.slice(0, -4);
  }
  head = head.trim();
  if (head.endsWith(")") && head.includes("(")) {
    const open = head.indexOf("(");
    set.name = head.slice(0, open).trim();
    set.species = head.slice(open + 1, -1).trim();
  } else {
    set.species = head;
    set.name = "";
  }
}

export function parsePsExport(text: string): ParseResult {
  const findings: ParseFinding[] = [];
  const sets: ParsedSet[] = [];
  let teamName: string | null = null;
  let headerCount = 0;
  let cur: ParsedSet | null = null;

  const lines = text.replace(/^\uFEFF/, "").split(/\r\n?|\n/);
  for (let i = 0; i < lines.length; i++) {
    const lineNo = i + 1;
    const line = lines[i].trim();
    if (line === "" || line === "---") {
      cur = null;
      continue;
    }
    if (line.startsWith("===")) {
      // team-backup header: `=== [format] Folder/Name ===`
      headerCount += 1;
      if (headerCount > 1) {
        findings.push({ severity: "error", code: "ps-multiple-teams", line: lineNo });
      } else {
        const m = /^===\s*(?:\[[^\]]*\]\s*)?(.*?)\s*===$/.exec(line);
        const name = m?.[1].split("/").pop()?.trim();
        if (name) teamName = name;
      }
      cur = null;
      continue;
    }
    if (!cur) {
      if (looksLikeProperty(line)) {
        findings.push({
          severity: "error",
          code: "ps-header-expected",
          line: lineNo,
          text: line,
        });
        continue;
      }
      cur = { name: "", species: "", item: "", ability: "", moves: [] };
      sets.push(cur);
      parseHeader(line, cur);
      continue;
    }
    if (!parseProperty(line, lineNo, cur, findings)) {
      findings.push({
        severity: "error",
        code: "ps-line-unknown",
        line: lineNo,
        text: line,
      });
    }
  }

  // resolve a `Hidden Power: <Type>` annotation onto an untyped HP move
  for (const set of sets) {
    const hpType = (set as { _hpType?: string })._hpType;
    delete (set as { _hpType?: string })._hpType;
    if (!hpType) continue;
    const i = set.moves.findIndex((m) => toId(m) === "hiddenpower");
    if (i >= 0) set.moves[i] = `Hidden Power ${hpType}`;
  }

  if (sets.length === 0) {
    findings.push({ severity: "error", code: "ps-empty", line: 1 });
  }
  return { sets, findings, teamName };
}
