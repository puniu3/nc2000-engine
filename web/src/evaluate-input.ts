import { ToolError } from "./tool-errors";
import { Battle, getDex, getValidator } from "./engine";
import { parsePsExport } from "./ps-import";
import { findingAnchor, findingText, type Finding } from "./findings";
import {
  normalizeWeights,
  type EvaluationTeam,
  type EvaluationOpponent,
} from "./evaluate-core";

export interface OpponentDraft {
  id: string;
  weight: string;
  text: string;
}
const explain = (f: Finding) =>
  [findingAnchor(f), findingText(f)].filter(Boolean).join(": ");

class TeamFindingError extends Error {
  constructor(private findings: Finding[]) {
    super("Invalid team");
  }
  toString(): string {
    return this.findings.map(explain).join("\n");
  }
}

export function readTeam(text: string): EvaluationTeam {
  if (!text.trim()) throw new ToolError("enterTeam");
  let raw: unknown;
  if (/^\s*[\[{]/.test(text)) raw = JSON.parse(text);
  else {
    const parsed = parsePsExport(text);
    if (parsed.findings.length)
      throw new TeamFindingError(parsed.findings);
    raw = parsed.sets;
  }
  if (raw && typeof raw === "object" && "sets" in raw) raw = raw.sets;
  if (!Array.isArray(raw) || raw.length < 1 || raw.length > 6)
    throw new ToolError("teamSize");
  for (const [i, set] of raw.entries()) {
    if (!set || typeof set !== "object" || typeof set.species !== "string")
      throw new ToolError("missingSpecies", i + 1);
    if (
      set.level !== undefined &&
      (!Number.isInteger(set.level) || set.level < 1 || set.level > 100)
    )
      throw new ToolError("invalidLevel", i + 1);
    if (
      set.moves !== undefined &&
      (!Array.isArray(set.moves) ||
        set.moves.some((m: unknown) => typeof m !== "string") ||
        set.moves.length > 4)
    )
      throw new ToolError("invalidMoves", i + 1);
  }
  raw = raw.map((set) => ({
    ...set,
    moves: (set.moves ?? []).filter((m: string) => m.trim()),
  }));
  const result = JSON.parse(
    getValidator().canonicalizeTeam(JSON.stringify(raw)),
  ) as { team: unknown[]; applied: Finding[]; errors: Finding[] };
  const fatal = result.errors.filter((f) =>
    [
      "json-invalid",
      "species-unknown",
      "move-unknown",
      "item-unknown",
    ].includes(f.code),
  );
  if (fatal.length) throw new TeamFindingError(fatal);
  if (!Array.isArray(result.team))
    throw new ToolError("invalidTeam");
  const relaxed = result.errors.some(
    (f) => f.code === "level-sum" || f.code === "level-sum-highest",
  );
  const battle = new Battle(
    getDex(),
    JSON.stringify(result.team),
    JSON.stringify(result.team),
    "1,2,3,4",
  );
  battle.free();
  return {
    sets: result.team,
    warnings: result.errors.map(explain),
    fixes: result.applied.map(explain),
    relaxed,
  };
}

export function readOpponents(drafts: OpponentDraft[]): EvaluationOpponent[] {
  const seen = new Set<string>();
  const entries = drafts.map((d) => {
    const id = d.id.trim();
    if (!id || seen.has(id))
      throw new ToolError("uniqueOpponentNames");
    seen.add(id);
    if (!d.weight.trim())
      throw new ToolError("missingWeight", id);
    return { id, weight: Number(d.weight), ...readTeam(d.text) };
  });
  return normalizeWeights(entries);
}

export function importDistribution(text: string): OpponentDraft[] {
  const raw = JSON.parse(text);
  if (!raw || !Array.isArray(raw.teams))
    throw new ToolError("invalidDistribution");
  const drafts = raw.teams.map(
    (e: { id?: unknown; weight?: unknown; sets?: unknown }) => {
      if (
        !e ||
        typeof e.id !== "string" ||
        typeof e.weight !== "number" ||
        !Array.isArray(e.sets)
      )
        throw new ToolError("incompleteOpponent");
      return {
        id: e.id,
        weight: String(e.weight),
        text: JSON.stringify(e.sets, null, 2),
      };
    },
  );
  readOpponents(drafts);
  return drafts;
}

export function exportDistribution(drafts: OpponentDraft[]): string {
  const teams = drafts.map((d) => ({
    id: d.id,
    weight: Number(d.weight),
    sets: JSON.parse(d.text),
  }));
  return JSON.stringify({ teams }, null, 2);
}

export { sha256 } from "./evaluate-hash";
