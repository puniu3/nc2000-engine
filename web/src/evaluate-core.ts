import { ToolError, type ToolIssue } from "./tool-errors";
export interface EvaluationTeam {
  sets: unknown[];
  warnings: string[];
  fixes: string[];
  relaxed: boolean;
}
export interface EvaluationOpponent extends EvaluationTeam {
  id: string;
  weight: number;
}
export interface EvaluationConfig {
  build: string;
  beliefHash: string;
  player: EvaluationTeam;
  opponents: EvaluationOpponent[];
  iterations: number;
  c: number;
  turnLimit: number;
}
export interface GameResult {
  playerSide: 0 | 1;
  outcome: "win" | "loss" | "tie" | "cap";
  turns: number;
}
export interface PairResult {
  index: number;
  opponent: string;
  battleSeed: number;
  agentSeeds: [number, number];
  games: [GameResult, GameResult];
}
export interface EvaluationRun {
  version: 1;
  id: string;
  createdAt: string;
  config: EvaluationConfig;
  seed: number;
  targetPairs: number;
  pairs: PairResult[];
}
export type WorkerRequest =
  | { type: "start"; run: EvaluationRun; beliefJson: string }
  | { type: "ack" };
export type WorkerResponse =
  | {
      type: "progress";
      pair: number;
      game: number;
      turn: number;
      side: number;
      iterations: number;
    }
  | { type: "pair"; pair: PairResult }
  | { type: "done" }
  | { type: "error"; message: string; issue?: ToolIssue };

export function normalizeWeights<T extends { weight: number }>(
  entries: T[],
): T[] {
  if (!entries.length) throw new ToolError("noOpponents");
  if (entries.some((e) => !Number.isFinite(e.weight) || e.weight < 0))
    throw new ToolError("invalidWeight");
  const max = Math.max(...entries.map((e) => e.weight));
  if (max === 0)
    throw new ToolError("zeroWeights");
  const total = entries.reduce((s, e) => s + e.weight / max, 0);
  return entries.map((e) => ({ ...e, weight: e.weight / max / total }));
}

export function seedWord(seed: number, index: number): number {
  let x = (seed + Math.imul(index + 1, 0x9e3779b9)) >>> 0;
  x = Math.imul(x ^ (x >>> 16), 0x21f0aaad);
  x = Math.imul(x ^ (x >>> 15), 0x735a2d97);
  return (x ^ (x >>> 15)) >>> 0;
}

export function pairSchedule(
  seed: number,
  index: number,
  opponents: { id: string; weight: number }[],
) {
  const mix = normalizeWeights(opponents).filter((e) => e.weight > 0);
  const draw = seedWord(seed, index * 4) / 2 ** 32;
  let acc = 0;
  const opponent =
    mix.find((e) => {
      acc += e.weight;
      return draw < acc;
    }) ?? mix[mix.length - 1];
  return {
    index,
    opponent: opponent.id,
    battleSeed: seedWord(seed, index * 4 + 1),
    agentSeeds: [
      seedWord(seed, index * 4 + 2),
      seedWord(seed, index * 4 + 3),
    ] as [number, number],
  };
}

export function score(game: GameResult): number {
  return game.outcome === "win" ? 1 : game.outcome === "loss" ? 0 : 0.5;
}

export function summarize(pairs: PairResult[]) {
  const games = pairs.flatMap((p) => p.games);
  const counts = { win: 0, loss: 0, tie: 0, cap: 0 };
  games.forEach((g) => counts[g.outcome]++);
  const means = pairs.map((p) => (score(p.games[0]) + score(p.games[1])) / 2);
  const mean = means.length
    ? means.reduce((a, b) => a + b, 0) / means.length
    : null;
  let interval: [number, number] | null = null;
  if (means.length >= 2 && mean !== null) {
    const variance =
      means.reduce((s, x) => s + (x - mean) ** 2, 0) / (means.length - 1);
    const margin = 1.96 * Math.sqrt(variance / means.length);
    interval = [Math.max(0, mean - margin), Math.min(1, mean + margin)];
  }
  return {
    ...counts,
    games: games.length,
    mean,
    interval,
    winRate: games.length ? counts.win / games.length : null,
  };
}

export function sameConfig(a: EvaluationConfig, b: EvaluationConfig): boolean {
  const identity = (config: EvaluationConfig) => ({
    build: config.build,
    beliefHash: config.beliefHash,
    iterations: config.iterations,
    c: config.c,
    turnLimit: config.turnLimit,
    player: { sets: config.player.sets, relaxed: config.player.relaxed },
    opponents: config.opponents.map((e) => ({
      id: e.id,
      sets: e.sets,
      relaxed: e.relaxed,
    })),
  });
  return (
    JSON.stringify(identity(a)) === JSON.stringify(identity(b)) &&
    a.opponents.every(
      (e, i) => Math.abs(e.weight - b.opponents[i].weight) < 1e-14,
    )
  );
}

export function appendPair(
  run: EvaluationRun,
  pair: PairResult,
): EvaluationRun {
  if (pair.index !== run.pairs.length || pair.games.length !== 2)
    throw new ToolError("resultOrderError");
  return { ...run, pairs: [...run.pairs, pair] };
}

export function resultsCsv(run: EvaluationRun): string {
  const q = (v: unknown) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const rows: unknown[][] = [
    [
      "section",
      "opponent",
      "games",
      "wins",
      "losses",
      "ties",
      "caps",
      "win_rate",
      "score",
      "ci95_low",
      "ci95_high",
    ],
  ];
  for (const id of [null, ...run.config.opponents.map((e) => e.id)]) {
    const s = summarize(
      id === null ? run.pairs : run.pairs.filter((p) => p.opponent === id),
    );
    rows.push([
      "summary",
      id ?? "overall",
      s.games,
      s.win,
      s.loss,
      s.tie,
      s.cap,
      s.winRate,
      s.mean,
      s.interval?.[0],
      s.interval?.[1],
    ]);
  }
  rows.push(
    [],
    [
      "pair",
      "opponent",
      "player_side",
      "outcome",
      "score",
      "turns",
      "battle_seed",
      "player_agent_seed",
      "opponent_agent_seed",
    ],
  );
  run.pairs.forEach((p) =>
    p.games.forEach((g) =>
      rows.push([
        p.index,
        p.opponent,
        g.playerSide + 1,
        g.outcome,
        score(g),
        g.turns,
        p.battleSeed,
        ...p.agentSeeds,
      ]),
    ),
  );
  return "\ufeff" + rows.map((row) => row.map(q).join(",")).join("\r\n");
}
