import { test, expect } from "@playwright/test";
import {
  appendGame,
  completedGames,
  completedCount,
  normalizeWeights,
  pairSchedule,
  sameConfig,
  summarize,
  resultsCsv,
  type EvaluationConfig,
  type EvaluationRun,
  type PairResult,
} from "../src/evaluate-core";
import { readDoor } from "../src/info-mode";

const pair = (
  index: number,
  a: "win" | "loss" | "tie" | "cap",
  b = a,
): PairResult => ({
  ...pairSchedule(19, index, [{ id: "one", weight: 1 }]),
  games: [
    { playerSide: 0, outcome: a, turns: 1, log: ["|turn|1"], kifu: "NC2-first", selections: [] },
    { playerSide: 1, outcome: b, turns: 2, log: ["|turn|2"], kifu: "NC2-second", selections: [] },
  ],
});
const team = {
  sets: [{ species: "Snorlax", level: 55 }],
  warnings: ["team size"],
  fixes: ["default EVs"],
  relaxed: false,
};
const config: EvaluationConfig = {
  build: "abc",
  beliefHash: "def",
  player: team,
  opponents: [{ ...team, id: "one", weight: 1 }],
  iterations: 3000,
  previewIterations: 729000,
  c: 0.4,
  turnLimit: 500,
};
const run: EvaluationRun = {
  version: 2,
  id: "test",
  createdAt: "2026-09-27",
  config,
  seed: 19,
  targetPairs: 4,
  pairs: [],
};

test("evaluator URL is isolated and blind", () => {
  expect(readDoor("?evaluate&nash&solver")).toBe("evaluate");
  expect(readDoor("?evaluate=false&nash")).toBe("nash");
  expect(readDoor("")).toBe("play");
  expect(readDoor("?blind")).toBe("play");
  expect(readDoor("?blind=0")).toBe("play");
  expect(readDoor("?fork=4296-t11&evaluate")).toBe("fork");
  expect(readDoor("?fork")).toBe("fork");
});
test("weights normalize without overflow and reject invalid distributions", () => {
  expect(
    normalizeWeights([
      { weight: Number.MAX_VALUE },
      { weight: Number.MAX_VALUE },
    ]),
  ).toEqual([{ weight: 0.5 }, { weight: 0.5 }]);
  for (const input of [
    [],
    [{ weight: 0 }],
    [{ weight: -1 }],
    [{ weight: NaN }],
    [{ weight: Infinity }],
  ])
    expect(() => normalizeWeights(input)).toThrow();
});
test("opponent draws and seeds reproduce across chunking, respecting zero weights", () => {
  const mix = [
    { id: "zero", weight: 0 },
    { id: "one", weight: 1 },
    { id: "three", weight: 3 },
  ];
  const full = Array.from({ length: 10000 }, (_, i) =>
    pairSchedule(9182, i, mix),
  );
  expect(full.some((p) => p.opponent === "zero")).toBe(false);
  const rate = full.filter((p) => p.opponent === "one").length / full.length;
  expect(rate).toBeGreaterThan(0.23);
  expect(rate).toBeLessThan(0.27);
  expect([pairSchedule(9182, 5, mix), pairSchedule(9182, 6, mix)]).toEqual(
    full.slice(5, 7),
  );
});
test("paired interval uses pairs, keeps caps distinct and scores ties as half", () => {
  expect(summarize([]).mean).toBeNull();
  expect(summarize([pair(0, "win")]).interval).toBeNull();
  const s = summarize([
    pair(0, "win"),
    pair(1, "loss"),
    pair(2, "tie"),
    pair(3, "cap"),
  ]);
  expect(s).toMatchObject({
    games: 8,
    win: 2,
    loss: 2,
    tie: 2,
    cap: 2,
    mean: 0.5,
    winRate: 0.25,
  });
  expect(s.interval![0]).toBeCloseTo(0.5 - 1.96 * Math.sqrt(0.5 / 3 / 4));
  expect(
    summarize([pair(0, "win", "loss"), pair(1, "loss", "win")]).interval,
  ).toEqual([0.5, 0.5]);
});
test("resume identity ignores canonicalization notices but rejects changed conditions", () => {
  expect(
    sameConfig(config, { ...config, player: { ...team, fixes: [] } }),
  ).toBe(true);
  for (const patch of [
    { iterations: 10000 },
    { previewIterations: 81000 },
    { build: "other" },
    { beliefHash: "other" },
    { player: { ...team, relaxed: true } },
  ])
    expect(sameConfig(config, { ...config, ...patch })).toBe(false);
});
test("completed games survive a half pair, enter statistics only as a pair, and cannot be duplicated", () => {
  const { games, ...scheduled } = pair(0, "win", "loss");
  const first = { ...scheduled, game: games[0] };
  const second = { ...scheduled, game: games[1] };
  expect(() => appendGame(run, second)).toThrow();
  const half = appendGame(run, first);
  expect(completedCount(half)).toBe(1);
  expect(completedGames(half)).toEqual([first]);
  expect(summarize(half.pairs).games).toBe(0);
  expect(half.pending!.games[0].kifu).toBe("NC2-first");
  expect(resultsCsv(half)).toContain('"false"');
  expect(() => appendGame(half, first)).toThrow();
  expect(() => appendGame(half, { ...second, battleSeed: 0 })).toThrow();
  const next = appendGame(half, second);
  expect(next.pairs).toHaveLength(1);
  expect(completedCount(next)).toBe(2);
  expect(next.pending).toBeUndefined();
  expect(completedGames(next)).toEqual([first, second]);
  expect(() => appendGame(next, first)).toThrow();
  expect(resultsCsv(next)).toContain(
    `"0","one","2","loss","0","2","${scheduled.battleSeed}","${scheduled.agentSeeds[0]}","${scheduled.agentSeeds[1]}","true"`,
  );
});

test("HTML logs remain readable, escape user text, mark caps and follow the player's side", async () => {
  const { logsHtml } = await import("../src/evaluate-export");
  const { setLocale } = await import("../src/i18n");
  setLocale("ja", false);
  const p = pair(0, "cap");
  for (const game of p.games) {
    game.log = ["|move|p1a: Snorlax|Tackle|p2a: Magikarp", "|move|p2a: Magikarp|Splash|p1a: Snorlax", "|<script>alert(1)</script>"];
  }
  const record = { ...run, pairs: [p] };
  const text = logsHtml(record, undefined, () => '<img src=x onerror="alert(1)">');
  expect(text).toContain("&lt;img");
  expect(text).not.toContain("<img");
  expect(text).not.toContain("<script");
  expect(text).toContain("&lt;script&gt;");
  expect(text).toContain("勝敗は未確定");
  expect(text).toContain("NC2-first");
  expect(text).toContain("NC2-second");
  const [first, second] = text.split('<section id="game-2">');
  expect(first).toContain("あいての Magikarp");
  expect(second).toContain("あいての Snorlax");
});

test("portable SHA-256 matches native hashing for UTF-8, padding boundaries and the belief pool", async () => {
  const { createHash } = await import("node:crypto");
  const { readFileSync } = await import("node:fs");
  const { sha256Portable } = await import("../src/evaluate-hash");
  const corpus = readFileSync(
    new URL("../../data/belief-pool-v1/belief-pool.json", import.meta.url),
    "utf8",
  );
  for (const value of [
    "",
    "abc",
    "カビゴン🦋",
    ...[1, 55, 56, 63, 64, 65, 127, 128, 129, 1000].map((n) => "a".repeat(n)),
    corpus,
  ]) {
    expect(sha256Portable(value)).toBe(
      createHash("sha256").update(value).digest("hex"),
    );
  }
});
