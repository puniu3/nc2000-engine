import { ToolError } from "./tool-errors";
import init, {
  Dex,
  Battle,
  BlindSearcher,
  deriveBattleSeed,
} from "../../crates/wasm/pkg-web/nc2000_wasm";
import {
  pairSchedule,
  type EvaluationRun,
  type GameResult,
  type WorkerRequest,
  type WorkerResponse,
} from "./evaluate-core";

const send = (message: WorkerResponse) => postMessage(message);
const yieldTask = () => new Promise<void>((resolve) => setTimeout(resolve, 0));
let acknowledge: (() => void) | undefined;
let started = false;
onmessage = (event: MessageEvent<WorkerRequest>) => {
  if (event.data.type === "ack") {
    acknowledge?.();
    acknowledge = undefined;
    return;
  }
  if (started) return;
  started = true;
  void run(event.data.run, event.data.beliefJson).catch((e) =>
    send({ type: "error", message: String(e), ...(e instanceof ToolError ? { issue: e.issue } : {}) }),
  );
};

async function run(run: EvaluationRun, beliefJson: string) {
  await init();
  const dex = new Dex();
  try {
    for (let index = run.pairs.length; index < run.targetPairs; index++) {
      const scheduled = pairSchedule(run.seed, index, run.config.opponents);
      const opponent = run.config.opponents.find(
        (e) => e.id === scheduled.opponent,
      )!;
      const games: GameResult[] = [];
      for (const playerSide of [0, 1] as const) {
        const teams =
          playerSide === 0
            ? [run.config.player, opponent]
            : [opponent, run.config.player];
        const battle = new Battle(
          dex,
          JSON.stringify(teams[0].sets),
          JSON.stringify(teams[1].sets),
          deriveBattleSeed(scheduled.battleSeed),
        );
        const agents: BlindSearcher[] = [];
        try {
          teams.forEach((t, side) =>
            battle.setPreviewLevelCap(side, t.relaxed ? undefined : 155),
          );
          for (let side = 0; side < 2; side++)
            agents.push(
              new BlindSearcher(
                battle,
                side,
                beliefJson,
                scheduled.agentSeeds[side === playerSide ? 0 : 1],
                run.config.c,
                undefined,
              ),
            );
          let lastProgress = 0;
          let lastYield = performance.now();
          while (!battle.outcome() && battle.turn() < run.config.turnLimit) {
            const needs = JSON.parse(battle.needsChoice()) as boolean[];
            if (!needs.some(Boolean))
              throw new ToolError("missingRequests");
            const choices: (string | undefined)[] = [];
            for (let side = 0; side < 2; side++) {
              if (!needs[side]) continue;
              const agent = agents[side];
              agent.observe(battle);
              for (let n = 0; n < run.config.iterations; n += 32) {
                agent.step(Math.min(32, run.config.iterations - n));
                if (performance.now() - lastProgress > 500) {
                  send({
                    type: "progress",
                    pair: index,
                    game: playerSide + 1,
                    turn: battle.turn(),
                    side,
                    iterations: agent.iterations(),
                  });
                  lastProgress = performance.now();
                }
                if (performance.now() - lastYield >= 16) {
                  await yieldTask();
                  lastYield = performance.now();
                }
              }
              choices[side] = agent.best();
              if (!choices[side])
                throw new ToolError("noLegalAction", side + 1);
            }
            for (let side = 0; side < 2; side++)
              if (choices[side]) battle.applyChoice(side, choices[side]!);
          }
          const outcome = battle.outcome();
          games.push({
            playerSide,
            turns: battle.turn(),
            outcome: !outcome
              ? "cap"
              : outcome === "tie"
                ? "tie"
                : outcome === `p${playerSide + 1}`
                  ? "win"
                  : "loss",
          });
        } finally {
          agents.forEach((a) => a.free());
          battle.free();
        }
      }
      const ack = new Promise<void>((resolve) => {
        acknowledge = resolve;
      });
      send({
        type: "pair",
        pair: { ...scheduled, games: games as [GameResult, GameResult] },
      });
      await ack;
    }
    send({ type: "done" });
  } finally {
    dex.free();
  }
}
