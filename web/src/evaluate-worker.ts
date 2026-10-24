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
      for (const playerSide of [0, 1] as const) {
        // The main thread already holds this game, including its log.
        if (run.pending?.index === index && playerSide === 0) continue;
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
        let game: GameResult;
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
          const selections: GameResult["selections"] = [];
          while (!battle.outcome() && battle.turn() < run.config.turnLimit) {
            const needs = JSON.parse(battle.needsChoice()) as boolean[];
            if (!needs.some(Boolean))
              throw new ToolError("missingRequests");
            const choices: (string | undefined)[] = [];
            for (let side = 0; side < 2; side++) {
              if (!needs[side]) continue;
              const agent = agents[side];
              const preview = JSON.parse(battle.legalChoices(side))[0]?.kind === "team";
              const budget = preview ? run.config.previewIterations : run.config.iterations;
              agent.observe(battle);
              // No compiled lookup, table injection or preview cache. Both
              // sides search every selection using the common live budget.
              for (let n = 0; n < budget; n += 32) {
                agent.step(Math.min(32, budget - n));
                if (performance.now() - lastProgress > 500) {
                  send({
                    type: "progress",
                    pair: index,
                    game: playerSide + 1,
                    turn: battle.turn(),
                    side,
                    iterations: agent.iterations(),
                    budget,
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
              if (preview) selections.push({ side, input: choices[side]!, iterations: agent.iterations() });
            }
            for (let side = 0; side < 2; side++)
              if (choices[side]) battle.applyChoice(side, choices[side]!);
          }
          const outcome = battle.outcome();
          game = {
            playerSide,
            turns: battle.turn(),
            outcome: !outcome
              ? "cap"
              : outcome === "tie"
                ? "tie"
                : outcome === `p${playerSide + 1}`
                  ? "win"
                  : "loss",
            log: JSON.parse(battle.takeNewLog()),
            // Review the user's party, including after the side swap.
            kifu: battle.exportKifu(false, playerSide),
            selections,
          };
        } finally {
          agents.forEach((a) => a.free());
          battle.free();
        }
        const ack = new Promise<void>((resolve) => {
          acknowledge = resolve;
        });
        send({ type: "game", record: { ...scheduled, game } });
        await ack;
      }
    }
    send({ type: "done" });
  } finally {
    dex.free();
  }
}
