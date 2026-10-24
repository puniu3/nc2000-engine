import { completedGames, type EvaluationRun, type GameRecord, type GameResult } from "./evaluate-core";
import { Narrator } from "./narrate";
import { locale } from "./i18n";
import { toolText } from "./tool-strings";

export const outcomeLabel = (outcome: GameResult["outcome"]) => toolText(({
  win: "gameWin", loss: "gameLoss", tie: "gameTie", cap: "gameCap",
} as const)[outcome]);

const escapeHtml = (value: unknown) => String(value).replace(/[&<>"']/g, c => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[c]!);

/** Self-contained, script-free HTML. All user-controlled text is escaped;
 * narration is materialized now so it stays readable after engine updates. */
export function logsHtml(
  run: EvaluationRun,
  records: GameRecord[] = completedGames(run),
  opponentLabel: (id: string) => string = id => id,
): string {
  const title = toolText("evaluationLogsTitle");
  const body = records.map(({ game, ...pair }) => {
    const number = pair.index * 2 + game.playerSide + 1;
    const narration = new Narrator(game.playerSide).render(game.log);
    const status = pair.index < run.pairs.length ? toolText("countedGame") : toolText("pendingGame");
    return `<section id="game-${number}">
<h2>${toolText("gameNumber", number)} — ${escapeHtml(opponentLabel(pair.opponent))} — ${outcomeLabel(game.outcome)}</h2>
<p>${toolText("logGameDetails", game.playerSide + 1, game.turns, status)}</p>
${game.outcome === "cap" ? `<p>${toolText("logCap", run.config.turnLimit)}</p>` : ""}
<div class="log">${narration.map(entry => `<p class="${entry.kind}">${escapeHtml(entry.text)}</p>`).join("\n")}</div>
<details><summary>${toolText("logKifu")}</summary><pre>${escapeHtml(game.kifu)}</pre></details>
<details><summary>${toolText("rawLog")}</summary><pre>${escapeHtml(game.log.join("\n"))}</pre></details>
<details><summary>${toolText("gameConditions")}</summary><pre>${escapeHtml(JSON.stringify({
      ...pair, playerSide: game.playerSide, selections: game.selections,
    }, null, 2))}</pre></details>
</section>`;
  }).join("\n");
  return `<!doctype html>
<html lang="${locale()}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${title}</title><style>
body{font:16px/1.7 system-ui,sans-serif;color:#202820;background:#f7f9f5;max-width:960px;margin:auto;padding:24px}
section{background:white;border:1px solid #cdd6c8;border-radius:12px;padding:20px;margin:24px 0}
h1{font-size:1.6rem}h2{font-size:1.2rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85rem}
summary{cursor:pointer;padding:8px 0}.log p{margin:4px 0}.turn{font-weight:bold;border-top:1px solid #ddd;padding-top:12px;margin-top:16px!important}
.minor{color:#52604f}.result{font-weight:bold}a{color:#315d31}
</style></head><body><h1>${title}</h1>
<p>${escapeHtml(toolText("logStarted", run.createdAt, records.length))}</p>
<p>${toolText("logBudgets", run.config.previewIterations.toLocaleString(), run.config.iterations.toLocaleString())}</p>
<p>${toolText("logPerspective")}</p>
<nav aria-label="${toolText("gameIndex")}">${records.map(({ index, game }) => {
    const n = index * 2 + game.playerSide + 1;
    return `<a href="#game-${n}">${toolText("gameNumber", n)}</a>`;
  }).join(" · ")}</nav>
${body}
<details><summary>${toolText("evaluationSettings")}</summary><pre>${escapeHtml(JSON.stringify({
    version: run.version, id: run.id, seed: run.seed, config: run.config,
  }, null, 2))}</pre></details>
</body></html>`;
}
