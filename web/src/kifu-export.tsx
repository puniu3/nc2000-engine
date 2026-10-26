import { useToolMessage } from "./tool-message";
import { toolText } from "./tool-strings";
import { useEffect, useState } from "preact/hooks";
import type { Battle } from "./engine";
import type { InfoMode } from "./info-mode";
import type { StateView } from "./types";
import { copyKifu, kifuUrl } from "./kifu-code";
import "./kifu.css";

export interface WinRatePoint {
  turn: number;
  /** Expected win rate from the player's (side 0) perspective, in [0, 1]. */
  winRate: number;
}

export function KifuExport(props: { battle: Battle; mode: InfoMode; state: StateView; winRates?: WinRatePoint[]; ended?: boolean }) {
  const [code, setCode] = useState("");
  const [notice, setNotice] = useToolMessage("");
  const [manual, setManual] = useState(false);
  useEffect(() => {
    setManual(false);
    setNotice("");
    try {
      setCode(props.battle.exportKifu(props.mode === "open", 1));
    } catch (error) {
      console.error("kifu export", error);
      setCode("");
      setNotice(() => toolText("exportReplayError"));
    }
  }, [props.battle, props.mode, props.state]);

  async function copy() {
    const copied = await copyKifu(code);
    setManual(!copied);
    setNotice(() => copied ? toolText("replayCopied") : toolText("copyReplayFallback"));
  }

  const points = props.ended && props.winRates && props.winRates.length > 0 ? props.winRates : null;

  return <section class="kifu-tool kp-export-shell" aria-labelledby="kifu-export-title">
    <div class="kp-card">
      {points && code && <WinRateChart points={points} code={code} outcome={props.state.outcome} />}
      <h2 id="kifu-export-title">{toolText("saveBattle")}</h2>
      <p class="kp-muted">{toolText("saveBattleHelp")}</p>
      <div class="kp-actions">
        <button class="primary" disabled={!code} onClick={() => void copy()}>{toolText("copyReplay")}</button>
        {code && <a class="kp-button" href={kifuUrl(code)} target="_blank" rel="noopener">{toolText("tryAnotherMove")}<span class="sr-only">{toolText("newTab")}</span></a>}
      </div>
      <p class="kp-status" role="status">{notice}</p>
      {manual && <div class="kp-copy-fallback">
        <label for="kifu-copy-code">{toolText("replayToCopy")}</label>
        <textarea id="kifu-copy-code" readOnly value={code} rows={4} onFocus={e => e.currentTarget.select()} />
        <p class="kp-muted">{toolText("manualCopyHelp")}</p>
      </div>}
    </div>
  </section>;
}

function WinRateChart(props: { points: WinRatePoint[]; code: string; outcome: "p1" | "p2" | "tie" | null }) {
  // Collapse same-turn points (e.g. post-faint switch in the same turn) to the first decision of that turn
  const byTurn = new Map<number, WinRatePoint>();
  for (const p of props.points) {
    if (!byTurn.has(p.turn)) byTurn.set(p.turn, p);
  }
  const turns = [...byTurn.values()];
  const W = 560;
  const H = 160;
  const padL = 36;
  const padR = 16;
  const padT = 14;
  const padB = 26;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const xOf = (i: number) => turns.length <= 1 ? padL + plotW / 2 : padL + (i / (turns.length - 1)) * plotW;
  const yOf = (r: number) => padT + (1 - Math.max(0, Math.min(1, r))) * plotH;
  const poly = turns.map((p, i) => `${xOf(i).toFixed(1)},${yOf(p.winRate).toFixed(1)}`).join(" ");

  return <div class="kp-chart-wrap" data-testid="winrate-chart">
    <h2>{toolText("winRateChartTitle")}</h2>
    <p class="kp-muted">{toolText("winRateChartHelp")}</p>
    <svg class="kp-chart-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={toolText("winRateChartTitle")}>
      <line x1={padL} y1={yOf(1)} x2={W - padR} y2={yOf(1)} class="kp-chart-grid" />
      <line x1={padL} y1={yOf(0.5)} x2={W - padR} y2={yOf(0.5)} class="kp-chart-mid" />
      <line x1={padL} y1={yOf(0)} x2={W - padR} y2={yOf(0)} class="kp-chart-grid" />
      <text x={padL - 6} y={yOf(1) + 4} class="kp-chart-axis" text-anchor="end">100%</text>
      <text x={padL - 6} y={yOf(0.5) + 4} class="kp-chart-axis" text-anchor="end">50%</text>
      <text x={padL - 6} y={yOf(0) + 4} class="kp-chart-axis" text-anchor="end">0%</text>
      {turns.length > 1 && <polyline fill="none" class="kp-chart-line" points={poly} />}
      {turns.map((p, i) => {
        const cx = xOf(i);
        const cy = yOf(p.winRate);
        const pct = Math.round(p.winRate * 100);
        const href = kifuUrl(props.code, { side: 0, turn: p.turn });
        return <a key={p.turn} href={href} target="_blank" rel="noopener" aria-label={toolText("winRatePointLabel", p.turn, pct)}>
          <circle cx={cx} cy={cy} r={6} class="kp-chart-dot" />
          <text x={cx} y={H - 6} class="kp-chart-turn" text-anchor="middle">T{p.turn}</text>
        </a>;
      })}
    </svg>
    <div class="kp-chart-turns">
      {turns.map((p) => {
        const pct = Math.round(p.winRate * 100);
        return <a key={p.turn} class="kp-turn-chip" href={kifuUrl(props.code, { side: 0, turn: p.turn })} target="_blank" rel="noopener">
          <span>T{p.turn}</span>
          <strong>{pct}%</strong>
        </a>;
      })}
    </div>
  </div>;
}
