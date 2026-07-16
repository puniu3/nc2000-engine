// Team select: pick your team from the meta pool, then the opponent's
// (specific team or random-from-pool). Also hosts the device benchmark —
// the M9 gate ("skuct:3000 within 2-3 s/move") is certified per device by
// tapping it.

import { useEffect, useRef, useState } from "preact/hooks";
import type { BotInfo, MetaPool, PoolTeam } from "./types";
import { STRENGTHS } from "./app";
import { BotWorker } from "./bot";

function provenanceLine(t: PoolTeam): string {
  const p = t.provenance;
  const bits: string[] = [];
  if (p.player) bits.push(p.player);
  if (p.placement) bits.push(p.placement);
  if (p.event) bits.push(p.event);
  return bits.join(" · ") || p.source || "";
}

function TeamCard(props: {
  team: PoolTeam;
  index: number;
  selected: boolean;
  onTap: () => void;
}) {
  const { team, index, selected } = props;
  return (
    <button
      class={`team-card ${selected ? "selected" : ""}`}
      onClick={props.onTap}
      data-team={index}
    >
      <div class="team-card-head">
        <span class="team-rank">#{index + 1}</span>
        <span class="team-id">{team.id}</span>
        <span class="team-tier">{team.tier}</span>
      </div>
      <div class="team-prov">{provenanceLine(team)}</div>
      <div class="team-species">
        {team.species.map((sp, i) => (
          <span class="species-chip" key={i}>
            {sp} <small>L{team.levels[i]}</small>
          </span>
        ))}
      </div>
    </button>
  );
}

// Fixed workload: identical battle + searcher seeds + iteration count on
// every device, so the numbers are directly comparable. 10k iterations at
// an in-battle root ~= 4-7 s on 2025 hardware.
const BENCH_ITERS = 10000;
const BENCH_BATTLE_SEED = "1,2,3,4";
const BENCH_SEARCH_SEED = 123456789;
const GATE_ITERS = 3000; // the M9 gate strength (skuct:3000 == mcts:3000)
const GATE_SECONDS = 3;

type BenchState =
  | { s: "idle" }
  | { s: "running"; done: number; total: number }
  | { s: "done"; itersPerSec: number; secPerMove: number }
  | { s: "error"; msg: string };

function DeviceBench(props: { pool: MetaPool }) {
  const [state, setState] = useState<BenchState>({ s: "idle" });
  const aliveRef = useRef(true);
  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);

  async function run() {
    if (state.s === "running") return;
    setState({ s: "running", done: 0, total: BENCH_ITERS });
    const bot = new BotWorker();
    try {
      const r = await bot.bench(
        JSON.stringify(props.pool.teams[0].sets),
        JSON.stringify(props.pool.teams[1].sets),
        BENCH_BATTLE_SEED,
        BENCH_SEARCH_SEED,
        BENCH_ITERS,
        (done, total) => {
          if (aliveRef.current) setState({ s: "running", done, total });
        },
      );
      const itersPerSec = r.iters / (r.ms / 1000);
      if (aliveRef.current)
        setState({
          s: "done",
          itersPerSec,
          secPerMove: GATE_ITERS / itersPerSec,
        });
    } catch (e) {
      if (aliveRef.current) setState({ s: "error", msg: String(e) });
    } finally {
      bot.terminate();
    }
  }

  return (
    <div class="bench-box">
      <div class="bench-head">
        <span class="bench-title">Device benchmark</span>
        <button
          class="ghost bench-btn"
          disabled={state.s === "running"}
          onClick={() => void run()}
        >
          {state.s === "running"
            ? `Running… ${Math.round((state.done / state.total) * 100)}%`
            : state.s === "idle"
              ? "Run (~5 s)"
              : "Run again"}
        </button>
      </div>
      {state.s === "done" && (
        <div
          class={`bench-result ${state.secPerMove <= GATE_SECONDS ? "pass" : "fail"}`}
          data-iters-per-sec={Math.round(state.itersPerSec)}
          data-sec-per-move={state.secPerMove.toFixed(2)}
        >
          {Math.round(state.itersPerSec)} iterations/s — a{" "}
          {GATE_ITERS / 1000}k think ≈ {state.secPerMove.toFixed(2)} s/move.
          Strength gate (≤{GATE_SECONDS} s):{" "}
          {state.secPerMove <= GATE_SECONDS ? "PASS" : "MISS"}
        </div>
      )}
      {state.s === "error" && <div class="bench-result fail">{state.msg}</div>}
      <div class="bench-note">
        Fixed search workload ({BENCH_ITERS / 1000}k iterations, fixed seeds)
        — comparable across devices.
      </div>
    </div>
  );
}

export function TeamSelect(props: {
  pool: MetaPool;
  strength: number;
  onStrength: (n: number) => void;
  botInfo: BotInfo;
  onBotInfo: (v: BotInfo) => void;
  onStart: (humanIdx: number, botIdx: number | "random") => void;
}) {
  const [humanIdx, setHumanIdx] = useState<number | null>(null);
  const [botIdx, setBotIdx] = useState<number | "random">("random");
  const [step, setStep] = useState<0 | 1>(0);
  const teams = props.pool.teams;

  return (
    <div class="screen select-screen">
      <header class="app-header">
        <h1>NC2000</h1>
        <span class="subtitle">Gen 2 · human vs bot</span>
        <label class="strength-label">
          Bot strength{" "}
          <select
            value={props.strength}
            onChange={(e) =>
              props.onStrength(Number((e.target as HTMLSelectElement).value))
            }
          >
            {STRENGTHS.map((s) => (
              <option value={s.iters} key={s.iters}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <label class="strength-label botinfo-label">
          Bot information{" "}
          <select
            value={props.botInfo}
            onChange={(e) =>
              props.onBotInfo(
                (e.target as HTMLSelectElement).value as BotInfo,
              )
            }
          >
            <option value="fair">Fair (blind)</option>
            <option value="xray">X-ray (perfect info)</option>
          </select>
        </label>
      </header>
      <div class="botinfo-note">
        {props.botInfo === "fair"
          ? "Fair: the bot sees only what a human opponent would — your movesets and items stay hidden until revealed in play."
          : "X-ray: the bot reads your exact sets, moves and items — perfect information."}
      </div>

      <div class="select-bar">
        <div class={`select-slot ${step === 0 ? "current" : ""}`}>
          <span class="slot-label">You</span>
          <span class="slot-value">
            {humanIdx === null ? "—" : teams[humanIdx].id}
          </span>
        </div>
        <span class="vs">vs</span>
        <div class={`select-slot ${step === 1 ? "current" : ""}`}>
          <span class="slot-label">Bot</span>
          <span class="slot-value">
            {botIdx === "random" ? "random from pool" : teams[botIdx].id}
          </span>
        </div>
        <button
          class="primary start-btn"
          disabled={humanIdx === null}
          onClick={() => props.onStart(humanIdx!, botIdx)}
        >
          Start battle
        </button>
      </div>

      <h2>
        {step === 0 ? "Choose your team" : "Choose the opponent's team"}
      </h2>
      {step === 1 && (
        <button
          class={`team-card random-card ${botIdx === "random" ? "selected" : ""}`}
          onClick={() => setBotIdx("random")}
        >
          Random from pool (34 teams)
        </button>
      )}
      <div class="team-list">
        {teams.map((t, i) => (
          <TeamCard
            key={t.id}
            team={t}
            index={i}
            selected={step === 0 ? humanIdx === i : botIdx === i}
            onTap={() => {
              if (step === 0) {
                setHumanIdx(i);
                setStep(1);
                window.scrollTo({ top: 0 });
              } else {
                setBotIdx(i);
              }
            }}
          />
        ))}
      </div>

      <DeviceBench pool={props.pool} />
    </div>
  );
}
