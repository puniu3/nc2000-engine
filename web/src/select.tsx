// Team select: pick your team from the meta pool, then the opponent's
// (specific team or random-from-pool).

import { useState } from "preact/hooks";
import type { MetaPool, PoolTeam } from "./types";
import { STRENGTHS } from "./app";

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

export function TeamSelect(props: {
  pool: MetaPool;
  strength: number;
  onStrength: (n: number) => void;
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
      </header>

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
    </div>
  );
}
