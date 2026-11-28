// psense.lib.net/_/PDINPUT.cgi encoder & automated browser importer.
//
// Specification of `PDINPUT.cgi` (EUC-JP, Apache 1.3.39):
// 1. Cookie storage:
//    User parties are stored in HTTP cookies `PD0`, `PD1`, `PD2`, ... under
//    path `/_/`. `CBAN` (1-indexed party number in the UI) maps to `PD{CBAN-1}`.
//    Active parties begin with `1_x_` followed by 6 `_x_`-delimited Pokémon
//    records; deleted slots hold `2`.
// 2. Apache 1.3 header limit:
//    `LimitRequestFieldSize` is 8190 bytes. Because each 6-Pokémon party is
//    ~455 bytes (`PDn=1_x_...`), at most 17 parties (~7.7 KB) can reside in
//    `Cookie:` simultaneously before Apache returns `400 Bad Request`.
//    The 15-team `team-pool-v4` catalog fits in a single batch (~6.8 KB).
// 3. 26-field Pokémon record (`_` delimiter):
//    [0]      National Dex number (1..251)
//    [1]      Level (50..55)
//    [2..5]   Atk, Def, Spe, Spc DVs in uppercase hex (`0`..`F`)
//    [6..10]  HP, Atk, Def, Spe, Spc stat-experience effort levels (`0`..`63`, `floor(ev/4)`)
//    [11..14] Move 1..4 IDs (`dex.moves[id].num + 1`, or `0` if empty; all Hidden Power variants use 238)
//    [15..18] Move 1..4 max PP (`base_pp + floor(base_pp / 5) * ppup`, including 64 for base-40 PP moves)
//    [19..22] Move 1..4 PP Up counts (`0`..`3`)
//    [23]     Max HP (`floor(((base_hp + hp_dv) * 2 + do_hp) * level / 100) + level + 10`)
//    [24]     Status (`0`)
//    [25]     Gen 2 item internal hex ID in decimal (`0` if no item)
// 4. Cross-origin import mechanisms on `PDINPUT.cgi`:
//    - Direct slot write (1 POST per party):
//      POST `ACTION=NEW&ACTION2=SAVED&PBAN=<1..17>&PD=<mon1_x_.._x_mon5>&PC=5&B1=%B4%B0%CE%BB`
//      with the 6th Pokémon's EUC-JP form fields sets `Set-Cookie: PD{PBAN-1}=1_x_<mon1..6>`
//      immediately without requiring `SameSite=Lax` intermediate cookies.
//    - Saved-HTML import (`ACTION=RECEPT`):
//      POST `ACTION=RECEPT&C=1_x_<mon1..6>` in a top-level window/tab chains
//      `RECEPT -> RECEPT_BRIDGE -> RECEPT_SAVE` via auto-submitted forms and
//      appends the party to the first empty slot (`PD0..PD11`).

import { toId } from "./i18n";
import type { PoolTeam } from "./types";

export const PDINPUT_URL = "http://psense.lib.net/_/PDINPUT.cgi";

/** Apache 1.3.39 LimitRequestFieldSize is 8190 bytes; 17 6-mon parties (~7.7 KB)
 * is the maximum safe batch size before Cookie header overflow (HTTP 400). */
export const PDINPUT_MAX_BATCH_SIZE = 17;
export const PDINPUT_COOKIE_LIMIT_BYTES = 8000;

/** Gen 2 internal item IDs (decimal) used by psense.lib.net/_/PDINPUT.cgi. */
export const PDINPUT_ITEM_CODES: Record<string, number> = {
  "": 0,
  masterball: 2,
  ultraball: 3,
  brightpowder: 4,
  greatball: 5,
  pokeball: 6,
  moonstone: 9,
  firestone: 23,
  thunderstone: 24,
  waterstone: 25,
  luckypunch: 31,
  leafstone: 35,
  metalpowder: 36,
  quickclaw: 74,
  psncureberry: 75,
  softsand: 77,
  sharpbeak: 78,
  przcureberry: 79,
  burntberry: 80,
  iceberry: 81,
  poisonbarb: 82,
  kingsrock: 83,
  bitterberry: 84,
  mintberry: 85,
  silverpowder: 89,
  mysticwater: 96,
  twistedspoon: 97,
  blackbelt: 99,
  blackglasses: 103,
  pinkbow: 105,
  stick: 106,
  nevermeltice: 108,
  magnet: 109,
  miracleberry: 110,
  spelltag: 114,
  miracleseed: 118,
  thickclub: 119,
  focusband: 120,
  hardstone: 126,
  charcoal: 139,
  berryjuice: 140,
  scopelens: 141,
  metalcoat: 144,
  dragonfang: 145,
  leftovers: 147,
  mysteryberry: 151,
  dragonscale: 152,
  berserkgene: 153,
  heavyball: 158,
  mail: 159,
  flowermail: 159,
  levelball: 160,
  lureball: 161,
  fastball: 162,
  lightball: 164,
  friendball: 165,
  moonball: 166,
  loveball: 167,
  sunstone: 170,
  polkadotbow: 171,
  upgrade: 173,
  berry: 174,
  goldberry: 175,
  sportball: 178,
  parkball: 178,
  surfmail: 182,
  litebluemail: 183,
  portraitmail: 184,
  lovelymail: 185,
  eonmail: 186,
  morphmail: 187,
  blueskymail: 188,
  musicmail: 189,
  miragemail: 190,
};

/** Gen 2 / PBS `PDINPUT.cgi` item spelling overrides where `data/i18n-ja.json`
 * uses a generic or Gen 3+ name (`mail` -> `はながらボール`, `sportball` -> `パークボール`). */
export const PDINPUT_ITEM_NAME_OVERRIDES: Record<string, string> = {
  blackglasses: "くろいめがね",
  mail: "はながらボール",
  flowermail: "はながらボール",
  sportball: "パークボール",
  parkball: "パークボール",
};

/** Canonical Gen 2 Hidden Power DV spreads (Atk/Def DVs; Spe=15, Spc=15, HP derived). */
export const GEN2_HIDDEN_POWER_DVS: Record<string, { atk: number; def: number }> = {
  fighting: { atk: 12, def: 12 },
  flying: { atk: 12, def: 13 },
  poison: { atk: 12, def: 14 },
  ground: { atk: 12, def: 15 },
  rock: { atk: 13, def: 12 },
  bug: { atk: 13, def: 13 },
  ghost: { atk: 13, def: 14 },
  steel: { atk: 13, def: 15 },
  fire: { atk: 14, def: 12 },
  water: { atk: 14, def: 13 },
  grass: { atk: 14, def: 14 },
  electric: { atk: 14, def: 15 },
  psychic: { atk: 15, def: 12 },
  ice: { atk: 15, def: 13 },
  dragon: { atk: 15, def: 14 },
  dark: { atk: 15, def: 15 },
};

/** Items that are inert in Gym Leader's Castle PBS (`PS.cgi`) due to Gen 2 bugs.
 * NEVER silently auto-rewrite these to their functional counterparts (`iceberry` /
 * `dragonscale`), as that can cause Item Clause collisions. */
export const CASTLE_INERT_ITEMS: Record<string, { replacement: string; reason: string }> = {
  bitterberry: {
    replacement: "iceberry",
    reason: "Bitter Berry (ID 84) is inert in Castle PBS; Ice Berry (ID 81) cures both burn and confusion.",
  },
  dragonfang: {
    replacement: "dragonscale",
    reason: "Dragon Fang (ID 145) is inert in Castle PBS; Dragon Scale (ID 152) boosts Dragon moves by 1.1x.",
  },
};

export interface PdinputDexSpecies {
  num: number;
  name: string;
  baseStats: { hp: number; atk: number; def: number; spa: number; spd: number; spe: number };
}

export interface PdinputDexMove {
  num: number;
  name: string;
  pp: number;
  noPPBoosts?: boolean;
}

export interface PdinputDex {
  species: Record<string, PdinputDexSpecies>;
  moves: Record<string, PdinputDexMove>;
}

export interface PdinputI18nJa {
  species: Record<string, string>;
  moves: Record<string, string>;
  items: Record<string, string>;
}

export interface SetInput {
  species?: string;
  name?: string;
  level?: number;
  item?: string;
  moves?: string[];
  /** Per-move PP Up counts (`0..=3`, clamped; `noPPBoosts` moves forced to `0`). */
  ppUps?: number[];
  /** Default PP Up count for moves not overridden by `ppUps` (default `3`). */
  ppUp?: number;
  ivs?: { hp?: number; atk?: number; def?: number; spa?: number; spd?: number; spe?: number };
  evs?: { hp?: number; atk?: number; def?: number; spa?: number; spd?: number; spe?: number };
}

export interface EncodedMon {
  /** 26-field `_`-delimited PDINPUT record for this Pokémon. */
  record: string;
  speciesId: string;
  jpSpecies: string;
  level: number;
  jpMoves: [string, string, string, string];
  ppUps: [number, number, number, number];
  jpItem: string;
  doValues: [number, number, number, number, number];
  koHex: string;
  maxHp: number;
}

export interface EncodedTeam {
  id: string;
  tier: string;
  /** Full `PDn` cookie value (`1_x_<mon1>_x_..._x_<mon6>`). */
  cookieValue: string;
  /** `_x_`-joined records of the first 5 Pokémon (used as hidden `PD` in single-POST slot writes). */
  firstFivePd: string;
  /** The 6th Pokémon's form fields (used with `firstFivePd` in single-POST slot writes). */
  sixthMon: EncodedMon;
  mons: EncodedMon[];
}

export interface CastleItemAuditFinding {
  severity: "warning" | "error";
  code: "item-castle-inert" | "item-clause";
  slot: number;
  species: string;
  item: string;
  itemCode?: number;
  suggestedItem?: string;
  message: string;
}

export interface CastleItemAuditReport {
  ok: boolean;
  warnings: CastleItemAuditFinding[];
  errors: CastleItemAuditFinding[];
  findings: CastleItemAuditFinding[];
}

/** Non-mutating Castle item audit for `PDINPUT.cgi` exports (§7-C.2):
 * emits warnings on inert `bitterberry` (ID 84) / `dragonfang` (ID 145)
 * and errors on Item Clause duplicates without ever silently rewriting items. */
export function auditCastleTeamForPdinput(
  sets: readonly SetInput[],
): CastleItemAuditReport {
  const findings: CastleItemAuditFinding[] = [];
  const seenByCode = new Map<number, number>();
  const seenById = new Map<string, number>();

  for (let slot = 0; slot < sets.length; slot++) {
    const s = sets[slot] ?? {};
    const species = toId(s.species ?? s.name ?? "");
    const itemId = toId(s.item ?? "");
    if (!itemId) continue;

    const itemCode = PDINPUT_ITEM_CODES[itemId];
    const inert = CASTLE_INERT_ITEMS[itemId];
    if (inert) {
      findings.push({
        severity: "warning",
        code: "item-castle-inert",
        slot,
        species,
        item: itemId,
        itemCode,
        suggestedItem: inert.replacement,
        message: inert.reason,
      });
    }

    const dupSlot =
      itemCode !== undefined ? seenByCode.get(itemCode) : seenById.get(itemId);
    if (dupSlot !== undefined) {
      findings.push({
        severity: "error",
        code: "item-clause",
        slot,
        species,
        item: itemId,
        itemCode,
        message: `Duplicate held item "${itemId}" (also on slot ${dupSlot + 1}) violates Item Clause.`,
      });
    } else {
      if (itemCode !== undefined) seenByCode.set(itemCode, slot);
      seenById.set(itemId, slot);
    }
  }

  const warnings = findings.filter((f) => f.severity === "warning");
  const errors = findings.filter((f) => f.severity === "error");
  return { ok: errors.length === 0, warnings, errors, findings };
}

function detectTypedHiddenPower(rawMoves: readonly string[]): string | undefined {
  for (const rawMove of rawMoves) {
    if (!rawMove) continue;
    const mId = toId(rawMove);
    if (mId.startsWith("hiddenpower") && mId.length > "hiddenpower".length) {
      const hpType = mId.slice("hiddenpower".length);
      if (hpType in GEN2_HIDDEN_POWER_DVS) return hpType;
    }
  }
  return undefined;
}

function shouldAutofillHiddenPowerDvs(ivs: SetInput["ivs"]): boolean {
  if (!ivs) return true;
  const values = [ivs.hp, ivs.atk, ivs.def, ivs.spa, ivs.spd, ivs.spe].filter(
    (v): v is number => typeof v === "number",
  );
  return values.length === 0 || values.every((v) => v === 31);
}

export function encodeMonForPdinput(
  rawSet: unknown,
  dex: PdinputDex,
  ja: PdinputI18nJa,
): EncodedMon {
  const s = (rawSet ?? {}) as SetInput;
  const spId = toId(s.species ?? s.name ?? "");
  const sp = dex.species[spId];
  if (!sp) throw new Error(`Unknown species: ${s.species ?? s.name ?? "?"}`);
  const jpSpecies = ja.species[spId];
  if (!jpSpecies) throw new Error(`Missing JP species name for ${spId}`);

  const level = typeof s.level === "number" ? s.level : 50;
  const rawMoves = Array.isArray(s.moves) ? s.moves : [];
  const typedHp = detectTypedHiddenPower(rawMoves);

  const ivs = s.ivs ?? {};
  let atkDv: number;
  let defDv: number;
  let speDv: number;
  let spcDv: number;
  if (typedHp && shouldAutofillHiddenPowerDvs(s.ivs)) {
    const canonical = GEN2_HIDDEN_POWER_DVS[typedHp];
    atkDv = canonical.atk;
    defDv = canonical.def;
    speDv = 15;
    spcDv = 15;
  } else {
    const rawSpc = ivs.spa ?? ivs.spd ?? 31;
    atkDv = Math.max(0, Math.min(15, Math.floor((ivs.atk ?? 31) / 2)));
    defDv = Math.max(0, Math.min(15, Math.floor((ivs.def ?? 31) / 2)));
    speDv = Math.max(0, Math.min(15, Math.floor((ivs.spe ?? 31) / 2)));
    spcDv = Math.max(0, Math.min(15, Math.floor(rawSpc / 2)));
  }
  const hpDv =
    ((atkDv & 1) << 3) |
    ((defDv & 1) << 2) |
    ((speDv & 1) << 1) |
    (spcDv & 1);

  const evs = s.evs ?? {};
  const rawSpcEv = evs.spa ?? evs.spd ?? 255;
  const doHp = Math.max(0, Math.min(63, Math.floor((evs.hp ?? 255) / 4)));
  const doAtk = Math.max(0, Math.min(63, Math.floor((evs.atk ?? 255) / 4)));
  const doDef = Math.max(0, Math.min(63, Math.floor((evs.def ?? 255) / 4)));
  const doSpe = Math.max(0, Math.min(63, Math.floor((evs.spe ?? 255) / 4)));
  const doSpc = Math.max(0, Math.min(63, Math.floor(rawSpcEv / 4)));

  const moveCodes: number[] = [];
  const movePps: number[] = [];
  const ppUps: [number, number, number, number] = [0, 0, 0, 0];
  const jpMoves: [string, string, string, string] = ["", "", "", ""];

  const compactMoves: Array<{ rawMove: string; origIdx: number }> = [];
  for (let i = 0; i < rawMoves.length && compactMoves.length < 4; i++) {
    const rawMove = rawMoves[i];
    if (typeof rawMove === "string" && rawMove.trim().length > 0) {
      compactMoves.push({ rawMove, origIdx: i });
    }
  }

  for (let i = 0; i < 4; i++) {
    const entry = compactMoves[i];
    if (entry) {
      let mId = toId(entry.rawMove);
      if (mId.startsWith("hiddenpower")) mId = "hiddenpower";
      const mv = dex.moves[mId];
      if (!mv) throw new Error(`Unknown move: ${entry.rawMove}`);
      const jpMv = ja.moves[mId];
      if (!jpMv) throw new Error(`Missing JP move name for ${mId}`);
      const noBoosts = mv.noPPBoosts === true || mId === "sketch" || mId === "struggle";
      const requestedPpUp =
        Array.isArray(s.ppUps) && typeof s.ppUps[entry.origIdx] === "number"
          ? s.ppUps[entry.origIdx]
          : typeof s.ppUp === "number"
            ? s.ppUp
            : 3;
      const ppUp = noBoosts
        ? 0
        : Math.max(0, Math.min(3, Math.trunc(requestedPpUp)));
      // PDINPUT.cgi uses `dex_num + 1` and `base_pp + floor(base_pp / 5) * ppUp`
      // (including 64 PP for base-40 PP moves).
      moveCodes.push(mv.num + 1);
      movePps.push(mv.pp + Math.floor(mv.pp / 5) * ppUp);
      ppUps[i] = ppUp;
      jpMoves[i] = jpMv;
    } else {
      moveCodes.push(0);
      movePps.push(0);
      ppUps[i] = 0;
      jpMoves[i] = "";
    }
  }

  const baseHp = sp.baseStats.hp;
  const maxHp =
    Math.floor(((baseHp + hpDv) * 2 + doHp) * level / 100) + level + 10;

  const itemId = toId(s.item ?? "");
  const itemCode = PDINPUT_ITEM_CODES[itemId];
  if (itemCode === undefined) {
    throw new Error(`Unsupported item for PDINPUT.cgi: ${s.item ?? ""}`);
  }
  const jpItem = itemId
    ? (PDINPUT_ITEM_NAME_OVERRIDES[itemId] ?? ja.items[itemId] ?? "")
    : "";

  const hex = (n: number) => n.toString(16).toUpperCase();
  const koHex = `${hex(atkDv)}${hex(defDv)}${hex(speDv)}${hex(spcDv)}`;

  const fields = [
    sp.num,
    level,
    hex(atkDv),
    hex(defDv),
    hex(speDv),
    hex(spcDv),
    doHp,
    doAtk,
    doDef,
    doSpe,
    doSpc,
    ...moveCodes,
    ...movePps,
    ...ppUps,
    maxHp,
    0,
    itemCode,
  ];

  return {
    record: fields.join("_"),
    speciesId: spId,
    jpSpecies,
    level,
    jpMoves,
    ppUps,
    jpItem,
    doValues: [doHp, doAtk, doDef, doSpe, doSpc],
    koHex,
    maxHp,
  };
}

export function encodeTeamForPdinput(
  team: Pick<PoolTeam, "id" | "sets"> & { tier?: string },
  dex: PdinputDex,
  ja: PdinputI18nJa,
): EncodedTeam {
  if (!Array.isArray(team.sets) || team.sets.length !== 6) {
    throw new Error(`Team ${team.id} must have 6 Pokémon (got ${team.sets?.length ?? 0})`);
  }
  const mons = team.sets.map((s) => encodeMonForPdinput(s, dex, ja));
  const cookieValue = "1_x_" + mons.map((m) => m.record).join("_x_");
  const firstFivePd = mons.slice(0, 5).map((m) => m.record).join("_x_");
  return {
    id: team.id,
    tier: team.tier ?? "",
    cookieValue,
    firstFivePd,
    sixthMon: mons[5],
    mons,
  };
}

/** Form fields for a single-POST write into slot `pban` (1-indexed, sets `PD{pban-1}`).
 * Submitting this form with `accept-charset="EUC-JP"` to `PDINPUT.cgi` writes all
 * 6 Pokémon into `PD{pban-1}` in a single HTTP request. */
export function buildSinglePostSlotFields(
  team: EncodedTeam,
  pban: number,
): Record<string, string> {
  const m6 = team.sixthMon;
  return {
    sim: "",
    ACTION: "NEW",
    ACTION2: "SAVED",
    PD: team.firstFivePd,
    PC: "5",
    PBAN: String(pban),
    LV: String(m6.level),
    SYU: m6.jpSpecies,
    WAZA1: m6.jpMoves[0],
    PPUP1: String(m6.ppUps[0]),
    WAZA2: m6.jpMoves[1],
    PPUP2: String(m6.ppUps[1]),
    WAZA3: m6.jpMoves[2],
    PPUP3: String(m6.ppUps[2]),
    WAZA4: m6.jpMoves[3],
    PPUP4: String(m6.ppUps[3]),
    ITEM: m6.jpItem,
    DO1: String(m6.doValues[0]),
    DO2: String(m6.doValues[1]),
    DO3: String(m6.doValues[2]),
    DO4: String(m6.doValues[3]),
    DO5: String(m6.doValues[4]),
    KO: m6.koHex,
    B1: "完了",
  };
}

/** Compute the exact HTTP `Cookie:` header byte length for a batch of encoded teams. */
export function cookieHeaderBytes(teams: EncodedTeam[]): number {
  if (teams.length === 0) return 0;
  return teams
    .map((t, i) => `PD${i}=${t.cookieValue}`)
    .join("; ").length;
}

/** Split a list of encoded teams into safe batches that fit within Apache's
 * 8190-byte Cookie header limit (at most 17 teams per batch). */
export function splitIntoPdinputBatches(
  teams: EncodedTeam[],
  maxPerBatch = PDINPUT_MAX_BATCH_SIZE,
  maxBytes = PDINPUT_COOKIE_LIMIT_BYTES,
): EncodedTeam[][] {
  const batches: EncodedTeam[][] = [];
  let current: EncodedTeam[] = [];
  for (const t of teams) {
    const next = [...current, t];
    if (
      current.length > 0 &&
      (next.length > maxPerBatch || cookieHeaderBytes(next) > maxBytes)
    ) {
      batches.push(current);
      current = [t];
    } else {
      current = next;
    }
  }
  if (current.length > 0) batches.push(current);
  return batches;
}

/** Generate a bookmarklet / console one-liner that writes the given batch into
 * `document.cookie` on `http://psense.lib.net/_/` and clears any stale higher slots. */
export function buildCookieInstallerScript(
  teams: EncodedTeam[],
  clearUpToSlot = 35,
): string {
  const entries = teams.map((t) => t.cookieValue);
  return `javascript:(()=>{if(location.hostname!=='psense.lib.net'){alert('psense.lib.net/_/PDINPUT.cgi で実行してください');return;}const p=${JSON.stringify(entries)};const exp='expires=Tue, 19 Jan 2038 03:14:07 GMT; path=/_/';for(let i=0;i<${clearUpToSlot};i++){document.cookie='PD'+i+'=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/_/';}p.forEach((v,i)=>{document.cookie='PD'+i+'='+v+'; '+exp;});location.href='http://psense.lib.net/_/PDINPUT.cgi';})()`;
}
