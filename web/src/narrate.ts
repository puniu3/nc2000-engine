// Protocol-log narration — a direct port of the settled CLI renderer
// (crates/bot/examples/play.rs, LogView/render_line), viewer = the human
// side. Foe HP renders as %, own HP exact; |split|pN groups pick the
// secret (real-HP) line for the viewer's own side and the shared line for
// the foe.
//
// M13: the line templates live in per-locale tables (EN_T / JA_T below,
// picked at render time), and every name crossing a template is mapped
// through the i18n lookups (species/moves/items from data/i18n-ja.json,
// conditions/statuses/stats from the hand tables) with English fallback.
// The Japanese register is the official-game kana style
// (「ピカチュウは 10まんボルトを つかった!」). Unknown tags keep the
// raw-line fallback in both locales.

import type { LogEntry } from "./types";
import {
  condName,
  itemName,
  locale,
  moveName,
  speciesName,
  statLongName,
  statusName,
  toId,
} from "./i18n";

const MAJOR_TAGS = new Set([
  "move",
  "switch",
  "drag",
  "faint",
  "cant",
  "-prepare",
]);

/** Everything locale-specific about a narration line. `who` strings are
 * already foe-prefixed and name-translated when they reach a template. */
interface Templates {
  turn(n: string): string;
  foe(name: string): string;
  used(who: string, move: string): string;
  sentOut(who: string, hp: string): string;
  dragged(who: string, hp: string): string;
  hpLine(who: string, hp: string, from: string): string;
  fainted(who: string): string;
  status(who: string, code: string, from: string): string;
  statusOther(who: string, raw: string): string;
  cured(who: string, status: string): string;
  boost(who: string, stat: string, up: boolean, sharp: boolean): string;
  cant(who: string, why: string): string;
  cantOther(who: string, reason: string): string;
  crit(): string;
  superEffective(): string;
  resisted(): string;
  immune(who: string): string;
  missed(who: string): string;
  failed(): string;
  nothing(): string;
  hitCount(n: string): string;
  preparing(who: string, move: string): string;
  mustRecharge(who: string): string;
  condStart(who: string, condId: string, cond: string, from: string): string;
  condEnd(who: string, condId: string, cond: string, from: string): string;
  activate(who: string, cond: string): string;
  fieldActivate(condId: string, cond: string): string;
  transformed(who: string, target: string): string;
  copiedBoost(who: string, target: string): string;
  clearAllBoost(): string;
  weather(id: string, name: string): string;
  side(mine: boolean): string;
  sideStart(owner: string, cond: string): string;
  sideEnd(owner: string, cond: string): string;
  holds(who: string, item: string, from: string): string;
  endItem(who: string, item: string, ate: boolean, from: string): string;
  win(winnerIsViewer: boolean, rawName: string): string;
  tie(): string;
}

const EN_T: Templates = {
  turn: (n) => `Turn ${n}`,
  foe: (name) => `Foe ${name}`,
  used: (who, move) => `${who} used ${move}!`,
  sentOut: (who, hp) => `${who} was sent out (${hp})`,
  dragged: (who, hp) => `${who} was dragged out (${hp})`,
  hpLine: (who, hp, from) => `${who}: ${hp}${from}`,
  fainted: (who) => `${who} fainted!`,
  status: (who, code, from) => {
    const verbs: Record<string, string> = {
      brn: "was burned",
      par: "was paralyzed",
      slp: "fell asleep",
      frz: "was frozen solid",
      psn: "was poisoned",
      tox: "was badly poisoned",
    };
    return `${who} ${verbs[code]}!${from}`;
  },
  statusOther: (who, raw) => `${who} status: ${raw}`,
  cured: (who, status) => `${who} was cured of ${status}!`,
  boost: (who, stat, up, sharp) =>
    `${who}'s ${stat} ${up ? "rose" : "fell"}${sharp ? " sharply" : ""}!`,
  cant: (who, why) => {
    const whys: Record<string, string> = {
      slp: "is fast asleep",
      par: "is fully paralyzed",
      frz: "is frozen solid",
      flinch: "flinched",
      recharge: "must recharge",
    };
    return `${who} ${whys[why]}!`;
  },
  cantOther: (who, reason) => `${who} can't move (${reason})!`,
  crit: () => "A critical hit!",
  superEffective: () => "It's super effective!",
  resisted: () => "It's not very effective...",
  immune: (who) => `It doesn't affect ${who}...`,
  missed: (who) => `${who}'s attack missed!`,
  failed: () => "But it failed!",
  nothing: () => "But nothing happened!",
  hitCount: (n) => `Hit ${n} time(s)!`,
  preparing: (who, move) => `${who} is preparing ${move}...`,
  mustRecharge: (who) => `${who} must recharge!`,
  condStart: (who, _id, cond, from) => `${who}: ${cond} started${from}`,
  condEnd: (who, _id, cond, from) => `${who}: ${cond} ended${from}`,
  activate: (who, cond) => `${who}: ${cond}`,
  fieldActivate: (_id, cond) => `${cond} affects everyone!`,
  transformed: (who, target) => `${who} transformed into ${target}!`,
  copiedBoost: (who, target) => `${who} copied ${target}'s stat changes!`,
  clearAllBoost: () => "All stat changes were eliminated!",
  weather: (id, name) =>
    id === "none" ? "The weather cleared." : `Weather: ${name}`,
  side: (mine) => (mine ? "Your side" : "Foe side"),
  sideStart: (owner, cond) => `${owner}: ${cond} started`,
  sideEnd: (owner, cond) => `${owner}: ${cond} ended`,
  holds: (who, item, from) => `${who} holds ${item}${from}`,
  endItem: (who, item, _ate, from) => `${who} used up its ${item}${from}`,
  win: (_isViewer, rawName) => `${rawName} wins!`,
  tie: () => "Tie",
};

// Official-game kana register: spaces between phrases, names substituted
// from the JP tables. Per-condition start/end phrasings for the common
// volatiles; the generic form covers the rest.
const JA_START: Record<string, (who: string) => string> = {
  substitute: (w) => `${w}の みがわりが あらわれた!`,
  confusion: (w) => `${w}は こんらんした!`,
  leechseed: (w) => `${w}に やどりぎのタネが うえつけられた!`,
  encore: (w) => `${w}は アンコールを うけた!`,
  attract: (w) => `${w}は メロメロに なった!`,
  curse: (w) => `${w}は のろいを かけられた!`,
  disable: (w) => `${w}は かなしばりに あった!`,
  nightmare: (w) => `${w}は あくむを みはじめた!`,
  meanlook: (w) => `${w}は にげられなくなった!`,
  focusenergy: (w) => `${w}は きあいを ためている!`,
  destinybond: (w) => `${w}は あいてを みちづれに しようとしている!`,
  perish3: (w) => `${w}の ほろびのカウントが 3に なった!`,
  perish2: (w) => `${w}の ほろびのカウントが 2に なった!`,
  perish1: (w) => `${w}の ほろびのカウントが 1に なった!`,
  perish0: (w) => `${w}の ほろびのカウントが 0に なった!`,
};

const JA_END: Record<string, (who: string) => string> = {
  substitute: (w) => `${w}の みがわりは きえてしまった…`,
  confusion: (w) => `${w}の こんらんが とけた!`,
  disable: (w) => `${w}の かなしばりが とけた!`,
  encore: (w) => `${w}の アンコールが とけた!`,
  attract: (w) => `${w}の メロメロが なおった!`,
};

const JA_T: Templates = {
  turn: (n) => `ターン ${n}`,
  foe: (name) => `あいての ${name}`,
  used: (who, move) => `${who}は ${move}を つかった!`,
  sentOut: (who, hp) => `${who}が くりだされた! (${hp})`,
  dragged: (who, hp) => `${who}が ひきずりだされた! (${hp})`,
  hpLine: (who, hp, from) => `${who}: ${hp}${from}`,
  fainted: (who) => `${who}は たおれた!`,
  status: (who, code, from) => {
    const verbs: Record<string, string> = {
      brn: "やけどを おった",
      par: "まひして わざが でにくくなった",
      slp: "ねむってしまった",
      frz: "こおりついた",
      psn: "どくを あびた",
      tox: "もうどくを あびた",
    };
    return `${who}は ${verbs[code]}!${from}`;
  },
  statusOther: (who, raw) => `${who}の じょうたい: ${raw}`,
  cured: (who, status) => `${who}の ${status}が なおった!`,
  boost: (who, stat, up, sharp) =>
    `${who}の ${stat}が ${
      up
        ? sharp
          ? "ぐーんと あがった"
          : "あがった"
        : sharp
          ? "がくっと さがった"
          : "さがった"
    }!`,
  cant: (who, why) => {
    const whys: Record<string, string> = {
      slp: "ぐうぐう ねむっている",
      par: "からだが しびれて うごけない",
      frz: "こおってしまって うごけない",
      flinch: "ひるんで わざが だせなかった",
      recharge: "こうげきの はんどうで うごけない",
    };
    return `${who}は ${whys[why]}!`;
  },
  cantOther: (who, reason) => `${who}は うごけない! (${reason})`,
  crit: () => "きゅうしょに あたった!",
  superEffective: () => "こうかは ばつぐんだ!",
  resisted: () => "こうかは いまひとつのようだ…",
  immune: (who) => `${who}には こうかが ないようだ…`,
  missed: (who) => `${who}の こうげきは はずれた!`,
  failed: () => "しかし うまく きまらなかった!",
  nothing: () => "しかし なにも おこらなかった!",
  hitCount: (n) => `${n}かい あたった!`,
  preparing: (who, move) => `${who}は ${move}を じゅんびしている…`,
  mustRecharge: (who) => `${who}は こうげきの はんどうで うごけない!`,
  condStart: (who, id, cond, from) =>
    (JA_START[id]?.(who) ?? `${who}の ${cond}が はじまった!`) + from,
  condEnd: (who, id, cond, from) =>
    (JA_END[id]?.(who) ?? `${who}の ${cond}が なくなった!`) + from,
  activate: (who, cond) => `${who}: ${cond}`,
  fieldActivate: (id, cond) =>
    id === "perishsong"
      ? "ほろびのうたが ひびきわたった!"
      : `${cond}が みんなに はたらいた!`,
  transformed: (who, target) => `${who}は ${target}に へんしんした!`,
  copiedBoost: (who, target) =>
    `${who}は ${target}の のうりょくへんかを コピーした!`,
  clearAllBoost: () => "すべての のうりょくへんかが もとに もどった!",
  weather: (id, name) => {
    switch (id) {
      case "none":
        return "てんきは もとに もどった。";
      case "raindance":
        return "あめが ふりだした!";
      case "sunnyday":
        return "ひざしが つよくなった!";
      case "sandstorm":
        return "すなあらしが ふきはじめた!";
      default:
        return `てんき: ${name}`;
    }
  },
  side: (mine) => (mine ? "じぶんの じんち" : "あいての じんち"),
  sideStart: (owner, cond) => `${owner}に ${cond}!`,
  sideEnd: (owner, cond) => `${owner}の ${cond}が なくなった!`,
  holds: (who, item, from) => `${who}は ${item}を もっている${from}`,
  endItem: (who, item, ate, from) =>
    ate
      ? `${who}は ${item}を たべた!${from}`
      : `${who}は ${item}を つかいきった!${from}`,
  win: (isViewer) =>
    isViewer ? "あなたの しょうり!" : "あいての しょうり!",
  tie: () => "ひきわけ",
};

/** cant reasons / status codes both template tables know. */
const CANT_REASONS = new Set(["slp", "par", "frz", "flinch", "recharge"]);
const STATUS_CODES = new Set(["brn", "par", "slp", "frz", "psn", "tox"]);

export class Narrator {
  constructor(private viewer: number) {}

  private t(): Templates {
    return locale() === "ja" ? JA_T : EN_T;
  }

  /** Render a chunk of raw protocol lines into display entries. */
  render(lines: string[]): LogEntry[] {
    const out: LogEntry[] = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      // |split|pN: next two lines are the secret (real HP) then shared
      // (/48-scaled) variants of the same event.
      if (line.startsWith("|split|p")) {
        const splitSide = line.charCodeAt(8) - 0x31; // '1' -> 0
        const pick =
          this.viewer === splitSide ? lines[i + 1] : lines[i + 2];
        if (pick !== undefined) {
          const e = this.renderLine(pick);
          if (e) out.push(e);
        }
        i += 3;
        continue;
      }
      const e = this.renderLine(line);
      if (e) out.push(e);
      i += 1;
    }
    return out;
  }

  /** "p1a: Gastly" -> [side, "Gastly"] */
  private parseRef(r: string): [number, string] {
    const side = r.startsWith("p2") ? 1 : 0;
    const idx = r.indexOf(": ");
    return [side, idx >= 0 ? r.slice(idx + 2) : r];
  }

  private who(r: string): string {
    const [side, name] = this.parseRef(r);
    const display = speciesName(name);
    return side === this.viewer ? display : this.t().foe(display);
  }

  /** "102/211 par" / "0 fnt" -> display form; % for a foe mon. */
  private fmtHp(token: string, monSide: number): string {
    const sp = token.indexOf(" ");
    const hp = sp >= 0 ? token.slice(0, sp) : token;
    const status = sp >= 0 ? token.slice(sp + 1) : "";
    let core = hp;
    if (monSide !== this.viewer) {
      const slash = hp.indexOf("/");
      if (slash >= 0) {
        const n = Number(hp.slice(0, slash));
        const d = Number(hp.slice(slash + 1));
        if (Number.isFinite(n) && Number.isFinite(d) && d > 0) {
          core = `${Math.max(1, Math.round((n / d) * 100))}%`;
        }
      }
    }
    return status ? `${core} ${statusName(status)}` : core;
  }

  /** Translate a [from]/effect payload: strip the "move:"/"item:"
   * namespace, then try statuses, items, conditions/moves — raw last. */
  private effectName(e: string): string {
    const s = stripEffect(e);
    if (STATUS_CODES.has(s)) return statusName(s);
    const asItem = itemName(s);
    if (asItem !== s) return asItem;
    return condName(s);
  }

  private renderLine(line: string): LogEntry | null {
    const T = this.t();
    const parts = line.split("|");
    if (parts.length < 2 || parts[1] === "") return null;
    const tag = parts[1];
    const arg = (i: number) => parts[i] ?? "";
    const fromPart = parts.find((p) => p.startsWith("[from]"));
    const from = fromPart
      ? `  (${this.effectName(fromPart.replace(/^\[from\] /, ""))})`
      : "";
    const who = (r: string) => this.who(r);

    let kind: LogEntry["kind"] = MAJOR_TAGS.has(tag) ? "major" : "minor";
    let text: string;
    switch (tag) {
      case "turn":
        return { kind: "turn", text: T.turn(arg(2)) };
      case "move":
        text = T.used(who(arg(2)), moveName(arg(3)));
        break;
      case "switch":
      case "drag": {
        const [side] = this.parseRef(arg(2));
        const hp = this.fmtHp(arg(4), side);
        text =
          tag === "drag"
            ? T.dragged(who(arg(2)), hp)
            : T.sentOut(who(arg(2)), hp);
        break;
      }
      case "-damage":
      case "-heal":
      case "-sethp": {
        const [side] = this.parseRef(arg(2));
        text = T.hpLine(who(arg(2)), this.fmtHp(arg(3), side), from);
        break;
      }
      case "faint":
        text = T.fainted(who(arg(2)));
        break;
      case "-status":
        text = STATUS_CODES.has(arg(3))
          ? T.status(who(arg(2)), arg(3), from)
          : T.statusOther(who(arg(2)), arg(3));
        break;
      case "-curestatus":
        text = T.cured(who(arg(2)), statusName(arg(3)));
        break;
      case "-boost":
      case "-unboost": {
        const n = Number(arg(4)) || 1;
        text = T.boost(
          who(arg(2)),
          statLongName(arg(3)),
          tag === "-boost",
          n >= 2,
        );
        break;
      }
      case "cant":
        text = CANT_REASONS.has(arg(3))
          ? T.cant(who(arg(2)), arg(3))
          : T.cantOther(who(arg(2)), this.effectName(arg(3)));
        break;
      case "-crit":
        text = T.crit();
        break;
      case "-supereffective":
        text = T.superEffective();
        break;
      case "-resisted":
        text = T.resisted();
        break;
      case "-immune":
        text = T.immune(who(arg(2)));
        break;
      case "-miss":
        text = T.missed(who(arg(2)));
        break;
      case "-fail":
        text = T.failed();
        break;
      case "-nothing":
        text = T.nothing();
        break;
      case "-hitcount":
        text = T.hitCount(arg(3));
        break;
      case "-prepare":
        text = T.preparing(who(arg(2)), moveName(arg(3)));
        break;
      case "-mustrecharge":
        text = T.mustRecharge(who(arg(2)));
        break;
      case "-start": {
        const id = toId(stripEffect(arg(3)));
        text = T.condStart(who(arg(2)), id, this.effectName(arg(3)), from);
        break;
      }
      case "-end": {
        const id = toId(stripEffect(arg(3)));
        text = T.condEnd(who(arg(2)), id, this.effectName(arg(3)), from);
        break;
      }
      case "-activate":
      case "-singlemove":
      case "-singleturn":
        text = T.activate(who(arg(2)), this.effectName(arg(3)));
        break;
      case "-fieldactivate": {
        const id = toId(stripEffect(arg(2)));
        text = T.fieldActivate(id, this.effectName(arg(2)));
        break;
      }
      case "-transform":
        text = T.transformed(who(arg(2)), who(arg(3)));
        break;
      case "-copyboost":
        text = T.copiedBoost(who(arg(2)), who(arg(3)));
        break;
      case "-clearallboost":
        text = T.clearAllBoost();
        break;
      case "-weather": {
        if (parts.some((p) => p === "[upkeep]")) return null;
        const id = toId(arg(2));
        text = T.weather(id === "" ? "none" : id, condName(arg(2)));
        break;
      }
      case "-sidestart":
      case "-sideend": {
        const [side] = this.parseRef(arg(2));
        const owner = T.side(side === this.viewer);
        const cond = this.effectName(arg(3));
        text =
          tag === "-sidestart"
            ? T.sideStart(owner, cond)
            : T.sideEnd(owner, cond);
        break;
      }
      case "-item":
        text = T.holds(who(arg(2)), itemName(arg(3)), from);
        break;
      case "-enditem":
        text = T.endItem(
          who(arg(2)),
          itemName(arg(3)),
          parts.some((p) => p === "[eat]"),
          from,
        );
        break;
      case "-message":
        text = arg(2);
        break;
      case "-hint":
        text = `(${arg(2)})`;
        break;
      case "win": {
        const winnerSide = arg(2) === "P2" ? 1 : 0;
        return {
          kind: "result",
          text: T.win(winnerSide === this.viewer, arg(2)),
        };
      }
      case "tie":
        return { kind: "result", text: T.tie() };
      // init/noise lines
      case "player":
      case "teamsize":
      case "gen":
      case "gametype":
      case "tier":
      case "rule":
      case "start":
      case "clearpoke":
      case "poke":
      case "teampreview":
      case "upkeep":
      case "t:":
      case "rated":
      case "-anim":
        return null;
      default:
        text = `. ${line}`; // unknown: keep visible, semi-raw
    }
    return { kind, text };
  }
}

function stripEffect(e: string): string {
  if (e.startsWith("move: ")) return e.slice(6);
  if (e.startsWith("item: ")) return e.slice(6);
  return e;
}
