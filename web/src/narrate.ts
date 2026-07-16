// Protocol-log narration — a direct port of the settled CLI renderer
// (crates/bot/examples/play.rs, LogView/render_line), viewer = the human
// side. Foe HP renders as %, own HP exact; |split|pN groups pick the
// secret (real-HP) line for the viewer's own side and the shared line for
// the foe.

import type { LogEntry } from "./types";

const MAJOR_TAGS = new Set([
  "move",
  "switch",
  "drag",
  "faint",
  "cant",
  "-prepare",
]);

export class Narrator {
  constructor(private viewer: number) {}

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
    return side === this.viewer ? name : `Foe ${name}`;
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
    return status ? `${core} ${status}` : core;
  }

  private renderLine(line: string): LogEntry | null {
    const parts = line.split("|");
    if (parts.length < 2 || parts[1] === "") return null;
    const tag = parts[1];
    const arg = (i: number) => parts[i] ?? "";
    const fromPart = parts.find((p) => p.startsWith("[from]"));
    const from = fromPart
      ? `  (${stripEffect(fromPart.replace(/^\[from\] /, ""))})`
      : "";
    const who = (r: string) => this.who(r);

    let kind: LogEntry["kind"] = MAJOR_TAGS.has(tag) ? "major" : "minor";
    let text: string;
    switch (tag) {
      case "turn":
        return { kind: "turn", text: `Turn ${arg(2)}` };
      case "move":
        text = `${who(arg(2))} used ${arg(3)}!`;
        break;
      case "switch":
      case "drag": {
        const [side] = this.parseRef(arg(2));
        const verb = tag === "drag" ? "was dragged out" : "was sent out";
        text = `${who(arg(2))} ${verb} (${this.fmtHp(arg(4), side)})`;
        break;
      }
      case "-damage":
      case "-heal":
      case "-sethp": {
        const [side] = this.parseRef(arg(2));
        text = `${who(arg(2))}: ${this.fmtHp(arg(3), side)}${from}`;
        break;
      }
      case "faint":
        text = `${who(arg(2))} fainted!`;
        break;
      case "-status": {
        const verbs: Record<string, string> = {
          brn: "was burned",
          par: "was paralyzed",
          slp: "fell asleep",
          frz: "was frozen solid",
          psn: "was poisoned",
          tox: "was badly poisoned",
        };
        const verb = verbs[arg(3)];
        text = verb
          ? `${who(arg(2))} ${verb}!${from}`
          : `${who(arg(2))} status: ${arg(3)}`;
        break;
      }
      case "-curestatus":
        text = `${who(arg(2))} was cured of ${arg(3)}!`;
        break;
      case "-boost":
      case "-unboost": {
        const stats: Record<string, string> = {
          atk: "Attack",
          def: "Defense",
          spa: "Sp. Atk",
          spd: "Sp. Def",
          spe: "Speed",
          accuracy: "accuracy",
          evasion: "evasion",
        };
        const stat = stats[arg(3)] ?? arg(3);
        const n = Number(arg(4)) || 1;
        const dir = tag === "-boost" ? "rose" : "fell";
        const adv = n >= 2 ? " sharply" : "";
        text = `${who(arg(2))}'s ${stat} ${dir}${adv}!`;
        break;
      }
      case "cant": {
        const whys: Record<string, string> = {
          slp: "is fast asleep",
          par: "is fully paralyzed",
          frz: "is frozen solid",
          flinch: "flinched",
          recharge: "must recharge",
        };
        const why =
          whys[arg(3)] ?? `can't move (${stripEffect(arg(3))})`;
        text = `${who(arg(2))} ${why}!`;
        break;
      }
      case "-crit":
        text = "A critical hit!";
        break;
      case "-supereffective":
        text = "It's super effective!";
        break;
      case "-resisted":
        text = "It's not very effective...";
        break;
      case "-immune":
        text = `It doesn't affect ${who(arg(2))}...`;
        break;
      case "-miss":
        text = `${who(arg(2))}'s attack missed!`;
        break;
      case "-fail":
        text = "But it failed!";
        break;
      case "-nothing":
        text = "But nothing happened!";
        break;
      case "-hitcount":
        text = `Hit ${arg(3)} time(s)!`;
        break;
      case "-prepare":
        text = `${who(arg(2))} is preparing ${arg(3)}...`;
        break;
      case "-mustrecharge":
        text = `${who(arg(2))} must recharge!`;
        break;
      case "-start":
        text = `${who(arg(2))}: ${stripEffect(arg(3))} started${from}`;
        break;
      case "-end":
        text = `${who(arg(2))}: ${stripEffect(arg(3))} ended${from}`;
        break;
      case "-activate":
      case "-singlemove":
      case "-singleturn":
        text = `${who(arg(2))}: ${stripEffect(arg(3))}`;
        break;
      case "-fieldactivate":
        text = `${stripEffect(arg(2))} affects everyone!`;
        break;
      case "-transform":
        text = `${who(arg(2))} transformed into ${who(arg(3))}!`;
        break;
      case "-copyboost":
        text = `${who(arg(2))} copied ${who(arg(3))}'s stat changes!`;
        break;
      case "-clearallboost":
        text = "All stat changes were eliminated!";
        break;
      case "-weather":
        if (parts.some((p) => p === "[upkeep]")) return null;
        text =
          arg(2) === "none" ? "The weather cleared." : `Weather: ${arg(2)}`;
        break;
      case "-sidestart":
      case "-sideend": {
        const [side] = this.parseRef(arg(2));
        const owner = side === this.viewer ? "Your side" : "Foe side";
        const what = tag === "-sidestart" ? "started" : "ended";
        text = `${owner}: ${stripEffect(arg(3))} ${what}`;
        break;
      }
      case "-item":
        text = `${who(arg(2))} holds ${arg(3)}${from}`;
        break;
      case "-enditem":
        text = `${who(arg(2))} used up its ${arg(3)}${from}`;
        break;
      case "-message":
        text = arg(2);
        break;
      case "-hint":
        text = `(${arg(2)})`;
        break;
      case "win":
        return { kind: "result", text: `${arg(2)} wins!` };
      case "tie":
        return { kind: "result", text: "Tie" };
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
  return e.startsWith("move: ") ? e.slice(6) : e;
}
