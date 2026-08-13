// Information mode — how much of the opponent the two sides may see — and
// the door the page load came in by.
//
//   "open"  : OPEN TEAM SHEET (the shipped default) — both sides' sets are
//             public, only selection is hidden. The bot pins the human's
//             exact team, so its belief is degenerate.
//   "blind" : both sides see only the opponent's six species / levels /
//             types and the public battle log. Sets stay private on both
//             sides; the bot falls back to pool imputation (+ optional
//             belief prior).
//
// The URL is the only door, and there are three of them:
//
//   /        open — the M12 product screen, unchanged and un-nudged.
//   ?blind   the blind experiment, with its setup modal (pool + prior).
//   ?nash    META-NASH v1's conclusion: blind rules, but the opponent is
//            drawn from the solved three-team mixture and NOTHING is
//            configurable — no pool swap, no belief prior. See nash-mix.ts.
//
// Blind and nash are experiments riding along in a public build, so the
// start screen shows no switch for either — a visitor without the link
// gets the M12 screen. Nothing is persisted either: a stored preference
// would outlive the link that set it, silently leaving a first-time visitor
// in an experiment with no UI to get back out of.
//
// Information mode is a FUNCTION of the door rather than a second thing to
// read, because "nash implies blind" has to hold at every call site: a
// reader that consulted `?blind` alone would answer "open" for a nash URL
// and hand the human the bot's sets. One reader, one derivation.
//
// A Game freezes the mode at start, so this is read on the start screen
// only and a running battle's information structure cannot move under it.

export type InfoMode = "open" | "blind";

/** Which of the three URL doors this page load came in by. */
export type Door = "open" | "blind" | "nash";

/**
 * `?nash`, `?nash=1`, `?nash=yes` → nash; otherwise `?blind`, `?blind=1`,
 * `?blind=yes` → blind; neither → open.
 *
 * `?nash=0` / `?blind=0` (and `=false`) read as the door being shut: a link
 * that has been passed around is easier to neutralize by editing one
 * character than by surgery on the query string.
 *
 * Nash is tested first so `?blind&nash` is nash rather than an argument
 * about ordering. It is the stricter of the two — a nash page has no setup
 * controls at all — so resolving the overlap towards it can only ever
 * subtract configuration, never hand a visitor something the plainer door
 * would have withheld.
 *
 * `search` defaults to `location.search` and exists as a parameter so the
 * rule can be exercised without a document.
 */
export function readDoor(search?: string): Door {
  const query =
    search ?? (typeof location === "undefined" ? "" : location.search);
  const params = new URLSearchParams(query);
  if (isOpen(params.get("nash"))) return "nash";
  if (isOpen(params.get("blind"))) return "blind";
  return "open";
}

/** The information policy a door implies. Nash is blind play with a fixed
 * opponent distribution, so it maps to "blind" — every downstream reader of
 * `InfoMode` (game.tsx, the pickers, the team sheets) needs no notion of
 * nash at all, and cannot forget to handle it. */
export function infoModeOf(door: Door): InfoMode {
  return door === "open" ? "open" : "blind";
}

/** This page load's information mode. Kept as the one-call form for callers
 * that only ever needed the policy. */
export function readInfoMode(search?: string): InfoMode {
  return infoModeOf(readDoor(search));
}

/** A door parameter is open when present and not explicitly negated. */
function isOpen(v: string | null): boolean {
  if (v === null) return false;
  const t = v.trim().toLowerCase();
  return t !== "0" && t !== "false";
}
