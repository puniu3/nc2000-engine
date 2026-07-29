// Information mode — how much of the opponent the two sides may see.
//
//   "open"  : OPEN TEAM SHEET (the shipped default) — both sides' sets are
//             public, only selection is hidden. The bot pins the human's
//             exact team, so its belief is degenerate.
//   "blind" : both sides see only the opponent's six species / levels /
//             types and the public battle log. Sets stay private on both
//             sides; the bot falls back to pool imputation (+ optional
//             belief prior).
//
// The URL is the only door: `?blind` opens it and nothing else does. Blind
// is an experiment riding along in a public build, so the start screen
// shows no switch for it — a visitor without the link gets the M12 screen,
// unchanged and un-nudged. Nothing is persisted either: a stored preference
// would outlive the link that set it, silently leaving a first-time visitor
// in the experiment with no UI to get back out of.
//
// A Game freezes the mode at start, so this is read on the start screen
// only and a running battle's information structure cannot move under it.

export type InfoMode = "open" | "blind";

/**
 * `?blind`, `?blind=1`, `?blind=yes` → blind; no `blind` parameter → open.
 * `?blind=0` and `?blind=false` also read as open: a link that has been
 * passed around is easier to neutralize by editing one character than by
 * surgery on the query string.
 *
 * `search` defaults to `location.search` and exists as a parameter so the
 * rule can be exercised without a document.
 */
export function readInfoMode(search?: string): InfoMode {
  const query =
    search ?? (typeof location === "undefined" ? "" : location.search);
  const v = new URLSearchParams(query).get("blind");
  if (v === null) return "open";
  const t = v.trim().toLowerCase();
  return t === "0" || t === "false" ? "open" : "blind";
}
