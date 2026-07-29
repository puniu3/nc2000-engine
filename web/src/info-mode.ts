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
// The choice is a per-browser preference, not part of a battle: a Game
// captures the mode at start so toggling mid-battle cannot change the
// information structure of a running game. Storage failures degrade to the
// default — a missing preference must never block play.

export type InfoMode = "open" | "blind";

const LS_KEY = "nc2000-info-mode";

/** The default stays "open" (M12 product policy); blind is opt-in. */
export function loadInfoMode(): InfoMode {
  try {
    return localStorage.getItem(LS_KEY) === "blind" ? "blind" : "open";
  } catch {
    return "open";
  }
}

export function storeInfoMode(m: InfoMode): void {
  try {
    localStorage.setItem(LS_KEY, m);
  } catch {
    /* storage unavailable: the mode still holds this session */
  }
}
