// UI-4 live announcer: two off-screen aria-live regions updated
// imperatively — never VDOM-diffed, so Preact re-renders can neither
// re-announce old text nor clobber a pending announcement. One polite
// region carries batched battle narration and decision prompts; one
// assertive region carries the outcome banner.
//
// Repeat-safety: screen readers only announce on *change*, so an identical
// consecutive message (e.g. "Your turn" twice) must still register — each
// write clears the region first and sets the text on a short timer.

let polite: HTMLElement | null = null;
let assertive: HTMLElement | null = null;

function mk(mode: "polite" | "assertive"): HTMLElement {
  const el = document.createElement("div");
  el.id = `sr-live-${mode}`;
  el.className = "sr-announcer";
  el.setAttribute("aria-live", mode);
  el.setAttribute("aria-atomic", "true");
  document.body.appendChild(el);
  return el;
}

/** Create the live regions (idempotent). Runs at app startup: screen
 * readers only track regions that existed before the update. */
export function initAnnouncer(): void {
  if (typeof document === "undefined" || polite) return;
  polite = mk("polite");
  assertive = mk("assertive");
}

function write(el: HTMLElement | null, text: string): void {
  if (!el || !text) return;
  el.textContent = "";
  const target = el;
  window.setTimeout(() => {
    target.textContent = text;
  }, 30);
}

/** Queue a polite announcement (read when the screen reader is idle). */
export function announce(text: string): void {
  write(polite, text);
}

/** Interrupting announcement — outcome banner only. */
export function announceAssertive(text: string): void {
  write(assertive, text);
}
