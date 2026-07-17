// UI-3 shared toggletip layer. Anchors are ordinary elements carrying
// class "has-note" plus a visually-hidden child span.bn-note with the note
// text; `noteRef` (a ref callback for that span) assigns it a unique id and
// wires aria-describedby on the anchor, so screen readers get the note with
// zero JS and the visible bubble is pure presentation (aria-hidden).
//
// Reveal paths:
//   - hover (only on hover-capable fine pointers) and keyboard focus;
//   - touch: long-press ~500ms. A quick tap is untouched — the timer just
//     never fires — and when the long-press DOES fire, the one click that
//     the release would deliver is swallowed (capture phase) so holding a
//     battle move button never chooses the move.
// Dismissal: release after sliding off the anchor, scroll anywhere, tap
// away, Escape, focus loss, viewport resize.
//
// The bubble is appended to the anchor's nearest <dialog> when it sits in
// the top layer (a body-level bubble would be dimmed behind ::backdrop),
// else to document.body, and positioned `fixed`, clamped to the viewport.

let seq = 0;

/** Ref callback for the hidden span.bn-note inside a noted anchor. */
export function noteRef(el: HTMLElement | null): void {
  if (!el || el.id) return;
  el.id = `bn-${++seq}`;
  const anchor = el.closest(".has-note");
  if (anchor) anchor.setAttribute("aria-describedby", el.id);
}

const LONG_PRESS_MS = 500;
const SLOP_PX = 8;
const GAP_PX = 8;
const EDGE_PX = 6;

type Mode = "hover" | "focus" | "press";

let bubble: HTMLDivElement | null = null;
let activeAnchor: HTMLElement | null = null;
let activeMode: Mode | null = null;

let pressTimer: number | null = null;
let pressAnchor: HTMLElement | null = null;
let pressX = 0;
let pressY = 0;
let suppressNextClick = false;

function noteTextOf(anchor: HTMLElement): string {
  return anchor.querySelector(".bn-note")?.textContent ?? "";
}

function hide(): void {
  bubble?.remove();
  bubble = null;
  activeAnchor = null;
  activeMode = null;
}

function cancelPress(): void {
  if (pressTimer !== null) {
    clearTimeout(pressTimer);
    pressTimer = null;
  }
  pressAnchor = null;
}

function show(anchor: HTMLElement, mode: Mode): void {
  const text = noteTextOf(anchor);
  if (!text) return;
  hide();
  const tip = document.createElement("div");
  tip.className = "bn-tip";
  tip.textContent = text;
  tip.setAttribute("aria-hidden", "true");
  // Anchors inside an open <dialog> live in the top layer; the bubble must
  // join them there or it renders dimmed behind the ::backdrop.
  (anchor.closest("dialog") ?? document.body).appendChild(tip);
  const r = anchor.getBoundingClientRect();
  const vw = window.innerWidth;
  const bw = tip.offsetWidth;
  const bh = tip.offsetHeight;
  let top = r.top - bh - GAP_PX;
  if (top < EDGE_PX) top = Math.min(r.bottom + GAP_PX, window.innerHeight - bh - EDGE_PX);
  // Center over the anchor — but a sheet row spans the panel, so cap the
  // reference span to keep the bubble near the (left-aligned) name.
  const refW = Math.min(r.width, 280);
  const left = Math.min(
    Math.max(r.left + refW / 2 - bw / 2, EDGE_PX),
    Math.max(vw - bw - EDGE_PX, EDGE_PX),
  );
  tip.style.top = `${Math.max(top, EDGE_PX)}px`;
  tip.style.left = `${left}px`;
  bubble = tip;
  activeAnchor = anchor;
  activeMode = mode;
}

function anchorOf(t: EventTarget | null): HTMLElement | null {
  return t instanceof Element ? t.closest<HTMLElement>(".has-note") : null;
}

let installed = false;

/** Install the document-level listeners (idempotent). */
export function initTooltips(): void {
  if (installed || typeof document === "undefined") return;
  installed = true;
  const hoverable = window.matchMedia("(hover: hover) and (pointer: fine)");

  document.addEventListener("pointerover", (e) => {
    if (!hoverable.matches || e.pointerType !== "mouse") return;
    const a = anchorOf(e.target);
    if (a && a !== activeAnchor) show(a, "hover");
  });
  document.addEventListener("pointerout", (e) => {
    if (activeMode !== "hover" || !activeAnchor) return;
    const to = e.relatedTarget;
    if (!(to instanceof Element) || !activeAnchor.contains(to)) hide();
  });

  document.addEventListener("focusin", (e) => {
    const a = anchorOf(e.target);
    if (a) {
      // Keyboard-driven focus only: a tap that happens to focus a button
      // (Android Chrome) must not pop the note — hover/long-press own the
      // pointer paths. :focus-visible is exactly that distinction.
      let keyboard = true;
      try {
        keyboard = a.matches(":focus-visible") || !!a.querySelector(":focus-visible");
      } catch {
        /* engine without :focus-visible: fall back to always showing */
      }
      if (keyboard) show(a, "focus");
    } else if (activeMode === "focus") hide();
  });
  document.addEventListener("focusout", () => {
    if (activeMode === "focus") hide();
  });

  document.addEventListener("pointerdown", (e) => {
    const a = anchorOf(e.target);
    // Tap-away (and tapping the anchor again) dismisses an open toggletip.
    if (activeAnchor && activeMode !== "hover") hide();
    else if (activeAnchor && e.pointerType !== "mouse") hide();
    if (e.pointerType === "mouse" || !a) return;
    pressAnchor = a;
    pressX = e.clientX;
    pressY = e.clientY;
    pressTimer = window.setTimeout(() => {
      pressTimer = null;
      const held = pressAnchor;
      pressAnchor = null;
      if (held) {
        show(held, "press");
        // The finger is still down: swallow the click this release
        // produces so a long-press never activates the anchor.
        suppressNextClick = true;
      }
    }, LONG_PRESS_MS);
  });
  document.addEventListener("pointermove", (e) => {
    if (pressTimer === null) return;
    if (Math.hypot(e.clientX - pressX, e.clientY - pressY) > SLOP_PX)
      cancelPress();
  });
  document.addEventListener("pointerup", (e) => {
    cancelPress();
    if (activeMode === "press" && activeAnchor) {
      // Touch pointerup targets lie under implicit pointer capture (they
      // report the pointerdown element), so judge release-outside by
      // coordinates: released off the anchor = dismiss, on it = the
      // toggletip stays.
      const r = activeAnchor.getBoundingClientRect();
      const inside =
        e.clientX >= r.left && e.clientX <= r.right &&
        e.clientY >= r.top && e.clientY <= r.bottom;
      if (!inside) hide();
      // Whether or not a click follows this release (it may land on a
      // common ancestor), the suppression flag must not leak into the
      // next tap.
      if (suppressNextClick) {
        window.setTimeout(() => {
          suppressNextClick = false;
        }, 100);
      }
    }
  });
  document.addEventListener("pointercancel", () => {
    cancelPress();
    if (activeMode === "press") hide();
    suppressNextClick = false;
  });

  document.addEventListener(
    "click",
    (e) => {
      if (!suppressNextClick) return;
      suppressNextClick = false;
      e.preventDefault();
      e.stopPropagation();
    },
    true,
  );

  // Long-press on some platforms raises a context menu; while a press
  // gesture is being tracked (or just fired), keep it quiet.
  document.addEventListener("contextmenu", (e) => {
    if (pressTimer !== null || activeMode === "press") e.preventDefault();
  });

  document.addEventListener(
    "scroll",
    () => {
      cancelPress();
      if (activeAnchor) hide();
    },
    { capture: true, passive: true },
  );
  window.addEventListener("resize", hide);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") hide();
  });
}
