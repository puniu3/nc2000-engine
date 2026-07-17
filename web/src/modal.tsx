// Modal: a real <dialog> driven by mount/unmount — the parent owns the
// open state; mounting shows the dialog (showModal: native focus trap +
// top layer + ::backdrop), unmounting closes it and returns focus to the
// element that had focus at open time. Esc arrives as the native
// "cancel" event and is routed to onClose so the parent stays the single
// source of truth; backdrop taps close only when the press started AND
// ended on the backdrop (drag-selecting text out of a field must not
// dismiss). Body scroll is locked while open.

import type { ComponentChildren } from "preact";
import { useLayoutEffect, useRef } from "preact/hooks";
import { ui } from "./i18n";

let seq = 0;

export function Modal(props: {
  title: string;
  onClose: () => void;
  children: ComponentChildren;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useRef(`modal-title-${++seq}`);
  const downOnBackdrop = useRef(false);

  // Layout effect: the dialog must be open before first paint (the [open]
  // display rule keeps a closed dialog hidden, but no closed frame should
  // ever be shown).
  useLayoutEffect(() => {
    const dlg = ref.current!;
    const opener = document.activeElement;
    dlg.showModal();
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prevOverflow;
      if (dlg.open) dlg.close();
      if (opener instanceof HTMLElement) opener.focus();
    };
  }, []);

  return (
    <dialog
      class="modal"
      ref={ref}
      aria-labelledby={titleId.current}
      onCancel={(e) => {
        e.preventDefault(); // don't let the dialog close itself…
        props.onClose(); // …the parent unmounts us, which closes it
      }}
      onPointerDown={(e) => {
        downOnBackdrop.current = e.target === ref.current;
      }}
      onClick={(e) => {
        if (e.target === ref.current && downOnBackdrop.current) props.onClose();
      }}
    >
      <div class="modal-head">
        <h2 id={titleId.current} class="modal-title">
          {props.title}
        </h2>
        <button class="ghost modal-close" onClick={props.onClose}>
          {ui().close}
        </button>
      </div>
      <div class="modal-body">{props.children}</div>
    </dialog>
  );
}
