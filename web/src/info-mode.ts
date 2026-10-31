/** The information policy a record was made under. Live play is always
 * "blind"; "open" names the retired open-sheet mode and survives only so
 * records made under it (forks, kifu) replay with their own semantics. */
export type InfoMode = "open" | "blind";
/** `play` is `/`; `?blind` is kept as an alias of it. */
export type Door = "play" | "nash" | "solver" | "evaluate" | "fork";
export type NashVariant = "new" | "v3";

export function readDoor(search?: string): Door {
  const query = search ?? (typeof location === "undefined" ? "" : location.search);
  const params = new URLSearchParams(query);
  for (const door of ["fork", "evaluate", "solver", "nash"] as const) {
    if (isOpen(params.get(door))) return door;
  }
  return "play";
}

/** `?nash=v3` or `?nash=old` selects the preserved v3 mixture; bare `?nash`
 * (or any other open value) selects the new Full-Combination Preview Nash. */
export function readNashVariant(search?: string): NashVariant {
  const query = search ?? (typeof location === "undefined" ? "" : location.search);
  const params = new URLSearchParams(query);
  const v = (params.get("nash") ?? "").trim().toLowerCase();
  return v === "v3" || v === "old" ? "v3" : "new";
}

function isOpen(v: string | null): boolean {
  if (v === null) return false;
  const t = v.trim().toLowerCase();
  return t !== "0" && t !== "false";
}
