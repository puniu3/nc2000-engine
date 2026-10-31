// Where the bot's party comes from on the play door when no pool file is
// loaded: the catalog by its draw weights, the new solved Nash mixture, or
// the preserved v3 Nash mixture. A loaded pool file overrides all of them
// (app.tsx). Kept per browser like the other setup choices; anything
// unreadable falls back to the catalog.

export type DrawSource = "catalog" | "nash" | "nash-v3";

const KEY = "nc2000-bot-draw";

export function loadDrawSource(): DrawSource {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "nash" || v === "nash-v3") return v;
    return "catalog";
  } catch {
    return "catalog";
  }
}

export function storeDrawSource(source: DrawSource): void {
  try {
    if (source === "nash" || source === "nash-v3")
      localStorage.setItem(KEY, source);
    else localStorage.removeItem(KEY);
  } catch {
    /* storage unavailable: the choice still holds this session */
  }
}
