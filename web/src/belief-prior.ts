// M18: the community belief prior, as the browser keeps it — the raw table
// TEXT plus the file name it came from.
//
// The text is stored verbatim and handed to wasm unparsed: the searcher
// (`BlindSearcher.setBeliefPrior`) is the only thing that decides what a
// table means, and re-encoding it here would let this layer silently
// disagree with the engine's parse. So this module is deliberately dumb —
// it names the bytes and stores them.
//
// Policy (crates/bot/src/prior.rs:491): a prior is NEVER auto-loaded. It
// exists only because the user explicitly picked a file (or the sample),
// which is exactly what a localStorage record is: an earlier explicit act
// that the user can clear. Nothing here fetches anything.
//
// Same house style as custom-teams.ts: try/catch everywhere, a corrupt
// record reads back as "no prior", write failures degrade silently or —
// where the caller can act on it — as a returned reason string.

export interface StoredPrior {
  /** Display name; the file's name, or the sample table's file name. */
  name: string;
  /** The table's JSON source text, verbatim (parsed only by wasm). */
  json: string;
}

const LS_KEY = "nc2000-belief-prior";

/** Refuse tables that cannot plausibly fit localStorage (typically a ~5 MB
 * per-origin budget shared with saved teams and picks). Checked before the
 * write so the failure is a clean message instead of a QuotaExceededError
 * that also nukes the previous record. */
export const PRIOR_MAX_BYTES = 2_000_000;

export function loadStoredPrior(): StoredPrior | null {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw) as Partial<StoredPrior>;
    if (!p || typeof p.name !== "string" || typeof p.json !== "string")
      return null;
    return { name: p.name, json: p.json };
  } catch {
    return null;
  }
}

/**
 * Persist a picked table. Returns `null` on success, or a short reason
 * string on failure — the caller surfaces it via `ui().priorLoadFailed`
 * and does NOT adopt the table (a prior the browser cannot keep would
 * silently vanish on reload, which is worse than refusing it).
 *
 * Reasons are terse English; they name a storage fault, not a table
 * defect (table defects come from the wasm probe's warnings).
 */
export function storePrior(name: string, json: string): string | null {
  if (!json.trim()) return "empty file";
  const bytes = byteLength(json);
  if (bytes > PRIOR_MAX_BYTES)
    return `table too large (${bytes} bytes, limit ${PRIOR_MAX_BYTES})`;
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({ name, json }));
    return null;
  } catch (e) {
    return `could not be saved (${String(e)})`;
  }
}

export function clearStoredPrior(): void {
  try {
    localStorage.removeItem(LS_KEY);
  } catch {
    /* storage unavailable: nothing was persisted to begin with */
  }
}

/** UTF-8 size; TextEncoder is universal in the browsers this build targets,
 * but the fallback keeps the cap meaningful in any stripped-down runtime
 * (UTF-16 length under-counts multi-byte text, never over-counts it). */
function byteLength(s: string): number {
  try {
    return new TextEncoder().encode(s).length;
  } catch {
    return s.length;
  }
}
