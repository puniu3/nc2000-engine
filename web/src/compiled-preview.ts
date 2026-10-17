import artifactJson from "../../data/compiled-preview-v1.json?raw";

interface CompiledPreview {
  format: string;
  build: string;
  c: number;
  priorSha256: string;
  dexSha256: string;
  iterations: number;
  policies: Record<string, string[]>;
}
const artifact = JSON.parse(artifactJson) as CompiledPreview;
// Every catalog team occurs as an exact own team in the complete artifact.
// Recover its public gender/item-presence facts without inspecting a human's
// hidden set. Ambiguous public rosters conservatively use live search.
const publicRosters = new Map<string, string | null>();
for (const key of Object.keys(artifact.policies)) {
  const [own] = JSON.parse(key) as [unknown[][]];
  const roster = JSON.stringify(own.map(p => [p[0], p[1]]));
  const facts = JSON.stringify(own.map(p => [p[0], p[1], p[2], p[8] !== null]));
  publicRosters.set(roster, publicRosters.has(roster) && publicRosters.get(roster) !== facts ? null : facts);
}

export async function compiledPreviewMatches(poolJson: string, dexJson: string): Promise<boolean> {
  if (!__COMPILED_PREVIEW_ENABLED__) return false;
  if (artifact.format !== "nc2000-compiled-preview-v1") return false;
  const digest = async (text: string) => {
    const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return Array.from(new Uint8Array(hash), n => n.toString(16).padStart(2, "0")).join("");
  };
  const [prior, dex] = await Promise.all([digest(poolJson), digest(dexJson)]);
  return prior === artifact.priorSha256 && dex === artifact.dexSha256;
}

/** Both the key and build identity come from the engine. The key contains
 * only our own set and the opponent's public roster, never their hidden set.
 * Sampling the precomputed seeds preserves search-seed diversity. */
export function compiledPreviewPick(build: string, key: string | undefined, seed: number, c: number, cacheInfo?: string): string | undefined {
  if (!__COMPILED_PREVIEW_ENABLED__) return undefined;
  if (build !== artifact.build || c !== artifact.c || key === undefined || !cacheInfo) return undefined;
  const [, foe] = JSON.parse(key);
  const observed = JSON.parse(cacheInfo).opponentPublic as unknown[][];
  if (publicRosters.get(JSON.stringify(foe)) !== JSON.stringify(observed.map(p => p.slice(0, 4)))) return undefined;
  const choices = artifact.policies[key];
  return choices?.length ? choices[(seed >>> 0) % choices.length] : undefined;
}
