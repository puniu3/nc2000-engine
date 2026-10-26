export function extractKifu(text: string): string | null {
  const match = text.match(/NC2-[A-Za-z0-9_-]+/);
  return match?.[0] ?? null;
}

export function kifuUrl(code: string, opts?: { side?: number; round?: number; turn?: number }): string {
  const url = new URL(location.href);
  const params = new URLSearchParams();
  params.set("fork", "");
  if (opts?.side !== undefined) params.set("side", String(opts.side));
  if (opts?.round !== undefined) params.set("round", String(opts.round));
  if (opts?.turn !== undefined) params.set("turn", String(opts.turn));
  url.search = `?${params.toString().replace(/fork=&?/, "fork&").replace(/&$/, "")}`;
  url.hash = `kifu=${code}`;
  return url.href;
}

export async function copyKifu(code: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(code);
      return true;
    }
  } catch {}
  const field = document.createElement("textarea");
  field.value = code;
  field.style.cssText = "position:fixed;left:-10000px;top:0";
  const previous = document.activeElement as HTMLElement | null;
  document.body.append(field);
  try {
    field.select();
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    field.remove();
    previous?.focus({ preventScroll: true });
  }
}
