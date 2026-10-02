export interface EndgameSettings {
  solverFraction: number;
  milliseconds: number;
  maxNodes?: number;
}

export function endgameSettings(search = globalThis.location?.search ?? ""): EndgameSettings | undefined {
  const params = new URLSearchParams(search);
  const option = params.get("endgame");
  if (option === null) return undefined;
  if (!["0", "5", "10", "20"].includes(option)) throw new Error("endgame must be 0, 5, 10, or 20");
  const milliseconds = Number(params.get("endgame-ms") ?? 2300);
  if (!Number.isFinite(milliseconds) || milliseconds < 1 || milliseconds > 6000)
    throw new Error("endgame-ms must be between 1 and 6000");
  const maxNodes = Number(params.get("endgame-nodes") ?? 12000);
  if (!Number.isSafeInteger(maxNodes) || maxNodes < 1 || maxNodes > 12000)
    throw new Error("endgame-nodes must be between 1 and 12000");
  return { solverFraction: Number(option) / 100, milliseconds, maxNodes };
}
