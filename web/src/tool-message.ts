import { useState } from "preact/hooks";
import { ToolError } from "./tool-errors";
import { toolText } from "./tool-strings";

export type ToolMessage = string | (() => string);
export function errorText(error: unknown): string {
  return error instanceof ToolError ? toolText(error.issue.key, ...error.issue.args) : toolText("unexpectedError", String(error));
}

export function useToolMessage(initial: string | null = ""): [string, (message: ToolMessage | null) => void] {
  const [message, setMessage] = useState<ToolMessage>(initial ?? "");
  return [typeof message === "function" ? message() : message, next => setMessage(() => next ?? "")];
}
