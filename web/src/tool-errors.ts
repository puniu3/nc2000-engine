import type { ToolTextKey } from "./tool-strings";

export interface ToolIssue {
  key: ToolTextKey;
  args: (string | number)[];
}

export class ToolError extends Error {
  readonly issue: ToolIssue;
  constructor(key: ToolTextKey, ...args: (string | number)[]) {
    super(key);
    this.issue = { key, args };
  }
}
