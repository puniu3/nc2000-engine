import { expect, test } from "@playwright/test";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const wasm = require("../../crates/wasm/pkg-node/nc2000_wasm.js");
const fixture = JSON.parse(readFileSync(new URL("../../fixtures/corpus-v1/full/battle-001.json", import.meta.url), "utf8"));
const dex = new wasm.Dex();
const battle = new wasm.Battle(dex, JSON.stringify(fixture.p1team), JSON.stringify(fixture.p2team), fixture.seed);
for (let i = 0; i < 100 && battle.outcome() === undefined; i++) {
  const needs = JSON.parse(battle.needsChoice());
  const picks = needs.map((n: boolean, s: number) => n ? JSON.parse(battle.legalChoices(s))[0].input : null);
  for (const s of [0, 1]) if (picks[s]) battle.applyChoice(s, picks[s]);
}
const code: string = battle.exportKifu(false, 1);
const kifu = new wasm.Kifu(dex, code);
const scenes = JSON.parse(kifu.scenes()).scenes;
const compareIndex = scenes.findLastIndex((s: {round: number}) => JSON.parse(kifu.scene(s.round)).choices.length > 1);
kifu.free(); battle.free(); dex.free();

test.use({ locale: "ja-JP", viewport: { width: 390, height: 844 }, launchOptions: { args: ["--mute-audio"] } });

test("pasted HTML restores a scene, supports keyboard selection and resumes a game", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(String(e)));
  page.on("dialog", d => void d.accept());
  await page.goto("/?fork");
  await page.getByLabel("対戦記録を貼り付ける", { exact: true }).fill(`<html><p>対戦のログ</p><pre>${code}</pre></html>`);
  await page.getByRole("button", { name: "記録を読み込む", exact: true }).click();
  await expect(page.getByRole("heading", { name: "どこから試しますか？" })).toBeFocused();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByLabel("再開する場面").selectOption(String(compareIndex));
  const option = page.locator('input[name="alternative"]').last();
  await option.focus();
  await page.keyboard.press("Space");
  await expect(option).toBeChecked();
  await page.getByRole("button", { name: "この手で対戦を始める" }).click();
  await expect(page.locator(".fork-battle")).toBeVisible();
  await expect(page.locator(".move-btn, .switch-btn").first()).toBeVisible();
  await page.locator(".move-btn, .switch-btn").first().click();
  await expect(page.locator(".move-btn, .switch-btn, .end-banner").first()).toBeVisible();
  if (await page.getByRole("button", { name: "投了", exact: true }).isVisible())
    await page.getByRole("button", { name: "投了", exact: true }).click();
  else await page.getByRole("button", { name: "場面と手を選び直す" }).click();
  await expect(page.getByRole("heading", { name: "どこから試しますか？" })).toBeFocused();
  await expect(page.getByLabel("再開する場面")).toHaveValue(String(compareIndex));
  expect(errors).toEqual([]);
});

test("URL imports and a paired bot comparison run without technical controls", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", e => errors.push(String(e)));
  await page.addInitScript(() => Object.defineProperty(navigator, "hardwareConcurrency", { value: 2 }));
  await page.goto(`/?fork&kifu=${code}`);
  await expect(page.getByRole("heading", { name: "どこから試しますか？" })).toBeVisible();
  await page.getByLabel("再開する場面").selectOption(String(compareIndex));
  const original = await page.locator('input[name="alternative"]').first().getAttribute("value");
  await page.locator('input[name="alternative"]').last().check();
  await page.getByRole("radio", { name: /bot同士で比べる/ }).check();
  await expect(page.locator('input[name="alternative"]:checked')).not.toHaveValue(original!);
  await page.getByRole("button", { name: "この2つの手を比べる" }).click();
  await expect(page.getByRole("heading", { name: "元の手と別の手を比べる" })).toBeFocused();
  await page.getByRole("button", { name: "停止", exact: true }).click();
  await page.getByRole("button", { name: "結果を消去" }).click();
  await page.getByLabel("試行数", { exact: true }).fill("1");
  await page.getByRole("button", { name: "検証を開始" }).click();
  await expect(page.getByTestId("arena-progress")).toContainText("1 / 1 試行", { timeout: 120_000 });
  await expect(page.locator(".kp-arena-results .kp-card")).toHaveCount(2);
  await expect(page.getByLabel("思考量（1手あたり）")).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test("a normal bot game copies only code and its link imports the same record", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/");
  await page.getByRole("button", { name: /start battle|対戦開始/i }).click();
  for (let i = 0; i < 3; i++) await page.locator('.pick-head[aria-pressed="false"][aria-disabled="false"]').first().click();
  await page.getByRole("button", { name: /Confirm picks|選出を確定/ }).click();
  await expect(page.locator(".battle-screen")).toBeVisible();
  // A lead knocked out on turn 1 owes a forced switch inside that same turn,
  // so keep answering until the turn number moves.
  await expect(async () => {
    const choice = page.locator(".move-btn, .switch-btn").first();
    if (await choice.isVisible()) await choice.click();
    await expect(page.locator(".turn-label")).not.toHaveText(/(?:ターン|Turn) 1$/, { timeout: 2000 });
  }).toPass({ timeout: 60_000 });
  await page.getByRole("button", { name: "棋譜", exact: true }).click();
  await page.getByRole("button", { name: "棋譜をコピー", exact: true }).click();
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(copied).toMatch(/^NC2-[A-Za-z0-9_-]+$/);
  const href = await page.getByRole("link", { name: /別の手を試す/ }).getAttribute("href");
  expect(new URLSearchParams(new URL(href!).hash.slice(1)).get("kifu")).toBe(copied);
  await page.evaluate(() => {
    Object.defineProperty(navigator, "clipboard", { value: undefined });
    document.execCommand = () => false;
  });
  await page.getByRole("button", { name: "棋譜をコピー", exact: true }).click();
  const manual = page.getByLabel("コピーする棋譜", { exact: true });
  await expect(manual).toHaveValue(copied);
  await manual.focus();
  expect(await manual.evaluate((node: HTMLTextAreaElement) => node.selectionEnd - node.selectionStart)).toBe(copied.length);
  await page.goto(href!);
  await expect(page.getByRole("heading", { name: "どこから試しますか？" })).toBeVisible();
});

test("missing or damaged code keeps the input and offers a recoverable error", async ({ page }) => {
  await page.goto("/?fork");
  const input = page.getByLabel("対戦記録を貼り付ける", { exact: true });
  for (const text of ["日本語のログだけ", code.slice(0, -20)]) {
    await input.fill(text);
    await page.getByRole("button", { name: "記録を読み込む", exact: true }).click();
    await expect(page.locator("#kp-import-status")).not.toBeEmpty();
    await expect(input).toHaveValue(text);
    await expect(input).toBeFocused();
  }
});
