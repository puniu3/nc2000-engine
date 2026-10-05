import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { setLocale } from "../src/i18n";
import { ToolError } from "../src/tool-errors";
import { toolText } from "../src/tool-strings";

const base = process.env.NC2000_E2E_BASE ?? "/";
const player = [{ species: "Mewtwo", level: 100, moves: ["Psychic"] }];
const opponent = [{ species: "Magikarp", level: 1, moves: ["Splash"] }];
const language = (page: Page) => page.locator(".tool-language-bar select");

async function englishEvaluation(page: Page) {
  await page.goto(`${base}?evaluate`);
  await expect(page.getByRole("button", { name: "Add Pokémon" })).toBeEnabled();
  await page.getByText("Import from text", { exact: true }).click();
  await page.getByRole("textbox", { name: "Your team", exact: true }).fill(JSON.stringify(player));
  await page.getByLabel("Opponent settings file", { exact: true }).setInputFiles({
    name: "opponents.json", mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({ teams: [{ id: "fish", weight: 1, sets: opponent }] })),
  });
}

for (const route of ["", "?blind", "?nash", "?solver", "?evaluate", "?fork", "?fork&advanced", "?kifu-preview"]) {
  test(`English entry ${route || "/"}`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", e => errors.push(String(e)));
    await page.goto(base + route);
    await expect(page.locator("html")).toHaveAttribute("lang", "en");
    await expect(page.getByRole("heading").first()).toBeVisible();
    if (["?evaluate", "?fork", "?fork&advanced", "?kifu-preview"].includes(route)) {
      await expect(language(page)).toHaveValue("en");
      expect(await page.locator("main").innerText()).not.toMatch(/[ぁ-んァ-ヶ一-龠]/);
    }
    expect(errors).toEqual([]);
  });
}

test("saved locale overrides browser language across tools and reloads", async ({ browser }) => {
  const context = await browser.newContext({ locale: "ja-JP" });
  const page = await context.newPage();
  await page.goto(`${base}?evaluate`);
  await expect(language(page)).toHaveValue("ja");
  await language(page).selectOption("en");
  await expect(page).toHaveTitle("NC2000 — Team evaluation");
  await page.reload();
  await expect(language(page)).toHaveValue("en");
  await page.goto(`${base}?fork`);
  await expect(language(page)).toHaveValue("en");
  await expect(page).toHaveTitle("NC2000 — Try another move");
  await language(page).selectOption("ja");
  await page.goto(base);
  await expect(page.locator(".lang-select")).toHaveValue("ja");
  await context.close();
});

test("unsupported browser language and invalid saved locale fall back to English", async ({ browser }) => {
  const context = await browser.newContext({ locale: "fr-FR" });
  await context.addInitScript(() => localStorage.setItem("nc2000-locale", "invalid"));
  const page = await context.newPage();
  await page.goto(`${base}?evaluate`);
  await expect(language(page)).toHaveValue("en");
  await context.close();
});

test("blocked storage still uses browser language and allows manual switching", async ({ browser }) => {
  const context = await browser.newContext({ locale: "ja-JP" });
  await context.addInitScript(() => {
    Object.defineProperty(window, "localStorage", { get() { throw new Error("blocked"); } });
  });
  const page = await context.newPage();
  await page.goto(`${base}?evaluate`);
  await expect(language(page)).toHaveValue("ja");
  await language(page).selectOption("en");
  await expect(page.getByRole("heading", { name: "Evaluate your team's strength" })).toBeVisible();
  await context.close();
});

test("editor names, validation, and prior notices follow the selected language", async ({ page }) => {
  await englishEvaluation(page);
  const text = page.getByRole("textbox", { name: "Your team", exact: true });
  await text.fill('[{"species":"Snorlax","level":101}]');
  await expect(page.getByRole("alert")).toContainText("level must be an integer");
  await language(page).selectOption("ja");
  await expect(page.getByRole("alert")).toContainText("レベルは1〜100");
  await expect(page.getByRole("combobox", { name: "自分のパーティ1匹目のポケモン", exact: true })).toHaveValue("カビゴン");
  await language(page).selectOption("en");
  await expect(page.getByRole("combobox", { name: "Your team, Pokémon 1 species", exact: true })).toHaveValue("Snorlax");
  await page.getByLabel("Opponent settings file", { exact: true }).setInputFiles({
    name: "bad.json", mimeType: "application/json", buffer: Buffer.from('{"teams":[]}'),
  });
  await expect(page.getByRole("alert").first()).toContainText("Specify at least one opponent");
  await language(page).selectOption("ja");
  await expect(page.getByRole("alert").first()).toContainText("対戦相手を1件以上");
});

test("switching during evaluation preserves results, resume, and exported identities", async ({ page }) => {
  await englishEvaluation(page);
  await page.getByLabel("Number of games", { exact: true }).fill("20");
  await page.getByRole("button", { name: "Start evaluation", exact: true }).click();
  await expect(page.locator(".eval-progress")).not.toBeEmpty();
  await language(page).selectOption("ja");
  await expect(page.getByRole("button", { name: "停止", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "停止", exact: true }).click();
  await expect(page.locator(".eval-progress")).toContainText("停止しました");
  await language(page).selectOption("en");
  await expect(page.locator(".eval-progress")).toContainText("Stopped.");
  await expect(page.getByRole("button", { name: "Resume", exact: true })).toBeEnabled();
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Save results to a file", exact: true }).click();
  const run = JSON.parse(readFileSync((await (await download).path())!, "utf8"));
  expect(run.config.player.sets[0].species).toBe("Mewtwo");
  expect(run.config.opponents[0].id).toBe("fish");
  expect(run.version).toBe(1);
  await page.getByRole("button", { name: "Resume", exact: true }).click();
  await expect(page.locator(".eval-progress")).toContainText("All requested games have finished", { timeout: 120_000 });
  expect(await page.locator("main").innerText()).not.toMatch(/[ぁ-んァ-ヶ一-龠]/);
});

test("replay errors retranslate while preserving pasted input", async ({ page }) => {
  await page.goto(`${base}?fork`);
  await page.getByLabel("Paste a battle record", { exact: true }).fill("NC2-invalid");
  await page.getByRole("button", { name: "Load record", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("Could not read the complete replay");
  await language(page).selectOption("ja");
  await expect(page.getByRole("status")).toContainText("棋譜を最後まで");
  await expect(page.getByLabel("対戦記録を貼り付ける", { exact: true })).toHaveValue("NC2-invalid");
});

test("known worker errors retain their keys and translate on the receiving side", () => {
  const issue = JSON.parse(JSON.stringify(new ToolError("noLegalAction", 2).issue));
  setLocale("en", false);
  expect(toolText(issue.key, ...issue.args)).toBe("P2: no legal action is available.");
  setLocale("ja", false);
  expect(toolText(issue.key, ...issue.args)).toBe("P2: 選択できる手がありません。");
  setLocale("en", false);
});

for (const loc of ["en-US", "ja-JP"]) {
  test(`tool pages fit a narrow viewport in ${loc}`, async ({ browser }) => {
    const context = await browser.newContext({ locale: loc, viewport: { width: 390, height: 844 } });
    const page = await context.newPage();
    for (const query of ["?evaluate", "?fork", "?fork&advanced", "?fork=4296-t11"]) {
      await page.goto(base + query);
      await expect(page.getByRole("heading").first()).toBeVisible();
      if (query === "?fork=4296-t11") await expect(page.getByRole("heading", { name: "4296 T11" })).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), query).toBe(true);
    }
    await context.close();
  });
}

test("fork language changes preserve the active game and update move names and narration", async ({ page }) => {
  const workers: string[] = [];
  page.on("worker", worker => workers.push(worker.url()));
  await page.goto(`${base}?fork=4296-t11`);
  await expect(page.getByRole("heading", { name: "4296 T11" })).toBeVisible();
  await expect(page.locator(".fork-arms")).toContainText("Ampharos");
  await language(page).selectOption("ja");
  await expect(page.locator(".fork-arms")).toContainText("デンリュウ");
  await page.getByRole("button", { name: "対局を始める", exact: true }).click();
  await expect(page.locator(".move-btn, .switch-btn").first()).toBeVisible();
  const count = workers.length;
  await language(page).selectOption("en");
  await expect(page.getByRole("button", { name: "Forfeit", exact: true })).toBeVisible();
  await expect(page.locator(".turn-label")).toContainText("Turn 11");
  expect(await page.locator(".fork-battle").innerText()).not.toMatch(/[ぁ-んァ-ヶ一-龠]/);
  expect(workers).toHaveLength(count);
});
