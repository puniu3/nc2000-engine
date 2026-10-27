import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

const live = JSON.parse(
  readFileSync(new URL("../../data/live-search.json", import.meta.url), "utf8"),
) as { defaultStrength: string; battle: { strong: number } };

test("play doors always use the fixed strong search budget without a strength selector", async ({ page }) => {
  expect(live.defaultStrength).toBe("strong");
  expect(live.battle.strong).toBe(81000);
  await page.addInitScript(() => localStorage.setItem("nc2000-locale", "en"));
  for (const url of ["/", "/?nash", "/?blind"]) {
    await page.goto(url);
    await expect(page.locator(".start-title")).toBeVisible();
    await expect(page.getByRole("combobox", { name: "Bot strength", exact: true })).toHaveCount(0);
    await expect(page.locator(".strength-choice")).toHaveCount(0);
  }
});
