import { expect, test } from "@playwright/test";

test("strength persists across reloads and play doors", async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("nc2000-locale", "en"));
  await page.goto("/");
  const strength = page.getByRole("combobox", { name: "Bot strength", exact: true });
  await expect(strength).toHaveValue("strong");
  await strength.selectOption("standard");
  await page.reload();
  await expect(strength).toHaveValue("standard");
  await page.goto("/?nash");
  await expect(strength).toHaveValue("standard");
  await strength.selectOption("strong");
  await page.goto("/?blind");
  await expect(strength).toHaveValue("strong");
  await expect(page.locator(".strength-note")).toContainText("does not make it weaker");
  await page.evaluate(() => localStorage.setItem("nc2000-strength", "unknown-tier"));
  await page.reload();
  await expect(strength).toHaveValue("strong");
});
