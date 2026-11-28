// Built-in catalog (`team-pool-v4`), belief-prior loading, evaluator link,
// and graceful retirement of `?nash`.
//
// The 10-shell (15-team) strong pool is the sole built-in pool on the play
// door; Nash mixture selection and `?nash` mode have been withdrawn. What
// this suite verifies:
//
//   1. `/` and legacy `/?nash` open the normal blind play screen with the
//      15-team catalog, no Nash draw selector, and no Nash mixture panel;
//   2. the belief candidate pool is the shipped opponent prior
//      (data/belief-pool-v3): fetched and load-bearing (failing closed);
//   3. the catalog party picker lists all 15 teams in catalog order with
//      readable names and expandable set details;
//   4. the start screen links to the party evaluator (`?evaluate`).

import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

interface CatalogTeamJson {
  id: string;
  drawWeight: number;
  sets: { species: string; item?: string; moves: string[]; level?: number }[];
}

const catalog = JSON.parse(
  readFileSync(
    new URL("../../data/team-pool-v4/team-pool.json", import.meta.url),
    "utf8",
  ),
) as { teams: CatalogTeamJson[] };

async function seedStorage(page: Page) {
  await page.addInitScript(() => {
    if (sessionStorage.getItem("nc2000-e2e-seeded") === "1") return;
    sessionStorage.setItem("nc2000-e2e-seeded", "1");
    localStorage.setItem("nc2000-locale", "en");
    localStorage.removeItem("nc2000-team-pool-v4");
    localStorage.removeItem("nc2000-custom-teams");
    localStorage.removeItem("nc2000-start-picks");
    localStorage.removeItem("nc2000-belief-prior");
  });
}

function guardConsole(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  return errors;
}

test("Nash mode is withdrawn: / and legacy ?nash open the normal 10-shell pool screen", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  // Seed a stale Nash draw preference from an earlier build to verify cleanup.
  await page.addInitScript(() => {
    localStorage.setItem("nc2000-bot-draw", "nash");
  });

  for (const url of ["/", "/?nash", "/?nash=v3", "/?blind"]) {
    await page.goto(url);
    await expect(page.locator(".start-title")).toBeVisible();
    await expect(page.locator('[data-testid="mode-banner"]')).toContainText("Blind");
    await expect(page.locator(".start-col .party-btn")).toHaveCount(2);
    await expect(page.locator('[data-party="settings"] .party-value')).toContainText(
      `Built-in (${catalog.teams.length} parties)`,
    );
    await page.locator('[data-party="settings"]').click();
    await expect(page.locator('[data-draw="nash"]')).toHaveCount(0);
    await expect(page.locator('[data-testid="nash-mix"]')).toHaveCount(0);
    await expect(page.locator('[data-testid="old-nash-details"]')).toHaveCount(0);
    await page.locator("dialog.modal .modal-head button").click();
  }

  expect(
    await page.evaluate(() => localStorage.getItem("nc2000-bot-draw")),
  ).toBeNull();
  expect(errors).toEqual([]);
});

test("the belief pool is the shipped prior, fetched by the play door and load-bearing", async ({
  page,
}) => {
  const errors = guardConsole(page);
  const beliefUrls: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("belief-pool-v3/belief-pool.json"))
      beliefUrls.push(r.url());
  });
  const gotBelief = page.waitForResponse(
    (r) => r.url().includes("belief-pool-v3/belief-pool.json") && r.ok(),
  );
  await page.goto("/");
  await gotBelief;
  await expect(page.locator('[data-party="settings"]')).toBeVisible();
  expect(beliefUrls.length).toBeGreaterThan(0);
  expect(errors).toEqual([]);

  // Load-bearing: a page that cannot get the file fails closed.
  await page.route("**/belief-pool-v3/**", (r) => r.abort());
  await page.goto("/");
  await expect(page.locator(".error-box")).toBeVisible();
  await page.unroute("**/belief-pool-v3/**");
});

test("the catalog party picker shows readable names and expandable set details for all 15 teams", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  await page.goto("/");

  await page.locator('[data-party="human"]').click();
  await expect(page.locator("dialog.modal .pool-category-title")).toHaveCount(0);
  await expect(page.locator("dialog.modal .team-prov")).toHaveCount(0);
  await expect(page.locator("dialog.modal .team-tier")).toHaveCount(0);

  const pickerCards = page.locator("dialog.modal [data-team]");
  await expect(pickerCards).toHaveCount(catalog.teams.length);
  await expect(
    pickerCards
      .filter({ hasText: "pb131-c3r-fbhera-magzap-refcloy" })
      .locator(".team-name"),
  ).toHaveText("Magnet 55 Zapdos · 55 Tyranitar · EndRev Heracross (pb-131)");
  await expect(
    pickerCards
      .filter({ hasText: "apex-c3r-fbhera-twstar-refcloy" })
      .locator(".team-name"),
  ).toHaveText(
    "55 RestTalk Snorlax · Boom 4-Atk Gengar · Reflect Cloyster · TWave Starmie (apex A)",
  );
  await expect(
    pickerCards
      .filter({ has: page.locator(".team-id", { hasText: /^pb-6$/ }) })
      .locator(".team-name"),
  ).toHaveText("55 Zapdos Machamp WA (Shawa)");
  await expect(
    pickerCards.filter({ hasText: "sample-12" }).locator(".team-name"),
  ).toHaveText("55 Snorlax Umbreon");

  // Expand sets detail on a card to verify moves and items are visible.
  const targetCard = pickerCards.filter({
    hasText: "pb131-c3r-fbhera-magzap-refcloy",
  });
  await targetCard.locator(".team-sets-summary").click();
  await expect(targetCard.locator(".team-sets-content")).toBeVisible();
  await expect(targetCard.locator(".team-set-row")).toHaveCount(6);
  await expect(targetCard.locator(".team-sets-content")).toContainText("Magnet");
  await page.locator("dialog.modal .modal-head button").click();

  expect(errors).toEqual([]);
});

test("the start screen links to the party evaluator", async ({ page }) => {
  await seedStorage(page);
  await page.goto("/");
  const link = page.locator('.start-link[href*="?evaluate"]');
  await expect(link).toHaveAttribute("href", /\?evaluate$/);
  await link.click();
  await expect(
    page.getByRole("heading", { name: "Evaluate your team's strength" }),
  ).toBeVisible();
  await expect(page.getByLabel("Number of games", { exact: true })).toHaveValue("32");
  await expect(page.getByLabel("Search iterations", { exact: true })).toHaveCount(0);
});
