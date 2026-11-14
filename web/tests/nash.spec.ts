// `?nash` — start on normal screen with new Nash mixture as bot draw.
//
// Nash is blind play with the opponent's team drawn from the solved mixture.
// `?nash` sets the bot's draw source to the new Full-Combination preview Nash,
// while keeping the normal start screen intact (settings button, human party picker,
// evaluator link). What this suite verifies:
//
//   1. the query: `?nash` opens normal screen with new Nash selected;
//      `?nash=v3` selects the preserved old v3 mixture;
//      `?nash=0` or `/` defaults to catalog unless configured;
//   2. the mixture inside setup is the artifact's, weights and all, read-only
//      and set-free;
//   3. the team actually drawn is one of the mixture's arms, matched
//      species and level against the file;
//   4. a loaded custom pool file does not override ?nash's bot selection,
//      but is preserved when returning to `/`;
//   5. the belief candidate pool is the shipped opponent prior
//      (data/belief-pool-v3): fetched and load-bearing.
//
// Selectors depended on: [data-testid="mode-banner"|"nash-mix"|
// "belief-chip"], [data-party="human"|"settings"], and the
// shipped .start-col/.party-btn/.team-card/.species-chip structure.

import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

interface SetJson {
  species: string;
  item?: string;
  moves: string[];
  level?: number;
}

interface NashTeamJson {
  id: string;
  weight: number;
  sets: SetJson[];
}

interface CustomRecord {
  id: string;
  name: string;
  sets: SetJson[];
  species: string[];
  levels: number[];
  savedAt: number;
}

const artifact = JSON.parse(
  readFileSync(
    new URL("../../data/kodoku-r3/conclusion-equilibrium-deme.json", import.meta.url),
    "utf8",
  ),
) as { teams: NashTeamJson[] };

const prevR1Artifact = JSON.parse(
  readFileSync(
    new URL("../../data/kodoku-r3/conclusion-equilibrium-r1.json", import.meta.url),
    "utf8",
  ),
) as { teams: NashTeamJson[] };

const oldArtifact = JSON.parse(
  readFileSync(
    new URL("../../data/meta-nash-v3/pool-artifact.json", import.meta.url),
    "utf8",
  ),
) as { teams: NashTeamJson[] };

const pool = JSON.parse(
  readFileSync(
    new URL("../../data/meta-pool-v0/meta-pool.json", import.meta.url),
    "utf8",
  ),
) as { teams: { id: string; sets: SetJson[] }[] };

const prior = readFileSync(
  new URL("../../data/belief-prior-v0.sample.json", import.meta.url),
  "utf8",
);

function toId(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/** The roster line a preview column shows for a team: species and level,
 * sorted, so a drawn arm can be recognized without depending on the order
 * the six are listed in. */
function roster(sets: SetJson[]): string[] {
  return sets.map((s) => `${toId(s.species)}:${s.level ?? 55}`).sort();
}

/** The same off-pool party blind.spec.ts uses: three sets from each of two
 * pool teams, so every set is a verbatim legal pool set (learnsets, DVs,
 * items and species all inherited) while the six together match no pool
 * team's signature. That mismatch is the point — it is what puts the bot's
 * belief into fallback, which is the only state a belief prior can govern
 * and therefore the only state in which claim (4) can fail. Built from the
 * pool file rather than pasted, because a hand-written export has to
 * re-clear Item Clause and the learnsets by hand every time either moves. */
const mixedSets: SetJson[] = [
  ...[1, 2, 4].map((i) => pool.teams[4].sets[i]),
  ...[1, 2, 5].map((i) => pool.teams[29].sets[i]),
];

const beliefPool = JSON.parse(
  readFileSync(
    new URL("../../data/belief-pool-v3/belief-pool.json", import.meta.url),
    "utf8",
  ),
) as { teams: { id: string; sets: SetJson[] }[] };

/** Must be empty, or the bot identifies the party by signature, the belief
 * never falls back, and a dead prior chip would "pass" for the wrong
 * reason. Asserted in the test that depends on it. */
const mixTwins = [...artifact.teams, ...beliefPool.teams]
  .filter((t) => sameSpeciesSet(t.sets, mixedSets))
  .map((t) => t.id);

function sameSpeciesSet(a: SetJson[], b: SetJson[]): boolean {
  const ids = (sets: SetJson[]) => sets.map((s) => toId(s.species)).sort();
  return ids(a).join(",") === ids(b).join(",");
}

const customNash: CustomRecord = {
  id: "custom-nash",
  name: "Nash Custom",
  sets: mixedSets,
  species: mixedSets.map((s) => s.species),
  levels: mixedSets.map((s) => s.level ?? 55),
  savedAt: 1,
};

/** One-shot seeding, same idiom as blind.spec.ts: the init script runs on
 * every navigation, so a sessionStorage flag keeps a second goto from
 * undoing it. The prior is seeded ON PURPOSE here — the point of claim (4)
 * is that a nash game ignores it. */
async function seedStorage(
  page: Page,
  opts: { withPrior?: boolean; withCustom?: boolean } = {},
) {
  await page.addInitScript(
    ({ priorJson, record }) => {
      if (sessionStorage.getItem("nc2000-e2e-seeded") === "1") return;
      sessionStorage.setItem("nc2000-e2e-seeded", "1");
      localStorage.setItem("nc2000-locale", "en");
      localStorage.removeItem("nc2000-team-pool-v2");
      localStorage.removeItem("nc2000-bot-draw");
      if (record) {
        localStorage.setItem("nc2000-custom-teams", JSON.stringify([record]));
        localStorage.setItem(
          "nc2000-start-picks",
          JSON.stringify({ human: { kind: "custom", id: record.id } }),
        );
      } else {
        localStorage.removeItem("nc2000-custom-teams");
        localStorage.removeItem("nc2000-start-picks");
      }
      if (priorJson)
        localStorage.setItem(
          "nc2000-belief-prior",
          JSON.stringify({ name: "sample.json", json: priorJson }),
        );
      else localStorage.removeItem("nc2000-belief-prior");
    },
    {
      priorJson: opts.withPrior ? prior : "",
      record: opts.withCustom ? customNash : null,
    },
  );
}

/** Blind play skips the baked pair tables, so unlike blind.spec.ts this
 * suite expects no 404s at all. */
function guardConsole(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  return errors;
}

async function choosePreview(page: Page) {
  for (let n = 0; n < 3; n++) {
    const candidates = page.locator(
      '.pick-head[aria-pressed="false"][aria-disabled="false"]',
    );
    await expect(candidates.first()).toBeVisible();
    await candidates.first().click();
  }
  const confirm = page.getByRole("button", { name: "Confirm picks" });
  await expect(confirm).toBeEnabled();
  await confirm.click();
  await expect(page.locator(".battle-screen")).toBeVisible({ timeout: 90_000 });
}

test("?nash starts in the normal screen with the new Nash party selected as bot", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);

  await page.goto("/?nash");
  await expect(page.locator(".start-title")).toBeVisible();
  // Normal screen: blind banner is shown, not a special door banner
  await expect(page.locator('[data-testid="mode-banner"]')).toContainText(
    "Blind",
  );
  // Normal controls: human party and settings button
  await expect(page.locator('[data-party="human"]')).toHaveCount(1);
  await expect(page.locator('[data-party="settings"]')).toHaveCount(1);
  await expect(page.locator('[data-party="nash"]')).toHaveCount(0);
  await expect(page.locator('[data-party="bot"]')).toHaveCount(0);
  await expect(page.locator(".start-col .party-btn")).toHaveCount(2);

  // Settings button reports that the new Nash mixture is chosen as opponent
  const row = page.locator('[data-party="settings"] .party-value');
  await expect(row).toContainText(`Solved mixture (${artifact.teams.length}`);

  // Opening settings shows the Nash mixture panel and the draw choice pressed
  await page.locator('[data-party="settings"]').click();
  await expect(page.locator('[data-draw="nash"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  const panel = page.locator('[data-testid="nash-mix"]');
  await expect(panel).toBeVisible();
  await expect(panel.locator(".team-card")).toHaveCount(artifact.teams.length);
  await page.locator("dialog.modal .modal-head button").click();

  // `?nash=0` and `?blind` default to catalog
  for (const url of ["/?nash=0", "/?blind", "/"]) {
    await page.goto(url);
    await expect(page.locator('[data-party="settings"]')).toHaveCount(1);
    await expect(page.locator('[data-party="nash"]')).toHaveCount(0);
    await expect(page.locator('[data-party="bot"]')).toHaveCount(0);
    await expect(page.locator('[data-party="settings"] .party-value')).toContainText(
      "Built-in",
    );
  }

  expect(errors).toEqual([]);
});

test("the mixture panel inside setup is the artifact, read-only and set-free", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  await page.goto("/?nash");
  await page.locator('[data-party="settings"]').click();

  const panel = page.locator('[data-testid="nash-mix"]');
  await expect(panel).toBeVisible();
  await expect(panel.locator(".team-card")).toHaveCount(artifact.teams.length);

  // Weights are the file's, renormalized, and shown to one decimal.
  const total = artifact.teams.reduce((a, t) => a + t.weight, 0);
  for (const t of artifact.teams) {
    const card = panel.locator(`[data-nash="${t.id}"]`);
    await expect(card).toHaveCount(1);
    await expect(card.locator(".team-rank")).toHaveText(
      `${((t.weight / total) * 100).toFixed(1)}%`,
    );
    // Every species, and only those.
    await expect(card.locator(".species-chip")).toHaveCount(t.sets.length);
    for (const set of t.sets)
      await expect(card).toContainText(set.species.split("-")[0]);
  }

  // A readout, not a picker: no set body anywhere
  await expect(panel.locator(".set-detail")).toHaveCount(0);
  await expect(panel.locator("[data-move]")).toHaveCount(0);
  await expect(panel.locator("[data-item]")).toHaveCount(0);

  expect(errors).toEqual([]);
});

test("the drawn opponent is one arm of the mixture, and no prior reaches the game", async ({
  page,
}) => {
  const errors = guardConsole(page);
  // A prior IS in storage, left there by an earlier play-door visit.
  expect(mixTwins).toEqual([]);
  await seedStorage(page, { withPrior: true, withCustom: true });
  await page.goto("/?nash");
  await expect(page.locator('[data-party="human"] .party-value')).toHaveText(
    customNash.name,
  );
  await page.getByRole("button", { name: "Start battle" }).click();

  const foe = page.locator(".preview-cols > section").first();
  await expect(foe.locator("[data-mon]")).toHaveCount(6);
  // Species and level read off the same row, never zipped from two lists:
  // a foe roster is six rows, and pairing by index across two queries would
  // pass on a page that had lost one of them.
  const shown = (
    await foe.locator("[data-mon]").evaluateAll((els) =>
      els.map((e) => {
        // First "L<n>" in the row: the level is printed in the visible head
        // and again in the screen-reader label, so a bare digit strip reads
        // "5050".
        const level = /L(\d+)/.exec(
          e.querySelector(".mon-level")?.textContent ?? "",
        );
        return `${e.getAttribute("data-mon") ?? ""}:${level ? level[1] : "?"}`;
      }),
    )
  ).sort();
  const arms = artifact.teams.map((t) => roster(t.sets).join(","));
  expect(arms).toContain(shown.join(","));

  await choosePreview(page);

  // The belief is live (the mixed party is off-pool, so it falls back)
  const chip = page.locator('[data-testid="belief-chip"]');
  await expect(chip).toBeVisible();
  await expect(chip).toContainText("off-pool");
  await expect(chip).not.toContainText("prior");

  expect(errors).toEqual([]);
});

test("the belief pool is the shipped prior, fetched by every door and load-bearing", async ({
  page,
}) => {
  // (a) ?nash and the play door fetch the prior and boot on it.
  const errors = guardConsole(page);
  const beliefUrls: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("belief-pool-v3/belief-pool.json"))
      beliefUrls.push(r.url());
  });
  const gotBelief = page.waitForResponse(
    (r) => r.url().includes("belief-pool-v3/belief-pool.json") && r.ok(),
  );
  await page.goto("/?nash");
  await gotBelief;
  await expect(page.locator('[data-party="settings"]')).toBeVisible();
  expect(beliefUrls.length).toBeGreaterThan(0);
  beliefUrls.length = 0;
  await page.goto("/");
  await expect(page.locator('[data-party="settings"]')).toBeVisible();
  expect(beliefUrls.length).toBeGreaterThan(0);
  expect(errors).toEqual([]);

  // (b) load-bearing: a page that cannot get the file fails closed (the
  // boot error box), never a quiet game under a plainer prior.
  for (const url of ["/?nash", "/"]) {
    await page.route("**/belief-pool-v3/**", (r) => r.abort());
    await page.goto(url);
    await expect(page.locator(".error-box")).toBeVisible();
    await page.unroute("**/belief-pool-v3/**");
  }
});

test("the play door's Nash draw choice draws a mixture arm and persists", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  await page.goto("/");
  await page.locator('[data-party="settings"]').click();
  await expect(page.locator('[data-draw="catalog"]')).toHaveAttribute("aria-pressed", "true");
  await page.locator('[data-draw="nash"]').click();
  await expect(page.locator('[data-draw="nash"]')).toHaveAttribute("aria-pressed", "true");
  // The mixture is shown where it is chosen, weights and all.
  await expect(page.locator('[data-testid="nash-mix"] .team-card')).toHaveCount(
    artifact.teams.length,
  );
  await page.locator("dialog.modal .modal-head button").click();
  const setup = page.locator('[data-party="settings"] .party-value');
  await expect(setup).toContainText(`Solved mixture (${artifact.teams.length}`);
  await page.reload();
  await expect(setup).toContainText(`Solved mixture (${artifact.teams.length}`);

  await page.getByRole("button", { name: "Start battle" }).click();
  const foe = page.locator(".preview-cols > section").first();
  await expect(foe.locator("[data-mon]")).toHaveCount(6);
  const shown = (
    await foe.locator("[data-mon]").evaluateAll((els) =>
      els.map((e) => {
        const level = /L(\d+)/.exec(e.querySelector(".mon-level")?.textContent ?? "");
        return `${e.getAttribute("data-mon") ?? ""}:${level ? level[1] : "?"}`;
      }),
    )
  ).sort();
  const arms = artifact.teams.map((t) => roster(t.sets).join(","));
  expect(arms).toContain(shown.join(","));
  await page.locator(".preview-actions .quit-btn").click();

  // Back to the built-in parties: the choice is stored, and clears.
  await page.locator('[data-party="settings"]').click();
  await page.locator('[data-draw="catalog"]').click();
  await page.locator("dialog.modal .modal-head button").click();
  await expect(setup).not.toContainText("Solved mixture");
  expect(await page.evaluate(() => localStorage.getItem("nc2000-bot-draw"))).toBeNull();

  // Catalog party picker groups parties by category with subheadings,
  // gives parties readable names, omits redundant shoulder badges, and provides
  // expandable set details so camouflage variants can be inspected.
  await page.locator('[data-party="human"]').click();
  const categoryHeadings = page.locator("dialog.modal .pool-category-title");
  await expect(categoryHeadings).toHaveText([
    "Nash equilibrium",
    "Evolved",
    "Classic & community",
  ]);
  await expect(page.locator("dialog.modal .team-prov")).toHaveCount(0);
  await expect(page.locator("dialog.modal .team-tier")).toHaveCount(0);
  const pickerCards = page.locator("dialog.modal [data-team]");
  await expect(
    pickerCards.filter({ hasText: "apex-c08-surgical-142" }).locator(".team-name"),
  ).toHaveText("Rain Cloyster · HP-Bug Marowak 55 Snorlax");
  await expect(
    pickerCards.filter({ hasText: "red-v10-s1105-r2-s0142" }).locator(".team-name"),
  ).toHaveText("Rain Cloyster · EQ Heracross 55 Snorlax");
  await expect(
    pickerCards.filter({ hasText: "apex-c04-bug-3atkstar-boomgen" }).locator(".team-name"),
  ).toHaveText("Explosion Cloyster · Boom Gengar 55 Snorlax");
  await expect(
    pickerCards.filter({ hasText: "red-v10-s1105-r1-s0148" }).locator(".team-name"),
  ).toHaveText("Explosion Cloyster · HP-Fight 55 Snorlax");
  await expect(
    pickerCards.filter({ hasText: "sample-08" }).locator(".team-name"),
  ).toHaveText("Zapdos Tank (Boom Exeggutor / Sample 08)");
  await expect(
    pickerCards.filter({ hasText: "red-v5-r1-s0097" }).locator(".team-name"),
  ).toHaveText("55 Tyranitar Heracross");

  // Expand sets detail on a card to verify moves and items are visible.
  const targetCard = pickerCards.filter({ hasText: "red-v10-s1105-r2-s0142" });
  await targetCard.locator(".team-sets-summary").click();
  await expect(targetCard.locator(".team-sets-content")).toBeVisible();
  await expect(targetCard.locator(".team-set-row")).toHaveCount(6);
  await expect(targetCard.locator(".team-sets-content")).toContainText("Rain Dance");
  await page.locator("dialog.modal .modal-head button").click();

  expect(errors).toEqual([]);
});

test("the start screen links to the party evaluator", async ({ page }) => {
  await seedStorage(page);
  await page.goto("/");
  const link = page.locator('.start-link[href*="?evaluate"]');
  await expect(link).toHaveAttribute("href", /\?evaluate$/);
  await link.click();
  await expect(page.getByRole("heading", { name: "Evaluate your team's strength" })).toBeVisible();
  await expect(page.getByLabel("Number of games", { exact: true })).toHaveValue("32");
  await expect(page.getByLabel("Search iterations", { exact: true })).toHaveCount(0);
});

test("the previous two Nash mixtures (r1 and v3) are collapsed under details and remain accessible", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);

  await page.goto("/?nash=v3");
  await expect(page.locator(".start-title")).toBeVisible();
  await expect(page.locator('[data-party="settings"]')).toBeVisible();
  const row = page.locator('[data-party="settings"] .party-value');
  await expect(row).toContainText(`Old mixture v3 (${oldArtifact.teams.length}`);

  await page.locator('[data-party="settings"]').click();
  await expect(page.locator('[data-testid="old-nash-details"]')).toHaveJSProperty("open", true);
  await expect(page.locator('[data-draw="nash-v3"]')).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator('[data-testid="nash-mix"] .team-card')).toHaveCount(
    oldArtifact.teams.length,
  );
  await page.locator("dialog.modal .modal-head button").click();

  // On `/` with catalog default, the previous mixtures are collapsed under <details>.
  await page.evaluate(() => localStorage.removeItem("nc2000-bot-draw"));
  await page.goto("/");
  await page.locator('[data-party="settings"]').click();
  const details = page.locator('[data-testid="old-nash-details"]');
  await expect(details).toHaveJSProperty("open", false);
  await details.locator("summary").click();
  await expect(details).toHaveJSProperty("open", true);

  // Select r1 mixture
  await page.locator('[data-draw="nash-r1"]').click();
  await expect(page.locator('[data-draw="nash-r1"]')).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator('[data-testid="nash-mix"] .team-card')).toHaveCount(
    prevR1Artifact.teams.length,
  );

  // Select v3 mixture
  await page.locator('[data-draw="nash-v3"]').click();
  await expect(page.locator('[data-draw="nash-v3"]')).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator('[data-testid="nash-mix"] .team-card')).toHaveCount(
    oldArtifact.teams.length,
  );
  await page.locator("dialog.modal .modal-head button").click();
  await expect(page.locator('[data-party="settings"] .party-value')).toContainText(
    `Old mixture v3 (${oldArtifact.teams.length}`,
  );
  expect(errors).toEqual([]);
});

