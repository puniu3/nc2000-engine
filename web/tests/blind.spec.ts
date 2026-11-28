// Blind play — the only live information policy — and the belief prior.
//
// `/` and its alias `?blind` are the same door (info-mode.ts): the start
// screen is title/subtitle, Start battle, a one-line banner, your party,
// and the setup button — no opponent row (the banner's "a random opponent
// each battle" is the whole of what such a row could say), and no open-sheet
// surface anywhere. Built-in parties are the catalog (data/team-pool-v2), the
// same ids and exact sets the bot draws from.
//
// What this suite mostly has to prove is a NEGATIVE — the opponent's sets
// never reach the DOM while the battle is live — plus the one positive that
// depends on the same machinery: the belief prior actually reaching the
// bot's imputation. Both hang on the human side being a party the bot
// CANNOT identify from its six public species: against a known party the
// belief pins the exact team by signature, never falls back, and the prior
// — which governs the fallback roster only — stays dead. Hence the
// hand-mixed custom party below.
//
// Selectors depended on: [data-testid="mode-banner"], [data-party="human"|
// "settings"] with their .party-value, [data-testid="pool-file"|prior-file
// |prior-sample|prior-report|prior-clear|belief-chip|reveal-foe], plus the
// .start-col/.party-btn and .preview-cols/.team-sheets/.mon-sheet/
// .set-detail structure. Depended on by their ABSENCE: [data-party="bot"|
// "bot-random"|"pool"|"prior"] and [data-testid="mode-row"] must exist
// nowhere.

import { expect, test, type Locator, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

interface SetJson {
  species: string;
  item?: string;
  moves: string[];
  level?: number;
}

interface PoolTeamJson {
  id: string;
  export: string;
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

/** The retired bundled 32: only a source of legal sets for the mixed
 * custom party below. */
const pool = JSON.parse(
  readFileSync(
    new URL("../../data/meta-pool-v0/meta-pool.json", import.meta.url),
    "utf8",
  ),
) as { teams: PoolTeamJson[] };

/** The catalog: the built-in lists and what the bot draws from. */
const catalog = JSON.parse(
  readFileSync(
    new URL("../../data/team-pool-v4/team-pool.json", import.meta.url),
    "utf8",
  ),
) as { teams: PoolTeamJson[] };

/** The shipped opponent prior every searcher identifies against. */
const beliefPool = JSON.parse(
  readFileSync(
    new URL("../../data/belief-pool-v3/belief-pool.json", import.meta.url),
    "utf8",
  ),
) as { teams: PoolTeamJson[] };

const catalogLine = `Built-in (${catalog.teams.length} parties)`;

// Three sets from each of the two pool teams the open-sheet suite already
// uses for their short, decisive games. Every set stays a verbatim legal
// pool set (learnsets/DVs inherited), species and items stay unique across
// the six (species clause / item clause), and the three lightest sum to
// 150 <= 155 (Max Total Level), so the preview picker always has a legal
// 3-pick. Levels in party order are 55/50/50/55/50/50 — the greedy picker
// in choosePreview lands on 55+50+50 = 155 exactly.
//   hc75-top8-tsuru: Machamp L55 / Golem L50 / Marowak L50
//   sample-16:       Gengar L55 / Poliwrath L50 / Snorlax L50
const mixedSets: SetJson[] = [
  ...[1, 2, 4].map((i) => pool.teams[4].sets[i]),
  ...[1, 2, 5].map((i) => pool.teams[29].sets[i]),
];

const customBlind: CustomRecord = {
  id: "custom-blind",
  name: "Blind Custom",
  sets: mixedSets,
  species: mixedSets.map((s) => s.species),
  levels: mixedSets.map((s) => s.level ?? 55),
  savedAt: 1,
};

/** Known teams whose six species are exactly the mixed party's — must be
 * empty, or the bot identifies the "custom" opponent by signature and both
 * the fallback and the prior go untested. Asserted in the tests that rely
 * on it so the failure names the cause instead of showing a dead chip. */
const mixTwins = [...catalog.teams, ...beliefPool.teams]
  .filter((t) => sameSpeciesSet(t.sets, mixedSets))
  .map((t) => t.id);

function sameSpeciesSet(a: SetJson[], b: SetJson[]): boolean {
  const ids = (sets: SetJson[]) => sets.map((s) => toId(s.species)).sort();
  return ids(a).join(",") === ids(b).join(",");
}

function toId(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}

interface SeedOpts {
  customs?: CustomRecord[];
  picks?: unknown;
}

/** One-shot seeding: the init script runs on every navigation, so a
 * sessionStorage flag keeps a reload — or a second goto with a different
 * query string — from undoing what the test itself changed through the UI.
 * The team pool is cleared because every foe id here is looked up in the
 * catalog file read above. */
async function seedStorage(page: Page, opts: SeedOpts = {}) {
  await page.addInitScript(
    ({ records, initialPicks }) => {
      if (sessionStorage.getItem("nc2000-e2e-seeded") === "1") return;
      sessionStorage.setItem("nc2000-e2e-seeded", "1");
      localStorage.setItem("nc2000-locale", "en");
      localStorage.setItem("nc2000-custom-teams", JSON.stringify(records));
      localStorage.removeItem("nc2000-belief-prior");
      localStorage.removeItem("nc2000-team-pool-v4");
      if (initialPicks === undefined)
        localStorage.removeItem("nc2000-start-picks");
      else
        localStorage.setItem(
          "nc2000-start-picks",
          JSON.stringify(initialPicks),
        );
    },
    {
      records: opts.customs ?? [],
      initialPicks: opts.picks,
    },
  );
}

/** Console errors, plus any request for a baked pair table: no door may
 * read one any more. */
function guardConsole(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    errors.push(m.text());
  });
  page.on("request", (r) => {
    if (r.url().includes("/preview-tables-v0/"))
      errors.push(`pair-table request: ${r.url()}`);
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  return errors;
}

/** Every set of `team`, expanded and compared field by field inside
 * `section` — the same strictness as the open-sheet suite's
 * expectExactSets, minus its heading check (blind reveals the foe id in
 * the heading only after the game, so the caller reads it there first). */
async function expectExactSets(section: Locator, sets: SetJson[]) {
  for (const set of sets) {
    const head = section.locator(`[data-mon="${toId(set.species)}"]`);
    await expect(head).toHaveCount(1);
    if ((await head.getAttribute("aria-expanded")) !== "true")
      await head.click();
    const sheet = head.locator("xpath=..");
    if (set.item)
      await expect(sheet.locator(`[data-item="${toId(set.item)}"]`)).toHaveCount(
        1,
      );
    const actualMoves = await sheet.locator("[data-move]").evaluateAll((els) =>
      els.map((e) => e.getAttribute("data-move") ?? "").sort(),
    );
    expect(actualMoves).toEqual(set.moves.map(toId).sort());
  }
}

/** No set body anywhere in `section`, and no affordance that could open
 * one: the blind foe rows are static divs (no expand toggle, no
 * aria-expanded button), so even a click leaves the DOM sets-free. */
async function expectNoSetDetail(section: Locator) {
  await expect(section.locator(".set-detail")).toHaveCount(0);
  await expect(section.locator("[data-expand]")).toHaveCount(0);
  await expect(section.locator("[aria-expanded]")).toHaveCount(0);
  await expect(section.locator("button.mon-sheet-head")).toHaveCount(0);
  // Species / level / types stay public — that is the blind contract, not
  // an omission — so the six rows must still be there.
  await expect(section.locator("[data-mon]")).toHaveCount(6);
  await section.locator("[data-mon]").first().click();
  await expect(section.locator(".set-detail")).toHaveCount(0);
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

async function playToOutcome(page: Page) {
  let decisions = 0;
  const deadline = Date.now() + 10 * 60 * 1000;
  while (decisions < 300 && Date.now() < deadline) {
    if (await page.locator(".end-banner").isVisible()) return;
    const moves = page.locator(".move-btn");
    const moveCount = await moves.count();
    if (moveCount > 0) {
      const scores = await moves.evaluateAll((buttons) =>
        buttons.map((b) => {
          const text = b.querySelector(".move-bp")?.textContent ?? "0";
          return Number(text.match(/\d+/)?.[0] ?? 0);
        }),
      );
      let best = 0;
      for (let i = 1; i < scores.length; i++)
        if (scores[i] > scores[best]) best = i;
      await moves.nth(best).click();
      decisions++;
      await page.waitForTimeout(100);
      continue;
    }
    const switches = page.locator(".switch-btn");
    if ((await switches.count()) > 0) {
      await switches.first().click();
      decisions++;
      await page.waitForTimeout(100);
      continue;
    }
    await page.waitForTimeout(100);
  }
  throw new Error(
    `battle did not reach an outcome (${decisions} human decisions)`,
  );
}

// Deliberately NOT describe.configure({ mode: "serial" }): each test seeds
// its own storage in a fresh context, so an early failure must not hide
// the other cases' verdicts. The harness is single-worker anyway.

// ----------------------------------------------------------- the door
/** Everything any earlier revision put on the start screen that the blind
 * product must not have: the mode toggle, the open-mode opponent picker,
 * and the separate pool / prior buttons the one setup modal replaced. */
const RETIRED_SELECTORS = [
  '[data-testid="mode-row"]',
  '[data-party="bot"]',
  '[data-party="bot-random"]',
  '[data-party="pool"]',
  '[data-party="prior"]',
];

async function expectBlindScreen(page: Page) {
  await expect(page.locator(".start-screen")).toBeVisible();
  for (const sel of RETIRED_SELECTORS)
    await expect(page.locator(sel), `the play door must not render ${sel}`)
      .toHaveCount(0);
  const banner = page.locator('[data-testid="mode-banner"]');
  await expect(banner).toBeVisible();
  // One line, carrying both facts the screen says nowhere else: the sets
  // are hidden in both directions, and the opponent is redrawn every
  // battle.
  await expect(banner).toContainText(/blind/i);
  await expect(banner).toContainText(/random/i);
  // Two rows: your party, and the setup button, whose value line is the
  // resting state — the catalog, no prior (a loaded prior IS a file name).
  await expect(page.locator(".start-col .party-btn")).toHaveCount(2);
  const setup = page.locator('[data-party="settings"] .party-value');
  await expect(setup).toContainText(catalogLine);
  await expect(setup).not.toContainText(".json");
}

test("/ and ?blind are the same blind door, and nothing about it is stored", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  for (const url of ["/", "/?blind", "/?blind=0"]) {
    await page.goto(url);
    await expectBlindScreen(page);
  }
  // The built-in list renders catalog parties in catalog order, and Random
  // draws from exactly that many parties.
  await page.locator('[data-party="human"]').click();
  await expect(page.locator(".random-card")).toContainText(
    `Random (${catalog.teams.length} parties)`,
  );
  const ids = await page
    .locator("dialog.modal .team-card[data-team] .team-id")
    .allInnerTexts();
  expect(ids).toEqual(catalog.teams.map((t) => t.id));
  await page.locator("dialog.modal .modal-head button").click();
  expect(
    await page.evaluate(() =>
      Object.keys(localStorage).filter((k) => /mode|info/i.test(k)),
    ),
  ).toEqual([]);
  expect(errors).toEqual([]);
});

test("a pool file saved by the open-sheet build is dropped, not restored", async ({
  page,
}) => {
  const errors = guardConsole(page);
  const stale = {
    name: "old-pool.json",
    json: JSON.stringify({ teams: [pool.teams[0], pool.teams[1]] }),
  };
  await page.addInitScript((record) => {
    if (sessionStorage.getItem("nc2000-e2e-seeded") === "1") return;
    sessionStorage.setItem("nc2000-e2e-seeded", "1");
    localStorage.setItem("nc2000-locale", "en");
    localStorage.setItem("nc2000-team-pool", JSON.stringify(record));
  }, stale);
  await page.goto("/");
  await expectBlindScreen(page);
  expect(
    await page.evaluate(() => localStorage.getItem("nc2000-team-pool")),
  ).toBeNull();
  expect(errors).toEqual([]);
});

// ------------------------------------------------------- contract B4-2/3/4
test("blind hides the foe's sets for the whole game, then reveals them at the end", async ({
  page,
}) => {
  const errors = guardConsole(page);
  expect(mixTwins, "the mixed party must not be a pool signature").toEqual([]);
  await seedStorage(page, {
    customs: [customBlind],
    picks: { human: { kind: "custom", id: customBlind.id } },
  });
  await page.goto("/");
  await expect(page.locator('[data-party="human"] .party-value')).toHaveText(
    customBlind.name,
  );
  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();

  const previewFoe = page.locator(".preview-cols > section").first();
  const previewMine = page.locator(".preview-cols > section").nth(1);

  await test.step("B4-2: preview shows the foe's species only", async () => {
    // No team id either — the section is named generically so the catalog
    // entry cannot be looked up by hand.
    await expect(previewFoe.getByRole("heading")).toHaveText(
      "Opponent's party",
    );
    await expect(page.locator(".sheet-hint")).toContainText("Blind");
    await expectNoSetDetail(previewFoe);
    // The player's OWN side is untouched: the expand toggle still opens
    // their own sets.
    await expect(previewMine.locator("[data-expand]")).toHaveCount(6);
    await previewMine.locator("[data-expand]").first().click();
    await expect(previewMine.locator(".set-detail")).toHaveCount(1);
  });

  await choosePreview(page);

  await test.step("B4-2: the in-battle sheets modal hides them too", async () => {
    await page.locator(".sheets-btn").click();
    await expectNoSetDetail(page.locator(".team-sheets > section").nth(1));
    // Own side keeps its full open sheet.
    await expect(
      page.locator(".team-sheets > section").first().locator("[data-mon]"),
    ).toHaveCount(6);
    await expect(
      page
        .locator(".team-sheets > section")
        .first()
        .locator("[aria-expanded]"),
    ).toHaveCount(6);
    await page.locator("dialog.modal .modal-head button").click();
  });

  await test.step("B4-3: the blind game plays to a decided outcome", async () => {
    await playToOutcome(page);
    await expect(page.locator(".end-banner")).toBeVisible();
  });

  await test.step("B4-4: the end screen reveals the foe's real sets", async () => {
    const reveal = page.locator('[data-testid="reveal-foe"]');
    await expect(reveal).toHaveText("Show opponent's sets");
    await reveal.click();
    const foeSheet = page.locator(".team-sheets > section").nth(1);
    // Decided: the section names the catalog party, so the reveal can be
    // checked against the real entry rather than against itself.
    const heading = (await foeSheet.locator("h3").innerText()).trim();
    const id = /^Opponent's party \((.+)\)$/.exec(heading)?.[1];
    expect(id, `foe heading should name the party: ${heading}`).toBeTruthy();
    const drawn = catalog.teams.find((t) => t.id === id);
    expect(drawn, `drawn opponent ${id} must be a catalog party`).toBeTruthy();
    await expectExactSets(foeSheet, drawn!.sets);
    await page.locator("dialog.modal .modal-head button").click();
  });

  expect(errors).toEqual([]);
});

// --------------------------------------------------------- contract B4-5
test("the sample belief prior loads, applies, and governs the bot's read", async ({
  page,
}) => {
  const errors = guardConsole(page);
  const samplePrior = JSON.parse(
    readFileSync(
      new URL("../../data/belief-prior-v0.sample.json", import.meta.url),
      "utf8",
    ),
  ) as { species: Record<string, unknown> };
  const speciesCount = Object.keys(samplePrior.species).length;
  expect(speciesCount).toBeGreaterThan(0);
  expect(mixTwins, "the mixed party must not be a pool signature").toEqual([]);
  await seedStorage(page, {
    customs: [customBlind],
    picks: { human: { kind: "custom", id: customBlind.id } },
  });
  await page.goto("/");

  const setup = page.locator('[data-party="settings"] .party-value');
  // The line at rest, captured rather than spelled out: the clear at the
  // end has to put it back exactly, and comparing against a copy of the
  // wording would only test that this file and i18n-strings.ts agree.
  const atRest = ((await setup.textContent()) ?? "").replace(/\s+/g, " ").trim();
  expect(atRest, "the setup line starts on the catalog").toContain(
    catalogLine,
  );

  const report = page.locator('[data-testid="prior-report"]');
  await page.locator('[data-party="settings"]').click();
  // One modal, two sections: the pool panel and the prior panel are behind
  // the same button now, so both file inputs are in this dialog.
  await expect(page.locator('[data-testid="pool-file"]')).toHaveCount(1);
  // The hand-pick path exists; this test drives the sample instead (no
  // file chooser needed, and the same code path behind it).
  await expect(page.locator('[data-testid="prior-file"]')).toHaveCount(1);
  await page.locator('[data-testid="prior-sample"]').click();
  await expect(report).toContainText(`${speciesCount} species`);
  await expect(report).toContainText("0 entries skipped");
  await expect(report).toContainText("Applied");
  await expect(report).not.toContainText("NOT applied");
  // Each verdict box belongs to the file it is about: loading a prior must
  // not raise the pool panel's box, whose "Rejected" would read as a
  // verdict on what just happened.
  await expect(page.locator('[data-testid="pool-report"]')).toHaveCount(0);
  await page.locator("dialog.modal .modal-head button").click();
  // The button's value line reports both halves; the prior half is now the
  // file, and the pool half is exactly where it was.
  await expect(setup).toContainText("belief-prior-v0.sample.json");
  await expect(setup).toContainText(catalogLine);

  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();
  await choosePreview(page);
  await expect(page.locator(".move-btn").first()).toBeVisible({
    timeout: 90_000,
  });

  const chip = page.locator('[data-testid="belief-chip"]');
  await expect(chip).toBeVisible();
  // The mixed party is off-pool by construction, so the belief must be in
  // fallback — which is the only branch the prior can reach.
  await expect(chip).toContainText("off-pool");
  const chipText = (await chip.innerText()).trim();
  const governed = /prior:\s*(\d+)\s*\/\s*(\d+)/.exec(chipText);
  expect(governed, `belief chip must carry the prior counter: ${chipText}`)
    .not.toBeNull();
  // Every species in the mixed party is in the sample table, so the prior
  // governs at least one fallback slot; 0/N would mean it never landed.
  expect(Number(governed![1])).toBeGreaterThanOrEqual(1);
  expect(Number(governed![2])).toBeGreaterThanOrEqual(Number(governed![1]));

  // Back to the start screen: a stored prior re-probes when the setup
  // modal mounts (so the user sees what is loaded without re-picking it),
  // and clears cleanly.
  await page.locator(".battle-screen .quit-btn").click();
  await page.locator('[data-party="settings"]').click();
  await expect(report).toContainText(`${speciesCount} species`);
  await page.locator('[data-testid="prior-clear"]').click();
  // All the way back to the line this screen opened with — not merely
  // "no longer the file name", which a half-cleared state would also pass.
  await expect(setup).toHaveText(atRest);
  expect(
    await page.evaluate(() => localStorage.getItem("nc2000-belief-prior")),
  ).toBeNull();
  // Clearing may or may not close the modal — the contract does not say,
  // and neither behavior is wrong.
  const close = page.locator("dialog.modal .modal-head button");
  if ((await close.count()) > 0) await close.click();
  expect(errors).toEqual([]);
});

// --------------------------------------------------- pinned catalog party
test("a pinned built-in party plays its exact catalog sets", async ({ page }) => {
  const errors = guardConsole(page);
  const pinned = catalog.teams[catalog.teams.length - 1];
  await seedStorage(page, { picks: { human: { kind: "pool", id: pinned.id } } });
  await page.goto("/");
  await expect(page.locator('[data-party="human"] .party-value')).toHaveText(
    pinned.id,
  );
  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();
  const previewMine = page.locator(".preview-cols > section").nth(1);
  const species = await previewMine
    .locator("[data-mon]")
    .evaluateAll((els) => els.map((e) => e.getAttribute("data-mon")));
  expect(species).toEqual(pinned.sets.map((s) => toId(s.species)));
  await page.locator(".preview-actions .quit-btn").click();
  await expect(page.locator(".start-screen")).toBeVisible();
  expect(errors).toEqual([]);
});

test("pick prior loads, validates, stores, and clears cleanly in setup", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  await page.goto("/");
  await page.evaluate(() => localStorage.removeItem("nc2000-pick-prior"));

  const setup = page.locator('[data-party="settings"] .party-value');
  const atRest = ((await setup.textContent()) ?? "").replace(/\s+/g, " ").trim();

  await page.locator('[data-party="settings"]').click();
  const pickInput = page.locator('[data-testid="pick-prior-file"]');
  const pickReport = page.locator('[data-testid="pick-prior-report"]');

  await pickInput.setInputFiles({
    name: "bad-pick-prior.json",
    mimeType: "application/json",
    buffer: Buffer.from('{"schema":"wrong"}'),
  });
  await expect(pickReport).toContainText("NOT applied");
  await expect(page.locator('[data-testid="prior-report"]')).toHaveCount(0);

  const foeTeam = catalog.teams[0];
  const ownTeam = catalog.teams[1];
  const validPickPrior = JSON.stringify({
    schema: "nc2000-pick-prior-v1",
    smoothing: 0.01,
    backoff_strength: 1.0,
    source: {},
    rows: [
      {
        team: foeTeam.id,
        enemy_preview: ownTeam.sets.map((s) => ({
          species: toId(s.species),
          level: s.level,
          gender: s.gender === "M" || s.gender === "F" ? s.gender : "",
          item: Boolean(s.item),
        })),
        side: 0,
        choices: [
          {
            species: [0, 1, 2].map((i) => toId(foeTeam.sets[i].species)),
            count: 50,
          },
        ],
      },
    ],
  });

  await pickInput.setInputFiles({
    name: "valid-pick-prior.json",
    mimeType: "application/json",
    buffer: Buffer.from(validPickPrior),
  });
  await expect(pickReport).toContainText("Applied");
  await expect(pickReport).toContainText("1 row, 1 team");
  await page.locator("dialog.modal .modal-head button").click();
  await expect(setup).toContainText("valid-pick-prior.json");
  await expect(setup).toContainText(catalogLine);

  await page.locator('[data-party="settings"]').click();
  await page.locator('[data-testid="pick-prior-clear"]').click();
  await expect(setup).toHaveText(atRest);
  expect(
    await page.evaluate(() => localStorage.getItem("nc2000-pick-prior")),
  ).toBeNull();
  const close = page.locator("dialog.modal .modal-head button");
  if ((await close.count()) > 0) await close.click();
  expect(errors).toEqual([]);
});
