// Blind information mode + belief prior (M18 experiment) — contract B4,
// revised for the URL-gated entry and the one Blind setup modal.
//
// Blind is no longer something the start screen offers. `?blind` is the
// only door (info-mode.ts), nothing about the mode is persisted, and open
// mode says nothing about modes at all — so every case here enters with
// page.goto("/?blind"), and the first one is about the door itself: that
// it opens, that `blind=0` closes it, and that the screen behind it is the
// shipped M12 screen with no trace of the experiment on it. That last part
// is the regression test the simplification pass asked for, so it names
// every element the experiment has ever added to this screen and asserts
// each one absent, dead testids included.
//
// The blind screen is five things and nothing else: title/subtitle, Start
// battle, a one-line banner, your party, and the Blind setup button. There
// is no opponent row at all — the banner's "a random opponent each battle"
// is the whole of what the deleted static row said — and no separate pool
// or prior button: both panels are sections of the single
// `[data-party="settings"]` modal.
//
// The six required cases run as four browser sessions: cases 2-4 are one
// full game (they are three assertions about the SAME battle — preview,
// outcome, post-game reveal — so replaying the game three times would buy
// nothing but minutes), marked off with test.step so the report still maps
// 1:1 onto the contract.
//
// What this suite mostly has to prove is a NEGATIVE — the opponent's sets
// never reach the DOM while the battle is live — plus the one positive that
// depends on the same machinery: the belief prior actually reaching the
// bot's imputation. Both hang on the human side being a party the bot
// CANNOT identify from its six public species: against a pool party the
// belief pins the exact team by signature (contract "決定済み 2" accepts
// this), never falls back, and the prior — which governs the fallback
// roster only — stays dead. Hence the hand-mixed custom party below.
//
// Written against the contract's testids/attributes rather than against a
// running app: the UI implementing them lands in parallel with this file.
// Selectors depended on (integration must check them):
// [data-testid="mode-banner"] (blind only), [data-party="human"|"bot"|
// "settings"] with their .party-value, [data-testid="pool-file"|prior-file
// |prior-sample|prior-report|prior-clear|belief-chip|reveal-foe], plus the
// shipped .start-col/.party-btn and .preview-cols/.team-sheets/.mon-sheet/
// .set-detail structure. Depended on by their ABSENCE, which is just as
// load-bearing here: [data-party="bot-random"|"pool"|"prior"] and
// [data-testid="mode-row"] must exist nowhere, in either mode.

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

const pool = JSON.parse(
  readFileSync(
    new URL("../../data/meta-pool-v0/meta-pool.json", import.meta.url),
    "utf8",
  ),
) as { teams: PoolTeamJson[] };

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

/** Pool teams whose six species are exactly the mixed party's — must be
 * empty, or the bot identifies the "custom" opponent by signature and both
 * the fallback and the prior go untested. Asserted in the tests that rely
 * on it so the failure names the cause instead of showing a dead chip. */
const mixTwins = pool.teams
  .filter((t) => sameSpeciesSet(t.sets, mixedSets))
  .map((t) => t.id);

function sameSpeciesSet(a: SetJson[], b: SetJson[]): boolean {
  const ids = (sets: SetJson[]) => sets.map((s) => toId(s.species)).sort();
  return ids(a).join(",") === ids(b).join(",");
}

function toId(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}

/** Start-screen pin for the opponent side. Blind ignores it (it always
 * draws from the pool), but seeding it keeps the record shape honest and
 * makes the same seed reusable if the URL drops `?blind`. */
const RANDOM_PICK = { kind: "random" } as const;

interface SeedOpts {
  customs?: CustomRecord[];
  picks?: unknown;
}

/** Same one-shot seeding idiom as custom-bot.spec.ts: the init script runs
 * on every navigation, so a sessionStorage flag keeps a reload — or a
 * second goto with a different query string — from undoing what the test
 * itself changed through the UI. The mode is NOT seeded: it lives in the
 * URL now, and this suite asserts that nothing writes it down. The team
 * pool is cleared because every foe id here is looked up in the bundled
 * pool file read above. */
async function seedStorage(page: Page, opts: SeedOpts = {}) {
  await page.addInitScript(
    ({ records, initialPicks }) => {
      if (sessionStorage.getItem("nc2000-e2e-seeded") === "1") return;
      sessionStorage.setItem("nc2000-e2e-seeded", "1");
      localStorage.setItem("nc2000-locale", "en");
      localStorage.setItem("nc2000-custom-teams", JSON.stringify(records));
      localStorage.removeItem("nc2000-belief-prior");
      localStorage.removeItem("nc2000-team-pool");
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

/** A pool-vs-pool game asks for its baked pair table, and `data/
 * preview-tables-v0/` ships README-only — fetchPairJson answers that 404
 * with null by design (the live search takes over), but Chromium still
 * logs the failed load, and the message text names no URL. This suite is
 * the first to run pool-vs-pool at all, which is why the open-sheet suite
 * never needed the exemption: it is scoped by the failing resource's URL,
 * so a real console error still fails the test. */
const EXPECTED_404 = "/preview-tables-v0/pair-";

function guardConsole(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() !== "error") return;
    if (m.location().url.includes(EXPECTED_404)) return;
    errors.push(m.text());
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

// --------------------------------------------------------- contract B4-1
/** Everything the experiment has ever put on the start screen, by the
 * selector it put there. `/` must have none of them: not the mode toggle
 * the feature started as, not the banner, not the static opponent row, and
 * not the pool / prior buttons the one setup modal replaced. Two of these
 * are dead testids from earlier rounds and are listed on purpose — a dead
 * control that survived a delete is exactly the thing that still looks
 * like working UI. */
const EXPERIMENT_SELECTORS = [
  '[data-testid="mode-row"]',
  '[data-testid="mode-banner"]',
  '[data-party="bot-random"]',
  '[data-party="settings"]',
  '[data-party="pool"]',
  '[data-party="prior"]',
];

async function expectShippedOpenScreen(page: Page) {
  await expect(page.locator(".start-screen")).toBeVisible();
  for (const sel of EXPERIMENT_SELECTORS)
    await expect(page.locator(sel), `open mode must not render ${sel}`)
      .toHaveCount(0);
  await expect(page.locator('[data-party="bot"]')).toHaveCount(1);
  // Nothing unnamed either: the M12 screen is Start battle plus exactly two
  // party rows, so a control this file has never heard of cannot slip in
  // under a testid it does not check.
  await expect(page.locator(".start-col .party-btn")).toHaveCount(2);
  // ...and no copy points at the door. Hiding the experiment from the
  // public build is the entire purpose of the revision, so "there is a
  // mode, add ?blind to get it" must not be discoverable here. Nothing
  // else on this screen can say "blind": both parties read "Random", and
  // no pool team id contains it.
  await expect(page.locator(".start-col")).not.toContainText(/blind/i);
}

test("blind is URL-gated, and the open screen keeps no trace of the experiment", async ({
  page,
}) => {
  const errors = guardConsole(page);
  // No saved parties here: the start screen's text is asserted below, and a
  // custom team named "Blind Custom" would be part of it.
  await seedStorage(page);

  await page.goto("/");
  await expectShippedOpenScreen(page);

  await page.goto("/?blind");
  const banner = page.locator('[data-testid="mode-banner"]');
  await expect(banner).toBeVisible();
  // One line, carrying both facts the screen says nowhere else: the sets
  // are hidden in both directions, and the opponent is redrawn every
  // battle. The second of those had a row of its own until this pass, and
  // the row is gone — so if the banner ever stops saying it, nothing does.
  await expect(banner).toContainText(/blind/i);
  await expect(banner).toContainText(/random/i);
  // The opponent is neither choosable nor listed: blind draws from the pool
  // at start and redraws on rematch.
  await expect(page.locator('[data-party="bot"]')).toHaveCount(0);
  await expect(page.locator('[data-party="bot-random"]')).toHaveCount(0);
  // One button for both of the experiment's settings, and its value line is
  // their resting state — the bundled pool, no prior. Asserting that the
  // line names no file is how "no prior" is checked without pinning the
  // wording of the empty case: a loaded prior IS a file name, which is what
  // B4-5 reads off this very element.
  const setup = page.locator('[data-party="settings"] .party-value');
  await expect(setup).toContainText(`Bundled (${pool.teams.length} teams)`);
  await expect(setup).not.toContainText(".json");
  await expect(page.locator('[data-party="pool"]')).toHaveCount(0);
  await expect(page.locator('[data-party="prior"]')).toHaveCount(0);
  // Two rows here as well: your party, and the setup button.
  await expect(page.locator(".start-col .party-btn")).toHaveCount(2);

  // A reload keeps blind — but only because the query string is still
  // there. Nothing was written down: a stored preference would outlive the
  // link that set it and strand a later visitor in the experiment.
  await page.reload();
  await expect(page.locator('[data-testid="mode-banner"]')).toBeVisible();
  await expect(page.locator('[data-party="bot"]')).toHaveCount(0);

  // `blind=0` reads as absent, so a link that has been passed around can
  // be defused by editing one character — and what it lands on is the same
  // untouched open screen, not a half-dressed one.
  await page.goto("/?blind=0");
  await expectShippedOpenScreen(page);
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
    picks: { human: { kind: "custom", id: customBlind.id }, bot: RANDOM_PICK },
  });
  await page.goto("/?blind");
  await expect(page.locator('[data-party="human"] .party-value')).toHaveText(
    customBlind.name,
  );
  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();

  const previewFoe = page.locator(".preview-cols > section").first();
  const previewMine = page.locator(".preview-cols > section").nth(1);

  await test.step("B4-2: preview shows the foe's species only", async () => {
    // No team id either — blind names the section generically so the pool
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
    // Decided: the section names the pool team, so the reveal can be
    // checked against the real entry rather than against itself.
    const heading = (await foeSheet.locator("h3").innerText()).trim();
    const id = /^Foe team \((.+)\)$/.exec(heading)?.[1];
    expect(id, `foe heading should name the pool team: ${heading}`).toBeTruthy();
    const drawn = pool.teams.find((t) => t.id === id);
    expect(drawn, `drawn opponent ${id} must be a pool team`).toBeTruthy();
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
  expect(mixTwins, "the mixed party must not be a pool signature").toEqual([]);
  await seedStorage(page, {
    customs: [customBlind],
    picks: { human: { kind: "custom", id: customBlind.id }, bot: RANDOM_PICK },
  });
  await page.goto("/?blind");

  const setup = page.locator('[data-party="settings"] .party-value');
  // The line at rest, captured rather than spelled out: the clear at the
  // end has to put it back exactly, and comparing against a copy of the
  // wording would only test that this file and i18n-strings.ts agree.
  const atRest = ((await setup.textContent()) ?? "").replace(/\s+/g, " ").trim();
  expect(atRest, "the setup line starts on the bundled pool").toContain(
    `Bundled (${pool.teams.length} teams)`,
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
  // 42 species is the sample table's content, so this also proves the
  // table was really parsed rather than merely stored.
  await expect(report).toContainText("42 species");
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
  await expect(setup).toContainText(`Bundled (${pool.teams.length} teams)`);

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
  await expect(report).toContainText("42 species");
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

// --------------------------------------------------------- contract B4-6
test("open mode keeps the shipped open-team-sheet behavior", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page, {
    picks: {
      human: { kind: "pool", id: pool.teams[29].id },
      bot: { kind: "pool", id: pool.teams[4].id },
    },
  });
  await page.goto("/");

  // Nothing blind is even offered: the experiment must not change the
  // default screen by merely existing. B4-1 proves that on an empty
  // profile; here it holds with both sides pinned to pool teams, which is
  // the state a returning player arrives in.
  for (const sel of EXPERIMENT_SELECTORS)
    await expect(page.locator(sel), `open mode must not render ${sel}`)
      .toHaveCount(0);
  await expect(page.locator('[data-party="bot"] .party-value')).toHaveText(
    pool.teams[4].id,
  );

  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();
  const previewFoe = page.locator(".preview-cols > section").first();
  await expect(previewFoe.getByRole("heading")).toHaveText(
    `Foe team (${pool.teams[4].id})`,
  );
  await expect(page.locator(".sheet-hint")).not.toContainText("Blind");
  // Open sheet: the foe's sets are readable, exactly as shipped.
  await expect(previewFoe.locator("[aria-expanded]")).toHaveCount(6);
  await expectExactSets(previewFoe, pool.teams[4].sets);
  await page.locator(".preview-actions .quit-btn").click();
  await expect(page.locator(".start-screen")).toBeVisible();
  expect(errors).toEqual([]);
});
