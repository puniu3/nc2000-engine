// Blind information mode + belief prior (M18 experiment) — contract B4.
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
// Selectors depended on (integration must check them): [data-mode],
// [data-party="bot-random"], [data-party="prior"], [data-testid="prior-file"
// |prior-sample|prior-report|prior-clear|belief-chip|reveal-foe], plus the
// shipped .preview-cols/.team-sheets/.mon-sheet/.set-detail structure.

import { expect, test, type Locator, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

type InfoMode = "open" | "blind";

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
 * makes the same seed reusable if the mode is flipped mid-test. */
const RANDOM_PICK = { kind: "random" } as const;

interface SeedOpts {
  customs?: CustomRecord[];
  picks?: unknown;
  /** Omitted = key removed, i.e. the shipped default ("open"). */
  mode?: InfoMode;
}

/** Same one-shot seeding idiom as custom-bot.spec.ts: the init script runs
 * on every navigation, so a sessionStorage flag keeps a reload from
 * undoing what the test itself changed through the UI. */
async function seedStorage(page: Page, opts: SeedOpts = {}) {
  await page.addInitScript(
    ({ records, initialPicks, mode }) => {
      if (sessionStorage.getItem("nc2000-e2e-seeded") === "1") return;
      sessionStorage.setItem("nc2000-e2e-seeded", "1");
      localStorage.setItem("nc2000-locale", "en");
      localStorage.setItem("nc2000-custom-teams", JSON.stringify(records));
      localStorage.removeItem("nc2000-belief-prior");
      if (mode === undefined) localStorage.removeItem("nc2000-info-mode");
      else localStorage.setItem("nc2000-info-mode", mode);
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
      mode: opts.mode,
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
test("blind replaces the opponent picker with a random draw, and persists", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page, { customs: [customBlind] });
  await page.goto("/");

  // Default is open (contract "決定済み 4"): the toggle reflects it and no
  // blind-only affordance exists yet.
  await expect(page.locator('[data-testid="mode-row"]')).toBeVisible();
  await expect(page.locator('[data-mode="open"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator('[data-mode="blind"]')).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await expect(page.locator('[data-party="bot"]')).toHaveCount(1);
  await expect(page.locator('[data-party="bot-random"]')).toHaveCount(0);
  await expect(page.locator('[data-party="prior"]')).toHaveCount(0);

  await page.locator('[data-mode="blind"]').click();

  // The opponent is no longer choosable: blind draws from the pool.
  await expect(page.locator('[data-party="bot"]')).toHaveCount(0);
  const botStatic = page.locator('[data-party="bot-random"]');
  await expect(botStatic).toBeVisible();
  await expect(botStatic).toContainText("Randomly drawn each battle");
  await expect(page.locator(".mode-note")).not.toBeEmpty();
  // The prior is blind-only (contract "決定済み 6"), and starts unset.
  await expect(page.locator('[data-party="prior"] .party-value')).toHaveText(
    "None",
  );
  expect(
    await page.evaluate(() => localStorage.getItem("nc2000-info-mode")),
  ).toBe("blind");

  await page.reload();
  await expect(page.locator('[data-mode="blind"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator('[data-party="bot"]')).toHaveCount(0);
  await expect(page.locator('[data-party="bot-random"]')).toBeVisible();

  // ...and back: switching to open restores the shipped start screen.
  await page.locator('[data-mode="open"]').click();
  await expect(page.locator('[data-party="bot"]')).toHaveCount(1);
  await expect(page.locator('[data-party="bot-random"]')).toHaveCount(0);
  await expect(page.locator('[data-party="prior"]')).toHaveCount(0);
  expect(
    await page.evaluate(() => localStorage.getItem("nc2000-info-mode")),
  ).toBe("open");
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
    mode: "blind",
    picks: { human: { kind: "custom", id: customBlind.id }, bot: RANDOM_PICK },
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
    mode: "blind",
    picks: { human: { kind: "custom", id: customBlind.id }, bot: RANDOM_PICK },
  });
  await page.goto("/");

  const report = page.locator('[data-testid="prior-report"]');
  await page.locator('[data-party="prior"]').click();
  // The hand-pick path exists; this test drives the sample instead (no
  // file chooser needed, and the same code path behind it).
  await expect(page.locator('[data-testid="prior-file"]')).toHaveCount(1);
  await page.locator('[data-testid="prior-sample"]').click();
  // 42 species is the sample table's content, so this also proves the
  // table was really parsed rather than merely stored.
  await expect(report).toContainText("42 species");
  await expect(report).toContainText("Applied");
  await expect(report).not.toContainText("NOT applied");
  await page.locator("dialog.modal .modal-head button").click();
  await expect(page.locator('[data-party="prior"] .party-value')).toHaveText(
    "belief-prior-v0.sample.json",
  );

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

  // Back to the start screen: a stored prior re-probes on open (so the
  // user sees what is loaded without re-picking it), and clears cleanly.
  await page.locator(".battle-screen .quit-btn").click();
  await page.locator('[data-party="prior"]').click();
  await expect(report).toContainText("42 species");
  await page.locator('[data-testid="prior-clear"]').click();
  await expect(page.locator('[data-party="prior"] .party-value')).toHaveText(
    "None",
  );
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

  // Nothing blind is even offered, and the preference is still unwritten:
  // the experiment must not change the default by merely existing.
  expect(
    await page.evaluate(() => localStorage.getItem("nc2000-info-mode")),
  ).toBeNull();
  await expect(page.locator('[data-mode="open"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator('[data-party="bot"] .party-value')).toHaveText(
    pool.teams[4].id,
  );
  await expect(page.locator('[data-party="bot-random"]')).toHaveCount(0);
  await expect(page.locator('[data-party="prior"]')).toHaveCount(0);

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
