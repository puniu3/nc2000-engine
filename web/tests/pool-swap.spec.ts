// Swappable team pool (UI revision P2/P3, retargeted onto the Blind setup
// modal): one file replaces the pool everywhere the pool is read — and
// only under `?blind`.
//
// The thing under test is a replacement, not a list — so every case here
// checks the swap through a second consumer as well as through the button
// that performed it: the start screen's team list (what the user picks
// from), the baked-pair-table decision inside game.tsx (what the engine
// side indexes by pool rank), and a full blind game whose belief chip
// reports the candidate set the WORKER was handed. A test that only read
// the setup button's caption would pass on a swap that changed nothing but
// a label.
//
// The pool panel is now the first half of the one `[data-party="settings"]`
// modal, and that button exists only in blind mode — so every case enters
// at /?blind. Open mode plays the bundled pool whatever is stored, which is
// the last case in this file: a hidden control that still lets a saved file
// rewrite the public team lists is exactly the trap the simplification pass
// was closing, and it would be invisible from the open screen.
//
// THE FIXTURE IS CUT FROM THE SERVED POOL, IN THE PAGE, AND THEN EDITED.
// A pool file written out in this repo would encode this suite's idea of
// the format and would keep passing after the app's idea of it moved;
// slicing the file the app itself fetches means the fixture cannot drift,
// and it also exercises the loader's derivation path for real — the slice
// carries only `{id, sets}` (the documented minimum), so the
// species/levels/tier/rank the team cards render must have been derived
// from the canonicalized sets. The edit (BUMP_TO, below) is what makes the
// slice a fixture instead of a copy.
//
// The rejected file is rejected for a reason THIS loader exists to catch:
// a team of five. It is valid JSON and a well-formed team object, so no
// parser refuses it — only the format's own "a party is exactly 6" rule
// does. Malformed JSON would prove nothing about the loader.
//
// Written against the contract's testids rather than against a running
// app (the UI lands in parallel). Selectors depended on (integration must
// check them): [data-party="settings"] with a .party-value, and inside the
// setup modal [data-testid="pool-file"] (an <input type=file>),
// [data-testid="pool-report"], [data-testid="pool-reset"]; plus the shipped
// [data-party="human"] picker, .team-card[data-team] > .team-id /
// .species-chip, [data-testid="belief-chip"], [data-testid="mode-banner"],
// and dialog.modal .modal-head button. Asserted ABSENT in open mode:
// [data-party="settings"], [data-testid="mode-banner"].

import { expect, test, type Page } from "@playwright/test";
import { Buffer } from "node:buffer";

/** Two teams out of the bundled pool, by index. Not the first two: these
 * are the pair the other e2e suites play their full games with, chosen
 * there for being attack-heavy and decisive, and the blind game at the
 * bottom of this file has to finish inside the suite budget. Which two
 * they are is otherwise irrelevant — the pool file is cut from whatever
 * the app serves. */
const FIXTURE_TEAMS = [4, 29];
const FIXTURE_NAME = "two-team-pool.json";

/**
 * The edit that makes each fixture team something the bundled pool does not
 * contain: one mon per team moves from level 50 to level 51.
 *
 * Without it this suite could not tell a real swap from half of one. A
 * verbatim slice IS a bundled team, so every assertion below would read the
 * same if the worker were still handed `bundled.poolJson` — the human would
 * be playing a team the bundled pool explains perfectly well.
 *
 * Level, and not the held item: the belief's preview filter matches a
 * candidate on (species, level, gender) and on item PRESENCE only
 * (`build_refs`, crates/bot/src/belief.rs), so trading Leftovers for a
 * different item is invisible to it until the item reveals itself mid-game,
 * whereas a level is public from team preview onward — which is where the
 * chip is read.
 *
 * +1 survives canonicalization and every rule that touches levels: the
 * format's range is 50..=55 (MIN_LEVEL/MAX_LEVEL, crates/engine/src/
 * validate.rs), species evolution floors and level-up move floors are
 * minima that a bump can only satisfy harder, and the two level-sum caps
 * bind from below — the three lightest still sum to 150 with the party's
 * other level-50 mons, so a legal 3-pick still exists (and the picker's own
 * fitsLevelCap keeps offering one).
 */
const BUMP_FROM = 50;
const BUMP_TO = 51;

/** Only the fields the fixture builder itself reads; the rest of each set
 * rides along through the JSON round-trip untouched. */
interface FixtureSet {
  species: string;
  level?: number;
}

interface Fixture {
  /** Team count of the bundled pool, read at runtime — the "unchanged"
   * baseline for the rejection case and the reset case. */
  bundled: number;
  /** Ids of the sliced teams, in file order. */
  ids: string[];
  /** "Golem L51" per team, in file order: what the bump actually did, for
   * failure messages worth reading. */
  bumped: string[];
  json: string;
}

/**
 * Build a pool file out of the pool the app is serving right now, then bump
 * one level per team (see BUMP_TO) so the result is a pool of its own.
 * `drop` cuts one Pokémon out of the team at that index, producing a file
 * that is well-formed everywhere except where the format has an opinion.
 *
 * Runs in the page so the bytes come from the same URL the app loads
 * (vite.config.ts maps `/data/` to the repo's data dir in both the dev and
 * the preview server).
 */
async function poolFixture(
  page: Page,
  opts: { drop?: number } = {},
): Promise<Fixture> {
  return page.evaluate(
    async ({ indexes, drop, from, to }) => {
      const res = await fetch("/data/meta-pool-v0/meta-pool.json");
      if (!res.ok) throw new Error(`pool fetch failed: ${res.status}`);
      const bundled = (await res.json()) as {
        teams: { id: string; sets: FixtureSet[] }[];
      };
      // Only id + sets: the minimum the contract accepts, and the shape the
      // Rust side actually reads (crates/bot/src/preview.rs). Deep copies —
      // the bump below must not reach the signatures it is checked against.
      const teams = indexes.map((i) => ({
        id: bundled.teams[i].id,
        sets: JSON.parse(JSON.stringify(bundled.teams[i].sets)) as FixtureSet[],
      }));

      const bumped = teams.map((t) => {
        const at = t.sets.findIndex((s) => (s.level ?? 55) === from);
        if (at < 0)
          throw new Error(`fixture team ${t.id}: no level-${from} mon to bump`);
        t.sets[at].level = to;
        return `${t.sets[at].species} L${to}`;
      });

      // What the bump has to achieve, checked rather than assumed: no
      // bundled team may still answer to a fixture team's public preview.
      // If one did, a worker left holding the bundled pool would find a
      // candidate for the human's party and the belief chip would read the
      // same in both worlds — which is the confusion this fixture exists to
      // remove.
      const signature = (sets: FixtureSet[]) =>
        sets
          .map(
            (s) =>
              `${s.species.toLowerCase().replace(/[^a-z0-9]/g, "")}@${
                s.level ?? 55
              }`,
          )
          .sort()
          .join(",");
      const bundledSigs = new Set(bundled.teams.map((t) => signature(t.sets)));
      for (const t of teams)
        if (bundledSigs.has(signature(t.sets)))
          throw new Error(
            `fixture team ${t.id} still shares a bundled team's species/level signature`,
          );

      if (drop !== undefined) teams[drop].sets = teams[drop].sets.slice(0, 5);
      return {
        bundled: bundled.teams.length,
        ids: teams.map((t) => t.id),
        bumped,
        json: JSON.stringify({ teams }),
      };
    },
    { indexes: FIXTURE_TEAMS, drop: opts.drop, from: BUMP_FROM, to: BUMP_TO },
  );
}

/** Same one-shot seeding idiom as the other suites: the init script runs on
 * every navigation, and the sessionStorage flag keeps the reload in the
 * first test — and the goto("/") in the last one — from wiping the pool
 * the test just loaded through the UI. */
async function seedStorage(page: Page) {
  await page.addInitScript(() => {
    if (sessionStorage.getItem("nc2000-e2e-seeded") === "1") return;
    sessionStorage.setItem("nc2000-e2e-seeded", "1");
    localStorage.setItem("nc2000-locale", "en");
    localStorage.removeItem("nc2000-team-pool");
    localStorage.removeItem("nc2000-start-picks");
    localStorage.removeItem("nc2000-custom-teams");
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

/** Requests for a baked pair table. A swapped pool must never produce one:
 * the baked files are indexed by the BUNDLED pool's rank, so pair-04-29
 * would describe two entirely different teams. (Nothing is baked today —
 * `data/preview-tables-v0/` is README-only — so a request would also 404
 * into the console error list below. Both facts are asserted, because the
 * bake could come back.) Two guards suppress the fetch here, not one:
 * game.tsx skips it for a custom pool AND for blind, and since the pool
 * control is blind-only there is no longer any way to reach a custom pool
 * in open mode — so this tracker no longer isolates the poolIsCustom guard,
 * it only holds the outcome that guard exists for. */
function trackPairRequests(page: Page): string[] {
  const seen: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/preview-tables-v0/pair-")) seen.push(r.url());
  });
  return seen;
}

/** The Blind setup button's value line: `<pool> · <prior>`. Everything in
 * this file reads the pool half of it, so the assertions are containment,
 * not equality — the prior half is B4-5's business. */
function setupValue(page: Page) {
  return page.locator('[data-party="settings"] .party-value');
}

/** Open the setup modal and hand the pool input a file. Playwright sets the
 * input's files directly, so it works on the sr-only input the label wraps
 * (the prior panel's pattern) without a native file dialog. */
async function loadPool(page: Page, name: string, json: string) {
  await page.locator('[data-party="settings"]').click();
  await page.locator('[data-testid="pool-file"]').setInputFiles({
    name,
    mimeType: "application/json",
    buffer: Buffer.from(json, "utf8"),
  });
}

/** The setup modal stays up after a load, accepted or refused, so its report
 * can be read; the party pickers close themselves on a pick. Tolerating
 * both is what lets one helper follow every modal in this file. */
async function closeModal(page: Page) {
  const close = page.locator("dialog.modal .modal-head button");
  if ((await close.count()) > 0) await close.click();
}

/** The human party picker's pool list — the swap's second consumer, and
 * the one the player actually chooses from. */
async function expectPickerPool(page: Page, fx: Fixture) {
  await page.locator('[data-party="human"]').click();
  const cards = page.locator("dialog.modal [data-team]");
  await expect(cards).toHaveCount(fx.ids.length);
  await expect(cards.locator(".team-id")).toHaveText(fx.ids);
  // Six species chips on the first card: the fixture carries no `species`
  // or `levels` field, so anything rendered here was derived from the
  // canonicalized sets.
  await expect(cards.first().locator(".species-chip")).toHaveCount(6);
  // And the derivation carried the bump through — the level the loader
  // wrote back is the edited one, not the file's own claim and not a
  // clamped-away 50. Without this, a canonicalizer that quietly normalized
  // levels would leave the belief-chip case below testing nothing.
  for (let i = 0; i < fx.ids.length; i++)
    await expect(cards.nth(i), `team ${fx.ids[i]} shows ${fx.bumped[i]}`)
      .toContainText(`L${BUMP_TO}`);
  await closeModal(page);
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
    if ((await moves.count()) > 0) {
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

// Not describe.configure({ mode: "serial" }): each test seeds its own
// storage in a fresh context, so an early failure must not hide the rest.

// ------------------------------------------------------- contract P3-1/2/3
// One session, because case 2 is "it is still there after a reload" — it
// has nothing to be still there unless case 1 ran in the same browser
// context first.
test("a pool file replaces the pool, survives a reload, and gives way to the bundled one", async ({
  page,
}) => {
  const errors = guardConsole(page);
  const pairRequests = trackPairRequests(page);
  await seedStorage(page);
  await page.goto("/?blind");
  await expect(page.locator(".start-screen")).toBeVisible();

  const fx = await poolFixture(page);
  const value = setupValue(page);
  const bundledLabel = `Bundled (${fx.bundled} teams)`;
  await expect(value).toContainText(bundledLabel);

  await test.step("P3-1: the file becomes the pool", async () => {
    await loadPool(page, FIXTURE_NAME, fx.json);
    await expect(value).toContainText(`${FIXTURE_NAME} (2 teams)`);
    // The panel is still up and says what it did. Two clauses of the
    // contract collided here — the modal is told to stay open with its
    // result, and app.tsx was told to key the start screen by pool name,
    // which would remount the screen and take the modal with it. It was
    // settled in favour of the modal (select.tsx reconciles the pinned
    // picks itself instead of being remounted), so the report surviving
    // adoption is an assertion, not a bonus.
    await expect(page.locator('[data-testid="pool-report"]')).toContainText(
      "2 teams accepted",
    );
    await closeModal(page);
    await expectPickerPool(page, fx);
    const stored = await page.evaluate(() =>
      localStorage.getItem("nc2000-team-pool"),
    );
    expect(stored, "an accepted pool is persisted").not.toBeNull();
    expect((JSON.parse(stored!) as { name: string }).name).toBe(FIXTURE_NAME);
  });

  await test.step("P3-1: a swapped pool never asks for a baked pair", async () => {
    // Both sides are drawn from the loaded pool, so both carry a poolIdx
    // and the historical condition (pool vs pool) is satisfied; the baked
    // files are indexed by bundled rank, and pair-04-29 under this pool
    // names two teams that are not these.
    await page.getByRole("button", { name: "Start battle" }).click();
    await expect(page.locator(".preview-screen")).toBeVisible();
    expect(pairRequests).toEqual([]);
    await page.locator(".preview-actions .quit-btn").click();
    await expect(page.locator(".start-screen")).toBeVisible();
  });

  await test.step("P3-2: the pool is still there after a reload", async () => {
    await page.reload();
    await expect(page.locator(".start-screen")).toBeVisible();
    await expect(value).toContainText(`${FIXTURE_NAME} (2 teams)`);
    await expectPickerPool(page, fx);
  });

  await test.step("P3-3: the bundled pool comes back", async () => {
    await page.locator('[data-party="settings"]').click();
    await page.locator('[data-testid="pool-reset"]').click();
    await expect(value).toContainText(bundledLabel);
    await expect(value).not.toContainText(FIXTURE_NAME);
    await closeModal(page);
    await page.locator('[data-party="human"]').click();
    await expect(page.locator("dialog.modal [data-team]")).toHaveCount(
      fx.bundled,
    );
    await closeModal(page);
    expect(
      await page.evaluate(() => localStorage.getItem("nc2000-team-pool")),
    ).toBeNull();
  });

  expect(errors).toEqual([]);
});

// --------------------------------------------------------- contract P3-4
test("a file with an unplayable team changes nothing", async ({ page }) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  await page.goto("/?blind");
  await expect(page.locator(".start-screen")).toBeVisible();

  // Team 0 of the file is five Pokémon; team 1 is a legal (bumped) team.
  // So this also pins the all-or-nothing rule: a file is not partially
  // adopted, and the good team does NOT become a one-team pool.
  const fx = await poolFixture(page, { drop: 0 });
  const value = setupValue(page);
  const bundledLabel = `Bundled (${fx.bundled} teams)`;
  await expect(value).toContainText(bundledLabel);

  await loadPool(page, "five-mons.json", fx.json);
  const report = page.locator('[data-testid="pool-report"]');
  await expect(report).toBeVisible();
  // team-pool.ts's own line (poolErrTeamSize), so the report is the
  // loader's verdict on the actual defect and not a generic failure box.
  await expect(report).toContainText(`Team ${fx.ids[0]}: 5 Pokémon`);
  await expect(report).not.toContainText("accepted");

  await expect(value).toContainText(bundledLabel);
  await closeModal(page);
  await page.locator('[data-party="human"]').click();
  await expect(page.locator("dialog.modal [data-team]")).toHaveCount(
    fx.bundled,
  );
  await closeModal(page);
  expect(
    await page.evaluate(() => localStorage.getItem("nc2000-team-pool")),
  ).toBeNull();
  expect(errors).toEqual([]);
});

// --------------------------------------------------------- contract P3-5
test("a swapped pool plays a full blind game", async ({ page }) => {
  const errors = guardConsole(page);
  const pairRequests = trackPairRequests(page);
  await seedStorage(page);
  await page.goto("/?blind");
  await expect(page.locator(".start-screen")).toBeVisible();

  const fx = await poolFixture(page);
  await loadPool(page, FIXTURE_NAME, fx.json);
  await expect(setupValue(page)).toContainText(`${FIXTURE_NAME} (2 teams)`);
  await closeModal(page);
  // Adopting a pool does NOT remount the start screen: app.tsx hands
  // StartScreen a new LoadedPool object and select.tsx re-runs loadPicks in
  // an effect, precisely so the modal holding the load report survives. The
  // mode is not state at all — app.tsx reads `?blind` once at module load —
  // so what this checks is the narrower thing that could still go wrong: a
  // pool swap that dropped the blind-only chrome from the re-render.
  await expect(page.locator('[data-testid="mode-banner"]')).toBeVisible();
  await expect(page.locator('[data-party="settings"]')).toHaveCount(1);
  await expect(page.locator('[data-party="bot"]')).toHaveCount(0);

  // Blind draws the opponent from the pool that was loaded, and the
  // searcher is handed the same normalized JSON — a pool the wasm side
  // could not read would fail here, at battle construction, not on screen.
  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();
  await choosePreview(page);

  // The belief chip is the only place the WORKER's pool becomes visible,
  // and it is what makes this case more than a re-run of P3-1. Both fixture
  // teams carry the level bump, so whichever one the human drew is a party
  // no bundled team can explain:
  //   worker holding the loaded pool  -> exactly one candidate survives the
  //                                      preview filter, "1 candidate";
  //   worker holding the bundled pool -> no candidate survives, the belief
  //                                      falls back and reads "off-pool".
  // The two worlds differ by one word, and only because of the bump — with
  // a verbatim slice both of them would say "1 candidate".
  await expect(page.locator(".move-btn").first()).toBeVisible({
    timeout: 90_000,
  });
  const chip = page.locator('[data-testid="belief-chip"]');
  await expect(chip).toBeVisible();
  await expect(chip).toHaveText("bot's read: 1 candidate");

  await playToOutcome(page);
  await expect(page.locator(".end-banner")).toBeVisible();

  // Blind alone would skip the pair fetch; so would the custom pool. Both
  // skips are in force here, and the console guard below is unexempted
  // precisely because neither can be missing.
  expect(pairRequests).toEqual([]);
  expect(errors).toEqual([]);
});

// ------------------------------------- open mode is pinned to the bundled pool
test("a pool loaded in blind never reaches the open screen", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  await page.goto("/?blind");
  await expect(page.locator(".start-screen")).toBeVisible();

  const fx = await poolFixture(page);
  await loadPool(page, FIXTURE_NAME, fx.json);
  await expect(setupValue(page)).toContainText(`${FIXTURE_NAME} (2 teams)`);
  await closeModal(page);
  await expectPickerPool(page, fx);

  // The same browser, one query string later. A stored pool that kept
  // rewriting the public team lists would be the worst version of this
  // feature: the visitor sees two unfamiliar teams, and the control that
  // did it — and the reset button that would undo it — are not on the
  // screen. So open plays the bundled pool, and says nothing about pools.
  await page.goto("/");
  await expect(page.locator(".start-screen")).toBeVisible();
  await expect(page.locator('[data-party="settings"]')).toHaveCount(0);
  await expect(page.locator('[data-testid="mode-banner"]')).toHaveCount(0);
  await expect(page.locator('[data-party="bot"]')).toHaveCount(1);
  await page.locator('[data-party="human"]').click();
  await expect(page.locator("dialog.modal [data-team]")).toHaveCount(
    fx.bundled,
  );
  await closeModal(page);
  // Not played, not deleted: the record is the user's, and open mode is a
  // policy about what it plays, not a cleanup pass. The trip back proves
  // it survived — and that this is the same file, not a re-derived label.
  expect(
    await page.evaluate(() => localStorage.getItem("nc2000-team-pool")),
  ).not.toBeNull();

  await page.goto("/?blind");
  await expect(setupValue(page)).toContainText(`${FIXTURE_NAME} (2 teams)`);
  await expectPickerPool(page, fx);
  expect(errors).toEqual([]);
});
