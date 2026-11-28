// Swappable team pool: in the setup modal one file replaces the team lists
// and the bot's draw — and NOT the bot's belief, which stays the shipped
// opponent prior (docs/TEAM-POOL-REBUILD-PLAN.md: own-team draws independent
// of the opponent belief, custom flow included). A loaded file is an
// explicit override of the catalog, shown on the setup button, and `?nash`
// never plays it.
//
// The thing under test is a replacement, not a list — so every case here
// checks the swap through a second consumer as well as through the button
// that performed it: the start screen's team list (what the user picks
// from) and a full game whose belief chip reports the candidate set the
// WORKER was handed. A test that only read the setup button's caption would
// pass on a swap that changed nothing but a label.
//
// THE FIXTURE IS CUT FROM THE SERVED CATALOG, IN THE PAGE, AND THEN EDITED.
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
// Selectors depended on: [data-party="settings"] with a .party-value, and
// inside the setup modal [data-testid="pool-file"] (an <input type=file>),
// [data-testid="pool-report"], [data-testid="pool-reset"]; plus the
// [data-party="human"] picker, .team-card[data-team] > .team-id /
// .species-chip, [data-testid="belief-chip"], [data-testid="mode-banner"],
// and dialog.modal .modal-head button.

import { expect, test, type Page } from "@playwright/test";
import { Buffer } from "node:buffer";

/** How many catalog parties the fixture slices: the first ones carrying a
 * level-50 mon to bump. Which they are is otherwise irrelevant — the pool
 * file is cut from whatever the app serves. */
const FIXTURE_SIZE = 2;
const FIXTURE_NAME = "two-team-pool.json";

/**
 * The edit that makes each fixture team something the prior does not
 * contain: one mon per team moves from level 50 to level 51.
 *
 * Without it this suite could not tell a real swap from half of one. A
 * verbatim slice IS a catalog party, so every assertion below would read the
 * same if the worker were handed the loaded file — the human would be
 * playing a team the prior explains perfectly well.
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
  /** Party count of the catalog, read at runtime — the "unchanged"
   * baseline for the rejection case and the reset case. */
  catalog: number;
  /** Ids of the sliced teams, in file order. */
  ids: string[];
  /** "Golem L51" per team, in file order: what the bump actually did, for
   * failure messages worth reading. */
  bumped: string[];
  json: string;
}

/**
 * Build a pool file out of the catalog the app is serving right now, then
 * bump one level per team (see BUMP_TO) so the result is a pool of its own.
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
    async ({ size, drop, from, to }) => {
      const get = async (url: string) => {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${url}: ${res.status}`);
        return (await res.json()) as {
          teams: { id: string; sets: FixtureSet[] }[];
        };
      };
      const catalog = await get("/data/team-pool-v4/team-pool.json");
      const prior = await get("/data/belief-pool-v3/belief-pool.json");
      // Only id + sets: the minimum the contract accepts, and the shape the
      // Rust side actually reads (crates/bot/src/preview.rs). Deep copies —
      // the bump below must not reach the signatures it is checked against.
      const teams = catalog.teams
        .filter((t) => t.sets.some((s) => (s.level ?? 55) === from))
        .slice(0, size)
        .map((t) => ({
          id: t.id,
          sets: JSON.parse(JSON.stringify(t.sets)) as FixtureSet[],
        }));
      if (teams.length < size)
        throw new Error(`catalog has fewer than ${size} parties with a level-${from} mon`);

      const bumped = teams.map((t) => {
        const at = t.sets.findIndex((s) => (s.level ?? 55) === from);
        if (at < 0)
          throw new Error(`fixture team ${t.id}: no level-${from} mon to bump`);
        t.sets[at].level = to;
        return `${t.sets[at].species} L${to}`;
      });

      // What the bump has to achieve, checked rather than assumed: no prior
      // team may still answer to a fixture team's public preview. If one
      // did, the worker would find a candidate for the human's party and the
      // belief chip would read the same in both worlds — which is the
      // confusion this fixture exists to remove.
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
      const priorSigs = new Set(prior.teams.map((t) => signature(t.sets)));
      for (const t of teams)
        if (priorSigs.has(signature(t.sets)))
          throw new Error(
            `fixture team ${t.id} still shares a prior team's species/level signature`,
          );

      if (drop !== undefined) teams[drop].sets = teams[drop].sets.slice(0, 5);
      return {
        catalog: catalog.teams.length,
        ids: teams.map((t) => t.id),
        bumped,
        json: JSON.stringify({ teams }),
      };
    },
    { size: FIXTURE_SIZE, drop: opts.drop, from: BUMP_FROM, to: BUMP_TO },
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
    localStorage.removeItem("nc2000-team-pool-v4");
    localStorage.removeItem("nc2000-start-picks");
    localStorage.removeItem("nc2000-custom-teams");
    localStorage.removeItem("nc2000-belief-prior");
  });
}

/** Console errors, plus any request for a baked pair table: no door may
 * read one. */
function guardConsole(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("request", (r) => {
    if (r.url().includes("/preview-tables-v0/"))
      errors.push(`pair-table request: ${r.url()}`);
  });
  return errors;
}

/** The setup button's value line: `<pool> · <prior>`. Everything in
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
test("a pool file replaces the lists, survives a reload, and gives way to the catalog", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  await page.goto("/");
  await expect(page.locator(".start-screen")).toBeVisible();

  const fx = await poolFixture(page);
  const value = setupValue(page);
  const catalogLabel = `Built-in (${fx.catalog} parties)`;
  await expect(value).toContainText(catalogLabel);

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
      localStorage.getItem("nc2000-team-pool-v4"),
    );
    expect(stored, "an accepted pool is persisted").not.toBeNull();
    expect((JSON.parse(stored!) as { name: string }).name).toBe(FIXTURE_NAME);
  });

  await test.step("P3-2: the pool is still there after a reload", async () => {
    await page.reload();
    await expect(page.locator(".start-screen")).toBeVisible();
    await expect(value).toContainText(`${FIXTURE_NAME} (2 teams)`);
    await expectPickerPool(page, fx);
  });

  await test.step("P3-3: the catalog comes back", async () => {
    await page.locator('[data-party="settings"]').click();
    await page.locator('[data-testid="pool-reset"]').click();
    await expect(value).toContainText(catalogLabel);
    await expect(value).not.toContainText(FIXTURE_NAME);
    await closeModal(page);
    await page.locator('[data-party="human"]').click();
    await expect(page.locator("dialog.modal [data-team]")).toHaveCount(
      fx.catalog,
    );
    await closeModal(page);
    expect(
      await page.evaluate(() => localStorage.getItem("nc2000-team-pool-v4")),
    ).toBeNull();
  });

  expect(errors).toEqual([]);
});

// --------------------------------------------------------- contract P3-4
test("a file with an unplayable team changes nothing", async ({ page }) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  await page.goto("/");
  await expect(page.locator(".start-screen")).toBeVisible();

  // Team 0 of the file is five Pokémon; team 1 is a legal (bumped) team.
  // So this also pins the all-or-nothing rule: a file is not partially
  // adopted, and the good team does NOT become a one-team pool.
  const fx = await poolFixture(page, { drop: 0 });
  const value = setupValue(page);
  const catalogLabel = `Built-in (${fx.catalog} parties)`;
  await expect(value).toContainText(catalogLabel);

  await loadPool(page, "five-mons.json", fx.json);
  const report = page.locator('[data-testid="pool-report"]');
  await expect(report).toBeVisible();
  // team-pool.ts's own line (poolErrTeamSize), so the report is the
  // loader's verdict on the actual defect and not a generic failure box.
  await expect(report).toContainText(`Team ${fx.ids[0]}: 5 Pokémon`);
  await expect(report).not.toContainText("accepted");

  await expect(value).toContainText(catalogLabel);
  await closeModal(page);
  await page.locator('[data-party="human"]').click();
  await expect(page.locator("dialog.modal [data-team]")).toHaveCount(
    fx.catalog,
  );
  await closeModal(page);
  expect(
    await page.evaluate(() => localStorage.getItem("nc2000-team-pool-v4")),
  ).toBeNull();
  expect(errors).toEqual([]);
});

// --------------------------------------------------------- contract P3-5
test("a swapped pool plays a full blind game", async ({ page }) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  await page.goto("/");
  await expect(page.locator(".start-screen")).toBeVisible();

  const fx = await poolFixture(page);
  await loadPool(page, FIXTURE_NAME, fx.json);
  await expect(setupValue(page)).toContainText(`${FIXTURE_NAME} (2 teams)`);
  await closeModal(page);
  // Adopting a pool does NOT remount the start screen: app.tsx hands
  // StartScreen a new LoadedPool object and select.tsx re-runs loadPicks in
  // an effect, precisely so the modal holding the load report survives.
  await expect(page.locator('[data-testid="mode-banner"]')).toBeVisible();
  await expect(page.locator('[data-party="settings"]')).toHaveCount(1);
  await expect(page.locator('[data-party="bot"]')).toHaveCount(0);

  // The opponent is drawn from the pool that was loaded — a pool the wasm
  // side could not read would fail here, at battle construction, not on
  // screen.
  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();
  await choosePreview(page);

  // The belief chip is the only place the WORKER's candidate set becomes
  // visible, and it is what makes this case more than a re-run of P3-1.
  // Both fixture teams carry the level bump, so whichever one the human
  // drew is a party only the loaded file can explain:
  //   worker holding the shipped prior -> no candidate survives, the belief
  //                                       falls back and reads "off-pool";
  //   worker holding the loaded pool   -> "1 candidate" — the file leaking
  //                                       into the belief, the defect.
  // The two worlds differ by one word, and only because of the bump.
  await expect(page.locator(".move-btn").first()).toBeVisible({
    timeout: 90_000,
  });
  const chip = page.locator('[data-testid="belief-chip"]');
  await expect(chip).toBeVisible();
  await expect(chip).toContainText("off-pool");

  await playToOutcome(page);
  await expect(page.locator(".end-banner")).toBeVisible();

  expect(errors).toEqual([]);
});
