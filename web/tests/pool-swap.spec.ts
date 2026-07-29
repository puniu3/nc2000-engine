// Swappable team pool (UI revision P2/P3): one file replaces the pool
// everywhere the pool is read.
//
// The thing under test is a replacement, not a list — so every case here
// checks the swap through a second consumer as well as through the button
// that performed it: the start screen's team list (what the user picks
// from), the baked-pair-table decision inside game.tsx (what the engine
// side indexes by pool rank), and a full blind game (the bot's draw and,
// behind it, the belief's candidate set). A test that only read the pool
// button's caption would pass on a swap that changed nothing but a label.
//
// THE FIXTURE IS CUT FROM THE SERVED POOL, IN THE PAGE. A pool file
// written out in this repo would encode this suite's idea of the format
// and would keep passing after the app's idea of it moved; slicing the
// file the app itself fetches means the fixture cannot drift, and it also
// exercises the loader's derivation path for real — the slice carries only
// `{id, sets}` (the documented minimum), so the species/levels/tier/rank
// the team cards render must have been derived from the canonicalized sets.
//
// The rejected file is rejected for a reason THIS loader exists to catch:
// a team of five. It is valid JSON and a well-formed team object, so no
// parser refuses it — only the format's own "a party is exactly 6" rule
// does. Malformed JSON would prove nothing about the loader.
//
// Written against the contract's testids rather than against a running
// app (the UI lands in parallel). Selectors depended on (integration must
// check them): [data-party="pool"] with a .party-value, and inside the
// pool modal [data-testid="pool-file"] (an <input type=file>),
// [data-testid="pool-report"], [data-testid="pool-reset"]; plus the
// shipped [data-party="human"] picker, .team-card[data-team] > .team-id /
// .species-chip, and dialog.modal .modal-head button.

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

interface Fixture {
  /** Team count of the bundled pool, read at runtime — the "unchanged"
   * baseline for the rejection case and the reset case. */
  bundled: number;
  /** Ids of the sliced teams, in file order. */
  ids: string[];
  json: string;
}

/**
 * Build a pool file out of the pool the app is serving right now. `drop`
 * cuts one Pokémon out of the team at that index, producing a file that is
 * well-formed everywhere except where the format has an opinion.
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
    async ({ indexes, drop }) => {
      const res = await fetch("/data/meta-pool-v0/meta-pool.json");
      if (!res.ok) throw new Error(`pool fetch failed: ${res.status}`);
      const bundled = (await res.json()) as {
        teams: { id: string; sets: unknown[] }[];
      };
      // Only id + sets: the minimum the contract accepts, and the shape the
      // Rust side actually reads (crates/bot/src/preview.rs).
      const teams = indexes.map((i) => ({
        id: bundled.teams[i].id,
        sets: bundled.teams[i].sets.slice(),
      }));
      if (drop !== undefined) teams[drop].sets = teams[drop].sets.slice(0, 5);
      return {
        bundled: bundled.teams.length,
        ids: teams.map((t) => t.id),
        json: JSON.stringify({ teams }),
      };
    },
    { indexes: FIXTURE_TEAMS, drop: opts.drop },
  );
}

/** Same one-shot seeding idiom as the other suites: the init script runs on
 * every navigation, and the sessionStorage flag keeps the reload in the
 * first test from wiping the pool that test just loaded through the UI. */
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
 * bake could come back.) */
function trackPairRequests(page: Page): string[] {
  const seen: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/preview-tables-v0/pair-")) seen.push(r.url());
  });
  return seen;
}

/** Open the pool modal and hand the file input a pool file. Playwright
 * sets the input's files directly, so it works on the sr-only input the
 * label wraps (the prior panel's pattern) without a native file dialog. */
async function loadPool(page: Page, name: string, json: string) {
  await page.locator('[data-party="pool"]').click();
  await page.locator('[data-testid="pool-file"]').setInputFiles({
    name,
    mimeType: "application/json",
    buffer: Buffer.from(json, "utf8"),
  });
}

/** The pool modal stays up after a load, accepted or refused, so its report
 * can be read; the party pickers close themselves on a pick. Tolerating
 * both is what lets one helper follow every modal in this file. */
async function closeModal(page: Page) {
  const close = page.locator("dialog.modal .modal-head button");
  if ((await close.count()) > 0) await close.click();
}

/** The human party picker's pool list — the swap's second consumer, and
 * the one the player actually chooses from. */
async function expectPickerPool(page: Page, ids: string[]) {
  await page.locator('[data-party="human"]').click();
  const cards = page.locator("dialog.modal [data-team]");
  await expect(cards).toHaveCount(ids.length);
  await expect(cards.locator(".team-id")).toHaveText(ids);
  // Six species chips on the first card: the fixture carries no `species`
  // or `levels` field, so anything rendered here was derived from the
  // canonicalized sets.
  await expect(cards.first().locator(".species-chip")).toHaveCount(6);
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
  await page.goto("/");
  await expect(page.locator(".start-screen")).toBeVisible();

  const fx = await poolFixture(page);
  const poolValue = page.locator('[data-party="pool"] .party-value');
  const bundledLabel = `Bundled (${fx.bundled} teams)`;
  await expect(poolValue).toHaveText(bundledLabel);

  await test.step("P3-1: the file becomes the pool", async () => {
    await loadPool(page, FIXTURE_NAME, fx.json);
    await expect(poolValue).toHaveText(`${FIXTURE_NAME} (2 teams)`);
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
    await expectPickerPool(page, fx.ids);
    const stored = await page.evaluate(() =>
      localStorage.getItem("nc2000-team-pool"),
    );
    expect(stored, "an accepted pool is persisted").not.toBeNull();
    expect((JSON.parse(stored!) as { name: string }).name).toBe(FIXTURE_NAME);
  });

  await test.step("P3-1: a swapped pool never asks for a baked pair", async () => {
    // Both sides are drawn from the loaded pool, so both carry a poolIdx
    // and the historical condition (pool vs pool) is satisfied — the only
    // thing between this game and a request for pair-00-01.json is
    // game.tsx's poolIsCustom guard.
    await page.getByRole("button", { name: "Start battle" }).click();
    await expect(page.locator(".preview-screen")).toBeVisible();
    expect(pairRequests).toEqual([]);
    await page.locator(".preview-actions .quit-btn").click();
    await expect(page.locator(".start-screen")).toBeVisible();
  });

  await test.step("P3-2: the pool is still there after a reload", async () => {
    await page.reload();
    await expect(page.locator(".start-screen")).toBeVisible();
    await expect(poolValue).toHaveText(`${FIXTURE_NAME} (2 teams)`);
    await expectPickerPool(page, fx.ids);
  });

  await test.step("P3-3: the bundled pool comes back", async () => {
    await page.locator('[data-party="pool"]').click();
    await page.locator('[data-testid="pool-reset"]').click();
    await expect(poolValue).toHaveText(bundledLabel);
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
  await page.goto("/");
  await expect(page.locator(".start-screen")).toBeVisible();

  // Team 0 of the file is five Pokémon; team 1 is a verbatim pool team and
  // perfectly legal. So this also pins the all-or-nothing rule: a file is
  // not partially adopted, and the good team does NOT become a one-team
  // pool.
  const fx = await poolFixture(page, { drop: 0 });
  const poolValue = page.locator('[data-party="pool"] .party-value');
  const bundledLabel = `Bundled (${fx.bundled} teams)`;
  await expect(poolValue).toHaveText(bundledLabel);

  await loadPool(page, "five-mons.json", fx.json);
  const report = page.locator('[data-testid="pool-report"]');
  await expect(report).toBeVisible();
  // team-pool.ts's own line (poolErrTeamSize), so the report is the
  // loader's verdict on the actual defect and not a generic failure box.
  await expect(report).toContainText(`Team ${fx.ids[0]}: 5 Pokémon`);
  await expect(report).not.toContainText("accepted");

  await expect(poolValue).toHaveText(bundledLabel);
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
  await expect(page.locator('[data-party="pool"] .party-value')).toHaveText(
    `${FIXTURE_NAME} (2 teams)`,
  );
  await closeModal(page);
  // The mode lives in the URL, and adopting a pool remounts the start
  // screen — so this is also where a remount that lost the mode would show.
  await expect(page.locator('[data-testid="mode-banner"]')).toBeVisible();
  await expect(page.locator('[data-party="bot-random"]')).toBeVisible();

  // Blind draws the opponent from the pool that was loaded, and the
  // searcher is handed the same normalized JSON — a pool the wasm side
  // could not read would fail here, at battle construction, not on screen.
  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();
  await choosePreview(page);
  await playToOutcome(page);
  await expect(page.locator(".end-banner")).toBeVisible();

  // Blind alone would skip the pair fetch; so would the custom pool. Both
  // skips are in force here, and the console guard below is unexempted
  // precisely because neither can be missing.
  expect(pairRequests).toEqual([]);
  expect(errors).toEqual([]);
});
