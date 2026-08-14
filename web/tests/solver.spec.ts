// `?solver` — the study board.
//
// Not a battle, so nothing the other three suites assert applies: there is
// no game, no opponent draw, no reveal at the end. What has to hold here is
// narrower and, for a teaching tool, more important:
//
//   1. the door: `?solver` opens it, `?solver=0` shuts it, and the product
//      page grows no trace of it (the same regression the blind suite makes
//      about its own experiment — a visitor without the link must get the
//      M12 screen, unchanged);
//   2. a position that is not yet a position cannot be analyzed, and the
//      screen says which part is missing rather than failing at the engine;
//   3. the round trip that matters: describe a board, press analyze, and get
//      every legal option scored — shares summing to 1, sorted by playouts;
//   4. the opponent's HIDDEN information stays hidden. Loading their roster
//      from a pool team copies six species and levels; their sets must not
//      reach the DOM, and no move of theirs may be named as known unless the
//      user typed it into "moves shown". This is the blind contract, and on
//      this screen it is the whole product claim;
//   5. the belief actually uses what it is given: a roster copied from a
//      pool team is identified, which is the difference between the solver
//      reasoning about that team and reasoning about any team at all.
//
// Selectors depended on: [data-testid="solver-own"|"solver-foe"|
// "solver-solve"|"solver-problems"|"solver-results"|"solver-actions"|
// "solver-matrix"|"solver-belief"|"solver-reset"], plus .solver-mon
// [data-species], .solver-active-btn and .solver-team-btn.

import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";

interface PoolTeamJson {
  id: string;
  species: string[];
  levels: number[];
  sets: { species: string; moves: string[]; item?: string }[];
}

const pool = JSON.parse(
  readFileSync(
    new URL("../../data/meta-pool-v0/meta-pool.json", import.meta.url),
    "utf8",
  ),
) as { teams: PoolTeamJson[] };

const MINE = 0;
const THEIRS = 1;

/** Load both rosters from pool teams and send one Pokémon out on each side —
 * the shortest complete position the screen accepts. */
async function fillPosition(page: Page, mine = MINE, theirs = THEIRS) {
  await page.goto("/?solver");
  await expect(page.getByTestId("solver-own")).toBeVisible();

  await page.locator('[data-testid="solver-own"] .solver-panel-head button').click();
  await page.locator(".solver-team-btn").nth(mine).click();
  await expect(page.locator('[data-testid="solver-own"] .solver-mon')).toHaveCount(6);

  await page.locator('[data-testid="solver-foe"] .solver-panel-head button').click();
  await page.locator(".solver-team-btn").nth(theirs).click();
  await expect(page.locator('[data-testid="solver-foe"] .solver-mon')).toHaveCount(6);

  await page.locator('[data-testid="solver-own"] .solver-active-btn').first().click();
  await page.locator('[data-testid="solver-foe"] .solver-active-btn').first().click();
}

test.beforeEach(async ({ page }) => {
  // Positions persist to localStorage on purpose; every case here starts
  // from an empty board so one test's edits cannot explain another's pass.
  await page.goto("/?solver");
  await page.evaluate(() => localStorage.removeItem("nc2000-solver-position"));
});

test("the door opens, closes, and leaves the product page alone", async ({ page }) => {
  await page.goto("/?solver");
  await expect(page.getByTestId("solver-own")).toBeVisible();
  await expect(page.getByTestId("solver-solve")).toBeVisible();

  await page.goto("/?solver=0");
  await expect(page.getByTestId("solver-own")).toHaveCount(0);
  await expect(page.getByRole("button", { name: /start battle|対戦開始/i })).toBeVisible();

  await page.goto("/");
  await expect(page.getByTestId("solver-own")).toHaveCount(0);
  await expect(page.getByTestId("solver-solve")).toHaveCount(0);
  // and nothing on the shipped screen advertises it
  await expect(page.locator("body")).not.toContainText("?solver");
});

test("an unfinished position is refused by name, not by the engine", async ({ page }) => {
  await page.goto("/?solver");
  const solve = page.getByTestId("solver-solve");
  await expect(solve).toBeDisabled();
  await expect(page.getByTestId("solver-problems")).toBeVisible();

  // teams loaded, nobody out yet: still refused, and for the new reason
  await page.locator('[data-testid="solver-own"] .solver-panel-head button').click();
  await page.locator(".solver-team-btn").nth(MINE).click();
  await page.locator('[data-testid="solver-foe"] .solver-panel-head button').click();
  await page.locator(".solver-team-btn").nth(THEIRS).click();
  await expect(solve).toBeDisabled();
  const problems = await page.getByTestId("solver-problems").innerText();
  expect(problems.toLowerCase()).toMatch(/out|場に出/);

  await page.locator('[data-testid="solver-own"] .solver-active-btn').first().click();
  await page.locator('[data-testid="solver-foe"] .solver-active-btn').first().click();
  await expect(page.getByTestId("solver-problems")).toHaveCount(0);
  await expect(solve).toBeEnabled();
});

test("every legal option comes back scored", async ({ page }) => {
  await fillPosition(page);
  await page.getByTestId("solver-solve").click();
  await expect(page.getByTestId("solver-results")).toBeVisible({ timeout: 120_000 });

  const rows = page.locator('[data-testid="solver-actions"] tbody tr');
  // four moves plus the two benched picks
  await expect(rows).toHaveCount(6);

  const data = await page.evaluate(() => {
    const tr = [...document.querySelectorAll('[data-testid="solver-actions"] tbody tr')];
    return tr.map((r) => {
      const cells = [...r.querySelectorAll("td")].map((c) => c.innerText.trim());
      return {
        label: cells[0],
        win: cells[1],
        worst: cells[2],
        mix: cells[3],
        share: cells[4],
        visits: cells[5],
      };
    });
  });
  const num = (s: string) => Number(s.replace(/[^0-9.]/g, ""));
  const visits = data.map((d) => num(d.visits));
  expect(visits.every((v) => v >= 0)).toBe(true);
  // sorted by playouts, descending — the display order IS the ranking
  expect([...visits].sort((a, b) => b - a)).toEqual(visits);
  const shares = data.map((d) => num(d.share));
  expect(Math.abs(shares.reduce((a, b) => a + b, 0) - 100)).toBeLessThan(1.5);
  for (const d of data) {
    const w = num(d.win);
    expect(w).toBeGreaterThanOrEqual(0);
    expect(w).toBeLessThanOrEqual(100);
    // the floor never beats the value against a mixture that contains it
    if (d.worst !== "—") expect(num(d.worst)).toBeLessThanOrEqual(w + 0.05);
  }
  // the position has a stated value, and it is one of the options' equity
  await expect(page.getByTestId("solver-value")).toBeVisible();
  const value = num(await page.getByTestId("solver-value").innerText());
  expect(Math.min(...data.map((d) => Math.abs(num(d.win) - value)))).toBeLessThan(0.15);

  // the joint view is what the marginal hides, so it has to be there
  await expect(page.getByTestId("solver-matrix")).toBeVisible();
  const cols = await page.locator('[data-testid="solver-matrix"] thead th').count();
  expect(cols).toBeGreaterThan(1);
});

test("their sets stay hidden, and only what was typed counts as shown", async ({
  page,
}) => {
  await fillPosition(page);
  await page.getByTestId("solver-solve").click();
  await expect(page.getByTestId("solver-results")).toBeVisible({ timeout: 120_000 });

  const foe = pool.teams[THEIRS];
  const body = await page.locator("body").innerText();

  // Their six species are public at preview and appear; their MOVES are not.
  for (const sp of foe.species) expect(body).toContain(sp);

  // Moves that belong to their sets and to nothing on our side must not be
  // named anywhere as a known fact. The searched line is allowed to name
  // assumed moves — it says so — so this checks the input and score surfaces.
  const ours = new Set(pool.teams[MINE].sets.flatMap((s) => s.moves));
  const theirsOnly = foe.sets.flatMap((s) => s.moves).filter((m) => !ours.has(m));
  const scored = await page.locator('[data-testid="solver-actions"]').innerText();
  for (const m of theirsOnly) expect(scored).not.toContain(m);

  // and the position we typed says nothing was shown
  const revealed = await page.evaluate(() => {
    const spec = JSON.parse(localStorage.getItem("nc2000-solver-position") ?? "{}");
    const foeSide = spec.sides?.[1 - (spec.side ?? 0)];
    return (foeSide?.mons ?? []).flatMap(
      (m: { revealed_moves?: string[] }) => m.revealed_moves ?? [],
    );
  });
  expect(revealed).toEqual([]);
});

test("a roster copied from a pool team is identified", async ({ page }) => {
  await fillPosition(page);
  await page.getByTestId("solver-solve").click();
  await expect(page.getByTestId("solver-results")).toBeVisible({ timeout: 120_000 });
  // Species, level, gender and item presence are all public at preview, and
  // all four are what the belief filters on: a roster that carries them
  // faithfully identifies its team, and one that does not silently drops the
  // solver into set-by-set imputation.
  await expect(page.getByTestId("solver-belief")).toContainText(/identified|特定済み/);
});

test("editing the position drops the answer it no longer explains", async ({ page }) => {
  await fillPosition(page);
  await page.getByTestId("solver-solve").click();
  await expect(page.getByTestId("solver-results")).toBeVisible({ timeout: 120_000 });

  // send a different Pokémon out: the report is about a board that no longer
  // exists, and a stale report beside a live board is the one lie this
  // screen cannot afford
  await page.locator('[data-testid="solver-own"] .solver-active-btn').nth(1).click();
  await expect(page.getByTestId("solver-results")).toHaveCount(0);

  await page.getByTestId("solver-reset").click();
  await expect(page.locator('[data-testid="solver-own"] .solver-mon')).toHaveCount(0);
});
