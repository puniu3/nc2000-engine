// `?nash` — META-NASH v1's conclusion mode.
//
// Nash is blind play with the opponent's team replaced by a draw from the
// solved three-team mixture (data/meta-nash-v1/pool-artifact.json) and every
// control removed. blind.spec.ts already owns the blind information
// contract — the foe's sets never reach the DOM, the reveal at the end —
// and nash rides on exactly the same machinery (info-mode.ts maps the nash
// door onto InfoMode "blind", so game.tsx cannot tell the two apart). What
// is left for this suite is the difference, which is four claims:
//
//   1. the door: `?nash` opens it, `?nash=0` shuts it, and `?blind` is not
//      it — the blind screen keeps its setup button and grows no mixture
//      row, so the two experiments cannot be confused for one another;
//   2. the mixture on screen is the artifact's, weights and all, read-only
//      and set-free — it is a readout, and the one thing it must never
//      become is a picker;
//   3. the team actually drawn is one of the three arms, matched species
//      and level for species and level against the file;
//   4. a belief prior sitting in localStorage from an earlier `?blind`
//      visit does NOT reach a nash game. That is the mode's "one shipped
//      configuration" promise, and it is the only one of the four that
//      state left over from another door could silently break.
//   5. the belief candidate pool is the belief-pool-v1 artifact
//      (EXP-PRIOR-EXPLOIT v1): the nash door fetches it, the other doors
//      never do, and it is load-bearing — a nash page that cannot get it
//      fails closed instead of quietly playing under the plainer prior.
//
// (4) is the reason this suite plays a whole game against a hand-mixed
// off-pool party: the prior only ever governs the fallback roster, so a
// pool opponent would leave the chip dead for reasons that have nothing to
// do with nash. blind.spec.ts proves the same seed DOES light the chip
// under `?blind`; the contrast is the assertion.
//
// Selectors depended on: [data-testid="mode-banner"|"nash-mix"|
// "belief-chip"], [data-party="human"|"bot"|"nash"|"settings"], and the
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
    new URL("../../data/meta-nash-v1/pool-artifact.json", import.meta.url),
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

/** Must be empty, or the bot identifies the party by signature, the belief
 * never falls back, and a dead prior chip would "pass" for the wrong
 * reason. Asserted in the test that depends on it. */
const mixTwins = pool.teams
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
      localStorage.removeItem("nc2000-team-pool");
      if (record) {
        localStorage.setItem("nc2000-custom-teams", JSON.stringify([record]));
        // Pin it on the human side. The opponent half of the record is
        // ignored by every blind-family door, nash included — the foe is
        // drawn, never picked.
        localStorage.setItem(
          "nc2000-start-picks",
          JSON.stringify({
            human: { kind: "custom", id: record.id },
            bot: { kind: "random" },
          }),
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

test("the nash door opens only on ?nash, and carries no controls", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);

  await page.goto("/?nash");
  await expect(page.locator(".start-title")).toBeVisible();
  await expect(page.locator('[data-testid="mode-banner"]')).toContainText(
    "solved mixture",
  );
  // Start / banner / your party / the mixture row. No Blind setup: nash
  // ships one configuration, so there is nothing on this screen to change.
  await expect(page.locator('[data-party="nash"]')).toHaveCount(1);
  await expect(page.locator('[data-party="human"]')).toHaveCount(1);
  await expect(page.locator('[data-party="settings"]')).toHaveCount(0);
  await expect(page.locator('[data-party="bot"]')).toHaveCount(0);
  await expect(page.locator(".start-col .party-btn")).toHaveCount(2);

  // The row's own line carries the whole distribution, so the odds are
  // readable without opening anything.
  const row = page.locator('[data-party="nash"] .party-value');
  for (const t of artifact.teams) await expect(row).toContainText(t.id);

  // `?nash=0` is the shipped open screen — banner gone, opponent row back,
  // and not a word about either experiment.
  await page.goto("/?nash=0");
  await expect(page.locator('[data-testid="mode-banner"]')).toHaveCount(0);
  await expect(page.locator('[data-party="nash"]')).toHaveCount(0);
  await expect(page.locator('[data-party="bot"]')).toHaveCount(1);

  // And `?blind` is still itself: setup button, no mixture row.
  await page.goto("/?blind");
  await expect(page.locator('[data-party="settings"]')).toHaveCount(1);
  await expect(page.locator('[data-party="nash"]')).toHaveCount(0);

  expect(errors).toEqual([]);
});

test("the mixture panel is the artifact, read-only and set-free", async ({
  page,
}) => {
  const errors = guardConsole(page);
  await seedStorage(page);
  await page.goto("/?nash");
  await page.locator('[data-party="nash"]').click();

  const panel = page.locator('[data-testid="nash-mix"]');
  await expect(panel).toBeVisible();
  await expect(panel.locator(".team-card")).toHaveCount(artifact.teams.length);

  // Weights are the file's, renormalized (the shipped three sum to 0.998),
  // and shown to one decimal.
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

  // A readout, not a picker: no button, no pressed state, and no set body
  // anywhere — the sets are blind until a game ends, exactly as they are
  // for any other foe.
  await expect(panel.locator("button")).toHaveCount(0);
  await expect(panel.locator("[aria-pressed]")).toHaveCount(0);
  await expect(panel.locator(".set-detail")).toHaveCount(0);
  await expect(panel.locator("[data-move]")).toHaveCount(0);
  await expect(panel.locator("[data-item]")).toHaveCount(0);

  expect(errors).toEqual([]);
});

test("the drawn opponent is one arm of the mixture, and no prior reaches the game", async ({
  page,
}) => {
  const errors = guardConsole(page);
  // A prior IS in storage, left there by an earlier `?blind` visit.
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

  // The belief is live (the mixed party is off-pool, so it falls back) —
  // and the prior half of the chip is absent, because nash never handed
  // the searcher a table. blind.spec.ts is where that same record lights
  // it up; here its silence is the point.
  const chip = page.locator('[data-testid="belief-chip"]');
  await expect(chip).toBeVisible();
  await expect(chip).toContainText("off-pool");
  await expect(chip).not.toContainText("prior:");

  expect(errors).toEqual([]);
});

test("the nash belief pool is belief-pool-v1, exclusive to the door and load-bearing", async ({
  page,
}) => {
  // (a) the nash door fetches the artifact and boots on it.
  const errors = guardConsole(page);
  const beliefUrls: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("belief-pool-v1/belief-pool.json"))
      beliefUrls.push(r.url());
  });
  const gotBelief = page.waitForResponse(
    (r) => r.url().includes("belief-pool-v1/belief-pool.json") && r.ok(),
  );
  await page.goto("/?nash");
  await gotBelief;
  await expect(page.locator('[data-testid="nash-mix"]')).toBeVisible();
  expect(beliefUrls.length).toBeGreaterThan(0);
  expect(errors).toEqual([]);

  // (b) neither plain door asks for it: the swap is the nash door's alone.
  beliefUrls.length = 0;
  await page.goto("/?blind");
  await expect(page.locator('[data-party="settings"]')).toBeVisible();
  await page.goto("/");
  await expect(page.locator(".start-col").first()).toBeVisible();
  expect(beliefUrls).toEqual([]);

  // (c) load-bearing: a nash page that cannot get the file fails closed
  // (the boot error box), never a quiet game under the plainer prior.
  await page.route("**/belief-pool-v1/**", (r) => r.abort());
  await page.goto("/?nash");
  await expect(page.locator(".error-box")).toBeVisible();
  await page.unroute("**/belief-pool-v1/**");
});
