import { expect, test, type Page } from "@playwright/test";
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

// Both are legal pool teams copied into custom records. Their attack-heavy,
// self-destruct-heavy sets keep the full-game UI gates short and decisive.
const teamA = pool.teams[4];
const teamB = pool.teams[29];

function custom(team: PoolTeamJson, id: string, name: string): CustomRecord {
  return {
    id,
    name,
    sets: team.sets,
    species: team.sets.map((s) => s.species),
    levels: team.sets.map((s) => s.level ?? 55),
    savedAt: 1,
  };
}

const customA = custom(teamA, "custom-a", "Custom A");

function toId(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}

async function seedStorage(
  page: Page,
  customs: CustomRecord[],
  picks?: unknown,
) {
  await page.addInitScript(
    ({ records, initialPicks }) => {
      if (sessionStorage.getItem("nc2000-e2e-seeded") === "1") return;
      sessionStorage.setItem("nc2000-e2e-seeded", "1");
      localStorage.setItem("nc2000-locale", "en");
      localStorage.setItem("nc2000-custom-teams", JSON.stringify(records));
      if (initialPicks === undefined)
        localStorage.removeItem("nc2000-start-picks");
      else
        localStorage.setItem(
          "nc2000-start-picks",
          JSON.stringify(initialPicks),
        );
    },
    { records: customs, initialPicks: picks },
  );
}

function guardConsole(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(String(e)));
  return errors;
}

async function expectExactSets(
  section: ReturnType<Page["locator"]>,
  team: { name: string; sets: SetJson[] },
) {
  await expect(section.getByRole("heading", { name: new RegExp(team.name) })).toBeVisible();
  for (const set of team.sets) {
    const head = section.locator(`[data-mon="${toId(set.species)}"]`);
    await expect(head).toHaveCount(1);
    if ((await head.getAttribute("aria-expanded")) !== "true")
      await head.click();
    const sheet = head.locator("xpath=..");
    if (set.item)
      await expect(sheet.locator(`[data-item="${toId(set.item)}"]`)).toHaveCount(1);
    const actualMoves = await sheet.locator("[data-move]").evaluateAll((els) =>
      els.map((e) => e.getAttribute("data-move") ?? "").sort(),
    );
    expect(actualMoves).toEqual(set.moves.map(toId).sort());
  }
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

test.describe.configure({ mode: "serial" });

test("custom-vs-custom persists, shows exact bot sets, and rematches after deletion", async ({
  page,
}) => {
  const errors = guardConsole(page);
  const pairRequests: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/preview-tables-v0/pair-")) pairRequests.push(r.url());
  });
  await seedStorage(page, [customA]);
  await page.goto("/");

  await page.locator('[data-party="human"]').click();
  await page.locator('[data-custom="custom-a"]').click();

  // Import from the opponent picker itself; a successful import pins that
  // new saved party to the bot while keeping the result visible.
  await page.locator('[data-party="bot"]').click();
  await page.getByRole("button", { name: "+ Import a custom team" }).click();
  await page.locator(".import-text").fill(teamB.export);
  await page.locator(".import-name").fill("Custom B");
  await page.locator(".import-btn").click();
  await expect(page.locator(".import-ok-note")).toContainText("Custom B");
  const stored = await page.evaluate(() => ({
    customs: JSON.parse(localStorage.getItem("nc2000-custom-teams") ?? "[]"),
    picks: JSON.parse(localStorage.getItem("nc2000-start-picks") ?? "{}"),
  }));
  const savedB = (stored.customs as CustomRecord[]).find(
    (t) => t.name === "Custom B",
  );
  expect(savedB).toBeTruthy();
  expect(stored.picks).toEqual({
    human: { kind: "custom", id: "custom-a" },
    bot: { kind: "custom", id: savedB!.id },
  });
  await expect(page.locator(`[data-custom="${savedB!.id}"]`)).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.locator("dialog.modal .modal-head button").click();

  // Pinned choices survive a reload by stable custom id.
  await page.reload();
  await expect(page.locator('[data-party="human"] .party-value')).toHaveText(
    "Custom A",
  );
  await expect(page.locator('[data-party="bot"] .party-value')).toHaveText(
    "Custom B",
  );
  await page.locator('[data-party="bot"]').click();
  await expect(page.locator(`[data-custom="${savedB!.id}"]`)).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.screenshot({ path: "/tmp/nc2000-bot-custom-picker.png" });
  await page.locator("dialog.modal .modal-head button").click();

  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();
  const previewFoe = page.locator(".preview-cols > section").first();
  await expectExactSets(previewFoe, { name: "Custom B", sets: savedB!.sets });
  await page.screenshot({
    path: "/tmp/nc2000-custom-v-custom-preview.png",
    fullPage: true,
  });
  expect(pairRequests).toEqual([]);

  // Simulate deletion from another tab after start. GameSpec owns both
  // snapshots, so this battle and Rematch must retain Custom B exactly.
  await page.evaluate((id) => {
    const records = JSON.parse(
      localStorage.getItem("nc2000-custom-teams") ?? "[]",
    ) as CustomRecord[];
    localStorage.setItem(
      "nc2000-custom-teams",
      JSON.stringify(records.filter((t) => t.id !== id)),
    );
  }, savedB!.id);

  await choosePreview(page);
  await page.locator(".sheets-btn").click();
  const battleFoe = page.locator(".team-sheets > section").nth(1);
  await expectExactSets(battleFoe, { name: "Custom B", sets: savedB!.sets });
  await page.locator("dialog.modal .modal-head button").click();
  await playToOutcome(page);

  await page.getByRole("button", { name: "Rematch" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();
  await expect(
    page.locator(".preview-cols > section").first().getByRole("heading", {
      name: /Custom B/,
    }),
  ).toBeVisible();
  await page.locator(".preview-actions .quit-btn").click();
  await expect(page.locator('[data-party="human"] .party-value')).toHaveText(
    "Custom A",
  );
  await expect(page.locator('[data-party="bot"] .party-value')).toHaveText(
    "Random",
  );

  // If both start-screen sides pin the same saved party, the in-UI delete
  // path invalidates both choices in one update.
  await page.locator('[data-party="bot"]').click();
  await page.locator('[data-custom="custom-a"]').click();
  await page.locator('[data-party="bot"]').click();
  const deleteA = page
    .locator('.custom-card:has([data-custom="custom-a"])')
    .locator(".delete-btn");
  await deleteA.click();
  await deleteA.click();
  await page.locator("dialog.modal .modal-head button").click();
  await expect(page.locator('[data-party="human"] .party-value')).toHaveText(
    "Random",
  );
  await expect(page.locator('[data-party="bot"] .party-value')).toHaveText(
    "Random",
  );
  expect(errors).toEqual([]);
});

test("pool-vs-custom completes without requesting a baked pair", async ({
  page,
}) => {
  const errors = guardConsole(page);
  const pairRequests: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/preview-tables-v0/pair-")) pairRequests.push(r.url());
  });
  await seedStorage(page, [customA], {
    human: { kind: "pool", id: teamB.id },
    bot: { kind: "custom", id: customA.id },
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();
  await expectExactSets(page.locator(".preview-cols > section").first(), {
    name: "Custom A",
    sets: customA.sets,
  });
  expect(pairRequests).toEqual([]);
  await choosePreview(page);
  await playToOutcome(page);
  await expect(page.locator(".end-banner")).toBeVisible();
  expect(errors).toEqual([]);
});

test("custom-vs-pool keeps the existing live-preview path", async ({
  page,
}) => {
  const errors = guardConsole(page);
  const pairRequests: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/preview-tables-v0/pair-")) pairRequests.push(r.url());
  });
  await seedStorage(page, [customA], {
    human: { kind: "custom", id: customA.id },
    bot: { kind: "pool", id: teamB.id },
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Start battle" }).click();
  await expect(page.locator(".preview-screen")).toBeVisible();
  await expectExactSets(page.locator(".preview-cols > section").first(), {
    name: teamB.id,
    sets: teamB.sets,
  });
  expect(pairRequests).toEqual([]);
  await choosePreview(page);
  await playToOutcome(page);
  await expect(page.locator(".end-banner")).toBeVisible();
  expect(errors).toEqual([]);
});
