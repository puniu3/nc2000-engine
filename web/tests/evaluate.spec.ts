import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import type { EvaluationRun } from "../src/evaluate-core";

test.use({ locale: "ja-JP", actionTimeout: 10000 });

const route = `${process.env.NC2000_E2E_BASE ?? "/"}?evaluate`;
/** The default opponents are the shipped Nash mixture, read from the same
 * file the page fetches so the suite follows a re-solved mixture. */
const nash = JSON.parse(
  readFileSync(
    new URL("../../data/meta-nash-v3/pool-artifact.json", import.meta.url),
    "utf8",
  ),
) as { teams: { id: string; weight: number; sets: unknown[] }[] };
const nashTotal = nash.teams.reduce((a, t) => a + t.weight, 0);
const player = [{ species: "Mewtwo", level: 100, moves: ["Psychic"] }];
const opponent = [{ species: "Magikarp", level: 1, moves: ["Splash"] }];

async function distribution(page: Page, sets = opponent) {
  await page.getByLabel("相手の設定ファイル").setInputFiles({
    name: "mix.json",
    mimeType: "application/json",
    buffer: Buffer.from(
      JSON.stringify({ teams: [{ id: "fish", weight: 1, sets }] }),
    ),
  });
  await expect(page.locator(".eval-opponent strong")).toHaveText(["fish"]);
}
async function exported(page: Page): Promise<EvaluationRun> {
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "結果をファイルに保存" }).click();
  return JSON.parse(readFileSync((await (await download).path())!, "utf8"));
}
async function boot(page: Page) {
  await page.goto(route);
  await expect(
    page.getByLabel("自分のパーティ", { exact: true }),
  ).toBeEnabled();
  await page.getByText("テキストから読み込む", { exact: true }).click();
}

test("warnings allow illegal parties, invalid data blocks execution", async ({
  page,
}) => {
  await boot(page);
  await page.getByLabel("自分のパーティ", { exact: true }).fill(
    JSON.stringify(
      Array.from({ length: 6 }, () => ({
        species: "Snorlax",
        level: 55,
        item: "Leftovers",
        moves: ["Spikes"],
      })),
    ),
  );
  await expect(
    page.getByText(
      "このパーティは、選ぶ3匹のレベル合計が155を超えていても対戦できます。",
      {
        exact: true,
      },
    ),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "計測を開始", exact: true }),
  ).toBeEnabled();
  await page
    .getByLabel("自分のパーティ", { exact: true })
    .fill('[{"species":"MissingNo","moves":["Tackle"]}]');
  await expect(
    page.getByRole("button", { name: "計測を開始", exact: true }),
  ).toBeDisabled();
  await page
    .getByLabel("自分のパーティ", { exact: true })
    .fill(JSON.stringify(player));
  await page
    .getByLabel("相手の設定ファイル")
    .setInputFiles({
      name: "invalid.json",
      mimeType: "application/json",
      buffer: Buffer.from(
        JSON.stringify({ teams: [{ id: "bad", weight: -1, sets: opponent }] }),
      ),
    });
  await expect(page.getByRole("alert")).toContainText("出やすさ");
  await expect(page.locator(".eval-opponent strong")).toHaveCount(nash.teams.length);
});

test("local wasm plays both sides, resumes in memory, separates changed configurations and exports", async ({
  page,
}) => {
  const failures: string[] = [];
  let workerUrl = "";
  page.on("worker", (w) => {
    workerUrl = w.url();
  });
  page.on("pageerror", (e) => failures.push(String(e)));
  await boot(page);
  await distribution(page);
  await page
    .getByLabel("自分のパーティ", { exact: true })
    .fill(JSON.stringify(player));
  await page.getByLabel("試合数", { exact: true }).fill("2");
  await expect(
    page.getByRole("button", { name: "計測を開始", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "計測を開始", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("計測が完了", {
    timeout: 120000,
  });
  const initial = await exported(page);
  expect(initial.pairs).toHaveLength(1);
  expect(initial.pairs[0].games.map((g) => g.playerSide)).toEqual([0, 1]);
  expect(initial.pairs[0].games.map((g) => g.outcome)).toEqual(["win", "win"]);
  await expect(
    page.getByRole("button", { name: "追加計測", exact: true }),
  ).toBeEnabled();
  await page.getByLabel("試合数", { exact: true }).fill("2");
  await page.getByRole("button", { name: "追加計測", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("計測が完了", {
    timeout: 120000,
  });
  const updated = await exported(page);
  expect(updated.pairs).toHaveLength(2);
  expect(updated.pairs[0]).toEqual(initial.pairs[0]);
  expect(updated.pairs[1].index).toBe(1);
  const replayed = await page.evaluate(
    async ({ url, run, base }) => {
      const beliefJson = await (
        await fetch(`${base}data/belief-pool-v3/belief-pool.json`)
      ).text();
      return new Promise((resolve, reject) => {
        const w = new Worker(url, { type: "module" });
        const pairs: unknown[] = [];
        w.onerror = (event) => {
          w.terminate();
          reject(event.message);
        };
        w.onmessage = (event) => {
          if (event.data.type === "pair") {
            pairs.push(event.data.pair);
            w.postMessage({ type: "ack" });
          }
          if (event.data.type === "done") {
            w.terminate();
            resolve(pairs);
          }
          if (event.data.type === "error") {
            w.terminate();
            reject(event.data.message);
          }
        };
        w.postMessage({
          type: "start",
          run: { ...run, pairs: [] },
          beliefJson,
        });
      });
    },
    { url: workerUrl, run: updated, base: process.env.NC2000_E2E_BASE ?? "/" },
  );
  expect(replayed).toEqual(updated.pairs);

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "結果をファイルに保存" }).click();
  const file = await download;
  expect(JSON.parse(readFileSync((await file.path())!, "utf8"))).toEqual(
    updated,
  );
  await distribution(page, [{ ...opponent[0], level: 2 }]);
  await expect(
    page.getByRole("button", { name: "追加計測", exact: true }),
  ).toBeDisabled();
  await distribution(page);
  await page.getByLabel("試合数", { exact: true }).fill("100");
  await page.getByRole("button", { name: "追加計測", exact: true }).click();
  await page.getByRole("button", { name: "停止", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "再開", exact: true }),
  ).toBeEnabled();
  expect((await exported(page)).pairs.length).toBeGreaterThanOrEqual(2);
  await page
    .getByLabel("自分のパーティ", { exact: true })
    .fill(JSON.stringify([{ ...player[0], level: 99 }]));
  await expect(
    page.getByRole("button", { name: "再開", exact: true }),
  ).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "別の計測を開始", exact: true }),
  ).toBeEnabled();
  expect(failures).toEqual([]);
});

test("results stay out of browser storage and reset on reload or tab closure", async ({
  page,
  context,
}) => {
  const downloads: string[] = [];
  page.on("download", (file) => downloads.push(file.suggestedFilename()));
  await boot(page);
  await distribution(page);
  await page
    .getByLabel("自分のパーティ", { exact: true })
    .fill(JSON.stringify(player));
  await page.getByLabel("試合数", { exact: true }).fill("2");
  await page.getByRole("button", { name: "計測を開始", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("計測が完了", {
    timeout: 120000,
  });
  expect(downloads).toEqual([]);
  expect(await page.evaluate(async () => ({
    local: localStorage.length,
    session: sessionStorage.length,
    databases: await indexedDB.databases(),
  }))).toEqual({ local: 0, session: 0, databases: [] });
  expect((await exported(page)).pairs).toHaveLength(1);
  expect(downloads).toHaveLength(1);

  await page.reload();
  await expect(page.getByLabel("自分のパーティ", { exact: true })).toBeEnabled();
  await expect(page.getByLabel("このタブの計測")).toHaveCount(0);
  await expect(page.getByLabel("自分のパーティ", { exact: true })).toHaveValue("");
  await expect(page.getByRole("button", { name: "結果をファイルに保存" })).toHaveCount(0);
  await page.getByText("テキストから読み込む", { exact: true }).click();
  await distribution(page);
  await page.getByLabel("自分のパーティ", { exact: true }).fill(JSON.stringify(player));
  await page.getByLabel("試合数", { exact: true }).fill("2");
  await page.getByRole("button", { name: "計測を開始", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("計測が完了");
  await page.close();

  const reopened = await context.newPage();
  await boot(reopened);
  await expect(reopened.getByLabel("このタブの計測")).toHaveCount(0);
  await expect(reopened.getByLabel("自分のパーティ", { exact: true })).toHaveValue("");
  await expect(reopened.locator(".eval-opponent strong")).toHaveCount(nash.teams.length);
  await expect(reopened.getByRole("button", { name: "計測を開始", exact: true })).toBeVisible();
  await expect(reopened.getByText("結果は自動保存されません。", { exact: false })).toBeVisible();
});

test("relaxed high-level preview survives blind search and reaches outcomes", async ({
  page,
}) => {
  await boot(page);
  await distribution(page);
  await page
    .getByLabel("自分のパーティ", { exact: true })
    .fill(JSON.stringify(Array.from({ length: 3 }, () => player[0])));
  await page.getByLabel("試合数", { exact: true }).fill("2");
  await expect(
    page.getByRole("button", { name: "計測を開始", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "計測を開始", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("計測が完了", {
    timeout: 120000,
  });
  const result = await exported(page);
  expect(result.config.player.relaxed).toBe(true);
  expect(result.pairs[0].games.map((g) => g.outcome)).toEqual(["win", "win"]);
});

test("ordinary six-Pokemon party completes against the default Nash mixture", async ({
  page,
}) => {
  await boot(page);
  const input = readFileSync(
    new URL("../../data/party-vs-nash-20260927/party.json", import.meta.url),
    "utf8",
  );
  await page.getByLabel("自分のパーティ", { exact: true }).fill(input);
  await page.getByLabel("試合数", { exact: true }).fill("2");
  await expect(
    page.getByRole("button", { name: "計測を開始", exact: true }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "計測を開始", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("計測が完了", {
    timeout: 180000,
  });
  const result = await exported(page);
  expect(result.config.player.relaxed).toBe(false);
  expect(result.config.player.warnings).toEqual([]);
  expect(nash.teams.map((t) => t.id)).toContain(result.pairs[0].opponent);
  expect(result.pairs[0].games).toHaveLength(2);
});

test("calculation and explicit export work with all browser storage disabled", async ({
  page,
}) => {
  await page.addInitScript(() => {
    for (const name of ["localStorage", "sessionStorage", "indexedDB"]) {
      Object.defineProperty(window, name, {
        get() { throw new Error("storage unavailable"); },
      });
    }
  });
  await boot(page);
  await distribution(page);
  await page
    .getByLabel("自分のパーティ", { exact: true })
    .fill(JSON.stringify(player));
  await page.getByLabel("試合数", { exact: true }).fill("2");
  await page.getByRole("button", { name: "計測を開始", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("計測が完了", {
    timeout: 120000,
  });
  await expect(page.getByRole("alert")).toHaveCount(0);
  expect((await exported(page)).pairs).toHaveLength(1);
  await expect(
    page.getByRole("button", { name: "結果をファイルに保存" }),
  ).toBeEnabled();
});

test("worker failure stops without inventing a game outcome", async ({
  page,
}) => {
  await page.route("**/assets/evaluate-worker-*.js", (route) => route.abort());
  await boot(page);
  await distribution(page);
  await page
    .getByLabel("自分のパーティ", { exact: true })
    .fill(JSON.stringify(player));
  await page.getByLabel("試合数", { exact: true }).fill("2");
  await page.getByRole("button", { name: "計測を開始", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("エラーで停止");
  await expect(page.getByRole("alert")).toBeVisible();
  expect((await exported(page)).pairs).toEqual([]);
});

test("Japanese party form works without secure-context APIs, preserving moves across members", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(Crypto.prototype, "subtle", { get: () => undefined });
    Object.defineProperty(Crypto.prototype, "randomUUID", { value: undefined });
  });
  await page.goto(route);
  const mon = page.getByRole("combobox", {
    name: "自分のパーティ1匹目のポケモン",
    exact: true,
  });
  await expect(mon).toBeEnabled();
  await expect(page.getByLabel("自分のパーティ", { exact: true })).toBeHidden();
  await mon.fill("みゅう");
  await page.getByRole("option", { name: "ミュウツー", exact: true }).click();
  await page
    .getByLabel("自分のパーティ1匹目のレベル", { exact: true })
    .fill("100");
  await page
    .getByRole("combobox", { name: "自分のパーティ1匹目の技1", exact: true })
    .fill("サイコキネシス");
  await page
    .getByRole("heading", { name: "自分のパーティ", exact: true })
    .click();
  await page
    .getByRole("button", { name: "＋ ポケモンを追加", exact: true })
    .first()
    .click();
  await page
    .getByRole("combobox", {
      name: "自分のパーティ2匹目のポケモン",
      exact: true,
    })
    .fill("カビゴン");
  await page
    .getByRole("combobox", { name: "自分のパーティ2匹目の技1", exact: true })
    .fill("のしかかり");
  await page
    .getByRole("button", { name: "1. ミュウツー", exact: true })
    .click();
  await expect(
    page.getByRole("combobox", {
      name: "自分のパーティ1匹目の技1",
      exact: true,
    }),
  ).toHaveValue("サイコキネシス");
  await page.getByRole("button", { name: "2. カビゴン", exact: true }).click();
  await page
    .getByRole("button", { name: "このポケモンを外す", exact: true })
    .first()
    .click();
  await distribution(page);
  await page.getByLabel("試合数", { exact: true }).fill("2");
  await page.getByRole("button", { name: "計測を開始", exact: true }).click();
  await expect(page.getByRole("status")).toContainText("計測が完了", {
    timeout: 120000,
  });
  const result = await exported(page);
  expect(result.config.player.sets).toHaveLength(1);
  expect(result.config.player.sets[0]).toMatchObject({
    species: "Mewtwo",
    level: 100,
    moves: ["Psychic"],
  });
  expect(result.pairs[0].games.map((g) => g.outcome)).toEqual(["win", "win"]);
  expect(result.config.beliefHash).toHaveLength(64);

});

test("editing an imported party preserves custom stats and exposes no technical input by default", async ({
  page,
}) => {
  await boot(page);
  const original = {
    species: "Raikou",
    level: 55,
    moves: ["Thunderbolt", "Hidden Power Ice"],
    ivs: { hp: 30, atk: 22, def: 26, spa: 30, spd: 30, spe: 30 },
    evs: { hp: 200, atk: 0, def: 128, spa: 255, spd: 255, spe: 255 },
    happiness: 17,
  };
  await page
    .getByLabel("自分のパーティ", { exact: true })
    .fill(JSON.stringify([original]));
  await page
    .getByRole("combobox", { name: "自分のパーティ1匹目の持ち物", exact: true })
    .fill("たべのこし");
  const raw = JSON.parse(
    await page.getByLabel("自分のパーティ", { exact: true }).inputValue(),
  );
  expect(raw[0]).toEqual({ ...original, item: "Leftovers" });
  await page.getByText("テキストから読み込む", { exact: true }).first().click();
  const visibleText = await page.locator("main").innerText();
  expect(visibleText).not.toMatch(/WASM|belief|反復|BLIND|正規化/);
  await expect(page.getByLabel("考える回数", { exact: true })).toHaveValue(
    "27000",
  );
});

test("opponent panel shows the mixture's parties and probabilities, with JSON as its only edit control", async ({
  page,
}) => {
  await page.goto(route);
  const panel = page.getByRole("region", { name: "対戦相手の設定" });
  await expect(panel.locator(".eval-opponent strong")).toHaveText(
    nash.teams.map((_, i) => `基本の相手${i + 1}`),
  );
  await expect(panel.locator(".eval-opponent span")).toHaveText(
    nash.teams.map((t) => `${((t.weight / nashTotal) * 100).toFixed(1)}%`),
  );
  await expect(panel.locator(".eval-opponent-roster li")).toHaveCount(
    nash.teams.length * 6,
  );
  await expect(panel.locator("input")).toHaveCount(1);
  await expect(panel.locator("input")).toHaveAttribute("type", "file");
  await expect(panel.locator("textarea, select")).toHaveCount(0);
  await expect(
    page.getByText("自動で補った項目", { exact: false }),
  ).toHaveCount(0);
  await expect(
    page.getByLabel("考える回数", { exact: true }).locator("option"),
  ).toHaveText([
    "3,000回(簡易計測)",
    "10,000回(簡易計測)",
    "27,000回(実際のボットと同じ)",
  ]);
  await panel.getByText("技・持ち物を見る", { exact: true }).first().click();
  await expect(
    panel.locator(".eval-opponent-sets").first().locator("strong"),
  ).toHaveCount(6);
});

test("saved opponent settings load back unchanged and edit the distribution", async ({
  page,
}) => {
  await page.goto(route);
  const panel = page.getByRole("region", { name: "対戦相手の設定" });
  await expect(panel.locator(".eval-opponent strong")).toHaveCount(
    nash.teams.length,
  );
  const download = page.waitForEvent("download");
  await panel
    .getByRole("button", { name: "相手の設定をファイルに保存" })
    .click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("evaluate-opponents.json");
  const text = readFileSync((await file.path())!, "utf8");
  const saved = JSON.parse(text) as {
    teams: { id: string; weight: number; sets: unknown[] }[];
  };
  expect(saved.teams.map((t) => t.id)).toEqual(nash.teams.map((t) => t.id));
  expect(saved.teams.every((t) => t.sets.length === 6)).toBe(true);

  saved.teams[0].id = "edited";
  saved.teams[0].weight = 0;
  await panel.getByLabel("相手の設定ファイル").setInputFiles({
    name: "evaluate-opponents.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(saved)),
  });
  await expect(panel.locator(".eval-opponent strong")).toHaveText([
    "edited",
    ...nash.teams.slice(1).map((_, i) => `基本の相手${i + 2}`),
  ]);
  await expect(panel.locator(".eval-opponent span").first()).toHaveText("0.0%");

  await panel.getByLabel("相手の設定ファイル").setInputFiles({
    name: "evaluate-opponents.json",
    mimeType: "application/json",
    buffer: Buffer.from(text),
  });
  await expect(panel.locator(".eval-opponent span")).toHaveText(
    nash.teams.map((t) => `${((t.weight / nashTotal) * 100).toFixed(1)}%`),
  );
});
