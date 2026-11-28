import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import {
  auditCastleTeamForPdinput,
  buildSinglePostSlotFields,
  encodeMonForPdinput,
  encodeTeamForPdinput,
  PDINPUT_ITEM_CODES,
  type PdinputDex,
  type PdinputI18nJa,
} from "../src/pdinput";

const base = process.env.NC2000_E2E_BASE ?? "/";

function loadDexAndJa(): { dex: PdinputDex; ja: PdinputI18nJa } {
  const dex = JSON.parse(
    readFileSync(new URL("../../data/gen2stadium2.json", import.meta.url), "utf8"),
  ) as PdinputDex;
  const ja = JSON.parse(
    readFileSync(new URL("../../data/i18n-ja.json", import.meta.url), "utf8"),
  ) as PdinputI18nJa;
  return { dex, ja };
}

test("data/i18n-ja.json and encodeMonForPdinput use PBS Gen 2 spelling くろいめがね for Black Glasses", () => {
  const { dex, ja } = loadDexAndJa();
  expect(ja.items.blackglasses).toBe("くろいめがね");
  const mon = {
    species: "Umbreon",
    level: 50,
    item: "Black Glasses",
    moves: ["Pursuit", "Charm", "Moonlight", "Baton Pass"],
  };
  const team = encodeTeamForPdinput({ id: "t-bg", sets: [mon, mon, mon, mon, mon, mon] }, dex, ja);
  expect(team.sixthMon.jpItem).toBe("くろいめがね");
  expect(buildSinglePostSlotFields(team, 1).ITEM).toBe("くろいめがね");
});

test("encodeMonForPdinput supports all Gen 2 Mail IDs (159, 182-190) and Sport/Park Ball (178)", () => {
  const { dex, ja } = loadDexAndJa();
  const cases: Array<[string, number, string]> = [
    ["Mail", 159, "はながらボール"],
    ["Flower Mail", 159, "はながらボール"],
    ["Surf Mail", 182, "なみのりメール"],
    ["Liteblue Mail", 183, "みずいろメール"],
    ["Portrait Mail", 184, "にがおえメール"],
    ["Lovely Mail", 185, "ラブリーメール"],
    ["Eon Mail", 186, "ブイブイメール"],
    ["Morph Mail", 187, "へんしんメール"],
    ["Bluesky Mail", 188, "あおぞらメール"],
    ["Music Mail", 189, "おんぷメール"],
    ["Mirage Mail", 190, "まぼろしメール"],
    ["Sport Ball", 178, "パークボール"],
    ["Park Ball", 178, "パークボール"],
  ];

  for (const [item, expectedCode, expectedJp] of cases) {
    const enc = encodeMonForPdinput(
      { species: "Snorlax", level: 50, item, moves: ["Body Slam", "Rest"] },
      dex,
      ja,
    );
    const fields = enc.record.split("_");
    expect(Number(fields[25])).toBe(expectedCode);
    expect(enc.jpItem).toBe(expectedJp);
  }
  expect(PDINPUT_ITEM_CODES.sportball).toBe(178);
  expect(PDINPUT_ITEM_CODES.miragemail).toBe(190);
});

test("encodeMonForPdinput auto-sets canonical Gen 2 DVs and derived HP DV for typed Hidden Power", () => {
  const { dex, ja } = loadDexAndJa();

  // Hidden Power Ice without explicit IVs -> Atk=F(15), Def=D(13), Spe=F(15), Spc=F(15), HP DV=15
  const hpIce = encodeMonForPdinput(
    { species: "Zapdos", level: 55, moves: ["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"] },
    dex,
    ja,
  );
  expect(hpIce.koHex).toBe("FDFF");
  // Zapdos base HP 90, hpDv 15, doHp 63, Lv55 -> floor(((90+15)*2+63)*55/100) + 55 + 10 = 215
  expect(hpIce.maxHp).toBe(215);

  // Hidden Power Fire without explicit IVs -> Atk=E(14), Def=C(12), Spe=F(15), Spc=F(15), HP DV=3
  const hpFire = encodeMonForPdinput(
    { species: "Raikou", level: 50, moves: ["Thunderbolt", "hiddenpowerfire"] },
    dex,
    ja,
  );
  expect(hpFire.koHex).toBe("ECFF");
  // Raikou base HP 90, hpDv 3, doHp 63, Lv50 -> floor(((90+3)*2+63)*50/100) + 50 + 10 = 184
  expect(hpFire.maxHp).toBe(184);

  // Explicit custom IVs must be preserved (e.g. HP Bug with low Atk DV 13, Def DV 13, Spe DV 14 -> HP DV 13)
  const custom = encodeMonForPdinput(
    {
      species: "Marowak",
      level: 55,
      moves: ["Earthquake", "Hidden Power Bug"],
      ivs: { atk: 26, def: 26, spe: 28, spa: 30, spd: 30 },
    },
    dex,
    ja,
  );
  expect(custom.koHex).toBe("DDEF");
});

test("encodeMonForPdinput supports variable ppUp, clamping, base-40 PP=64, and Sketch ppUp=0", () => {
  const { dex, ja } = loadDexAndJa();

  const enc = encodeMonForPdinput(
    {
      species: "Smeargle",
      level: 50,
      // Tackle has base PP 35; Pound/Scratch have base PP 35; Splash has base PP 40; Sketch has noPPBoosts
      moves: ["Splash", "Splash", "Spore", "Sketch"],
      ppUps: [3, 0, 1, 3],
    },
    dex,
    ja,
  );
  const fields = enc.record.split("_");
  expect(enc.ppUps).toEqual([3, 0, 1, 0]);
  expect(fields.slice(15, 19).map(Number)).toEqual([64, 40, 18, 1]);
  expect(fields.slice(19, 23).map(Number)).toEqual([3, 0, 1, 0]);

  const flatZero = encodeMonForPdinput(
    {
      species: "Snorlax",
      level: 55,
      moves: ["Curse", "Rest"],
      ppUp: 0,
    },
    dex,
    ja,
  );
  expect(flatZero.ppUps).toEqual([0, 0, 0, 0]);
  expect(flatZero.record.split("_").slice(15, 17).map(Number)).toEqual([10, 10]);
});

test("encodeMonForPdinput compacts empty-string move gaps while preserving ppUps alignment", () => {
  const { dex, ja } = loadDexAndJa();

  const enc = encodeMonForPdinput(
    {
      species: "Snorlax",
      level: 55,
      moves: ["Body Slam", "", "  ", "Rest"],
      ppUps: [2, 3, 3, 1],
    },
    dex,
    ja,
  );
  const fields = enc.record.split("_");
  // Move IDs compacted to slots 0 & 1, trailing 0s in slots 2 & 3 (no 0 gap before Rest)
  expect(fields.slice(11, 15).map(Number)).toEqual([
    dex.moves.bodyslam.num + 1,
    dex.moves.rest.num + 1,
    0,
    0,
  ]);
  // ppUps stay aligned to original indices: Body Slam=2 (15+6=21), Rest=1 (10+2=12)
  expect(enc.ppUps).toEqual([2, 1, 0, 0]);
  expect(fields.slice(15, 19).map(Number)).toEqual([21, 12, 0, 0]);
  expect(fields.slice(19, 23).map(Number)).toEqual([2, 1, 0, 0]);
  expect(enc.jpMoves).toEqual([ja.moves.bodyslam, ja.moves.rest, "", ""]);
});

test("auditCastleTeamForPdinput warns on inert Bitter Berry / Dragon Fang without mutating or colliding", () => {
  const sets = [
    { species: "Snorlax", level: 55, item: "Bitter Berry", moves: ["Body Slam"] },
    { species: "Zapdos", level: 50, item: "Ice Berry", moves: ["Thunderbolt"] },
    { species: "Dragonite", level: 55, item: "Dragon Fang", moves: ["Dragon Breath"] },
    { species: "Kingdra", level: 50, item: "Dragon Scale", moves: ["Surf"] },
    { species: "Cloyster", level: 50, item: "Leftovers", moves: ["Spikes"] },
    { species: "Umbreon", level: 50, item: "Gold Berry", moves: ["Toxic"] },
  ];
  const report = auditCastleTeamForPdinput(sets);
  expect(report.ok).toBe(true);
  expect(report.errors).toHaveLength(0);
  expect(report.warnings).toHaveLength(2);
  expect(report.warnings[0]).toMatchObject({
    code: "item-castle-inert",
    item: "bitterberry",
    itemCode: 84,
    suggestedItem: "iceberry",
  });
  expect(report.warnings[1]).toMatchObject({
    code: "item-castle-inert",
    item: "dragonfang",
    itemCode: 145,
    suggestedItem: "dragonscale",
  });
  // Input sets must remain unmutated
  expect(sets[0].item).toBe("Bitter Berry");
  expect(sets[2].item).toBe("Dragon Fang");

  // True Item Clause collision (including alias sportball == parkball == 178)
  const dupReport = auditCastleTeamForPdinput([
    { species: "Snorlax", item: "Sport Ball" },
    { species: "Zapdos", item: "Park Ball" },
  ]);
  expect(dupReport.ok).toBe(false);
  expect(dupReport.errors).toHaveLength(1);
  expect(dupReport.errors[0].code).toBe("item-clause");
});

test("?pdinput renders all 15 teams in 1 safe batch under 8KB and generates installer script", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto(`${base}?pdinput`);
  await expect(page.getByTestId("pdinput-page")).toBeVisible();
  await expect(page.getByTestId("pdinput-batch-1")).toBeVisible();
  await expect(page.getByTestId("pdinput-batch-2")).toHaveCount(0);

  // Batch 1 holds all 15 teams of team-pool-v4, under 8000 bytes
  const summary = page.getByTestId("pdinput-selection-summary");
  await expect(summary).toContainText("15 / 17");
  const b1Match = (await summary.textContent())?.match(/([\d,]+)\s*\/\s*8,000 B/);
  expect(b1Match).toBeTruthy();
  const b1Bytes = Number(b1Match![1].replace(/,/g, ""));
  expect(b1Bytes).toBeGreaterThan(6000);
  expect(b1Bytes).toBeLessThanOrEqual(8000);

  // Script textarea holds a valid javascript: one-liner with all 15 entries
  const scriptVal = await page.getByTestId("pdinput-script-text").inputValue();
  expect(scriptVal.startsWith("javascript:(()=>{")).toBe(true);
  expect(scriptVal).toContain("1_x_");

  expect(errors).toEqual([]);
});
