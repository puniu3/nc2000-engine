import { locale } from "./i18n";

export interface TeamNameEntry {
  ja: string;
  en: string;
}

export const CATALOG_TEAM_NAMES: Record<string, TeamNameEntry> = {
  // Curated 10-shell (15-team) strong pool (team-pool-v4)
  "pb131-c3r-fbhera-magzap-refcloy": {
    ja: "磁石雷55サンダー・55バンギ・鉢巻無し起死回生ヘラ (pb-131)",
    en: "Magnet 55 Zapdos · 55 Tyranitar · EndRev Heracross (pb-131)",
  },
  "s07-c8-mcoatrode-fbwak-delax-thunzap": {
    ja: "ミラコマルマイン・大文字55ガラ・55ケンタ・雷55サンダー (sample-07)",
    en: "Mirror Coat Electrode · FB 55 Marowak · 55 Tauros · Thunder 55 Zapdos (sample-07)",
  },
  "apex-c3r-fbhera-twstar-refcloy": {
    ja: "55寝言カビ・爆破4Wゲンガー・リフパル・電磁波スターミー (apex A)",
    en: "55 RestTalk Snorlax · Boom 4-Atk Gengar · Reflect Cloyster · TWave Starmie (apex A)",
  },
  "apex-c5-perishgar-3atkstar-fbwak-curcloy": {
    ja: "55寝言カビ・滅びゲンガー・鈍いパル・大文字ガラ (apex B)",
    en: "55 RestTalk Snorlax · Perish Gengar · Curse Cloyster · Fire Blast Marowak (apex B)",
  },
  "ayagen-m-zap50": {
    ja: "8番道路ジム改 (55ゲンガー・55フーディン・50サンダー)",
    en: "Route 8 Gym Rev (55 Gengar / 55 Alakazam / 50 Zapdos)",
  },
  "pb63-c3-restlax-4atkgar-wwzap": {
    ja: "55眠る鈍いカビ・4W55ゲンガー・吹き飛ばしサンダー (pb-63 A)",
    en: "55 Rest Curse Snorlax · 4-Atk 55 Gengar · WW Zapdos (pb-63 A)",
  },
  "pb63-c3b-sdlax-4atkgar-dpzap": {
    ja: "55自爆鈍いカビ・4W55ゲンガー・ドリくちサンダー (pb-63 B)",
    en: "55 SD Curse Snorlax · 4-Atk 55 Gengar · Drill Peck Zapdos (pb-63 B)",
  },
  "rraikou-m-cune50": {
    ja: "ソード＆シールド改 (リフ55ライコウ・55ガラ・50スイクン A)",
    en: "Sword & Shield Rev (Reflect 55 Raikou · 55 Marowak · 50 Suicune A)",
  },
  "rraikou-c1-roarkou-roarcune-fblax": {
    ja: "ソード＆シールド改 (吠える55ライコウ・吠えるスイクン・大文字カビ B)",
    en: "Sword & Shield Rev (Roar 55 Raikou · Roar Suicune · FB Snorlax B)",
  },
  "pb85-m-cune50": {
    ja: "寝言カビSA改 (55カビ・冷ビ寝言スイクン・リフサンダー A)",
    en: "55 Snorlax SA Rev (RestTalk Snorlax · IB Suicune · Reflect Zapdos A)",
  },
  "pb85-c5-roartalkcune-wwzap": {
    ja: "寝言カビSA改 (55カビ・吠える寝言スイクン・吹き飛ばしサンダー B)",
    en: "55 Snorlax SA Rev (RestTalk Snorlax · Roar Suicune · WW Zapdos B)",
  },
  "pb85-c3r-sdlax-refcloy": {
    ja: "寝言カビSA改 (55自爆鈍いカビ・リフパル・50スイクン C)",
    en: "55 Snorlax SA Rev (SD Curse Snorlax · Reflect Cloyster · Suicune C)",
  },
  "pb87-c5-fblax-dpzap-crunchttar": {
    ja: "ドリくち55サンダー・噛み砕く55バンギ・大文字カビ (pb-87)",
    en: "Drill Peck 55 Zapdos · Crunch 55 Tyranitar · FB Snorlax (pb-87)",
  },

  // Historical catalog teams (retained for display if loaded as a custom pool)
  "s0053-lefties-hpi": {
    ja: "残飯めざ氷55サンダー・大文字55ガラガラ (s0053)",
    en: "Lefties HP-Ice 55 Zapdos · Fire Blast 55 Marowak (s0053)",
  },
  "s0053-lefties-dp": {
    ja: "残飯ドリくち55サンダー・めざ虫55ガラガラ (s0053)",
    en: "Lefties Drill Peck 55 Zapdos · HP-Bug 55 Marowak (s0053)",
  },

  // Strict-LIVE verified top Tier-1 additions (team-pool-v3)
  "pb-84": {
    ja: "カジュアル寝言カビA (55ライコウ・55カビゴン)",
    en: "Casual Sleep Talk Snorlax A (55 Raikou / 55 Snorlax)",
  },
  "mjj-2014-offantom": {
    ja: "ナマステひつまぶし (55ポリゴン2・55リザ・55ケンタ)",
    en: "Namaste Hitsumabushi (55 Porygon2 / 55 Charizard / 55 Tauros)",
  },
  "pb-122": {
    ja: "55ライコウ・55ガラガラA (バトンサンダース)",
    en: "55 Raikou / 55 Marowak A (AgilityPass Jolteon)",
  },
  "pb-131": {
    ja: "55サンダー・55バンギラス (ハガネール・ヘラ)",
    en: "55 Zapdos / 55 Tyranitar (Steelix / Heracross)",
  },
  "pb-87": {
    ja: "サンダーバンギWA feat.まるころタンク",
    en: "Zapdos Tyranitar WA feat. Miltank",
  },
  "mjj-type-QA-moudoku": {
    ja: "致死量 (毒統一4エース)",
    en: "Lethal Dose (Poison Quad-55)",
  },
  "pb-63": {
    ja: "Who is ace? (55カビ・55フシギバナ・ハピナス)",
    en: "Who is Ace? (55 Snorlax / 55 Venusaur / Blissey)",
  },
  "pb-66": {
    ja: "カビパルサンダー自分用 (55カビ・ハガネール)",
    en: "55 Snorlax Cloyster Zapdos (Steelix / Gengar)",
  },
  "mjj-2015-ayagen": {
    ja: "8番道路ジム (55怪しい光ゲンガー・55フーディン)",
    en: "Route 8 Gym (55 Confuse Ray Gengar / 55 Alakazam)",
  },

  // Historical Nash equilibrium (Kodoku v14)
  "apex-c08-surgical-142": {
    ja: "雨パル・めざ虫ガラ・冷ビスタ55カビ",
    en: "Rain Cloyster · HP-Bug Marowak 55 Snorlax",
  },
  "red-v10-s1105-r2-s0142": {
    ja: "雨パル・地震ヘラ55カビ",
    en: "Rain Cloyster · EQ Heracross 55 Snorlax",
  },
  "apex-c04-bug-3atkstar-boomgen": {
    ja: "爆破パル・爆破ゲン・波乗りスタ55カビ",
    en: "Explosion Cloyster · Boom Gengar 55 Snorlax",
  },
  "red-v10-s1105-r1-s0148": {
    ja: "爆破パル・めざ闘ヘラ55カビ",
    en: "Explosion Cloyster · HP-Fight 55 Snorlax",
  },
  "sample-08": {
    ja: "サンダータンク (爆破ナッシー / Sample 08)",
    en: "Zapdos Tank (Boom Exeggutor / Sample 08)",
  },
  "rental-cban-6": {
    ja: "サンダータンク (粉ナッシー / レンタル)",
    en: "Zapdos Tank (Sleep Powder / Rental)",
  },
  "hc75-3rd-bean": {
    ja: "サンダータンク (分身ミル・身代わり / ビーン3位)",
    en: "Zapdos Tank (Double Team / Bean 3rd)",
  },

  // Evolved
  "red-v5-r1-s0097": {
    ja: "55バンギヘラ",
    en: "55 Tyranitar Heracross",
  },
  "deme-r1-00-s0030": {
    ja: "55バンギライコウ",
    en: "55 Tyranitar Raikou",
  },
  "red-v10-s1105-r2-s0155": {
    ja: "55カビミル受けループ",
    en: "55 Snorlax Miltank Stall",
  },
  "sib-r1-n4-red-v6-r1-s0064-s07": {
    ja: "ナンス晴れファイヤー",
    en: "Wobbuffet Sun Moltres",
  },

  // Classic & community
  "sample-07": {
    ja: "55サンダーケンタロス",
    en: "55 Zapdos Tauros",
  },
  "rental-cban-8": {
    ja: "モーダルバンギ",
    en: "Modal Tyranitar",
  },
  "sample-12": {
    ja: "55カビゴンブラッキー",
    en: "55 Snorlax Umbreon",
  },
  "sample-17-orig": {
    ja: "55ポリゴン2昆布",
    en: "55 Porygon2 Phazing",
  },
  "nb-sample-13-0": {
    ja: "55カビゴン電磁波",
    en: "55 Snorlax Thunder Wave",
  },
  "rental-cban-13": {
    ja: "ポニョ (55フーディンガラ)",
    en: "Ponyo (55 Alakazam Marowak)",
  },
  "mjj-2014-thdrcomb": {
    ja: "サンダー昆布",
    en: "Zapdos Phazing",
  },
  "sample-11": {
    ja: "55ライコウナッシー",
    en: "55 Raikou Exeggutor",
  },
  "sample-06": {
    ja: "キチカビ3 (55ライコウカビゴンWA)",
    en: "Kichikabi 3 (55 Raikou Snorlax WA)",
  },
  "mjj-2010-garap-a": {
    ja: "ガラプラス",
    en: "Gala-Lapras",
  },
  "mjj-2020-rraikou": {
    ja: "ソード＆シールド (55ライコウガラ)",
    en: "Sword & Shield (55 Raikou Marowak)",
  },
  "sample-02": {
    ja: "55ポリゴン2ルージュラ",
    en: "55 Porygon2 Jynx",
  },
  "hc75-top8-tsuru": {
    ja: "つる 晴れエンテイ",
    en: "Tsuru Sun Entei",
  },
  "sample-09": {
    ja: "55ニドキングドーブル",
    en: "55 Nidoking Smeargle",
  },
  "sample-22-orig": {
    ja: "55カイリューキングドラ",
    en: "55 Dragonite Kingdra",
  },
  "sample-14": {
    ja: "55カビゴン滅び偽装",
    en: "55 Snorlax Perish Trap",
  },
  "pb-100": {
    ja: "カビゴンSA+フォレトス (わんおー)",
    en: "55 Snorlax SA + Forretress (Wan-oh)",
  },
  "pb-85": {
    ja: "寝言カビSA最終版 (甘えるバトンブラッキー)",
    en: "55 Snorlax Sleep Talk SA (Charm BP Umbreon)",
  },
  "pb-62": {
    ja: "55地震寝言ピント♀カビSA (いばみがサンダー)",
    en: "55 Scope Lens EQ/DE Snorlax SA (SubSwagger Zapdos)",
  },
  "pb-6": {
    ja: "55サンダーリキーWA (しゃわ)",
    en: "55 Zapdos Machamp WA (Shawa)",
  },
  "pb-96": {
    ja: "55ヘラクロス+エレブーWA",
    en: "55 Heracross Electabuzz WA",
  },
  "pb-89": {
    ja: "55サイドンタンクWA",
    en: "55 Rhydon Miltank WA",
  },
  "pb-46": {
    ja: "55エンテイ・メガニウム・バンギTA",
    en: "55 Entei Meganium Tyranitar TA",
  },
  "pb-13": {
    ja: "芸術は爆発だ！ (55ポリ2+5大爆発)",
    en: "Art is an Explosion! (55 Porygon2 + 5 Boom)",
  },
};

export function teamDisplayName(id: string): string | null {
  const entry = CATALOG_TEAM_NAMES[id];
  if (!entry) return null;
  return locale() === "ja" ? entry.ja : entry.en;
}
