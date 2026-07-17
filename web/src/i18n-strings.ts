// UI string tables for the two locales (M13). Battle narration lives in
// narrate.ts; dex names in data/i18n-ja.json. Everything else the UI
// prints is here, keyed by a typed interface so en/ja can't drift apart.

export type Locale = "en" | "ja";

export interface UIStrings {
  // app shell
  loadingEngine: string;
  failedLoad: string;
  settingUp: string;
  // title / start screen
  subtitle: string;
  openSheetNote: string;
  startBattle: string;
  yourParty: string;
  oppParty: string;
  randomLabel: string;
  randomCard: (n: number) => string;
  chooseYours: string;
  chooseOpp: string;
  poolSection: string;
  close: string;
  languageLabel: string;
  // custom teams (M14)
  customBadge: string;
  customSection: string;
  addCustom: string;
  importTitle: string;
  importHelp: string;
  importPlaceholder: string;
  importNameLabel: string;
  importNamePlaceholder: string;
  importButton: string;
  importCancel: string;
  importedOk: (name: string) => string;
  appliedFixes: (n: number) => string;
  importErrors: (n: number) => string;
  deleteTeam: string;
  deleteConfirm: string;
  // device benchmark
  benchTitle: string;
  benchRun: string;
  benchAgain: string;
  benchRunning: (pct: number) => string;
  benchResult: (r: {
    ips: number;
    fullK: number;
    fullSec: string;
    gateK: number;
    gateSec: number;
    pass: boolean;
    sec: string;
  }) => string;
  benchNote: (kIters: number) => string;
  // team preview
  teamPreview: string;
  foeTeam: (id: string) => string;
  yourTeamPick: string;
  lead: string;
  confirmPicks: string;
  pickMore: (n: number) => string;
  levelSum: (sum: number, cap: number) => string;
  overLevelCap: (cap: number) => string;
  previewFromTable: string;
  previewFromSearch: string;
  // battle chrome
  quit: string;
  turnLabel: (n: number) => string;
  nLeft: (n: number) => string;
  fnt: string;
  switchLabel: string;
  foePrefix: string;
  fieldFoe: (cond: string) => string;
  fieldYou: (cond: string) => string;
  moveCat: (cat: "Physical" | "Special" | "Status") => string;
  bp: (n: number) => string;
  // thinking / waiting
  thinkChip: (doneK: string, budgetK: string) => string;
  ponderChip: (bonusK: string) => string;
  botThinking: (done: number, budget: number) => string;
  botFinishing: string;
  waitingBot: string;
  // end
  youWin: string;
  botWins: string;
  tie: string;
  rematch: string;
  newTeams: string;
}

const EN: UIStrings = {
  loadingEngine: "Loading engine…",
  failedLoad: "Failed to load",
  settingUp: "Setting up battle…",
  subtitle: "Gen 2 · human vs bot",
  openSheetNote:
    "Open team sheet: the bot sees your sets, and you can read its sets " +
    "in the team list — neither side sees which 3 the other picks until " +
    "they're revealed in battle.",
  startBattle: "Start battle",
  yourParty: "Your party",
  oppParty: "Opponent's party",
  randomLabel: "Random",
  randomCard: (n) => `Random from pool (${n} teams)`,
  chooseYours: "Choose your team",
  chooseOpp: "Choose the opponent's team",
  poolSection: "Meta pool teams",
  close: "Close",
  languageLabel: "Language",
  customBadge: "custom",
  customSection: "Your custom teams",
  addCustom: "+ Import a custom team",
  importTitle: "Import a custom team",
  importHelp:
    "Paste a team in the Pokémon Showdown teambuilder export format. " +
    "Fixable issues (missing gender, derived HP DV, …) are corrected " +
    "automatically; rule violations are listed below.",
  importPlaceholder:
    "Snorlax @ Leftovers\nLevel: 55\n- Body Slam\n- Rest\n…",
  importNameLabel: "Team name",
  importNamePlaceholder: "My team",
  importButton: "Import team",
  importCancel: "Close",
  importedOk: (name) => `Saved “${name}” — it plays under the open team sheet like any pool team.`,
  appliedFixes: (n) => `${n} automatic ${n === 1 ? "fix" : "fixes"} applied`,
  importErrors: (n) => `${n} ${n === 1 ? "problem" : "problems"} — fix and import again`,
  deleteTeam: "Delete",
  deleteConfirm: "Delete?",
  benchTitle: "Device benchmark",
  benchRun: "Run (~5 s)",
  benchAgain: "Run again",
  benchRunning: (pct) => `Running… ${pct}%`,
  benchResult: (r) =>
    `${r.ips} iterations/s — full strength (${r.fullK}k) ≈ ${r.fullSec} ` +
    `s/move, mostly hidden by pondering. Reference gate (${r.gateK}k ≤ ` +
    `${r.gateSec} s): ${r.pass ? "PASS" : "MISS"} (${r.sec} s)`,
  benchNote: (k) =>
    `Fixed search workload (${k}k iterations, fixed seeds) — comparable ` +
    `across devices.`,
  teamPreview: "Team preview",
  foeTeam: (id) => `Foe team (${id})`,
  yourTeamPick: "Your team — pick 3, lead first",
  lead: "Lead",
  confirmPicks: "Confirm picks",
  pickMore: (n) => `Pick ${n} more`,
  levelSum: (sum, cap) => `Total level ${sum}/${cap}`,
  overLevelCap: (cap) => `Over the total-level cap of ${cap}`,
  previewFromTable: "Opponent picks from the baked equilibrium table",
  previewFromSearch: "Opponent picks by live search (matchup not baked yet)",
  quit: "Quit",
  turnLabel: (n) => `Turn ${n}`,
  nLeft: (n) => `${n} left`,
  fnt: "fnt",
  switchLabel: "switch",
  foePrefix: "Foe ",
  fieldFoe: (c) => `Foe: ${c}`,
  fieldYou: (c) => `You: ${c}`,
  moveCat: (c) => c,
  bp: (n) => `${n} BP`,
  thinkChip: (d, b) => `thinking ${d}/${b}`,
  ponderChip: (x) => `pondering +${x}`,
  botThinking: (d, b) => `Bot is thinking… ${d} / ${b}`,
  botFinishing: "Bot is finishing up…",
  waitingBot: "Waiting for the bot…",
  youWin: "You win!",
  botWins: "The bot wins!",
  tie: "Tie",
  rematch: "Rematch",
  newTeams: "New teams",
};

const JA: UIStrings = {
  loadingEngine: "エンジンを読み込み中…",
  failedLoad: "読み込みに失敗しました",
  settingUp: "対戦を準備中…",
  subtitle: "第2世代 · 人間 vs ボット",
  openSheetNote:
    "オープンチームシート: ボットはあなたの構成(技・持ち物)を知って" +
    "おり、あなたもチーム一覧でボットの構成を読めます。どちらの側も、" +
    "相手がどの3体を選出したかは対戦中に明かされるまで見えません。",
  startBattle: "対戦開始",
  yourParty: "自分のパーティ",
  oppParty: "相手のパーティ",
  randomLabel: "ランダム",
  randomCard: (n) => `プールからランダム(全${n}チーム)`,
  chooseYours: "自分のチームを選ぶ",
  chooseOpp: "相手のチームを選ぶ",
  poolSection: "メタプールのチーム",
  close: "閉じる",
  languageLabel: "言語",
  customBadge: "カスタム",
  customSection: "自分のカスタムチーム",
  addCustom: "+ カスタムチームを取り込む",
  importTitle: "カスタムチームの取り込み",
  importHelp:
    "Pokémon Showdown のチームビルダーからエクスポートしたテキストを" +
    "貼り付けてください。自動修正できる項目(性別の補完、HPのDV導出など)は" +
    "取り込み時に修正され、ルール違反は下に一覧表示されます。",
  importPlaceholder:
    "Snorlax @ Leftovers\nLevel: 55\n- Body Slam\n- Rest\n…",
  importNameLabel: "チーム名",
  importNamePlaceholder: "マイチーム",
  importButton: "取り込む",
  importCancel: "閉じる",
  importedOk: (name) =>
    `「${name}」を保存しました — プールのチームと同じくオープンシートで対戦できます。`,
  appliedFixes: (n) => `自動修正 ${n}件`,
  importErrors: (n) => `問題 ${n}件 — 修正して再度取り込んでください`,
  deleteTeam: "削除",
  deleteConfirm: "削除する?",
  benchTitle: "端末ベンチマーク",
  benchRun: "実行(約5秒)",
  benchAgain: "もう一度",
  benchRunning: (pct) => `実行中… ${pct}%`,
  benchResult: (r) =>
    `${r.ips} 回/秒 — 最大強度(${r.fullK}k) ≈ ${r.fullSec} 秒/手` +
    `(ポンダリングでほぼ隠れます)。参照ゲート(${r.gateK}k ≤ ` +
    `${r.gateSec}秒): ${r.pass ? "PASS" : "MISS"} (${r.sec}秒)`,
  benchNote: (k) =>
    `固定探索ワークロード(${k}k回・固定シード)— 端末間で比較できます。`,
  teamPreview: "選出(見せ合い)",
  foeTeam: (id) => `相手のチーム(${id})`,
  yourTeamPick: "自分のチーム — 3体選ぶ(1体目が先発)",
  lead: "先発",
  confirmPicks: "選出を確定",
  pickMore: (n) => `あと${n}体`,
  levelSum: (sum, cap) => `合計レベル ${sum}/${cap}`,
  overLevelCap: (cap) => `合計レベルが${cap}を超えるため選べません`,
  previewFromTable: "相手の選出: 事前計算した均衡テーブル",
  previewFromSearch: "相手の選出: ライブ探索(この組み合わせは未計算)",
  quit: "やめる",
  turnLabel: (n) => `ターン ${n}`,
  nLeft: (n) => `残り${n}体`,
  fnt: "ひんし",
  switchLabel: "交代",
  foePrefix: "相手の ",
  fieldFoe: (c) => `相手: ${c}`,
  fieldYou: (c) => `自分: ${c}`,
  moveCat: (c) =>
    c === "Physical" ? "物理" : c === "Special" ? "特殊" : "変化",
  bp: (n) => `威力${n}`,
  thinkChip: (d, b) => `思考中 ${d}/${b}`,
  ponderChip: (x) => `先読み中 +${x}`,
  botThinking: (d, b) => `ボットの思考中… ${d} / ${b}`,
  botFinishing: "ボットが考えをまとめています…",
  waitingBot: "ボットを待っています…",
  youWin: "あなたの勝ち!",
  botWins: "ボットの勝ち!",
  tie: "ひきわけ",
  rematch: "再戦",
  newTeams: "チーム選択へ",
};

export const STRINGS: Record<Locale, UIStrings> = { en: EN, ja: JA };
