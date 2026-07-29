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
  // team preview
  teamPreview: string;
  foeTeam: (id: string) => string;
  previewTapHint: string;
  yourTeamPick: string;
  lead: string;
  confirmPicks: string;
  pickMore: (n: number) => string;
  levelSum: (sum: number, cap: number) => string;
  overLevelCap: (cap: number) => string;
  overCapChip: (cap: number) => string;
  detailsFor: (species: string) => string;
  previewFromTable: string;
  previewFromSearch: string;
  // open team sheets (UI-2)
  teamSheets: string;
  yourTeam: (id: string) => string;
  sheetNote: string;
  sheetItem: string;
  sheetNoItem: string;
  sheetGender: string;
  sheetHp: string;
  markPicked: string;
  markRevealed: string;
  markActive: string;
  markFainted: string;
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
  // information mode (M18 blind experiment). "open" is the historical mode
  // and stays the default; "blind" hides both sides' sets symmetrically —
  // each side gets only the other's six species/levels/types plus the public
  // battle log, and the opponent is drawn from the pool anew on each rematch.
  // Blind is reached only through `?blind` (info-mode.ts), so there is no
  // switch and no open-mode copy: open says nothing about modes at all, and
  // blind announces itself once with blindBanner + modeNoteBlind.
  // The *Blind keys are drop-in replacements for their open-mode neighbours
  // (openSheetNote, foeTeam, previewTapHint, sheetNote) — same slot, blind copy.
  blindBanner: string;
  modeNoteBlind: string;
  blindSheetNote: string;
  oppRandomBlind: string;
  foeTeamBlind: string;
  previewTapHintBlind: string;
  sheetNoteBlind: string;
  revealFoeTeam: string;
  // what the bot currently believes the hidden opponent is (blind only):
  // how many pool teams still match what it has seen, or "off-pool" once no
  // pool team can explain the observations and it falls back to imputation.
  beliefChipPool: (n: number) => string;
  beliefChipOff: string;
  priorChip: (n: number, total: number) => string;
  // community belief prior (M18): a distribution table that fills in an
  // unidentifiable opponent's sets. Loaded by hand, never automatically.
  priorLabel: string;
  priorNone: string;
  priorTitle: string;
  priorHelp: string;
  priorPick: string;
  priorSample: string;
  priorClear: string;
  priorSummary: (
    species: number,
    meanMoveSum: number,
    skipped: number,
  ) => string;
  priorApplied: string;
  priorNotApplied: string;
  priorWarnings: string;
  priorLoadFailed: (why: string) => string;
  // swappable team pool: one file replaces the pool everywhere it is read
  // (the bot's draw, the blind belief's candidates, both team lists), in
  // either information mode. poolHelp has to say all three, or the swap
  // looks like it only changes the list the user is looking at.
  poolLabel: string;
  poolBundled: (n: number) => string;
  poolLoaded: (name: string, n: number) => string;
  poolTitle: string;
  poolHelp: string;
  poolPick: string;
  poolReset: string;
  poolAccepted: (n: number) => string;
  poolRejected: string;
  poolMore: (n: number) => string;
  poolNotStored: (why: string) => string;
  // rejection lines from the pool loader (team-pool.ts). Read by someone
  // staring at a file they hand-edited, so each one names the team and the
  // defect; `why` in poolErrTeam is an anchored validator finding, in
  // poolErrJson the runtime's own parse message.
  poolErrJson: (why: string) => string;
  poolErrNoTeams: string;
  poolErrSets: (team: string) => string;
  poolErrTeamSize: (team: string, n: number) => string;
  poolErrDupId: (team: string) => string;
  poolErrTeam: (team: string, why: string) => string;
  // screen reader only (UI-4) — never rendered visibly
  srLevel: (n: number) => string;
  srGender: (g: string) => string;
  srBattleHeading: string;
  srBattleLog: string;
  srYourAction: string;
  srYourActive: string;
  srFoeActive: string;
  srNoItem: string;
  srItemHeld: (item: string) => string;
  srItemGone: (item: string) => string;
  srYourTurn: string;
  srChooseSwitch: string;
  srBotThinking: string;
  srSwitchTo: (species: string, hpPct: number) => string;
  srPicked: (order: number) => string;
  srDeleteFor: (name: string) => string;
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
  customSection: "Saved custom teams",
  addCustom: "+ Import a custom team",
  importTitle: "Import a custom team",
  importHelp:
    "Paste a team in the Pokémon Showdown teambuilder export format. " +
    "Fixable issues (missing gender, derived HP DV, …) are corrected " +
    "automatically; rule violations are listed below.",
  importPlaceholder: "Snorlax @ Leftovers\nLevel: 55\n- Body Slam\n- Rest\n…",
  importNameLabel: "Team name",
  importNamePlaceholder: "My team",
  importButton: "Import team",
  importCancel: "Close",
  importedOk: (name) =>
    `Saved “${name}” — it plays under the open team sheet like any pool team.`,
  appliedFixes: (n) => `${n} automatic ${n === 1 ? "fix" : "fixes"} applied`,
  importErrors: (n) =>
    `${n} ${n === 1 ? "problem" : "problems"} — fix and import again`,
  deleteTeam: "Delete",
  deleteConfirm: "Delete?",
  teamPreview: "Team preview",
  foeTeam: (id) => `Foe team (${id})`,
  previewTapHint:
    "Open team sheet — tap a foe Pokémon for its full set; on your side the ▸ button opens it.",
  yourTeamPick: "Your team — pick 3, lead first",
  lead: "Lead",
  confirmPicks: "Confirm picks",
  pickMore: (n) => `Pick ${n} more`,
  levelSum: (sum, cap) => `Total level ${sum}/${cap}`,
  overLevelCap: (cap) => `Over the total-level cap of ${cap}`,
  overCapChip: (cap) => `>${cap}`,
  detailsFor: (s) => `${s} — details`,
  previewFromTable: "Opponent picks from the baked equilibrium table",
  previewFromSearch: "Opponent picks by live search (matchup not baked yet)",
  teamSheets: "Team sheets",
  yourTeam: (id) => `Your team (${id})`,
  sheetNote:
    "Both full teams are open information. Which 3 the opponent picked " +
    "stays hidden until each Pokémon appears in battle.",
  sheetItem: "Item",
  sheetNoItem: "No item",
  sheetGender: "Gender",
  sheetHp: "Hidden Power",
  markPicked: "Picked",
  markRevealed: "Revealed",
  markActive: "Active",
  markFainted: "Fainted",
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
  blindBanner: "Blind mode",
  modeNoteBlind:
    "Blind: each side sees only the other's six species, levels and types " +
    "plus the public battle log — sets stay hidden both ways. The opponent " +
    "is drawn from the pool at random, and redrawn on every rematch.",
  blindSheetNote:
    "Blind: neither side sees the other's sets. The bot gets your six " +
    "species, levels and types and nothing more — exactly what you get " +
    "from it.",
  oppRandomBlind: "Randomly drawn each battle",
  foeTeamBlind: "Opponent's party",
  previewTapHintBlind:
    "Blind — only species, level and types are public on the foe side; on your side the ▸ button opens your own sets.",
  sheetNoteBlind:
    "The opponent's sets are hidden — the foe side lists species, level and " +
    "types only. Your own team is shown in full.",
  revealFoeTeam: "Show opponent's sets",
  beliefChipPool: (n) =>
    `bot's read: ${n} ${n === 1 ? "candidate" : "candidates"}`,
  beliefChipOff: "bot's read: off-pool",
  priorChip: (n, total) => `prior: ${n}/${total}`,
  priorLabel: "Belief prior",
  priorNone: "None",
  priorTitle: "Belief prior",
  priorHelp:
    "A distribution table the bot uses to fill in an opponent's unknown " +
    "sets. It only bites in blind mode against a team the bot cannot " +
    "identify — in practice, when you play a custom team.",
  priorPick: "Choose a table file…",
  priorSample: "Load the sample table",
  priorClear: "Clear",
  priorSummary: (species, meanMoveSum, skipped) =>
    `${species} species, mean move-probability sum ${meanMoveSum.toFixed(2)}, ` +
    `${skipped} ${skipped === 1 ? "entry" : "entries"} skipped`,
  priorApplied: "Applied",
  priorNotApplied: "NOT applied",
  priorWarnings: "Warnings",
  priorLoadFailed: (why) => `Could not load that table — ${why}`,
  poolLabel: "Team pool",
  poolBundled: (n) => `Bundled (${n} ${n === 1 ? "team" : "teams"})`,
  poolLoaded: (name, n) => `${name} (${n} ${n === 1 ? "team" : "teams"})`,
  poolTitle: "Team pool",
  poolHelp:
    "A pool file replaces the pool everywhere it is used: the teams the " +
    "bot is drawn from, the candidates it narrows the opponent down to in " +
    "blind mode, and the team lists for both sides on this screen. Same " +
    "JSON as the bundled pool — {\"teams\": [{\"id\": …, \"sets\": […]}]} is " +
    "the minimum, a bare array of teams also reads, and id / tier / rank " +
    "are filled in when missing. Every team is exactly 6 Pokémon and is " +
    "checked against this format's rules; if a single team cannot play, " +
    "the whole file is refused.",
  poolPick: "Choose a pool file…",
  poolReset: "Use the bundled pool",
  poolAccepted: (n) => `${n} ${n === 1 ? "team" : "teams"} accepted`,
  poolRejected: "Rejected — nothing was changed",
  poolMore: (n) => `…and ${n} more`,
  poolNotStored: (why) =>
    `Loaded for this session, but not saved — ${why}. It will be gone after a reload.`,
  poolErrJson: (why) => `Not valid JSON — ${why}`,
  poolErrNoTeams:
    "No teams in this file — expected {\"teams\": [ … ]} or a bare array of teams.",
  poolErrSets: (team) => `Team ${team}: no “sets” array`,
  poolErrTeamSize: (team, n) =>
    `Team ${team}: ${n} Pokémon — a team is exactly 6`,
  poolErrDupId: (team) => `Team ${team}: duplicate id — ids must be unique`,
  poolErrTeam: (team, why) => `Team ${team} — ${why}`,
  srLevel: (n) => `Level ${n}`,
  srGender: (g) => (g === "M" ? "Male" : g === "F" ? "Female" : "Genderless"),
  srBattleHeading: "Battle",
  srBattleLog: "Battle log",
  srYourAction: "Your action",
  srYourActive: "Your active Pokémon",
  srFoeActive: "Opponent's active Pokémon",
  srNoItem: "no held item",
  srItemHeld: (it) => `holding ${it}`,
  srItemGone: (it) => `no item, was holding ${it}`,
  srYourTurn: "Your turn — choose a move or a switch.",
  srChooseSwitch: "Choose your next Pokémon.",
  srBotThinking: "Opponent is thinking…",
  srSwitchTo: (species, hpPct) => `Switch to ${species} — HP ${hpPct}%`,
  srPicked: (order) =>
    order === 0 ? "picked as lead" : `picked, number ${order + 1}`,
  srDeleteFor: (name) => `Delete team ${name}`,
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
  customSection: "保存済みカスタムチーム",
  addCustom: "+ カスタムチームを取り込む",
  importTitle: "カスタムチームの取り込み",
  importHelp:
    "Pokémon Showdown のチームビルダーからエクスポートしたテキストを" +
    "貼り付けてください。自動修正できる項目(性別の補完、HPのDV導出など)は" +
    "取り込み時に修正され、ルール違反は下に一覧表示されます。",
  importPlaceholder: "Snorlax @ Leftovers\nLevel: 55\n- Body Slam\n- Rest\n…",
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
  teamPreview: "選出(見せ合い)",
  foeTeam: (id) => `相手のチーム(${id})`,
  previewTapHint:
    "オープンチームシート — 相手のポケモンはタップで構成を確認、自分の側は ▸ ボタンで開けます。",
  yourTeamPick: "自分のチーム — 3体選ぶ(1体目が先発)",
  lead: "先発",
  confirmPicks: "選出を確定",
  pickMore: (n) => `あと${n}体`,
  levelSum: (sum, cap) => `合計レベル ${sum}/${cap}`,
  overLevelCap: (cap) => `合計レベルが${cap}を超えるため選べません`,
  overCapChip: () => "超過",
  detailsFor: (s) => `${s}の詳細`,
  previewFromTable: "相手の選出: 事前計算した均衡テーブル",
  previewFromSearch: "相手の選出: ライブ探索(この組み合わせは未計算)",
  teamSheets: "チームシート",
  yourTeam: (id) => `自分のチーム(${id})`,
  sheetNote:
    "両チームの構成は公開情報です。相手がどの3体を選出したかは、その" +
    "ポケモンが場に出るまで分かりません。",
  sheetItem: "持ち物",
  sheetNoItem: "なし",
  sheetGender: "性別",
  sheetHp: "めざめるパワー",
  markPicked: "選出",
  markRevealed: "判明",
  markActive: "出場中",
  markFainted: "ひんし",
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
  blindBanner: "ブラインド",
  modeNoteBlind:
    "ブラインド: 互いに見えるのは相手6体の種族・レベル・タイプと公開" +
    "のバトルログだけで、構成(技・持ち物・DV)はどちらも非公開です。" +
    "相手チームはプールからランダムに引かれ、再戦のたびに引き直します。",
  blindSheetNote:
    "ブラインド: 互いの構成は非公開です。ボットに渡るのもあなたの6体の" +
    "種族・レベル・タイプだけで、条件は同じです。",
  oppRandomBlind: "毎回プールからランダム",
  foeTeamBlind: "相手のパーティ",
  previewTapHintBlind:
    "ブラインド — 相手側は種族・レベル・タイプのみ公開です。自分の側は ▸ ボタンで構成を開けます。",
  sheetNoteBlind:
    "相手の構成は非公開です。相手側は種族・レベル・タイプのみ、自分の" +
    "チームだけ全部表示されます。",
  revealFoeTeam: "相手の構成を見る",
  beliefChipPool: (n) => `ボットの読み: 候補${n}`,
  beliefChipOff: "ボットの読み: プール外",
  priorChip: (n, total) => `事前分布: ${n}/${total}`,
  priorLabel: "相手構成の事前分布",
  priorNone: "なし",
  priorTitle: "相手構成の事前分布",
  priorHelp:
    "相手の構成が読めないときに、どの技・持ち物がどれくらい出やすいか" +
    "を埋めるための分布表です。効くのはブラインドで、かつボットが相手" +
    "チームを特定できないとき — つまりあなたがカスタムチームを使うとき" +
    "だけです。",
  priorPick: "表ファイルを選ぶ…",
  priorSample: "サンプル表を読み込む",
  priorClear: "クリア",
  priorSummary: (species, meanMoveSum, skipped) =>
    `${species}種、技確率の平均合計 ${meanMoveSum.toFixed(2)}、` +
    `除外 ${skipped}件`,
  priorApplied: "適用されます",
  priorNotApplied: "適用されません",
  priorWarnings: "警告",
  priorLoadFailed: (why) => `表を読み込めませんでした — ${why}`,
  poolLabel: "チームプール",
  poolBundled: (n) => `同梱(${n}チーム)`,
  poolLoaded: (name, n) => `${name}(${n}チーム)`,
  poolTitle: "チームプール",
  poolHelp:
    "プールを使っている箇所はすべて、このファイルに置き換わります — " +
    "ボットが引くチーム、ブラインドでボットが相手を絞り込む候補、そして" +
    "この画面の自分・相手のチーム一覧です。形式は同梱プールと同じ JSON " +
    "で、{\"teams\": [{\"id\": …, \"sets\": […]}]} が最小。チームだけの配列" +
    "でも読めます(id・tier・rank は無ければこちらで補います)。各チーム" +
    "はちょうど6体で、この形式のルールに照らして検査し、対戦できない" +
    "チームが1つでもあればファイル全体を拒否します。",
  poolPick: "プールのファイルを選ぶ…",
  poolReset: "同梱プールに戻す",
  poolAccepted: (n) => `${n}チームを読み込みました`,
  poolRejected: "拒否しました — プールは元のままです",
  poolMore: (n) => `他${n}件`,
  poolNotStored: (why) =>
    `このセッションでは使えますが、保存できませんでした — ${why}。再読み込みで元に戻ります。`,
  poolErrJson: (why) => `JSON として読めません — ${why}`,
  poolErrNoTeams:
    "チームが1つもありません — {\"teams\": [ … ]} かチームの配列を渡してください。",
  poolErrSets: (team) => `チーム ${team}: sets がありません`,
  poolErrTeamSize: (team, n) => `チーム ${team}: ${n}体です(6体ちょうど)`,
  poolErrDupId: (team) => `チーム ${team}: id が重複しています`,
  poolErrTeam: (team, why) => `チーム ${team} — ${why}`,
  srLevel: (n) => `レベル${n}`,
  srGender: (g) => (g === "M" ? "オス" : g === "F" ? "メス" : "せいべつなし"),
  srBattleHeading: "対戦",
  srBattleLog: "バトルログ",
  srYourAction: "あなたの行動",
  srYourActive: "自分の場のポケモン",
  srFoeActive: "相手の場のポケモン",
  srNoItem: "もちものなし",
  srItemHeld: (it) => `もちもの ${it}`,
  srItemGone: (it) => `もちものなし(もとは${it})`,
  srYourTurn: "あなたの番です — 技か交代を選んでください。",
  srChooseSwitch: "次のポケモンを選んでください。",
  srBotThinking: "相手は考えている…",
  srSwitchTo: (species, hpPct) => `${species}に交代 — HP ${hpPct}%`,
  srPicked: (order) =>
    order === 0 ? "選出済み・先発" : `選出済み・${order + 1}番目`,
  srDeleteFor: (name) => `チーム「${name}」を削除`,
};

export const STRINGS: Record<Locale, UIStrings> = { en: EN, ja: JA };
