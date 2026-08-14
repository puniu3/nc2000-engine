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
  // battle log, and the opponent is drawn from the pool anew on each battle.
  // Blind is reached only through `?blind` (info-mode.ts), so open mode
  // prints none of this — no banner, no note, not the word "blind" anywhere.
  //
  // blindBanner is the entire banner: one line, the only thing a blind player
  // reads before pressing Start. It used to be a heading over a paragraph and
  // is now neither, so it has to carry both halves of the deal on its own —
  // sets hidden both ways, and a random opponent every battle. There is no
  // second line left to lean on, and the start screen deliberately has no
  // opponent row to say the second half instead.
  //
  // The *Blind keys are drop-in replacements for their open-mode neighbours
  // (openSheetNote, foeTeam, previewTapHint, sheetNote) — same slot, blind copy.
  blindBanner: string;
  blindSheetNote: string;
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
  // the blind start screen's one setup entry: a single button under "Your
  // party" opening a single modal that holds both panels — team pool on top,
  // belief prior below. Two buttons and two modals shipped first and the
  // owner cut them to one, so blind is Start / banner / your party / this and
  // nothing else. settingsValue is what the button reads at rest: the state
  // of both panels, so neither has to be opened to be checked.
  settingsLabel: string;
  settingsTitle: string;
  settingsValue: (pool: string, prior: string) => string;
  // META-NASH v1's conclusion mode (`?nash`, info-mode.ts): blind rules, but
  // the opponent is drawn from the solved three-team mixture and nothing on
  // the screen is configurable. nashBanner replaces blindBanner in the same
  // one-line slot; it has to say what blind's says (sets hidden both ways,
  // new opponent every battle) AND why the opponent row below it is now
  // worth opening. The mixture's weights are shown, deliberately: the claim
  // is that knowing the distribution does not help, so hiding it would be
  // demonstrating something weaker than what was proved.
  nashBanner: string;
  nashTitle: string;
  nashMixNote: string;
  nashSource: (file: string) => string;
  // community belief prior (M18): a distribution table that fills in an
  // unidentifiable opponent's sets. Loaded by hand, never automatically.
  priorLabel: string;
  priorNone: string;
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
  // swappable team pool: one file replaces the pool everywhere blind mode
  // reads it — the bot's draw, the belief's candidate set, the human team
  // list — and poolHelp has to say all three, or the swap looks like it only
  // changes the list the user happens to be looking at. The panel now lives
  // inside the blind setup modal and is unreachable from open mode, which is
  // pinned to the bundled pool; so poolHelp names blind outright (it is the
  // only mode that can see this text) and says open ignores the file.
  poolLabel: string;
  poolBundled: (n: number) => string;
  poolLoaded: (name: string, n: number) => string;
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
  //
  // The two cap lines fire before anything is parsed, so they name no team:
  // validation runs synchronously on the UI thread, and a file big enough to
  // freeze the tab has to be turned away at the door. Both print the limit as
  // a number, because "too large" without it leaves nothing to aim at.
  poolErrTooLarge: (bytes: number, limit: number) => string;
  poolErrTooManyTeams: (n: number, limit: number) => string;
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
  /** The study board (`?solver`). */
  solver: SolverStrings;
}

/** File sizes for the pool loader's cap line, in the unit a file manager
 * shows — the number has to be comparable with what the OS says the file
 * weighs, or the limit is unactionable. MB/KB read the same in both locales,
 * so this sits outside the tables. */
function fileSize(bytes: number): string {
  if (bytes < 1_000_000) return `${Math.max(1, Math.round(bytes / 1000))} KB`;
  return `${(bytes / 1_000_000).toFixed(1).replace(/\.0$/, "")} MB`;
}


/** Study-board copy. Kept in its own interface so the screen's strings can
 * be read as a unit — and so a translator can see, at a glance, that every
 * number the board prints is hedged the same way in both languages: the
 * search's estimate is an estimate, and a field that rests on an imputed
 * opponent set says so. */
export interface SolverStrings {
  title: string;
  intro: string;
  yourSide: string;
  foeSide: string;
  fieldSection: string;
  fromPool: string;
  fromPaste: string;
  fromCustom: string;
  clear: string;
  noTeamYet: string;
  noFoeYet: string;
  activeLabel: string;
  setActive: string;
  faintedLabel: string;
  broughtLabel: string;
  hpLabel: string;
  statusLabel: string;
  restLabel: string;
  boostsLabel: string;
  volatilesLabel: string;
  ppLabel: string;
  revealedLabel: string;
  revealedHelp: string;
  addMove: string;
  itemLabel: string;
  itemUnknown: string;
  itemNone: string;
  turnLabel: string;
  weatherLabel: string;
  weatherNone: string;
  forceSwitchLabel: string;
  trappedLabel: string;
  previewLabel: string;
  solve: string;
  solving: string;
  deeper: string;
  stop: string;
  budgetLabel: string;
  reset: string;
  save: string;
  load: string;
  problems: string;
  problemOwnTeam: string;
  problemFoeTeam: string;
  problemOwnActive: string;
  problemFoeActive: string;
  problemHp: string;
  problemFainted: string;
  rejected: (why: string) => string;
  // results
  resultTitle: string;
  estimateNote: string;
  iterationsDone: (n: number, ms: number) => string;
  beliefCount: (n: number) => string;
  beliefOffPool: string;
  colOption: string;
  colShare: string;
  colWinRate: string;
  colVisits: string;
  dominatedTag: string;
  matrixTitle: string;
  matrixHelp: string;
  matrixEmptyCell: string;
  damageTitle: string;
  damageMine: string;
  damageTheirs: string;
  damageAssumed: string;
  koAlways: string;
  koPossible: string;
  koNever: (hits: number) => string;
  koNone: string;
  critLabel: string;
  lineTitle: string;
  lineAssumed: string;
  lineHelp: string;
  lineUs: string;
  lineThem: string;
  understood: string;
  understoodHelp: string;
  unknownMon: (pos: number) => string;
  pickedLabel: string;
  seenLabel: string;
  itemHeldLabel: string;
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
  blindBanner:
    "Blind — neither side sees the other's sets; a random opponent is drawn " +
    "each battle.",
  blindSheetNote:
    "Blind: neither side sees the other's sets. The bot gets your six " +
    "species, levels and types and nothing more — exactly what you get " +
    "from it.",
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
  settingsLabel: "Blind setup",
  settingsTitle: "Blind setup",
  settingsValue: (pool, prior) => `${pool} · ${prior}`,
  nashBanner:
    "The solved mixture — neither side sees the other's sets; each battle " +
    "the bot draws one of three teams at the odds below. Build whatever you " +
    "like: knowing the mixture is the point.",
  nashTitle: "The solved mixture",
  nashMixNote:
    "META-NASH v1's shipped answer: against a best-response search given the " +
    "same budget, no team and no blend beat this one. The bot samples it " +
    "afresh every battle, so which of the three you are facing stays hidden " +
    "until the game ends — the odds do not.",
  nashSource: (file) => `Solution: ${file}`,
  priorLabel: "Belief prior",
  priorNone: "None",
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
  poolHelp:
    "A pool file replaces the pool everywhere blind mode uses it: the teams " +
    "the opponent is drawn from, the candidates the bot narrows that " +
    "opponent down to, and your own team list on this screen. Without " +
    "?blind the page always plays the bundled pool and ignores this file. " +
    "Same JSON as the " +
    "bundled pool — {\"teams\": [{\"id\": …, \"sets\": […]}]} is the minimum, " +
    "a bare array of teams also reads, and id / tier / rank are filled in " +
    "when missing. Every team is exactly 6 Pokémon and is checked against " +
    "this format's rules; if a single team cannot play, the whole file is " +
    "refused.",
  poolPick: "Choose a pool file…",
  poolReset: "Use the bundled pool",
  poolAccepted: (n) => `${n} ${n === 1 ? "team" : "teams"} accepted`,
  poolRejected: "Rejected — nothing was changed",
  poolMore: (n) => `…and ${n} more`,
  poolNotStored: (why) =>
    `Loaded for this session, but not saved — ${why}. After a reload you are ` +
    `back on the bundled pool.`,
  poolErrTooLarge: (bytes, limit) =>
    `Pool is ${fileSize(bytes)} — the limit is ${fileSize(limit)}.`,
  poolErrTooManyTeams: (n, limit) =>
    `${n} teams in this file — the limit is ${limit}.`,
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
  solver: {
    title: "Solver",
    intro:
      "Describe a position and the bot scores every option on it — under the same information it gets on ladder: your sets exactly, the opponent by what has been shown.",
    yourSide: "Your side",
    foeSide: "Their side",
    fieldSection: "Field",
    fromPool: "From the pool",
    fromPaste: "Paste a team",
    fromCustom: "Saved teams",
    clear: "Clear",
    noTeamYet: "Load your six sets to begin.",
    noFoeYet: "Name their six Pokémon (species and level are public at preview).",
    activeLabel: "Out",
    setActive: "Send out",
    faintedLabel: "Fainted",
    broughtLabel: "Brought",
    hpLabel: "HP %",
    statusLabel: "Status",
    restLabel: "from Rest",
    boostsLabel: "Stat changes",
    volatilesLabel: "On the field",
    ppLabel: "PP left",
    revealedLabel: "Moves shown",
    revealedHelp: "Only moves you have actually seen. Everything else stays hidden — which is the point.",
    addMove: "Add a move",
    itemLabel: "Item",
    itemUnknown: "Unknown",
    itemNone: "None (gone)",
    turnLabel: "Turn",
    weatherLabel: "Weather",
    weatherNone: "None",
    forceSwitchLabel: "You must switch (yours fainted)",
    trappedLabel: "You cannot switch out",
    previewLabel: "Team preview (choosing three)",
    solve: "Analyze",
    solving: "Thinking",
    deeper: "Think longer",
    stop: "Stop",
    budgetLabel: "Search",
    reset: "New position",
    save: "Save",
    load: "Load",
    problems: "Finish the position first:",
    problemOwnTeam: "load your six sets",
    problemFoeTeam: "name their six Pokémon",
    problemOwnActive: "say which of yours is out",
    problemFoeActive: "say which of theirs is out",
    problemHp: "an HP value is out of range",
    problemFainted: "a Pokémon at 0 HP is not marked fainted",
    rejected: (why) => `That position cannot happen: ${why}`,
    resultTitle: "What the bot sees",
    estimateNote:
      "Win rates are the search's own estimate, not a solved value. Read them next to the visit counts: a line it barely looked at barely means anything.",
    iterationsDone: (n, ms) => `${n.toLocaleString()} playouts in ${(ms / 1000).toFixed(1)}s`,
    beliefCount: (n) => (n === 1 ? "their team is identified" : `${n} teams still fit`),
    beliefOffPool: "no known team fits — imputing set by set",
    colOption: "Option",
    colShare: "Search share",
    colWinRate: "Win rate",
    colVisits: "Playouts",
    dominatedTag: "proven pointless",
    matrixTitle: "Against each of their replies",
    matrixHelp:
      "Your option down the side, their reply across the top. This is what the single number hides: a move can be strong into a stay and lose to a switch.",
    matrixEmptyCell: "never tried",
    damageTitle: "Damage",
    damageMine: "You hit them",
    damageTheirs: "They hit you",
    damageAssumed: "assumed set",
    koAlways: "always KOs",
    koPossible: "KOs on a high roll",
    koNever: (hits) => `${hits}HKO`,
    koNone: "no damage",
    critLabel: "crit",
    lineTitle: "How it expects this to go",
    lineAssumed: "assuming their set is",
    lineHelp:
      "One continuation the search actually visited, under one guess at their hidden set. Change the guess and the line changes.",
    lineUs: "you",
    lineThem: "them",
    understood: "The board it read",
    understoodHelp:
      "Their HP lands in the middle of the percentage you gave — that is all a percentage says.",
    unknownMon: (pos) => `bench ${pos}`,
    pickedLabel: "Picked",
    seenLabel: "Seen",
    itemHeldLabel: "holds an item",
  },
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
  blindBanner:
    "ブラインド — 互いの構成(技・持ち物)は非公開。相手チームは毎回" +
    "ランダムに引き直します。",
  blindSheetNote:
    "ブラインド: 互いの構成は非公開です。ボットに渡るのもあなたの6体の" +
    "種族・レベル・タイプだけで、条件は同じです。",
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
  settingsLabel: "ブラインド設定",
  settingsTitle: "ブラインド設定",
  settingsValue: (pool, prior) => `${pool} · ${prior}`,
  nashBanner:
    "結論の混合戦略 — 互いの構成(技・持ち物)は非公開。ボットは毎回、下の" +
    "確率で3チームから1つを引きます。あなたの編成は自由 — 混合を知られても" +
    "崩れないことが結論の中身です。",
  nashTitle: "結論の混合戦略",
  nashMixNote:
    "META-NASH v1 の出荷解です。同じ予算を与えた最適応答探索でも、単体の" +
    "チームでも混合でも、これを上回るものは出ませんでした。ボットは毎回" +
    "引き直すので、今どれと当たっているかは終局まで伏せられます — 確率は" +
    "最初から公開です。",
  nashSource: (file) => `解: ${file}`,
  priorLabel: "相手構成の事前分布",
  priorNone: "なし",
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
  poolHelp:
    "ブラインドでプールを使っている箇所はすべて、このファイルに置き換わり" +
    "ます — 相手チームの抽選元、ボットが相手を絞り込む候補、そしてこの" +
    "画面の自分のチーム一覧です。?blind の付かない通常の画面は常に同梱" +
    "プールで対戦し、このファイルを見ません。形式は同梱プールと同じ JSON " +
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
    `このセッションでは使えますが、保存できませんでした — ${why}。` +
    `再読み込みすると同梱プールに戻ります。`,
  poolErrTooLarge: (bytes, limit) =>
    `プールが${fileSize(bytes)}あります — 上限は${fileSize(limit)}です。`,
  poolErrTooManyTeams: (n, limit) =>
    `チームが${n}件あります — 上限は${limit}件です。`,
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
  solver: {
    title: "ソルバ",
    intro:
      "盤面を書くと、bot が全部の選択肢に点数を付ける。情報はラダーと同じ — 自分のセットは正確に、相手は見えたものだけ。",
    yourSide: "自分",
    foeSide: "相手",
    fieldSection: "場",
    fromPool: "プールから",
    fromPaste: "貼り付け",
    fromCustom: "保存したチーム",
    clear: "クリア",
    noTeamYet: "まず自分の6体のセットを読み込む。",
    noFoeYet: "相手の6体を入れる（種族とレベルは選出画面で公開されている）。",
    activeLabel: "場",
    setActive: "場に出す",
    faintedLabel: "ひんし",
    broughtLabel: "選出済み",
    hpLabel: "HP %",
    statusLabel: "状態",
    restLabel: "ねむる由来",
    boostsLabel: "ランク",
    volatilesLabel: "場の状態",
    ppLabel: "残りPP",
    revealedLabel: "判明した技",
    revealedHelp: "実際に見た技だけ。それ以外は隠れたまま — そこが肝心。",
    addMove: "技を追加",
    itemLabel: "道具",
    itemUnknown: "不明",
    itemNone: "無し（消費済み）",
    turnLabel: "ターン",
    weatherLabel: "天候",
    weatherNone: "なし",
    forceSwitchLabel: "交代を選ぶ場面（自分がひんし）",
    trappedLabel: "交代できない",
    previewLabel: "選出画面（3体を選ぶ）",
    solve: "解析",
    solving: "思考中",
    deeper: "もっと考える",
    stop: "中断",
    budgetLabel: "探索量",
    reset: "新しい盤面",
    save: "保存",
    load: "読み込み",
    problems: "まだ足りない:",
    problemOwnTeam: "自分の6体のセット",
    problemFoeTeam: "相手の6体",
    problemOwnActive: "自分のどれが場に出ているか",
    problemFoeActive: "相手のどれが場に出ているか",
    problemHp: "HP の値が範囲外",
    problemFainted: "HP 0 なのに「ひんし」になっていない",
    rejected: (why) => `その盤面は成立しない: ${why}`,
    resultTitle: "bot の見え方",
    estimateNote:
      "勝率は探索の推定値であって解ではない。試行回数と並べて読むこと — ほとんど見ていない手の数字はほとんど何も言っていない。",
    iterationsDone: (n, ms) => `${n.toLocaleString()} 回の playout / ${(ms / 1000).toFixed(1)}秒`,
    beliefCount: (n) => (n === 1 ? "相手のチームは特定済み" : `候補はまだ ${n} チーム`),
    beliefOffPool: "既知のチームに当てはまらない — 1体ずつ推定している",
    colOption: "選択肢",
    colShare: "探索の配分",
    colWinRate: "勝率",
    colVisits: "playout",
    dominatedTag: "無意味と証明済み",
    matrixTitle: "相手の手ごとの勝率",
    matrixHelp:
      "縦が自分の手、横が相手の手。1つの数字が潰してしまうのはここ — 居座りには強いが交代に負ける技、というのが見える。",
    matrixEmptyCell: "未探索",
    damageTitle: "ダメージ",
    damageMine: "自分→相手",
    damageTheirs: "相手→自分",
    damageAssumed: "推定セット",
    koAlways: "確定KO",
    koPossible: "乱数KO",
    koNever: (hits) => `確定${hits}発`,
    koNone: "ダメージ無し",
    critLabel: "急所",
    lineTitle: "この後の読み筋",
    lineAssumed: "相手のセットを次と仮定",
    lineHelp:
      "探索が実際に通った進行を1本。相手の隠れたセットを1通りに仮定しているので、仮定が変われば読み筋も変わる。",
    lineUs: "自分",
    lineThem: "相手",
    understood: "ソルバが読み取った盤面",
    understoodHelp:
      "相手の HP は入力した％の中央値になる — ％が言えるのはそこまで。",
    unknownMon: (pos) => `控え${pos}`,
    pickedLabel: "選出",
    seenLabel: "出てきた",
    itemHeldLabel: "道具あり",
  },
};

export const STRINGS: Record<Locale, UIStrings> = { en: EN, ja: JA };
