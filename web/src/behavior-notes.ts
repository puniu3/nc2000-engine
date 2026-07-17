// UI-3 special-behavior notes. One short line per move/item whose behavior
// is not evident from what the button already shows (name / type / category
// / BP / PP): callback-list moves from data/gen2stadium2.json plus
// data-field specials (priority, multi-hit, charge/recharge, self-KO,
// drain/recoil, OHKO, trapping, secondary effects, status/weather/screen
// setting, data-dependent power, call semantics) and the 38+4 held items
// with battle behavior. Plain-damage moves get NO entry — the tooltip
// layer only attaches where a note exists.
//
// Every line is written against THIS port's behavior (crates/engine
// handlers + the PS gen2/gen2stadium2 mod sources), not later-gen memory:
// e.g. Hyper Beam recharges even after a KO here, Counter answers any
// physical-category move (Hidden Power included), sleep from moves lasts
// 1-3 missed turns (Stadium), Haze also clears the brn/par stat cuts,
// Dragon Fang works and Dragon Scale does nothing (Stadium fix).
// Deliberately unnoted: Pay Day (coin message only, no battle effect).

export interface BehaviorNote {
  en: string;
  ja: string;
}

const N = (en: string, ja: string): BehaviorNote => ({ en, ja });

export const MOVE_NOTES: Record<string, BehaviorNote> = {
  // ------------------------------------------------- two-turn / recharge
  dig: N(
    "Two-turn move: underground turn 1 (Earthquake/Magnitude still hit it, for double damage).",
    "2ターン技: 1ターン目は地中に潜る(じしん/マグニチュードは2倍で命中)",
  ),
  fly: N(
    "Two-turn move: airborne turn 1 (Gust/Twister hit for double; Thunder/Whirlwind also hit).",
    "2ターン技: 1ターン目は空中(かぜおこし/たつまきは2倍で命中、かみなり/ふきとばしも当たる)",
  ),
  razorwind: N(
    "Two-turn move: charges turn 1. High critical-hit ratio.",
    "2ターン技: 1ターン目は溜め。急所に当たりやすい",
  ),
  skullbash: N(
    "Two-turn move: charges turn 1, raising Defense by 1.",
    "2ターン技: 1ターン目に溜めてぼうぎょ+1",
  ),
  skyattack: N(
    "Two-turn move: charges turn 1.",
    "2ターン技: 1ターン目は溜め",
  ),
  solarbeam: N(
    "Two-turn move: charges turn 1 (no charge in sun). Halved power in rain.",
    "2ターン技: 1ターン目は溜め(晴れなら溜めなし)。雨では威力半減",
  ),
  hyperbeam: N(
    "Must recharge the turn after it hits (even after a KO).",
    "命中した次のターンは反動で動けない(相手を倒しても)",
  ),
  // --------------------------------------------------------------- self-KO
  explosion: N(
    "User faints; the target's Defense is halved for the damage. Using it with your last Pokemon loses the game.",
    "自分はひんしになる(相手のぼうぎょを半分にして計算)。最後の1体で使うと負け",
  ),
  selfdestruct: N(
    "User faints; the target's Defense is halved for the damage. Using it with your last Pokemon loses the game.",
    "自分はひんしになる(相手のぼうぎょを半分にして計算)。最後の1体で使うと負け",
  ),
  // ------------------------------------------------------------------ OHKO
  fissure: N(
    "One-hit KO. Fails against a higher-level target.",
    "一撃必殺。相手のレベルが自分より高いと失敗",
  ),
  guillotine: N(
    "One-hit KO. Fails against a higher-level target.",
    "一撃必殺。相手のレベルが自分より高いと失敗",
  ),
  horndrill: N(
    "One-hit KO. Fails against a higher-level target.",
    "一撃必殺。相手のレベルが自分より高いと失敗",
  ),
  // -------------------------------------------------------------- priority
  quickattack: N("Priority +1: strikes first.", "先制技(優先度+1)"),
  machpunch: N("Priority +1: strikes first.", "先制技(優先度+1)"),
  extremespeed: N("Priority +1: strikes first.", "先制技(優先度+1)"),
  vitalthrow: N(
    "Never misses. Moves last (priority -1).",
    "必中。後攻(優先度-1)",
  ),
  counter: N(
    "Moves last. Returns double the damage of the physical-category hit taken this turn (Hidden Power counts as physical).",
    "後攻。このターン受けた物理技のダメージを2倍にして返す(めざめるパワーは物理扱い)",
  ),
  mirrorcoat: N(
    "Moves last. Returns double the damage of the special-category hit taken this turn.",
    "後攻。このターン受けた特殊技のダメージを2倍にして返す",
  ),
  roar: N(
    "Forces the target to switch to a random Pokemon. Fails unless the user moves last this turn.",
    "相手をランダムな控えと強制交代させる。このターン最後の行動でないと失敗",
  ),
  whirlwind: N(
    "Forces the target to switch to a random Pokemon. Fails unless the user moves last this turn.",
    "相手をランダムな控えと強制交代させる。このターン最後の行動でないと失敗",
  ),
  // -------------------------------------------------------- protect family
  protect: N(
    "Priority +2: blocks moves targeting the user this turn. Success halves with consecutive use.",
    "優先度+2。このターン相手の技を防ぐ(連続使用で成功率半減)",
  ),
  detect: N(
    "Priority +2: blocks moves targeting the user this turn. Success halves with consecutive use.",
    "優先度+2。このターン相手の技を防ぐ(連続使用で成功率半減)",
  ),
  endure: N(
    "Priority +2: survives any hit this turn with at least 1 HP. Success halves with consecutive use.",
    "優先度+2。このターンはHP1で必ず耐える(連続使用で成功率半減)",
  ),
  // --------------------------------------------------------------- healing
  recover: N("Restores 1/2 max HP.", "最大HPの半分を回復"),
  milkdrink: N("Restores 1/2 max HP.", "最大HPの半分を回復"),
  softboiled: N("Restores 1/2 max HP.", "最大HPの半分を回復"),
  rest: N(
    "Fully restores HP and status; the user sleeps for 2 turns.",
    "HPと状態異常を全回復し、2ターン眠る",
  ),
  moonlight: N(
    "Restores 1/2 max HP — full HP in sun, only 1/4 in rain or sandstorm.",
    "最大HPの半分を回復(晴れなら全回復、雨/砂あらしでは1/4)",
  ),
  morningsun: N(
    "Restores 1/2 max HP — full HP in sun, only 1/4 in rain or sandstorm.",
    "最大HPの半分を回復(晴れなら全回復、雨/砂あらしでは1/4)",
  ),
  synthesis: N(
    "Restores 1/2 max HP — full HP in sun, only 1/4 in rain or sandstorm.",
    "最大HPの半分を回復(晴れなら全回復、雨/砂あらしでは1/4)",
  ),
  // ----------------------------------------------------------------- drain
  absorb: N(
    "Restores half the damage dealt. Always misses against a Substitute.",
    "与えたダメージの半分を回復(みがわりには必ず失敗)",
  ),
  megadrain: N(
    "Restores half the damage dealt. Always misses against a Substitute.",
    "与えたダメージの半分を回復(みがわりには必ず失敗)",
  ),
  gigadrain: N(
    "Restores half the damage dealt. Always misses against a Substitute.",
    "与えたダメージの半分を回復(みがわりには必ず失敗)",
  ),
  leechlife: N(
    "Restores half the damage dealt. Always misses against a Substitute.",
    "与えたダメージの半分を回復(みがわりには必ず失敗)",
  ),
  dreameater: N(
    "Only works on a sleeping target. Restores half the damage dealt; always misses against a Substitute.",
    "相手がねむり状態のときだけ命中。与えたダメージの半分を回復(みがわりには失敗)",
  ),
  // ---------------------------------------------------------------- recoil
  doubleedge: N(
    "User takes 1/4 of the damage dealt as recoil.",
    "与えたダメージの1/4を反動で受ける",
  ),
  submission: N(
    "User takes 1/4 of the damage dealt as recoil.",
    "与えたダメージの1/4を反動で受ける",
  ),
  takedown: N(
    "User takes 1/4 of the damage dealt as recoil.",
    "与えたダメージの1/4を反動で受ける",
  ),
  struggle: N(
    "Used when no PP is left. User takes 1/4 of the damage dealt as recoil.",
    "PPが尽きたときに出る技。与えたダメージの1/4を反動で受ける",
  ),
  highjumpkick: N(
    "If it misses, the user takes 1/8 of the damage it would have dealt.",
    "外すと、与えるはずだったダメージの1/8を自分が受ける",
  ),
  jumpkick: N(
    "If it misses, the user takes 1/8 of the damage it would have dealt.",
    "外すと、与えるはずだったダメージの1/8を自分が受ける",
  ),
  // ------------------------------------------------------------- multi-hit
  barrage: N("Hits 2-5 times.", "2〜5回連続で攻撃"),
  bonerush: N("Hits 2-5 times.", "2〜5回連続で攻撃"),
  cometpunch: N("Hits 2-5 times.", "2〜5回連続で攻撃"),
  doubleslap: N("Hits 2-5 times.", "2〜5回連続で攻撃"),
  furyattack: N("Hits 2-5 times.", "2〜5回連続で攻撃"),
  furyswipes: N("Hits 2-5 times.", "2〜5回連続で攻撃"),
  pinmissile: N("Hits 2-5 times.", "2〜5回連続で攻撃"),
  spikecannon: N("Hits 2-5 times.", "2〜5回連続で攻撃"),
  doublekick: N("Hits twice.", "2回連続で攻撃"),
  bonemerang: N("Hits twice.", "2回連続で攻撃"),
  twineedle: N(
    "Hits twice; the last hit has a 20% chance to poison.",
    "2回連続で攻撃(2発目でどく20%)",
  ),
  triplekick: N(
    "Hits 1-3 times; power rises 10, 20, 30 with each hit.",
    "1〜3回連続で攻撃。威力は10→20→30と上がる",
  ),
  // ----------------------------------------------------- trapping / escape
  bind: N(
    "Traps the target for 2-5 turns: 1/16 max HP damage per turn, and it can't switch while the user stays in.",
    "相手を2〜5ターン拘束: 毎ターン最大HPの1/16ダメージ、使用者が場にいる間は交代不可",
  ),
  clamp: N(
    "Traps the target for 2-5 turns: 1/16 max HP damage per turn, and it can't switch while the user stays in.",
    "相手を2〜5ターン拘束: 毎ターン最大HPの1/16ダメージ、使用者が場にいる間は交代不可",
  ),
  firespin: N(
    "Traps the target for 2-5 turns: 1/16 max HP damage per turn, and it can't switch while the user stays in.",
    "相手を2〜5ターン拘束: 毎ターン最大HPの1/16ダメージ、使用者が場にいる間は交代不可",
  ),
  whirlpool: N(
    "Traps the target for 2-5 turns: 1/16 max HP damage per turn, and it can't switch while the user stays in.",
    "相手を2〜5ターン拘束: 毎ターン最大HPの1/16ダメージ、使用者が場にいる間は交代不可",
  ),
  wrap: N(
    "Traps the target for 2-5 turns: 1/16 max HP damage per turn, and it can't switch while the user stays in.",
    "相手を2〜5ターン拘束: 毎ターン最大HPの1/16ダメージ、使用者が場にいる間は交代不可",
  ),
  meanlook: N(
    "The target can no longer switch out (while the user remains in battle).",
    "相手は交代できなくなる(使用者が場にいる間)",
  ),
  spiderweb: N(
    "The target can no longer switch out (while the user remains in battle).",
    "相手は交代できなくなる(使用者が場にいる間)",
  ),
  // ---------------------------------------------------------- fixed damage
  seismictoss: N(
    "Deals damage equal to the user's level.",
    "自分のレベルと同じダメージを与える",
  ),
  nightshade: N(
    "Deals damage equal to the user's level.",
    "自分のレベルと同じダメージを与える",
  ),
  dragonrage: N("Always deals 40 damage.", "つねに40ダメージ"),
  sonicboom: N("Always deals 20 damage.", "つねに20ダメージ"),
  superfang: N(
    "Halves the target's current HP.",
    "相手の現在HPを半分にする",
  ),
  psywave: N(
    "Random damage between 1 and 1.5x the user's level.",
    "自分のレベルの1〜1.5倍のランダムダメージ",
  ),
  // -------------------------------------------------------- data-dependent
  hiddenpower: N(
    "Type and power are fixed by the user's DVs (shown values are the actual ones).",
    "タイプと威力は個体値(DV)で決まる(表示されている値で固定)",
  ),
  return: N(
    "Power scales with happiness (max 102).",
    "なつき度が高いほど高威力(最大102)",
  ),
  frustration: N(
    "Power scales inversely with happiness (max 102 at minimum happiness).",
    "なつき度が低いほど高威力(最大102)",
  ),
  magnitude: N(
    "Random power 10-150 (70 most often). Hits Dig users for double damage.",
    "威力10〜150でランダム(70が最頻)。地中の相手には2倍で命中",
  ),
  present: N(
    "Random: 40, 80 or 120 power — or heals the target 1/4 max HP (20%).",
    "ランダムで威力40/80/120、20%で相手のHPを1/4回復してしまう",
  ),
  // ------------------------------------------------------------ call moves
  metronome: N("Uses a random move.", "ランダムな技を繰り出す"),
  sleeptalk: N(
    "Only usable while asleep: uses one of the user's other moves at random (not two-turn moves).",
    "ねむり中のみ使用可。自分の他の技をランダムに繰り出す(溜め技は除く)",
  ),
  mirrormove: N(
    "Uses the move the target last used; fails if there is none.",
    "相手が最後に使った技を繰り出す(なければ失敗)",
  ),
  snore: N(
    "Only usable while asleep. 30% chance to flinch.",
    "ねむり中のみ使用可。命中時ひるみ30%",
  ),
  // ----------------------------------------------------- weather / screens
  raindance: N(
    "5 turns of rain: Water moves 1.5x, Fire and Solar Beam halved, Thunder never misses.",
    "5ターン雨: みず技1.5倍、ほのお技とソーラービーム半減、かみなり必中",
  ),
  sunnyday: N(
    "5 turns of sun: Fire moves 1.5x, Water halved, Solar Beam skips its charge, Thunder drops to 50% accuracy.",
    "5ターン晴れ: ほのお技1.5倍、みず技半減、ソーラービーム即発動、かみなりは命中率50%",
  ),
  sandstorm: N(
    "5 turns: non-Rock/Ground/Steel Pokemon take 1/8 max HP at the end of each turn.",
    "5ターン砂あらし: いわ/じめん/はがね以外は毎ターン最大HPの1/8ダメージ",
  ),
  reflect: N(
    "For 5 turns, your side's Defense is doubled against physical moves.",
    "5ターン、自分側のぼうぎょが2倍(物理ダメージ半減)",
  ),
  lightscreen: N(
    "For 5 turns, your side's Sp. Def is doubled against special moves.",
    "5ターン、自分側のとくぼうが2倍(特殊ダメージ半減)",
  ),
  safeguard: N(
    "For 5 turns, your side can't receive status conditions or confusion.",
    "5ターン、自分側は状態異常やこんらんにならない",
  ),
  spikes: N(
    "Lays spikes: non-Flying foes take 1/8 max HP when switching in (one layer; Rapid Spin removes it).",
    "まきびしを設置: 交代で出てきた相手(ひこう以外)に最大HP1/8のダメージ(1層のみ、こうそくスピンで解除)",
  ),
  rapidspin: N(
    "Frees the user's side from Spikes, binding moves and Leech Seed.",
    "まきびし・拘束技・やどりぎのタネを自分の側から取り除く",
  ),
  mist: N(
    "Protects the user from stat drops caused by the opponent (until it switches out).",
    "相手の技による能力ダウンを防ぐ(交代するまで)",
  ),
  // -------------------------------------------------------- status setters
  thunderwave: N(
    "Paralyzes the target (Ground-types are immune).",
    "相手をまひにする(じめんタイプには無効)",
  ),
  glare: N("Paralyzes the target.", "相手をまひにする"),
  stunspore: N("Paralyzes the target.", "相手をまひにする"),
  sing: N(
    "Puts the target to sleep for 1-3 turns.",
    "相手をねむりにする(1〜3ターン行動不能)",
  ),
  sleeppowder: N(
    "Puts the target to sleep for 1-3 turns.",
    "相手をねむりにする(1〜3ターン行動不能)",
  ),
  spore: N(
    "Puts the target to sleep for 1-3 turns.",
    "相手をねむりにする(1〜3ターン行動不能)",
  ),
  hypnosis: N(
    "Puts the target to sleep for 1-3 turns.",
    "相手をねむりにする(1〜3ターン行動不能)",
  ),
  lovelykiss: N(
    "Puts the target to sleep for 1-3 turns.",
    "相手をねむりにする(1〜3ターン行動不能)",
  ),
  poisonpowder: N("Poisons the target.", "相手をどくにする"),
  poisongas: N("Poisons the target.", "相手をどくにする"),
  toxic: N(
    "Badly poisons: damage starts at 1/16 max HP and grows each turn. Reverts to regular poison on switching out.",
    "もうどく: ダメージが最大HPの1/16から毎ターン増えていく(交代すると通常のどくに戻る)",
  ),
  confuseray: N("Confuses the target.", "相手をこんらんさせる"),
  supersonic: N("Confuses the target.", "相手をこんらんさせる"),
  sweetkiss: N("Confuses the target.", "相手をこんらんさせる"),
  swagger: N(
    "Raises the target's Attack by 2 and confuses it.",
    "相手のこうげきを2段階上げて、こんらんさせる",
  ),
  attract: N(
    "Infatuates an opposite-gender target: 50% chance it can't act each turn.",
    "性別が逆の相手をメロメロにする: 毎ターン50%で行動不能",
  ),
  nightmare: N(
    "While the target stays asleep, it loses 1/4 max HP each turn.",
    "ねむっている相手に毎ターン最大HPの1/4ダメージ",
  ),
  leechseed: N(
    "Seeds the target: 1/8 of its max HP is drained to the user's side each turn. Grass-types are immune.",
    "植え付けた相手から毎ターン最大HPの1/8を吸い取る(くさタイプには無効)",
  ),
  curse: N(
    "Ghost user: pays 1/2 max HP and the cursed target loses 1/4 max HP each turn. Otherwise: Atk/Def +1, Speed -1.",
    "ゴースト: HP半分と引き換えに相手へ毎ターン1/4ダメージ。それ以外: 攻撃/防御+1、素早さ-1",
  ),
  // ------------------------------------------------------- move manipulation
  disable: N(
    "Disables the move the target last used, for several turns.",
    "相手が最後に使った技を数ターン使えなくする",
  ),
  encore: N(
    "The target repeats its last move for 3-6 turns.",
    "相手は3〜6ターン、最後に使った技しか出せなくなる",
  ),
  spite: N(
    "Cuts the PP of the target's last move by 2-5.",
    "相手が最後に使った技のPPを2〜5減らす",
  ),
  mimic: N(
    "Copies the target's last move in Mimic's slot (until the user switches out).",
    "相手が最後に使った技を自分の技としてコピー(交代するまで)",
  ),
  sketch: N("Always fails in this format.", "この対戦形式ではつねに失敗する"),
  // ----------------------------------------------------------- self / misc
  bellydrum: N(
    "Halves the user's HP and maximizes Attack. Fails at half HP or below.",
    "HPを半分削って、こうげきを最大まで上げる(HPが半分以下だと失敗)",
  ),
  destinybond: N(
    "Until the user moves again: if a foe's attack KOs it, that foe faints too. Fails for your last Pokemon.",
    "次に行動するまで、相手の攻撃で倒されると相手も道連れにする(最後の1体では失敗)",
  ),
  perishsong: N(
    "All active Pokemon faint after 3 turns unless they switch out. Fails for your last Pokemon.",
    "場の全ポケモンは3ターン後にひんし(交代すれば回避。最後の1体では失敗)",
  ),
  haze: N(
    "Resets all stat changes on both sides — including the Attack/Speed cuts from burn/paralysis.",
    "両者の能力変化をすべて消す(やけど/まひによる能力低下も元に戻る)",
  ),
  bide: N(
    "Waits 2-3 turns, then returns double the damage taken meanwhile.",
    "2〜3ターン耐えたあと、受けたダメージの2倍を返す",
  ),
  rage: N(
    "Until the user acts again, every hit it takes raises its Attack by 1.",
    "次の行動まで、攻撃を受けるたびにこうげき+1",
  ),
  rollout: N(
    "Locks in for up to 5 turns; power doubles with each consecutive hit (doubled again after Defense Curl).",
    "最大5ターン連続で使い、当てるたびに威力2倍(まるくなるの後はさらに2倍)",
  ),
  furycutter: N(
    "Power doubles with each consecutive hit, up to 160.",
    "連続で当てるたびに威力2倍(最大160)",
  ),
  defensecurl: N(
    "Raises Defense by 1 and doubles the power of the user's Rollout.",
    "ぼうぎょ+1。以後の自分のころがるの威力が2倍になる",
  ),
  thrash: N(
    "Rampages for 2-3 turns, then the user becomes confused.",
    "2〜3ターン暴れ続け、そのあと自分がこんらんする",
  ),
  petaldance: N(
    "Rampages for 2-3 turns, then the user becomes confused.",
    "2〜3ターン暴れ続け、そのあと自分がこんらんする",
  ),
  outrage: N(
    "Rampages for 2-3 turns, then the user becomes confused.",
    "2〜3ターン暴れ続け、そのあと自分がこんらんする",
  ),
  reversal: N(
    "More power the lower the user's HP (up to 200 BP). Never crits; no damage range.",
    "残りHPが少ないほど高威力(最大200)。急所に当たらず、乱数幅なし",
  ),
  flail: N(
    "More power the lower the user's HP (up to 200 BP). Never crits; no damage range.",
    "残りHPが少ないほど高威力(最大200)。急所に当たらず、乱数幅なし",
  ),
  pursuit: N(
    "If the target switches out, hits it first at double power.",
    "相手が交代するとき、その前に2倍の威力で攻撃する",
  ),
  falseswipe: N(
    "Always leaves the target with at least 1 HP.",
    "相手のHPを必ず1残す",
  ),
  minimize: N(
    "Raises evasion by 1. Stomp hits a minimized target for double damage.",
    "回避率+1(ちいさくなった相手にふみつけは2倍で命中)",
  ),
  futuresight: N(
    "Typeless damage lands two turns after use.",
    "2ターン後にダメージが発生(タイプ相性の影響なし)",
  ),
  beatup: N(
    "Hits once for each healthy, status-free party member. Typeless damage.",
    "ひんし・状態異常でない手持ち1体につき1回攻撃(タイプ相性の影響なし)",
  ),
  painsplit: N(
    "Averages the user's and the target's HP.",
    "自分と相手のHPを足して半分ずつに分け合う",
  ),
  healbell: N(
    "Cures the whole party's status conditions.",
    "手持ち全員の状態異常を治す",
  ),
  batonpass: N(
    "Switches out, passing stat changes and volatile effects to the replacement.",
    "控えと交代し、能力変化などの状態を引き継ぐ",
  ),
  psychup: N(
    "Copies the target's stat changes onto the user.",
    "相手の能力変化を自分にコピーする",
  ),
  transform: N(
    "Transforms into the target: copies species, stats, types and moves (5 PP each).",
    "相手に変身: 種族・能力・タイプ・技をコピー(各技PP5)",
  ),
  conversion: N(
    "Changes the user's type to the type of one of its own moves (chosen at random).",
    "自分のタイプを、自分の技のどれかと同じタイプに変える(ランダム)",
  ),
  conversion2: N(
    "Changes the user's type to one that resists the target's last move (chosen at random).",
    "相手が最後に使った技に強いタイプへ自分を変化させる(ランダム)",
  ),
  substitute: N(
    "Pays 1/4 max HP for a substitute that blocks status and damage until it breaks.",
    "最大HPの1/4を消費してみがわりを作る(壊れるまで状態異常や攻撃を肩代わり)",
  ),
  focusenergy: N(
    "Raises the user's critical-hit ratio by one stage (until it switches out).",
    "急所率を1段階上げる(交代するまで)",
  ),
  foresight: N(
    "Normal/Fighting moves can now hit the Ghost target; its evasion boosts are ignored.",
    "ゴーストにノーマル/かくとう技が当たるようになり、相手の回避率アップも無視する",
  ),
  lockon: N(
    "The user's next move cannot miss the target.",
    "次に使う技が必中になる",
  ),
  mindreader: N(
    "The user's next move cannot miss the target.",
    "次に使う技が必中になる",
  ),
  thief: N(
    "Steals the target's held item if the user has none.",
    "自分が持ち物なしのとき、相手の持ち物を奪う",
  ),
  splash: N("Does nothing.", "何も起こらない"),
  teleport: N("Always fails in trainer battles.", "対戦ではつねに失敗する"),
  swift: N("Never misses.", "必中"),
  feintattack: N("Never misses.", "必中"),
  // --------------------------------------------- always-on secondary (100%)
  zapcannon: N("Always paralyzes on hit.", "命中すれば必ずまひ"),
  dynamicpunch: N("Always confuses on hit.", "命中すれば必ずこんらん"),
  icywind: N(
    "Always lowers the target's Speed by 1 on hit.",
    "命中時、必ず相手のすばやさ-1",
  ),
  mudslap: N(
    "Always lowers the target's accuracy by 1 on hit.",
    "命中時、必ず相手の命中率-1",
  ),
  // ------------------------------------------------------ chance secondary
  bodyslam: N("30% chance to paralyze.", "命中時まひ30%"),
  dragonbreath: N("30% chance to paralyze.", "命中時まひ30%"),
  lick: N("30% chance to paralyze.", "命中時まひ30%"),
  spark: N("30% chance to paralyze.", "命中時まひ30%"),
  thunder: N(
    "30% chance to paralyze. Never misses in rain; only 50% accurate in sun.",
    "命中時まひ30%。雨なら必中、晴れでは命中率50%",
  ),
  thunderbolt: N("10% chance to paralyze.", "命中時まひ10%"),
  thunderpunch: N("10% chance to paralyze.", "命中時まひ10%"),
  thundershock: N("10% chance to paralyze.", "命中時まひ10%"),
  fireblast: N("10% chance to burn.", "命中時やけど10%"),
  flamethrower: N("10% chance to burn.", "命中時やけど10%"),
  ember: N("10% chance to burn.", "命中時やけど10%"),
  firepunch: N("10% chance to burn.", "命中時やけど10%"),
  flamewheel: N(
    "10% chance to burn. Usable while frozen, thawing the user.",
    "命中時やけど10%。こおり状態でも使えて、自分のこおりが治る",
  ),
  sacredfire: N(
    "50% chance to burn. Usable while frozen, thawing the user.",
    "命中時やけど50%。こおり状態でも使えて、自分のこおりが治る",
  ),
  blizzard: N("10% chance to freeze.", "命中時こおり10%"),
  icebeam: N("10% chance to freeze.", "命中時こおり10%"),
  icepunch: N("10% chance to freeze.", "命中時こおり10%"),
  powdersnow: N("10% chance to freeze.", "命中時こおり10%"),
  poisonsting: N("30% chance to poison.", "命中時どく30%"),
  sludge: N("30% chance to poison.", "命中時どく30%"),
  sludgebomb: N("30% chance to poison.", "命中時どく30%"),
  smog: N("40% chance to poison.", "命中時どく40%"),
  triattack: N(
    "20% chance to paralyze, burn or freeze (chosen at random).",
    "命中時20%でまひ/やけど/こおりのどれか",
  ),
  bite: N("30% chance to flinch.", "命中時ひるみ30%"),
  headbutt: N("30% chance to flinch.", "命中時ひるみ30%"),
  lowkick: N("30% chance to flinch.", "命中時ひるみ30%"),
  rockslide: N("30% chance to flinch.", "命中時ひるみ30%"),
  rollingkick: N("30% chance to flinch.", "命中時ひるみ30%"),
  stomp: N("30% chance to flinch.", "命中時ひるみ30%"),
  boneclub: N("10% chance to flinch.", "命中時ひるみ10%"),
  hyperfang: N("10% chance to flinch.", "命中時ひるみ10%"),
  twister: N("20% chance to flinch.", "命中時ひるみ20%"),
  confusion: N("10% chance to confuse.", "命中時こんらん10%"),
  psybeam: N("10% chance to confuse.", "命中時こんらん10%"),
  dizzypunch: N("20% chance to confuse.", "命中時こんらん20%"),
  acid: N(
    "10% chance to lower the target's Defense by 1.",
    "命中時10%で相手のぼうぎょ-1",
  ),
  irontail: N(
    "30% chance to lower the target's Defense by 1.",
    "命中時30%で相手のぼうぎょ-1",
  ),
  rocksmash: N(
    "50% chance to lower the target's Defense by 1.",
    "命中時50%で相手のぼうぎょ-1",
  ),
  crunch: N(
    "20% chance to lower the target's Sp. Def by 1.",
    "命中時20%で相手のとくぼう-1",
  ),
  psychic: N(
    "10% chance to lower the target's Sp. Def by 1.",
    "命中時10%で相手のとくぼう-1",
  ),
  shadowball: N(
    "20% chance to lower the target's Sp. Def by 1.",
    "命中時20%で相手のとくぼう-1",
  ),
  aurorabeam: N(
    "10% chance to lower the target's Attack by 1.",
    "命中時10%で相手のこうげき-1",
  ),
  bubble: N(
    "10% chance to lower the target's Speed by 1.",
    "命中時10%で相手のすばやさ-1",
  ),
  bubblebeam: N(
    "10% chance to lower the target's Speed by 1.",
    "命中時10%で相手のすばやさ-1",
  ),
  constrict: N(
    "10% chance to lower the target's Speed by 1.",
    "命中時10%で相手のすばやさ-1",
  ),
  octazooka: N(
    "50% chance to lower the target's accuracy by 1.",
    "命中時50%で相手の命中率-1",
  ),
  ancientpower: N(
    "10% chance to raise all of the user's stats by 1.",
    "命中時10%で自分の全能力+1",
  ),
  metalclaw: N(
    "10% chance to raise the user's Attack by 1.",
    "命中時10%で自分のこうげき+1",
  ),
  steelwing: N(
    "10% chance to raise the user's Defense by 1.",
    "命中時10%で自分のぼうぎょ+1",
  ),
  // -------------------------------------------------------------- high crit
  aeroblast: N("High critical-hit ratio.", "急所に当たりやすい"),
  crabhammer: N("High critical-hit ratio.", "急所に当たりやすい"),
  crosschop: N("High critical-hit ratio.", "急所に当たりやすい"),
  karatechop: N("High critical-hit ratio.", "急所に当たりやすい"),
  razorleaf: N("High critical-hit ratio.", "急所に当たりやすい"),
  slash: N("High critical-hit ratio.", "急所に当たりやすい"),
  // ------------------------------------------------------------ stat moves
  swordsdance: N("Raises the user's Attack by 2.", "自分のこうげき+2"),
  meditate: N("Raises the user's Attack by 1.", "自分のこうげき+1"),
  sharpen: N("Raises the user's Attack by 1.", "自分のこうげき+1"),
  acidarmor: N("Raises the user's Defense by 2.", "自分のぼうぎょ+2"),
  barrier: N("Raises the user's Defense by 2.", "自分のぼうぎょ+2"),
  harden: N("Raises the user's Defense by 1.", "自分のぼうぎょ+1"),
  withdraw: N("Raises the user's Defense by 1.", "自分のぼうぎょ+1"),
  growth: N("Raises the user's Sp. Atk by 1.", "自分のとくこう+1"),
  amnesia: N("Raises the user's Sp. Def by 2.", "自分のとくぼう+2"),
  agility: N("Raises the user's Speed by 2.", "自分のすばやさ+2"),
  doubleteam: N("Raises the user's evasion by 1.", "自分の回避率+1"),
  growl: N("Lowers the target's Attack by 1.", "相手のこうげき-1"),
  charm: N("Lowers the target's Attack by 2.", "相手のこうげき-2"),
  leer: N("Lowers the target's Defense by 1.", "相手のぼうぎょ-1"),
  tailwhip: N("Lowers the target's Defense by 1.", "相手のぼうぎょ-1"),
  screech: N("Lowers the target's Defense by 2.", "相手のぼうぎょ-2"),
  stringshot: N("Lowers the target's Speed by 1.", "相手のすばやさ-1"),
  cottonspore: N("Lowers the target's Speed by 2.", "相手のすばやさ-2"),
  scaryface: N("Lowers the target's Speed by 2.", "相手のすばやさ-2"),
  flash: N("Lowers the target's accuracy by 1.", "相手の命中率-1"),
  kinesis: N("Lowers the target's accuracy by 1.", "相手の命中率-1"),
  sandattack: N("Lowers the target's accuracy by 1.", "相手の命中率-1"),
  smokescreen: N("Lowers the target's accuracy by 1.", "相手の命中率-1"),
  sweetscent: N("Lowers the target's evasion by 1.", "相手の回避率-1"),
};

export const ITEM_NOTES: Record<string, BehaviorNote> = {
  // ------------------------------------------------------ berries (gen 2)
  berry: N(
    "Restores 10 HP when at half HP or less. Single use.",
    "HPが半分以下になると10回復(1回きり)",
  ),
  goldberry: N(
    "Restores 30 HP when at half HP or less. Single use.",
    "HPが半分以下になると30回復(1回きり)",
  ),
  berryjuice: N(
    "Restores 20 HP when at half HP or less. Single use.",
    "HPが半分以下になると20回復(1回きり)",
  ),
  przcureberry: N(
    "Cures the holder's paralysis. Single use.",
    "まひを自動で治す(1回きり)",
  ),
  psncureberry: N(
    "Cures the holder's poison. Single use.",
    "どくを自動で治す(1回きり)",
  ),
  mintberry: N(
    "Wakes the holder from sleep. Single use.",
    "ねむりを自動で治す(1回きり)",
  ),
  iceberry: N(
    "Cures the holder's burn. Single use.",
    "やけどを自動で治す(1回きり)",
  ),
  burntberry: N(
    "Thaws the holder from freeze. Single use.",
    "こおりを自動で治す(1回きり)",
  ),
  bitterberry: N(
    "Cures the holder's confusion. Single use.",
    "こんらんを自動で治す(1回きり)",
  ),
  miracleberry: N(
    "Cures any status condition or confusion. Single use.",
    "状態異常とこんらんをなんでも自動で治す(1回きり)",
  ),
  mysteryberry: N(
    "Restores 5 PP to a move that hits 0 PP. Single use.",
    "PPが0になった技を5回復(1回きり)",
  ),
  // ---------------------------------------------------------- battle items
  leftovers: N(
    "Restores 1/16 max HP at the end of every turn.",
    "毎ターン終了時に最大HPの1/16を回復",
  ),
  quickclaw: N(
    "~23% chance each turn to move first within the priority bracket.",
    "毎ターン約23%で同じ優先度の中で先に行動できる",
  ),
  focusband: N(
    "~12% chance to survive a KO hit with 1 HP.",
    "約12%で、ひんしになる攻撃をHP1で耐える",
  ),
  kingsrock: N(
    "Adds a ~12% flinch chance to most damaging moves.",
    "多くの攻撃技に約12%のひるみ効果を追加",
  ),
  brightpowder: N(
    "Lowers the accuracy of moves against the holder by ~8%.",
    "相手の技の命中率を約8%下げる",
  ),
  berserkgene: N(
    "On switch-in: Attack +2, but the holder becomes confused. Single use.",
    "場に出るとこうげき+2、かわりにこんらんする(1回きり)",
  ),
  scopelens: N(
    "Raises the holder's critical-hit ratio by one stage.",
    "急所率を1段階上げる",
  ),
  luckypunch: N(
    "Chansey only: greatly raises its critical-hit ratio.",
    "ラッキー専用: 急所率が大きく上がる",
  ),
  stick: N(
    "Farfetch'd only: greatly raises its critical-hit ratio.",
    "カモネギ専用: 急所率が大きく上がる",
  ),
  thickclub: N(
    "Cubone/Marowak only: doubles Attack.",
    "カラカラ/ガラガラ専用: こうげきが2倍",
  ),
  lightball: N(
    "Pikachu only: doubles Sp. Atk.",
    "ピカチュウ専用: とくこうが2倍",
  ),
  metalpowder: N(
    "Ditto only: Defense and Sp. Def x1.5 (even while Transformed).",
    "メタモン専用: ぼうぎょ/とくぼうが1.5倍(へんしん後も有効)",
  ),
  mail: N(
    "No effect, but it cannot be stolen by Thief.",
    "効果はないが、どろぼうで奪われない",
  ),
  // ---------------------------------------------------- type-boost (1.1x)
  blackbelt: N("Fighting-type moves 1.1x power.", "かくとう技の威力1.1倍"),
  blackglasses: N("Dark-type moves 1.1x power.", "あく技の威力1.1倍"),
  charcoal: N("Fire-type moves 1.1x power.", "ほのお技の威力1.1倍"),
  dragonfang: N(
    "Dragon-type moves 1.1x power (works in Stadium 2, unlike GSC).",
    "ドラゴン技の威力1.1倍(スタジアム2では正しく機能する)",
  ),
  dragonscale: N(
    "No effect in Stadium 2 (the GSC Dragon boost bug is fixed).",
    "スタジアム2では効果なし(金銀のバグが修正されている)",
  ),
  hardstone: N("Rock-type moves 1.1x power.", "いわ技の威力1.1倍"),
  magnet: N("Electric-type moves 1.1x power.", "でんき技の威力1.1倍"),
  metalcoat: N("Steel-type moves 1.1x power.", "はがね技の威力1.1倍"),
  miracleseed: N("Grass-type moves 1.1x power.", "くさ技の威力1.1倍"),
  mysticwater: N("Water-type moves 1.1x power.", "みず技の威力1.1倍"),
  nevermeltice: N("Ice-type moves 1.1x power.", "こおり技の威力1.1倍"),
  pinkbow: N("Normal-type moves 1.1x power.", "ノーマル技の威力1.1倍"),
  polkadotbow: N("Normal-type moves 1.1x power.", "ノーマル技の威力1.1倍"),
  poisonbarb: N("Poison-type moves 1.1x power.", "どく技の威力1.1倍"),
  sharpbeak: N("Flying-type moves 1.1x power.", "ひこう技の威力1.1倍"),
  silverpowder: N("Bug-type moves 1.1x power.", "むし技の威力1.1倍"),
  softsand: N("Ground-type moves 1.1x power.", "じめん技の威力1.1倍"),
  spelltag: N("Ghost-type moves 1.1x power.", "ゴースト技の威力1.1倍"),
  twistedspoon: N("Psychic-type moves 1.1x power.", "エスパー技の威力1.1倍"),
};

import { locale, toId } from "./i18n";

/** Note for a move (display name or PS id); Hidden Power variants share
 * the generic entry. Null when the move has no special behavior. */
export function moveNote(move: string): string | null {
  let id = toId(move);
  if (id.startsWith("hiddenpower")) id = "hiddenpower";
  const n = MOVE_NOTES[id];
  return n ? n[locale()] : null;
}

/** Note for a held item (display name or PS id). */
export function itemNote(item: string): string | null {
  const n = ITEM_NOTES[toId(item)];
  return n ? n[locale()] : null;
}
