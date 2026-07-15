# Porting notes (working memory — distilled from PS source @ 79c04dcc)

Read once from PS; keep updated as the port progresses. All line refs are PS repo files.

## Merged script resolution for mod gen2stadium2 (chain: gen2stadium2 → gen2 → gen3 → gen4 → base)

- `pokemon.getStat` — **gen2stadium2** (par/4 only with `parspeeddrop` volatile, brn/2 only with `brnattackdrop`; clamp 1..999 [fastReturn stops here]; screens ×2 (def/reflect, spd/lightscreen); thickclub/lightball ×2, metalpowder ×1.5 — uses `species.name` not baseSpecies)
- `pokemon.boostBy` — **gen2** (boost up ignored if getStat(stat,false,true)==999; clamps ±6, returns delta)
- `pokemon.getActionSpeed` — **gen3** (getStat spe; quickClawRoll && quickclaw → 65535)
- `actions.runMove` — **gen2**; `actions.useMove/useMoveInner` — base wrapper + **gen3** inner
- `actions.tryMoveHit`, `actions.getDamage` — **gen2stadium2**; `actions.moveHit` — **gen2**
- `actions.calcRecoilDamage` — **gen3**: clampIntRange(floor(dmg*recoil0/recoil1), 1)
- `actions.runSwitch` — **gen4** (EntryHazard ev, SwitchIn ev, gen<=2: all active lastMove=null; if !side.faintedThisTurn && draggedIn!=turn → AfterSwitchInSelf ev; isStarted=true; Start singleEvent for ability+item)
- `Battle.boost` — **gen2stadium2** (no ChangeBoost/getCappedBoost; TryBoost ev; per-stat: boostBy, on success remove brnattackdrop (atk) / parspeeddrop (spe); msg -boost/-unboost + '[from] fullname' when effect non-Move; AfterEachBoost; final AfterBoost)
- `Battle.faintMessages` — **gen2stadium2** (Self-KO clause: both sides 0 left → side whose lastMove non-null LOSES (win(other)); side.lastMove set on selfdestruct in tryMoveHit; gen2 faint → queue.cancelMove all active)
- base `switchIn`/`dragIn` — sim/battle-actions.ts:62/162 (gen<=4: pokemon.lastItem = oldActive.lastItem, oldActive.lastItem=''; isDrag && gen==2 → draggedIn=turn; non-drag → queue.insertChoice runSwitch)

## Battle flow essentials

- Constructor: log t: (stripped), format rules with handlers beyond {onBegin,onTeamPreview,onBattleStart,onValidateRule,onValidateTeam,onChangeSet,onValidateSet} → field.addPseudoWeather(rule). For NC2000 exactly: maxtotallevel (onChooseTeam), stadiumsleepclause (onSetStatus), freezeclausemod (onSetStatus). `add('gametype','singles')`.
- setPlayer → `|player|p1|P1||`; both set → start(): foe links, `|gen|2`, `|tier|[Gen 2] NC 2000`, rules' onBegin in ruleset order → `|rule|` lines: Stadium Sleep, Freeze, Species, Item(=1), Endless Battle, Event Moves, Beat Up Nicknames. cancelmod: supportCancel=true. runPickTeam → teampreview.onTeamPreview: `|clearpoke`, `|poke|side|details|` per getAllPokemon, makeRequest('teampreview') → `|teampreview|3`. Then queue.addChoice({start}), midTurn=true.
- makeRequest(type): sets requestState, clearChoice all sides, activeRequest=null, (teampreview → add line), getRequests: switch→forceSwitch only sides w/ switchFlag else wait; move→sides w/ pokemonLeft.
- side.choose: 'team ...' single string; else split ','. chooseTeam: positions 0-based, onChooseTeam(maxtotallevel: validates total ≤155 when data given), push actions {choice:'team', index, pokemon, priority:-index}.
- battle.choose(side, input): side.choose + isChoiceDone; allChoicesDone → commitChoices: updateSpeed(); oldQueue saved; inputLog; side.commitChoices → queue.addChoice; clearRequest; queue.sort() (**speedSort, RNG on full ties**); append oldQueue; turnLoop().
- turnLoop: add(''), add t: (stripped); requestState=''; if !midTurn: insertChoice beforeTurn + addChoice residual, midTurn=true; while queue.shift(): runAction; if requestState||ended return; endTurn(); midTurn=false; queue.clear().
- endTurn: turn++; per active: moveThisTurn='', newlySwitched=false, moveLastTurnResult shift; turn!=1 → usedItemThisTurn/hurtThisTurn resets; moveSlots disabled=false; runEvent DisableMove + per-slot singleEvent DisableMove(activeMove); trapped=maybeTrapped=false; runEvent TrapPokemon; (knownType always true gen2 → runEvent MaybeTrapPokemon); fainted→continue; activeTurns++. faintedLastTurn=faintedThisTurn, faintedThisTurn=null. maybeTriggerEndlessBattleClause (turn≤100 no-op except gen1). add('turn', N). **gen2: quickClawRoll = randomChance(60,256) — RNG EVERY endTurn**. makeRequest('move').
- runAction: 'start': pokemonLeft=6→pokemon.length? (`if (side.pokemonLeft) side.pokemonLeft = side.pokemon.length` — after team truncation =3), `|teamsize|`; add('start'); BattleStart singleEvents (species conds: none); switchIn(pokemon[0], 0) each side. 'team': index 0 → clear side.pokemon; push; position=index; **return early**. 'move': actions.runMove. 'switch'/'instaswitch': switchIn(target, pokemon.position). 'runSwitch': actions.runSwitch. 'residual': add(''); clearActiveMove(true); updateSpeed(); fieldEvent('Residual'); if !ended add('upkeep'). 'beforeTurn': eachEvent BeforeTurn.
  After-switch phazing: forceSwitchFlag → dragIn. clearActiveMove. faintMessages(); ended→true. If !queue.peek() || (gen≤3 && next is move|residual) → checkFainted (fainted actives get switchFlag=true, status='fnt' — **status set to 'fnt' string in PS but our fixture essence status shows 'fnt'?** — fixture pokemon status essence for fainted mon: PS sets status='fnt' as ID... on switch-out `if (oldActive.fainted) oldActive.status=''`. Mid-turn snapshots can show 'fnt'.) Then next instaswitch → return false (skips Update). gen<5 eachEvent('Update'). switch requests: sides w/ switchFlag: if !canSwitch → clear flags; else BeforeSwitchOut ev per active w/ hp+switchFlag (skipBeforeSwitchOutEventFlag), faintMessages; makeRequest('switch') return true.
- resolveAction: orders team=1,start=2,instaswitch=3,beforeTurn=4,beforeTurnMove=5,runSwitch=101,switch=103,priorityChargeMove=107,move/default=200,residual=300. move: beforeTurnCallback → unshift beforeTurnMove; fractionalPriority = runEvent FractionalPriority (0). switch: switchFlag string → sourceEffect=move; switchFlag=false. move: getActiveMove; targetLoc null → getRandomTarget + getLocOf; originalTarget=getAtLoc. getActionSpeed(action): move → priority = dex priority, singleEvent+runEvent ModifyPriority, action.priority = priority + fractionalPriority; speed = pokemon.getActionSpeed() (else 1).
- insertChoice: pokemon.updateSpeed(); resolveAction(midTurn); scan comparePriority; tie range → battle.random(firstIndex, lastIndex+1) **RNG**.
- comparePriority: order (default 2^32) asc, priority desc, speed desc, subOrder asc, effectOrder asc.
- speedSort: selection sort, ties → prng.shuffle(list, sorted, sorted+n) (n-1 draws).

## Event system

- runEvent: findEventHandlers(target,event,source); onEffect unshift (sourceEffect callback, used by moveHit? no — used with onEffect=true param: spreadDamage runEvent('Damage',...,true)? — YES spreadDamage passes onEffect=true for Damage). sort: ['Invulnerability','TryHit','DamagingHit','EntryHazard'] → compareLeftToRightOrder (order asc, priority desc, index asc); fastExit → compareRedirectOrder; else speedSort (**RNG on ties**). Suppression checks gen2-relevant: Status effect on holder whose status changed → skip (both singleEvent+runEvent). Weather handlers when suppressingWeather — no (no abilities). relayVar number+non-negative-int → modify(relayVar, event.modifier) at end.
- findEventHandlers(pokemon target): findPokemonEventHandlers(target, on{E}); prefixed (unless BeforeTurn/Update/Weather...): allies (self incl): onAlly/onAny; foes: onFoe/onAny; source: onSource{E}; then side-level (target.side): sideConditions on{E} own / onFoe{E} foes / onAny; field: pseudoWeathers, weather, terrain; battle: format (formatData) + this.events (unused).
- findPokemonEventHandlers order: status → volatiles (obj insertion order!) → ability → item → species → slotConditions.
- resolvePriority: order=effect[`${cb}Order`]??false, priority=[`${cb}Priority`]??0, subOrder=[`${cb}SubOrder`]??0 else effectType default {Condition:2, Weather:5, Rule/Format:5, side cond:4, slot cond:3, Ability:7, Item:8}; SwitchIn/RedirectTarget → effectOrder=state.effectOrder; holder w/ getStat → speed=pokemon.speed (already updated); cb endsWith SwitchIn → speed -= speedOrder.indexOf(getFieldPositionValue())/(activePerHalf*2).
- fieldEvent(eventid[, targets]): Residual → getKey='duration'. handlers: field onField{E} + per side onSide{E} + per active (targets filter): pokemon on{E}, side-as-holder-for-active, field-as-holder, battle(format). speedSort. Loop: shift; fainted holder skip (unless slot cond); Residual + handler.end + state.duration: `--duration; if 0 → end() (removeVolatile/removeSideCondition/clearWeather/removePseudoWeather/clearStatus), continue`. Stale-state identity check (state replaced → skip). holder Side → 'Side'+event; Field → 'Field'+event. singleEvent(handlerEventid, ...customCallback). faintMessages() after EACH handler; ended → return.
- eachEvent: getAllActive speedSort by speed desc (**RNG ties**), runEvent each.
- singleEvent: eventDepth guard 8; Status-changed guard; `effect.on{Event}` exists → run w/ pushed effect/state ctx.
- initEffectState: effectOrder = counter++ iff (id && target && (target not Pokemon || isActive)) else 0.
- getCallback quirk (gen≥5 only) — N/A gen2.

## Damage / heal plumbing

- spreadDamage(effect): clamp ≥1; weather immunity check runStatusImmunity(weather id); runEvent('Damage', target, source, effect, damage, **onEffect=true**); target.damage(); `-damage` msg variants: partiallytrapped → '[from] ' + volatiles.partiallytrapped.sourceEffect.fullname + '[partiallytrapped]'; confused → '[from] confusion'; Move/no-name → plain; else '[from] name' (+ '[of] source' if source && source!==target); tox fullname → 'psn'. gen≤4 drain: clampIntRange(floor(damage*num/den),1) → heal( , 'drain') → '-heal ... [from] drain [of] source'. instafaint → faintMessages(true) then gen≤2 target.faint().
- battle.damage: defaults from event (target=event.target etc. when args null).
- directDamage: clamp≥1, target.damage, '-damage' ('[from] confusion' for confusion), target.fainted → faint.
- heal: d≤1→1, trunc; TryHeal ev; hp checks; target.heal; '-heal' msgs (drain: '[from] drain [of] source').
- pokemon.damage: hp-=d; ≤0 → faint(source,effect) (queue faintQueue, hp=0, switchFlag=false).
- getHealth: '0 fnt'; secret `hp/maxhp`; shared pixels floor(48*hp/maxhp)||1 + '/48'; status appended to both.
- boost msgs handled in stadium2 Battle.boost above (no [silent]/[from] for Move effects).

## Move pipeline (gen2 runMove → useMove(base wrapper sets moveThisTurnResult) → gen3 useMoveInner → stadium2 tryMoveHit → gen2 moveHit → stadium2 getDamage)

- runMove(gen2): getActiveMove; getTarget(pokemon, move, targetLoc); OverrideAction ev (unless sourceEffect/struggle) — sleeptalk M2; !target → getRandomTarget; setActiveMove; moveThisTurn sanity; **BeforeMove ev** fail → MoveAborted ev + clearActiveMove(true) + AfterMoveSelf runEvent (sleep/par residuals!); beforeMoveCallback (gen2 moves? confusion via BeforeMove not this; M2); lastDamage=0; lockedMove = getLockedMove()||getSemiLocked; if !locked: deductPP (fail unless struggle → `|cant|...|nopp`), moveUsed(move) (gen2: metronome/mimic/mirrormove/sketch/sleeptalk/transform → lastMove=null else lastMove=move; lastMoveEncore likewise); useMove; singleEvent AfterMove; if !move.selfSwitch && foe.active[0].hp → runEvent AfterMoveSelf (psn/brn/tox residuals fire here).
- useMoveInner(gen3): sourceEffect from battle.effect if set; move= getActiveMove; pokemon.lastMoveUsed=move; if battle.activeMove: move.priority=activeMove.priority; baseTarget=move.target; target undefined → getRandomTarget; self→pokemon; setActiveMove; singleEvent ModifyMove (move callbacks e.g. thunder/struggle/present/magnitude — M2) + runEvent ModifyMove; `|move|poke|MoveName|target` via addMove (+ '[from] EffectName' if sourceEffect); !target → '[notarget]' attr + '-notarget'; getMoveTargets → pressure DeductPP ev (no Pressure in gen2 — mons... actually Pressure ability doesn't exist as ability; skip); singleEvent TryMove + runEvent TryMove (twoturnmove/primordialsea etc.); singleEvent UseMoveMessage (magnitude); ignoreImmunity default (Status → true); selfdestruct==='always'? (gen2 selfdestruct=true not 'always'? check data — gen2 explosion selfdestruct: "always"? in gen2 dex it's boolean-ish; banned in NC2000 anyway); target branches: all/foeSide/allySide/allyTeam → tryMoveHit direct; single: lacksTarget/fainted (+adjacency) → '[notarget]'+'-notarget'; damage=tryMoveHit(target...); moveResult bookkeeping; !moveResult → singleEvent MoveFail (jumpkick M2); (no sheerforce) → singleEvent+runEvent AfterMoveSecondarySelf (frz thaw on defrost move).
- tryMoveHit(stadium2): boost tables pos [1,1.33,1.66,2,2.33,2.66,3] neg [1,0.75,0.6,0.5,0.43,0.36,0.33]; selfdestruct → faint(pokemon) + side.lastMove bookkeeping (self-ko clause); singleEvent PrepareHit (fail → '-fail target') + runEvent PrepareHit; singleEvent Try (futuresight/snore); field/side targets → runEvent TryHitField/TryHitSide → moveHit; runEvent Invulnerability (twoturnmove conditions) false → '[miss]' attr + '-miss|poke'; ignoreImmunity default; runImmunity(move, true) → '-immune'; singleEvent TryImmunity ('-immune'); runEvent TryHit ('-fail'); OHKO: level< → '-immune [ohko]' return; accuracy: runEvent Accuracy; if num: floor(acc*255/100); ohko: +2×leveldiff cap255; boosts (unless ignoreAccuracy/ignoreEvasion): float multiply then `Math.min(Math.floor(acc),255)`, max 1; runEvent ModifyAccuracy; max(acc,0); runEvent Accuracy again(!); miss if acc!==true && acc!==255 && !randomChance(acc,256) → '[miss]'+'-miss|poke'; multihit: 2-5 → sample([2,2,2,3,3,3,4,5]) else random(lo,hi+1); loop hits (slp break unless sleepUsable; moveHit; eachEvent Update); `-hitcount|target|i`; single: damage=moveHit; category!==Status → gotAttacked; ohko → '-ohko'; singleEvent+runEvent AfterMoveSecondary (frz brn-thaw); recoil && totalDamage && (own left>1 || foe left>1 || target.hp) → damage(calcRecoil, pokemon, target, 'recoil').
- moveHit(gen2): singleEvent TryHit(moveData) / TryHitSide / TryHitField (moves w/ onTryHit — swagger/roar/whirlwind/mindreader M2); runEvent TryPrimaryHit (substitute — M2 move but condition...sub is callback move M2) hitResult 0 → target=null; isSecondary && !moveData.self → hitResult=true; getDamage(pokemon,target,moveData); damage/0 && !fainted → battle.damage(damage,target,pokemon,move) (false → 'damage interrupted' fail); false/null → '-fail' if primary; boosts && !fainted → battle.boost(...) (stadium2 boost); heal → target.heal(round? `Math.round(target.maxhp * heal0/heal1)`... note base uses this exact code in gen2 moveHit: `target.heal(Math.round(...))` wait it's in the copied gen2 moveHit — no it says (moveData.heal): `const d = target.heal(Math.round(target.maxhp * moveData.heal[0] / moveData.heal[1]));` hmm actually gen2 moveHit line 369: heal → -heal msg; status → trySetStatus (fail && move.status → return); forceStatus; volatileStatus → addVolatile; sideCondition; weather → field.setWeather; pseudoWeather; forceSwitch/selfSwitch canSwitch → didSomething; Hit events: singleEvent Hit + (primary) runEvent Hit + singleEvent AfterHit; nothing-happened check → '-fail'; **moveData.self: if !isSecondary && self.boosts → battle.random(100) burn a roll**, then moveHit(pokemon,pokemon,move,self,isSecondary,true); secondaries (target.hp && runEvent TrySecondaryHit): per secondary: brn/frz + target hasType(move.type) → skip; flinch vs slp/frz target → skip (unless kingsrock); if !multihit || lastHit: effectChance=floor((chance||100)*255/100); undefined chance || randomChance(effectChance,256) → moveHit(target,...,secondary,true,isSelf) (else if 255 → hint 1/256); forceSwitch && target.hp && pokemon.hp && canSwitch → runEvent DragOut → dragIn (roar M2 has onTryHit); selfSwitch && pokemon.hp → switchFlag=move.id.
- getDamage(stadium2): runImmunity(move,true)→false; ohko → target.maxhp; damageCallback; damage==='level' → level; move.damage number; category; type '???' ok; basePower; basePowerCallback; !bp → bp===0 ? undefined : bp; clamp≥1; critRatio = runEvent ModifyCritRatio(move.critRatio||0) clamp 0..5; critMult=[0,16,8,4,3,2]; isCrit=move.willCrit ?? (critRatio ? randomChance(1,critMult[critRatio]) : false); isCrit && runEvent CriticalHit → moveHitData.crit=true; BasePower runEvent (confusion self-hit uses baseMoveType); clamp≥1; attacker/defender overrides; atk/spa def/spd by category; isCrit: '-crit|target' now(!before stat get); if attacker boosts[atk] <= defender boosts[def] → unboosted=noburndrop=true; attack=getStat(atkType,unboosted,noburndrop), defense=getStat(defType,unboosted); ignoreOffensive → getStat(true,true); ignoreDefensive likewise; (present M2 glitch); if atk≥256||def≥256: atk=clamp(floor(clamp(atk,1,999)/4),1), def likewise (stadium2 — NO %256 rollover); selfdestruct && def → def=clamp(floor(def/2),1); dmg = level*2 floor/5 +2 *bp *atk floor/def floor/50; crit ×2; runEvent ModifyDamage → Math.floor; clamp 1..997; +2; weather: water/rain ×1.5 floor; fire/rain or water/sun → floor/2 (solarbeam also halved in rain); STAB: type!=='???' && source.hasType(type) → += floor(dmg/2); typeMod=target.runEffectiveness(move): >0 → '-supereffective', ×2 (≥2 → ×4); <0 → '-resisted', /2 floor (≤-2 → /4); random: !noDamageVariance && dmg>1 → dmg *= random(217,256), floor/255; bp && floor(dmg)==0 → 1.
- runEffectiveness: per target type: dex.getEffectiveness (typechart damageTaken: 1→+1 weak, 2→-1 resist, 0/3→0) + singleEvent Effectiveness (move.onEffectiveness — none pure) + runEvent Effectiveness per-type.
- runImmunity(move,msg): ignoreImmunity (incl. Status default) → true; '???' → true; NegateImmunity ev (foresight M2); Ground→isGrounded — gen2: no items/abilities: !negate && hasType Flying → not grounded → immune; else dex.getImmunity(type, target types) damageTaken 3 → immune → '-immune'.
- runStatusImmunity: dex.getImmunity(status id vs types) (typechart psn/tox/brn/par/frz keys) + runEvent Immunity (sunnyday onImmunity frz false! — means can't freeze in sun).

## Conditions (merged, M1 set)

- brn(stadium2): onStart '-status brn' + addVolatile brnattackdrop; onAfterMoveSelf[P3] residualdmg helper; onSwitchIn addVolatile brnattackdrop; onAfterSwitchInSelf residualdmg. residualdmg helper: volatiles.residualdmg → dmg=clamp(floor(maxhp/16)*counter,1) damage(dmg, pokemon) + hint(...,once=true) else damage(clamp(floor(maxhp/8),1)).
- par(stadium2): onStart '-status par' + parspeeddrop; onBeforeMove[P2] randomChance(1,4) → 'cant par' false; onSwitchIn addVolatile parspeeddrop.
- slp(stadium2): onStart '-status slp [from] move: X' (Move source) else plain; time=random(2,5) (NO startTime in stadium2!); onBeforeMove[P10]: time--; ≤0 → cureStatus, else 'cant slp'; sleepUsable → undefined else false.
- frz(gen2 over base): onStart '-status frz' (base); onBeforeMove[P10 from base]: defrost move → undefined; 'cant frz' false (NO thaw roll in befmove); onAfterMoveSecondary: (move.secondary?.status==='brn'||move.statusRoll==='brn') → target.cureStatus(); onAfterMoveSecondarySelf: defrost → cure; onResidual[order 7]: randomChance(25,256) → cureStatus.
- psn(gen2): onStart '-status psn'; onAfterMoveSelf[P3] residualdmg; onAfterSwitchInSelf residualdmg.
- tox(gen2): onStart '-status tox' + ensure residualdmg volatile + counter=0; onAfterMoveSelf[P3]: dmg=clamp(floor(maxhp/16),1)*counter → damage(dmg,pokemon,pokemon); onSwitchIn: status='psn' + '-status|poke|psn|[silent]'; onAfterSwitchInSelf: damage(clamp(floor(maxhp/16),1)).
- confusion(stadium2 onStart/onBeforeMove + base onEnd '-end confusion'): onStart: lockedmove src → '[silent]'; time=random(2,6); onBeforeMove[P3 base]: time--; 0 → removeVolatile, return; '-activate confusion'; randomChance(1,2) → return; else 40bp typeless physical self-hit via actions.getDamage(pokemon,pokemon,fakemove{isConfusionSelfHit,noDamageVariance,willCrit:false}), directDamage(damage), return false.
- flinch(base): duration 1; onBeforeMove[P8] 'cant flinch' + runEvent Flinch, false.
- partiallytrapped(base+gen2): duration 5, durationCallback random(3,6); onStart '-activate|poke|move: X|[of] source' + boundDivisor=8 (no bindingband); onResidual[order 3, subOrder 1]: source gone (!isActive||hp≤0||!activeTurns) → delete volatile + '-end ... [partiallytrapped] [silent]'; else damage(baseMaxhp/8) (spreadDamage clamps ≥1 int? damage arg baseMaxhp/8 float → clampIntRange(x,1) — clampIntRange truncs); onEnd '-end|poke|MoveName|[partiallytrapped]'; onTrapPokemon: source active → tryTrap.
- residualdmg(gen2/stadium2): onStart counter=0 (set via target.volatiles data); onAfterMoveSelf[P100]: status in brn/psn/tox → counter++; onAfterSwitchInSelf same.
- brnattackdrop/parspeeddrop: pure marker volatiles (stadium2 conditions? they're defined where — they're referenced but defined in gen2stadium2/conditions? NOT in the file read! Check: they must be implicit (addVolatile with unknown condition creates state with no callbacks — dex.conditions.get returns empty condition with exists=false but PS addVolatile still creates state? getVolatile of nonexistent condition — PS conditions.get('brnattackdrop') returns a Condition with effectType Condition, exists false but id set. addVolatile works, no Start callback. essence shows volatiles.brnattackdrop:{id:'brnattackdrop', name:'brnattackdrop'?...}). Verify against fixture.
- raindance/sunnyday(base + gen2 onFieldResidualOrder=2): duration 5 (durationCallback checks damprock — no item → 5); onFieldStart '-weather RainDance'; onFieldResidual '-weather|RainDance|[upkeep]' + eachEvent('Weather'); onFieldEnd '-weather none'; onWeatherModifyDamage unused (gen2 getDamage inline); sunnyday onImmunity: frz → false.
- sandstorm(base + gen2): onFieldResidualOrder 2; onWeather(gen2): damage(target.baseMaxhp/8); onFieldResidual '-weather|Sandstorm|[upkeep]' + isWeather check + eachEvent('Weather'); damage via spreadDamage w/ Weather effect → immunity check runStatusImmunity('sandstorm') (typechart sandstorm 3 for Rock/Ground/Steel) → '-damage|poke|hp|[from] Sandstorm'.
- stall(gen2): duration 2; counter=127; onStallMove randomChance(counter,255); onRestart counter/=2 duration=2. (M2 — protect/detect/endure callback moves)
- lockedmove(gen2), twoturnmove(base), trapped(base), futuremove, choicelock, mustrecharge: M2 (callback moves only).
- Rules: stadiumsleepclause.onSetStatus: source ally → undefined; slp && any target.side.pokemon hp&&slp → '-message Sleep Clause activated. (In official formats...)' + false. freezeclausemod.onSetStatus: frz && any target side pokemon frz → '-message Freeze Clause activated.' + false.

## Misc parity landmines

- `Pokemon` ctor: gender fallback battle.sample(['M','F']) if set.gender empty && species genderless-not — fixture teams always carry gender (validator), but implement identically anyway. moveSlots pp: calculatePP(move, 3): pp*8/5, gen≤2 && pp==40 → -3. ivs &= 30 (even). details: `Name, L##, G`.
- statModify (modern formula, tr = u32 trunc): hp = tr(tr(2*base+iv+tr(ev/4)+100)*level/100 + 10); other = tr(tr(2*base+iv+tr(ev/4))*level/100 + 5); nature neutral (Serious).
- getMoveHitData keyed by target slot; crit flag per target.
- battle.lastDamage (gen1 only usage in spreadDamage; but pokemon.lastDamage set: source.lastDamage=targetDamage on Move damage; counter/mirrorcoat M2).
- attrLastMove appends to lastMoveLine (addMove sets it); '[still]' → clears target field in the move line (parts[4]='').
- faint: pokemon.faint pushes faintQueue; faintMessages(stadium2) processes: BeforeFaint ev, '|faint|poke', pokemonLeft--, Faint ev, End ability/item singleEvents, clearVolatile(false), fainted=true isActive=false; gen2 queue.cancelMove(all actives); Self-KO win checks; AfterFaint ev.
- switch-out msg: `|switch|p1a: Name|details|hp/maxhp` via addSplit (getFullDetails fn → split side): secret exact, shared pixels.
- teamsize: after team preview truncation runAction start sets side.pokemonLeft = side.pokemon.length (3).
- 'team' action: index 0 clears side.pokemon — the essence side.pokemon array = picked 3 only, positions 0..2.
- undo: battle.undoChoice — fixture may contain '>p1 undo' lines? generator: `if (m[2]==='undo') battle.undoChoice` — possible with RandomPlayerAI? It never sends undo. OK skip.
- eachEvent('Update') callers: after each multihit hit, runAction gen<5 each action, moveHit... Update handlers in gen2: none (abilities/items) — but STILL sorts actives (speedSort by speed — RNG on speed ties!). eachEvent sorts `actives` with comparator (a,b)=>b.speed-a.speed — ties → shuffle → **RNG consumption even with no handlers**. Must implement faithfully.
- runEvent with no handlers still runs (no RNG since speedSort(handlers) empty).
- fieldEvent Residual: `add('')` happens in runAction before fieldEvent. Residual handlers include state.duration entries EVEN WITHOUT callbacks (getKey='duration' — e.g. flinch duration1, partiallytrapped) → duration-- and end() when 0 (removeVolatile → onEnd fires '-end' messages).
- Residual end() call args: [effectHolder, effect.id] → pokemon.removeVolatile(id)/side.removeSideCondition/field.clearWeather/removePseudoWeather; weather end via field.clearWeather() ignores id.
- getRandomTarget(singles): self/all/allySide/allyTeam/adjacentAllyOrSelf → self; adjacentAlly → null; else foe.active[0] (even fainted).
- getTarget: fails-if-self check (normal/any/adjacentAlly targeting self w/o twoturnmove → null → getRandomTarget); target fainted ally → return fainted; foe fainted → getRandomTarget.
- getMoveTargets(singles normal): target fainted&&!ally → getRandomTarget retarget; retargetLastMove if changed; futuremove exception.

## Essence extraction contract (tools/gen-fixtures.js essence())

- scal(state): keys with typeof number/string/boolean, skipping `effectOrder`. Includes id, sourceSlot (string!), name (volatiles), duration (number), time/startTime/counter/stage/layers/move(string) etc. NOT source/target (objects), NOT undefined.
- pokemon: ident=fullname, species=species.id (toID), hp/maxhp/fainted/status(''), statusState scal, boosts full 7, item ''/id, lastItem, itemState scal, moves from moveSlots {id, pp, disabled: !!disabled}, volatiles mapScal, types array, transformed, active=isActive, position.
- side: pokemonLeft, sideConditions mapScal, active [fullname|null], pokemon in current side.pokemon order.
- field: weather ''/id, weatherState scal, pseudoWeather mapScal.
- battle: turn, prngSeed 4-limb decimal, requestState ''→? PS requestState '' | teampreview | move | switch. Snapshot taken right after choose → requestState 'move' normally; '' if ended.

## Lessons burned in during M1 (divergences found by the harness)

- **`actions.runSwitch` is the gen4 override, not the modern base**: plain
  `runEvent('EntryHazard')` + `runEvent('SwitchIn')` + (gen≤2: ALL actives'
  `lastMove = null`; `AfterSwitchInSelf` only when `!side.faintedThisTurn &&
  draggedIn !== turn`) + item Start singleEvent. NO allActive speedSort, NO
  fieldEvent('SwitchIn') — the base version consumes extra PRNG on speed ties
  (caught by battle-003 seed divergence).
- **MoveSlot objects are SHARED between `moveSlots` and `baseMoveSlots`**
  (`moveSlots = baseMoveSlots.slice()` is a shallow copy). pp/disabled/used
  mutations persist through `clearVolatile` (faint, switch-out). Rust mirrors
  writes into base_move_slots for `shared` slots.
- **Non-function condition callbacks are invisible to the exporter**: fnNames
  only records functions, so `mustrecharge.onLockMove: 'recharge'` (a string
  constant) is missing from the data callbacks list → `conditions::has_builtin`
  whitelists such constants.
- **`-end` prints `${sourceEffect}` = effect NAME ("Wrap"); `-damage [from]`
  prints `fullname` ("move: Wrap")** — partiallytrapped uses both.
- **'recharge' is a nonexistent pseudo-move**: PS resolves it via
  dex.moves.get → empty move; it reaches only BeforeMove where mustrecharge
  aborts it. Interned as a synthetic 268th move.
- **partiallytrapped boundDivisor is 16** (gen5 onStart in the merged chain),
  duration random(3,6) (gen2), onResidual checks `!trapper.activeTurns` too.
- **effectTypeOrder quirk**: 'Status' is NOT in PS's subOrder table → statuses
  get subOrder 0 and run BEFORE volatiles (subOrder 2) on priority ties
  (matters for BeforeMove: slp/frz P10 → flinch P8 → confusion P3 → par P2).

## Lessons burned in during M2 (divergences found by fresh-seed soak corpora)

- **The golden 30 exercise ~60% of callbacks; fresh-seed soaks are the real
  gate.** Each 50–100 battle batch (`gen-fixtures.js --seed <new>` + the sweep
  example) found 2–7 real bugs the fixed corpus could not: encore override,
  transform speed cache, baton-pass copyvolatile, bide-vs-Ghost, future-sight
  timing, metronome-called charge moves, teleport. Green plateaued after ~300
  battles.
- **Prefixed handlers dispatch under their collection name.** A listener found
  as `onSourceAccuracy`/`onFoeBeforeSwitchOut` must be dispatched as that
  name, not `on{Event}` — `Listener.callback_name` carries it.
- **`dex.conditions.get(<any move id>)` always resolves.** Moves without a
  `condition` block yield a nonexistent Condition whose NAME IS THE RAW ID
  (essence shows `volatiles.solarbeam.name == "solarbeam"`, but
  `furycutter.name == "Fury Cutter"` since it has a block). Every move id is
  interned as a runtime condition.
- **`getOverflowedTurnCount()` is `turn - 1` for gen < 8** — Future Sight
  resolves at the end of use-turn + 2, not + 1.
- **`setSpecies` inside `transformInto` caches `speed` from
  `spreadModify(newSpecies.baseStats, this.set)`** — the user's own
  level/DVs/stat-exp on the target's base Spe — and the later storedStats copy
  does NOT refresh it. Speed ties from this cache drive eachEvent shuffles.
- **After `OverrideAction` (encore) everything downstream uses the overridden
  move** — PP deduction, moveUsed, useMove. Keeping the original move id
  silently executes the wrong move.
- **Baton Pass:** `copyVolatileFrom` copies boosts + non-`noCopy` volatiles
  (linked volatiles re-point their partners), and the `|switch|` line prints
  `[from] ${sourceEffect}` = plain NAME ("Baton Pass"), not fullname.
- **Synthetic moveData objects have no `ignoreImmunity`** — bide's unleash is
  blocked by Ghost immunity even though the bide move itself has
  `ignoreImmunity: true`.
- **`teleport` carries a constant `onTry: false`** (silent fail, no message) —
  a second instance of the invisible-constant-callback trap after
  `mustrecharge.onLockMove`; `rollout.onLockMove` and `bide.onSemiLockMove`
  are the other two.
- **Items participate with effectTypeOrder subOrder 8** and their own
  order/priority nums (focusband `onDamagePriority: -40`, leftovers
  `onResidualOrder: 5`); focusband rolls its 30/256 BEFORE checking lethality
  (JS `&&` order), so it consumes PRNG on every move-damage event.
- **`stall.counter` goes fractional** (127 → 63.5 → …) — essence needs a
  float-capable scalar; `onStallMove` floors it.

## Perf facts measured during M3 (2026-07)

- **The `format!` machinery, not malloc, dominated turn time.** gdb-sampling
  the release bench showed fmt frames under `find_event_handlers` in ~60% of
  samples; swapping mimalloc in barely moved turns/s while allocs/turn was
  1585. Interning the composed callback names (`on{E}`/`onAlly{E}`/…/`{cb}Order`)
  behind a pointer-keyed thread-local cache took 5k → 10k turns/s. Lesson:
  never build event/callback identity with `format!` in a dispatcher.
- **run_event with zero handlers is side-effect-free** (no PRNG: speedSort of
  an empty list; relay returned unchanged since modifier stays 1) — so
  handler collection can be skipped for events no cond/item in the format can
  handle (`Dex::possible_callbacks`). eachEvent's active-list tie shuffle
  happens BEFORE run_event and is preserved. Worth only ~8% here because most
  fired events are format-possible; a per-battle live-condition index would be
  the next step.
- **Remaining structural costs** (why we're at ~10k turns/s, not 1e5+):
  string-keyed handler lookup iterating `Vec<String>` callbacks per volatile
  per prefix per event, and 153 allocs per `Battle::clone` (Strings/Vecs in
  `Pokemon`/`EffectState`). Both are M4: integer event/callback ids +
  flattened state.
- **gdb SIGINT sampling works where perf doesn't** (WSL2, yama ptrace_scope
  blocks attach): run the binary UNDER gdb batch with a script of
  `bt/continue` pairs and a background `pkill -INT` pulse.
