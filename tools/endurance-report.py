#!/usr/bin/env python3
import json,collections,html,math,random,datetime,shutil,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1];src=root/'findings/deployed-endurance-v1';out=root/'previews/deployed-endurance-v1';out.mkdir(parents=True,exist_ok=True)
rows=[]
for f in sorted((src/'games').glob('g[0-9].jsonl')):
 records=[json.loads(l) for l in f.read_text().splitlines()];head=records[0];ds=[r for r in records if r['type']=='decision'];end=src/'games'/f'{f.stem}.json';terminal=json.loads(end.read_text()) if end.exists() else None
 sides=[]
 for s in range(2):
  counts=collections.Counter();high_heal=[];frozen=0
  for d in ds:
   pick=dict(d['picks']).get(s)
   if not pick:continue
   counts[pick]+=1;v=d['state']['sides'][s]
   if v['active'] is None:continue
   p=v['party'][v['active']]
   if p['status']=='frz' and pick.startswith('move '):frozen+=1
   if pick in ['move softboiled','move recover','move milkdrink'] and p['hp']/p['maxhp']>.7 and p['status']!='frz':high_heal.append({'decision':d['decision'],'turn':d['turn'],'hp':p['hp'],'maxhp':p['maxhp'],'action':pick})
  sides.append({'choices':dict(counts),'healsAbove70Percent':high_heal,'frozenMoveChoices':frozen})
 rows.append({'id':f.stem,'teams':head['teamIds'],'decisions':len(ds),'turn':terminal['turn'] if terminal else ds[-1]['turn'],'outcome':terminal['outcome'] if terminal else None,'sides':sides})
pairs=[]
for trial in range(16):
 paths=[src/'continuations'/f'g4-d12-{arm}-{trial}.json' for arm in ['baseline','conserve-heal']]
 if not all(p.exists() for p in paths):continue
 a,b=[json.loads(p.read_text()) for p in paths]
 if any(r['score'] is None for r in [a,b]):continue
 pairs.append({'trial':trial,'baseline':a['score'],'alternative':b['score'],'delta':b['score']-a['score'],'turns':[a['endTurn'],b['endTurn']]})
cf={'completedPairs':len(pairs),'plannedPairs':16,'pairs':pairs}
if len(pairs)==16:
 deltas=[p['delta'] for p in pairs];rng=random.Random(20261017);boot=sorted(sum(rng.choices(deltas,k=16))/16 for _ in range(20000));pos=sum(d>0 for d in deltas);neg=sum(d<0 for d in deltas);n=pos+neg
 cf.update(baselineScore=sum(p['baseline'] for p in pairs)/16,alternativeScore=sum(p['alternative'] for p in pairs)/16,delta=sum(deltas)/16,pairedBootstrap95=[boot[500],boot[19499]],alternativeBetterPairs=pos,baselineBetterPairs=neg,exactPairedSignP=min(1,2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else 1)
 cf['endTurns']={'median':[statistics.median(p['turns'][s] for p in pairs) for s in [0,1]],'maximum':[max(p['turns'][s] for p in pairs) for s in [0,1]],'alternativeLongerPairs':sum(p['turns'][1]>p['turns'][0] for p in pairs),'alternativeShorterPairs':sum(p['turns'][1]<p['turns'][0] for p in pairs)}
 cf['outcomes']={arm:{label:sum(p[arm]==score for p in pairs) for label,score in [('wins',1),('draws',.5),('losses',0)]} for arm in ['baseline','alternative']}
summary={'updatedUTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'games':rows,'counterfactual':cf};(src/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
esc=html.escape
trs=''.join(f'<tr><td>{r["id"]}</td><td>{esc(" / ".join(r["teams"]))}</td><td>{r["turn"]}</td><td>{r["outcome"] or "実行中"}</td><td>{sum(len(s["healsAbove70Percent"]) for s in r["sides"])}</td><td>{sum(s["choices"].get("move counter",0) for s in r["sides"])}</td></tr>' for r in rows)
cftext=f'16組中 {len(pairs)} 組完了。全組が揃うまで改善の判定は行いません。'
if len(pairs)==16:cftext=f'通常の得点率 {cf["baselineScore"]:.1%}、代替戦略 {cf["alternativeScore"]:.1%}。得点率差は {100*cf["delta"]:+.2f}ポイント、ペア・ブートストラップ95%区間は {100*cf["pairedBootstrap95"][0]:+.2f}〜{100*cf["pairedBootstrap95"][1]:+.2f}ポイント。'
historical=src/'historical/audit.json'
histHtml=''
if historical.exists():
 h=json.loads(historical.read_text());histHtml=f'<p>元の129ターン戦と同じ分岐時点・乱数状態での検証では、代替戦略側は{h["endTurn"]}ターンで敗れました。元の戦略も129ターンで敗れているため、この1例では改善していません。選択によって将来の乱数消費順も変わるので、平均的な優劣は16組の比較で判断します。</p>'
complete=len(pairs)==16
phase='全16組の比較が完了しました。' if complete else '比較計算を継続中です。'
analysis=json.loads((src/'analysis.json').read_text())
probeRows=''.join('<tr><td>'+esc(r['case'])+' / P'+str(r['side']+1)+'</td>'+''.join('<td>'+esc({'move softboiled':'タマゴうみ','move icebeam':'れいとうビーム','move counter':'カウンター','move rest':'ねむる','move surf':'なみのり','move recover':'じこさいせい','move return':'おんがえし'}.get(r['choices'][str(n)],r['choices'][str(n)]))+'</td>' for n in [81000,243000,729000])+'</tr>' for r in analysis['budgetProbes'])
pairRows=''.join(f'<tr><td>{r["trial"]}</td><td>{r["baseline"]:g}</td><td>{r["alternative"]:g}</td><td>{r["turns"][0]} / {r["turns"][1]}</td></tr>' for r in pairs)
conclusion='回復PPの使い方に強化余地があるかを検証中です。高HPでの回復を攻撃へ置き換えるだけで強くなるとは、まだ判断できません。'
if complete:
 lo,hi=cf['pairedBootstrap95']
 conclusion=('この局面の条件付き比較では代替戦略に改善が見られました。' if lo>0 else '高HPでの回復を攻撃へ置き換える単純な戦略は、この16組では明確な優位を確認できませんでした。' if hi>=0 else 'この局面の条件付き比較では代替戦略が不利でした。')+' PP管理と長い交代反復が、次に検証すべき強化候補です。'
cfDetails=''
if complete:
 cfDetails=f'<p>代替の得点が高かった組は{cf["alternativeBetterPairs"]}、通常の得点が高かった組は{cf["baselineBetterPairs"]}。不一致組に対する両側の正確な符号検定 p={cf["exactPairedSignP"]:.3f}。95%区間は16組を単位とした20,000回のブートストラップです。探索・対戦seedの変動を調べたもので、別の構築や未知情報全体への区間ではありません。</p>'
 cfDetails+=f'<p>通常は{cf["outcomes"]["baseline"]["wins"]}勝・{cf["outcomes"]["baseline"]["draws"]}分・{cf["outcomes"]["baseline"]["losses"]}敗、代替は{cf["outcomes"]["alternative"]["wins"]}勝・{cf["outcomes"]["alternative"]["draws"]}分・{cf["outcomes"]["alternative"]["losses"]}敗。得点は勝ち1、引き分け0.5、負け0です。</p>'
 cfDetails+=f'<p>終了ターンの中央値は通常{cf["endTurns"]["median"][0]:g}、代替{cf["endTurns"]["median"][1]:g}。最長はそれぞれ{cf["endTurns"]["maximum"][0]}、{cf["endTurns"]["maximum"][1]}ターンでした。代替で長くなった組は{cf["endTurns"]["alternativeLongerPairs"]}、短くなった組は{cf["endTurns"]["alternativeShorterPairs"]}です。対戦の長さの変化だけでは戦略の強弱を判定しません。</p>'
longHtml=''
if complete:
 longGames=json.loads((src/'case-analysis.json').read_text())['longestCompletedGames']
 longRows=''.join(f'<tr><td>{g["trial"]} / {"通常" if g["arm"]=="baseline" else "代替"}</td><td>{g["endTurn"]}</td><td>{g["bothVoluntarilySwitchDecisions"]} / {g["longestJointSwitchRun"]["decisions"] if g["longestJointSwitchRun"] else 0}</td><td>{g["struggleSelections"][0]} / {g["struggleSelections"][1]}</td></tr>' for g in longGames)
 longHtml=f'<details><summary>最も長かった3戦の行動</summary><table><tr><th>組 / 戦略</th><th>終了ターン</th><th>同時交代 計 / 最長連続</th><th>わるあがき P1 / P2</th></tr>{longRows}</table><p><small>同時交代は、両側が通常の行動として交代を選んだ局面数です。ひんし後の強制交代は除いています。わるあがきは選択回数であり、実際に技が成功した回数ではありません。</small></p></details>'
auditPath=src/'continuation-audit.json'
audits=json.loads(auditPath.read_text()) if auditPath.exists() else []
auditText=f'元の8戦は356状態とPRNG、代表局面の63件の探索結果（全行動の訪問数・評価値・選択）を一致確認。代替比較は現在{len(audits)}戦、計{sum(x["decisions"] for x in audits)}局面の状態・合法手・終局を再生確認しました。'
body=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>公開botの耐久戦調査</title><style>body{{font:17px/1.8 system-ui,sans-serif;background:#f6f4ee;color:#282c28;max-width:1050px;margin:auto;padding:25px}}h1{{font-size:1.7rem}}h2{{font-size:1.2rem;margin-top:2rem}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #c8ccc5}}.scroll{{overflow:auto}}.note{{padding:15px;background:#e8ebe3;border-radius:8px}}a{{color:#345d48}}code{{font-size:.85em;overflow-wrap:anywhere}}small{{color:#555}}li{{margin-bottom:.6em}}</style><h1>公開botの耐久戦調査</h1><p class="note">公開版の「強」＝81,000回探索を使用。耐久構築の対戦を終局まで記録し、同じ局面から代替戦略を比較しました。</p><p><strong>{conclusion}</strong></p><p>{cftext}</p><p>更新: {esc(summary['updatedUTC'])}。{phase}</p><h2>調査の基準</h2><p>既存作業をローカルコミット <code>00b4371</code> に保全し、clean を確認して開始。2026年10月17日17:02 JSTに取得した<a href="https://puniu3.github.io/nc2000-engine/">公開版</a>のworker・WASM・チーム・相手の事前分布を保存しました。選出は公開版の事前計算表、戦闘中は双方81,000回です。観測には相手の隠れた技や持ち物を渡していません。</p><p>公開WASMのSHA-256: <code>3164238a3912eb7444a9cf24c49281980893e332221cfc8c27accb750bb0c4ab</code>。手元のビルドは異なるため、比較計算にも保存した公開WASMを使用しました。</p><h2>対戦記録</h2><div class="scroll"><table><tr><th>ID</th><th>構築</th><th>ターン</th><th>勝者</th><th>HP70%超の即時回復選択</th><th>カウンター選択</th></tr>{trs}</table></div><p><small>耐久向けの4組を2つのseedで調べる目的抽出です。全対戦環境の勝率・頻度を表す標本ではありません。回復回数は選択時点のHPで数え、凍っている間の回復選択は除外しています。</small></p><h2>129ターンのHPとPP</h2><p>g4は双方がハピナス・スイクン・バンギラスを選出したミラーです。以下はハピナスの推移を示します。</p><img src="g4-resources.svg" style="width:100%;height:auto" alt="ハピナスのHP、回復PP、攻撃PPの推移。満タンでの回復選択を×で表示。"><p><small>×は満タンでのタマゴうみ選択（計11回）。実際の処理では9回が不発、2回は先に受けた攻撃の35 HPを回復しました。帯はP1ハピナスの凍結期間です。図は各選択前の状態と最終ターン後の状態を示します。<a href="g4-resources.png">図のPNG</a></small></p><h2>確認した行動</h2><ul><li>g4の10ターン目：ハピナス同士で、片側は299/361 HPからタマゴうみを選びました。その後も高HPでの回復が続きます。</li><li>ハピナスが19ターン目に凍り、32ターン目の終了時に解凍。33ターン目には49 HPまで減っていましたが、回復して残りました。この間P1の攻撃PPは23のまま、P2は23から10へ減少しています。凍結中に倒し切れなければ、攻めた側のPP消費が後の不利につながり得ます。</li><li>長期戦ではカウンターの連続選択が発生します。特殊攻撃中心の対面なので直接ダメージを狙う手ではありませんが、攻撃技のPPを残す待機手としての意味があり得ます。</li><li>g2の30ターン目：45/206 HPのスイクンがれいとうビームを選択。次ターン、22 HPからねむるを選んで倒されました。ただし控えが残り、その側が最終的には勝っています。駒を失ったことだけでは戦略的な誤りとは断定できません。</li></ul><h2>代替戦略の比較</h2>{cfDetails}{histHtml}<p>10ターン目の増量探索で選ばれたれいとうビームを手がかりに、回復を抑える1種類のルールを固定して試しました。g4の10ターン目から将来の乱数を変え、通常戦略と「相手が凍っていないハピナスで、自分が凍っておらずHP60%超なら、タマゴうみをれいとうビームに置き換える」戦略を比較しました。その他の行動は同じ公開botです。両側とも81,000回探索します。</p><p>過去の全観測を復元し、分岐直前のログも取り込んでから将来乱数を変更します。各組で同じ将来乱数seed・探索seedを使いますが、選択が変われば乱数の消費順は変わります。実際に起きた隠れ状態に条件づけた分析であり、未知情報全体での最適性を示すものではありません。</p><p>公開エンジンの1000ターン上限に達すると、正式な引き分けになります（内部のturn値は1001）。別途、交代だけを繰り返すエンジン確認で検証しました。この確認は32戦の勝敗集計に含めていません。待ち時間による打ち切りや未完走とは区別しています。</p><details><summary>組ごとの結果（1＝対象側の勝ち、0＝負け、0.5＝引き分け）</summary><table><tr><th>seed組</th><th>通常</th><th>代替</th><th>終了ターン 通常 / 代替</th></tr>{pairRows}</table></details><p>経過の例では、seed組0は通常が119ターンで負け、代替が199ターンで勝ちました。一方seed組11は両方勝ちですが、通常31ターンに対して代替は143ターンです。代替側は両例とも29ターン目にれいとうビームのPPが尽き、その後は置換条件を満たせず、満タン回復も再発しました。勝敗の改善と対戦の短縮は別に評価する必要があります。</p><p>seed組4では、通常戦略が732ターンで敗れ、代替戦略は151ターンで勝ちました。通常戦略は222～704ターンの483ターン連続で双方が交代し、HP・PPが変わりませんでした。終局後の追加診断では、500ターン目の81,000回の選択を両側で再現し、243,000・729,000回に増やしても双方が交代を選びました。交代反復を減らすことが一般に得点率を上げるかは、別途検証が必要です。</p><p>seed組12では、通常戦略が779ターン連続、代替戦略が806ターン連続で双方交代を続け、いずれもその間のHP・PPは変わりませんでした。両方とも1000ターン上限で引き分けです。今回の回復→攻撃ルールでは、この長い交代反復は解消しませんでした。</p><h2>探索を増やすと変わるか</h2><p>同じ履歴・探索seedで81,000、243,000、729,000回の入れ子の探索を比較しました。元の8戦から選んだ4局面・両側の8件です。81,000回の結果は元の対戦と全行動の訪問数・評価値まで一致しています。</p><div class="scroll"><table><tr><th>局面 / 側</th><th>81,000</th><th>243,000</th><th>729,000</th></tr>{probeRows}</table></div><p>g4-d12は10ターン目、g4-d55は53ターン目です。10ターン目のP2は増量で攻撃へ変わる一方、53ターン目のP1は満タンでの回復を維持しました。今回試した9倍までの増量では、両方は解決しませんでした。選択が変わったこと自体は勝率改善の証明ではありません。</p><p>満タンで選んだ回復でも、同じターンに相手の攻撃が先に入れば回復効果があります。g4の45・46ターン目は実際に35 HPずつ回復しました。一方、53ターン目など9回は不発です。選択時HPや1回の不発だけで、同時選択の期待値が悪いとは断定しません。</p>{longHtml}<h2>強化候補と採用条件</h2><ol><li><strong>長期のPP収支と交代反復を評価する。</strong>回復の残り回数と、有効な攻撃の残り回数を対面・控え込みで扱う候補です。回復を一律に攻撃へ置き換えると、有効な攻撃PPを先に使い切る危険があります。また、短い先読みが見積もる継続価値と、毎ターン再探索した結果としての交代反復とのずれを調べる価値があります。原因はまだ切り分けていません。複数の耐久構築・局面で終局の得点率を比較して採用を判断します。</li><li><strong>凍結中の本当に等価な行動へ探索を分散させない。</strong>凍ったハピナスの1局面で64 seed×相手5行動を確認し、かみなりとれいとうビーム／タマゴうみは各320件すべて次の表示状態とPRNGが一致しました。ただしカウンターは優先度によって一致しません。隠れたエンジン状態も含む等価性を証明してから統合し、強さを別途測る必要があります。</li><li><strong>探索予算の配分を見直す。</strong>増量で回復から攻撃へ変わる接戦評価の局面は候補です。一律9倍でも満タン回復は残るため、予算配分と長期評価を分けて検証します。</li></ol><p>確認できたのは行動上の改善候補と、1種類の介入の条件付き成績です。より良い戦略が存在しないことや、この変更が一般に強いことまでは示していません。</p><h2>検証と再現性</h2><p>{auditText} クラウドで完走した2戦は、別環境でも先頭・中間・末尾の両側、計12選択の一致を確認しました。途中実装の結果は使用せず、修正後の32戦を比較対象に固定しました。{'32戦すべて終局まで確認済みです。' if complete else '未完走は勝敗に数えず、終局まで計算を継続しています。'}</p><p>調査データと再現スクリプトはリポジトリの <code>findings/deployed-endurance-v1</code>、<code>web/tests/endurance-*.mjs</code>、<code>tools/endurance-*.py</code> に保存しています。公開資産のハッシュ、全対戦ログ、反実仮想の条件、監査結果を含みます。製品コードの変更・push・公開デプロイは行っていません。</p><p><small>補助クラウド計算の使用料見積もりは約103円（クレジット適用前の概算で、請求額ではありません）。クラウドのジョブ・VMは削除確認済みです。</small></p></html>'''
(out/'index.html').write_text(body);Path('/tmp/nc2000-endurance-preview').mkdir(exist_ok=True);Path('/tmp/nc2000-endurance-preview/index.html').write_text(body);[shutil.copy2(out/f'g4-resources.{ext}',Path('/tmp/nc2000-endurance-preview')/f'g4-resources.{ext}') for ext in ['svg','png'] if (out/f'g4-resources.{ext}').exists()];print(json.dumps({'games':len(rows),'terminal':sum(r['outcome'] is not None for r in rows),'pairs':len(pairs)},ensure_ascii=False))
