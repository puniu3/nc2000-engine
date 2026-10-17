#!/usr/bin/env python3
"""Audit all frozen rows and render the shared-preview report (no dependencies)."""
import collections
import hashlib
import html
import json
import math
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def read(path):
    return json.loads(path.read_text())

def exact_p(positive, negative):
    n = positive + negative
    return min(1, 2 * sum(math.comb(n, i) for i in range(min(positive, negative) + 1)) / 2**n) if n else 1

def binomial_bounds(k, n, alpha=.025):
    # Equal-tail Clopper-Pearson; each marginal interval covers >=97.5%.
    # Taking their difference therefore covers >=95%, including correlated arms.
    def bisect(fun, target, increasing):
        a, b = 0., 1.
        for _ in range(70):
            p = (a+b)/2
            if (fun(p) < target) == increasing: a = p
            else: b = p
        return (a+b)/2
    lo = 0 if k == 0 else bisect(lambda p: sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k,n+1)), alpha/2, True)
    hi = 1 if k == n else bisect(lambda p: sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1)), alpha/2, False)
    return [lo,hi]

def collect():
    probes = sorted((read(p) for p in (HERE/'probes').glob('*.json')), key=lambda r:r['id'])
    assert len(probes)==34
    flips=[]
    for p in probes:
        assert len(p['rows'])==8
        for a,b in zip(p['rows'][::2],p['rows'][1::2]):
            assert a['seed']==b['seed'] and (a['n'],b['n'])==(27000,81000)
            for r in [a,b]:
                assert sum(r['visits'])==r['n'] and r['choice'] in r['actions']
                assert all(0<=m<=1 for m in r['means'])
            if a['choice']!=b['choice']:flips.append({'id':p['id'],'seed':a['seed'],'before':a['choice'],'after':b['choice']})
    selected=read(HERE/'continuation-selection.json')
    assert {r['id'] for r in selected}=={r['id'] for r in flips}
    groups=collections.defaultdict(list)
    for f in (HERE/'continuations').glob('*.json'):
        r=read(f);groups[r['id']].append(r)
    continuations=[]
    for c in selected:
        rows=sorted(groups[c['id']],key=lambda r:r['trial'])
        assert [r['trial'] for r in rows]==list(range(64))
        assert len({r['seed'] for r in rows})==64
        for r in rows:
            assert r['iters']==3000
            assert [a['action'] for a in r['rows']]==c['actions']
            assert all(a['result']['score'] in (0,1) for a in r['rows'])
            assert all(a['result']['decisions']<1600 for a in r['rows'])
        scores=[[r['rows'][i]['result']['score'] for r in rows] for i in (0,1)]
        d=[b-a for a,b in zip(*scores)];positive=sum(x>0 for x in d);negative=sum(x<0 for x in d)
        bounds=[binomial_bounds(int(sum(s)),64) for s in scores]
        continuations.append({**c,'trials':64,'wins':[int(sum(s)) for s in scores],
            'scores':[statistics.mean(s) for s in scores],'delta':statistics.mean(d),
            'conservative_difference_95':[bounds[1][0]-bounds[0][1],bounds[1][1]-bounds[0][0]],
            'improved_pairs':positive,'worsened_pairs':negative,'mcnemar_two_sided':exact_p(positive,negative),
            'bonferroni_six':min(1,6*exact_p(positive,negative))})
    confirmation=sorted((read(p) for p in (HERE/'confirmation').glob('*.json')),key=lambda r:r['trial'])
    assert [r['trial'] for r in confirmation]==list(range(32))
    assert [r['seed'] for r in confirmation]==list(range(2026102200,2026102232))
    assert all(r['iterations']==27000 for r in confirmation)
    assert all([a['action'] for a in r['rows']]==['switch 2','switch 3'] for r in confirmation)
    assert all(a['result']['score'] in (0,1) and a['result']['decisions']<1600 for r in confirmation for a in r['rows'])
    wins=[int(sum(r['rows'][i]['result']['score'] for r in confirmation)) for i in (0,1)]
    positive=sum(r['rows'][1]['result']['score']>r['rows'][0]['result']['score'] for r in confirmation)
    negative=sum(r['rows'][1]['result']['score']<r['rows'][0]['result']['score'] for r in confirmation)
    cb=[binomial_bounds(w,32) for w in wins]
    confirmed={'case':'g2-d7','trials':32,'iterations':27000,'wins':wins,'scores':[w/32 for w in wins],
        'delta':(wins[1]-wins[0])/32,'improved_pairs':positive,'worsened_pairs':negative,
        'mcnemar_two_sided':exact_p(positive,negative),'conservative_difference_95':[cb[1][0]-cb[0][1],cb[1][1]-cb[0][0]]}
    null=[read(p) for p in (HERE/'null').glob('*.json')]
    assert len(null)==4 and all(r['rows'][0]['result']==r['rows'][1]['result'] for r in null)
    timing=[json.loads(x) for x in (HERE/'public-browser-timing.jsonl').read_text().splitlines()]
    fixed=[r for r in timing if r['type']=='timing' and not r['ponder']]
    assert len(fixed)==15
    for r in fixed:assert r['iterations']==r['budget'] and r['memory']>0
    times=[]
    for n in [27000,81000,243000]:
        rows=[r for r in fixed if r['budget']==n]
        times.append({'iterations':n,'measurements':len(rows),
            'search_seconds':[min(r['search_ms'] for r in rows)/1000,max(r['search_ms'] for r in rows)/1000],
            'memory_mb':[min(r['memory'] for r in rows)/1e6,max(r['memory'] for r in rows)/1e6],
            'max_progress_gap_ms':max(r['max_progress_gap_ms'] for r in rows)})
    ponder=[r for r in timing if r['type']=='timing' and r['ponder']]
    assert len(ponder)==2 and ponder[0]['policy']==ponder[1]['policy']
    old=[json.loads(x) for x in (HERE/'browser-failed-prefix.jsonl').read_text().splitlines() if json.loads(x)['type']=='timing']
    for a in old:
        b=next(r for r in fixed if r['case']==a['case'] and r['budget']==a['budget'] and r['seed']==a['seed'])
        assert a['policy']==b['policy'] # Instantiation/memory instrumentation did not change these policies.
    parity=read(HERE/'public-local-parity.json')
    solvers=[read(p) for p in (HERE/'solver').glob('*.json')]
    assert len(solvers)==28
    tactics=[read(p) for p in (HERE/'tactics').glob('*.json')]
    positions=[read(p) for p in sorted((HERE/'positions').glob('*.json'))]
    assert len(tactics)==2 and len(positions)==4
    for r in positions:
        assert len(r['rows'])==12
        for row in r['rows']:assert row['policy']['iterations']==row['n']
    result={'schema':'nc2000-strength-investigation-report-v1','public_assets':read(HERE/'public-assets.json'),
        'public_data':read(HERE/'public-data.json'),'browser_timing':times,
        'ponder':[{k:r[k] for k in ['budget','iterations','request_ms','memory','choice']} for r in ponder],
        'parity':[{'case':r['case'],'same_choice':r['public_choice']==r['local_choice'],'same_policy':r['public_policy']==r['local_policy']} for r in parity],
        'source_games':16,'roots':34,'paired_searches':136,'flips':flips,'continuations':continuations,'confirmation':confirmed,
        'solver':{'roots':len(solvers),'width_met':sum(r['value'][1]-r['value'][0]<=.02000001 for r in solvers),
            'stops':dict(collections.Counter(r['stop'] for r in solvers)),
            'one_sided_heal':sum(r['one_sided_heal'].startswith('Ok') for r in solvers),
            'two_sided_heal':sum(r['two_sided_heal'].startswith('Ok') for r in solvers)},
        'perish_positions':[{'name':r['name'],'scope':r['scope'],'solver':r['solver'],'choices':{str(n):dict(collections.Counter(x['choice'] for x in r['rows'] if x['n']==n)) for n in [27000,81000,243000]}} for r in positions],
        'tactics':[{'case':r['case'],'eval':r['eval'],'generic_solver':r['generic_solver'],'proof_actions':r['proof']['actions'],'search_choices':dict(collections.Counter(x['choice'] for x in r['search']))} for r in tactics],
        'checks':{'corpus':read(HERE/'corpus-verify.json'),'null_paired_trials':4,'null_identical':True,'continuation_games':768,'confirmation_games':64,'continuation_caps':0,'invalid_actions':0,'memory_instrumentation_preview_policies_equal':True,'perish_regression_tests':4},
        'limitations':['Mac Chromium, not iPad Safari. WebKit launch timed out.','Public and local WASM differ; one of three root policies differs.','34 roots from16 fresh3000-iteration weighted-catalog games, enriched for endgames.','64 paired continuations per root at3000 iterations; conditional policy values, not optimal play.','Not a causal decomposition of earlier756 confirmation games.','Historical Perish positions are reconstructed; generic solver results conditional on a single imputed world.','Two tactic fixtures are synthetic, with Magikarp bench; no operational prevalence claim.'],
        'references':[
            {'title':'MCTS Solver (2008)','url':'https://staff.ru.is/yngvi/pdf/WinandsBS08.pdf'},
            {'title':'Information Set Monte Carlo Tree Search (2012)','url':'https://eprints.whiterose.ac.uk/id/eprint/75048/'},
            {'title':'Bounded value iteration for concurrent stochastic games (2025)','url':'https://arxiv.org/abs/2505.21087'}]}
    return result

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def render(j):
    timing=table(['探索回数','測定数','探索時間の範囲','Wasm確保メモリ'],[[f"{r['iterations']:,}",r['measurements'],f"{r['search_seconds'][0]:.2f}–{r['search_seconds'][1]:.2f}秒",f"{r['memory_mb'][0]:.1f}–{r['memory_mb'][1]:.1f} MB"] for r in j['browser_timing']])
    names={'g1-d0':'G1 選出：ガラガラ先発 → ラプラス先発','g14-d8':'G14 T6：バンギラス → ナッシー','g2-d7':'G2 T5：カビゴン → スターミー','g3-d29':'G3 T27：おんがえし ↔ れいとうビーム','g4-d0':'G4 選出：ミルタンク先発 → カイリキー先発','g4-d13':'G4 T10：ミルタンク → カイリキー'}
    cases=table(['局面・探索増加時の変化','初手Aの勝ち','初手Bの勝ち','差','改善/悪化ペア','6比較補正p'],[[names[r['id']],f"{r['wins'][0]}/64",f"{r['wins'][1]}/64",f"{r['delta']*100:+.1f}pt",f"{r['improved_pairs']}/{r['worsened_pairs']}",f"{r['bonferroni_six']:.4f}"] for r in j['continuations']])
    perish=table(['再構成局面','27,000回','81,000回','243,000回','汎用ソルバ'],[[r['name'],*[', '.join(f'{k}: {v}/4' for k,v in r['choices'][str(n)].items()) for n in [27000,81000,243000]],'[0,1]・ノード上限'] for r in j['perish_positions']])
    g2=next(r for r in j['continuations'] if r['id']=='g2-d7')
    confirm=j['confirmation']
    refs=''.join(f'<li><a href="{r["url"]}">{html.escape(r["title"])}</a></li>' for r in j['references'])
    return f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>NC2000 bot 強化調査</title>
<style>body{{background:#f0ece3;color:#302c27;max-width:69rem;margin:2rem auto;padding:0 1rem;font:17px/1.8 system-ui}}h1{{font-size:1.65rem}}h2{{font-size:1.25rem;margin-top:2.3rem}}h3{{font-size:1.05rem}}a{{color:#75502e}}table{{border-collapse:collapse;width:100%;font-size:.87rem}}th,td{{text-align:left;vertical-align:top;padding:.6rem;border-bottom:1px solid #c9c0b3}}.scroll{{overflow-x:auto}}.lead{{border-left:4px solid #877654;padding:.5rem 1rem}}code{{font-size:.85em}}small{{color:#655d53}}li{{margin:.45rem 0}}details{{border:1px solid #c9c0b3;padding:.6rem;margin:1rem 0}}</style>
<h1>NC2000 bot：探索予算・判断変化・ソルバの適用調査</h1>
<p><small>2026-10-16 / 公開版の配信実物を測定。研究用コードと測定を追加。公開設定・デプロイの変更なし。</small></p>
<p class="lead"><strong>81,000回は公開版でも動作し、Macでは約2～3秒です。ただし現行の先読みと重複するため、一律3倍化の利益は前の勝率差より小さくなり得ます。判断の改善例は交代先の選択で確認できました。ソルバは短い手順に絞る方向が有望ですが、PP耐久戦全般を解く実用性や強化効果は未確認です。</strong></p>
<h2>1. 公開版にどこまで適用できるか</h2>
<p>前の研究は手元の blind 81,000対27,000、確認756局で得点56.48%、逐次区間50.41～62.54%。先読みなし・カタログ一様分布です。製品は drawWeight による抽選と、人間の思考中の追加探索を使うため、そのまま製品の改善幅にはなりません。</p>
<p><a href="https://puniu3.github.io/nc2000-engine/">公開サイト</a>から取得したworkerとWasmを変更せずに実行し、メモリ取得だけをWebAssembly初期化に追加しました。公開workerは27,000回、c=0.4、先読み上限は予算の10倍です。チーム・事前分布データは手元と一致。Wasmは不一致で、同一条件の3局面照合では選択手は3件とも一致、全探索統計は2件だけ一致しました。公開版と手元を同じビルドとは扱いません。</p>
{timing}
<p>Mac Chromium 149、選出・初手・既存forkの3局面。27k/81kは各2乱数、243kは各1乱数です。範囲はこの少数測定の最小～最大で、p95や全対局の保証ではありません。メモリは各workerの線形メモリ確保量であり、タブ全体のRSSではありません。iPad Safariは未測定、Mac WebKitは起動タイムアウトでした。</p>
<p>同じ初手で人間が5秒考えると、27k設定でも81k設定でも<strong>183,750回</strong>まで進み、全探索統計と選択手が一致しました。最低予算の増加は、特に人間の即決時・botだけが選ぶ局面に効きます。長く考える対戦者には既存の先読みが既に追加計算を提供します。</p>
<p><strong>実装候補：</strong>候補Aは最低81k＋先読み上限270k、候補Bは最低27kのまま選出・交代局面を81kへ増やす構成です。両案とも最低予算と先読み上限を独立にします。訪問数が一手に集中したことだけで早期終了する案は、下記の反例があるため採りません。現コードの定数だけを81kへ変更すると先読み上限も810kへ増えます。243k常用は今回の7～10秒・約83～86MBから優先しません。81kの一律適用を確定する前に、公開ビルドそのもの・製品の抽選分布・先読みありの比較が必要です。端末別の上限は今回のMac測定から決められません。</p>
<h2>2. 具体的に何が変わり、改善したか</h2>
<p>現エンジンで drawWeight に従う16局を新規生成（両者3,000回）。先頭6局の選出と、各局の最初・最後の対象終盤から34局面を選び、各4乱数で27k→81kと同じ探索を延長しました。<strong>136比較中9件、6局面</strong>で手が変わりました。終盤に偏った診断標本であり、全対戦での変化率ではありません。</p>
<p>変化した全6局面で、各初手を固定して64組の新しい乱数で最後まで対局しました。以後は両側とも blind 3,000回です。初手の比較は同じ真の局面・同じ乱数を共有し、両者に相手の秘密情報を渡しません。探索中のQ値を正解として採点していません。</p>
{cases}
<p><strong>確認できた改善：</strong>G2 T5ではサンダーが倒れた後、27kで一度だけ出たカビゴン選択が81kでは消え、4乱数すべてスターミーになりました。初手を強制すると4勝対20勝（64組）、差+25.0ポイント。改善19組・悪化3組で、正確McNemar検定の6比較Bonferroni補正後p=0.00513。保守的な差の95%区間は{g2['conservative_difference_95'][0]*100:.1f}～{g2['conservative_difference_95'][1]*100:.1f}ポイントです。これはこの局面・継続方策での改善で、最善手の証明や81k bot全体の増分勝率ではありません。</p>
<p><strong>現行27kの継続探索でも独立追試：</strong>この1局面を選び、試行数を先に32組へ固定して別乱数で追試しました。カビゴン先出しは{confirm['wins'][0]}/32勝、スターミー先出しは{confirm['wins'][1]}/32勝（6.25%→62.5%）、差+56.25ポイント。改善{confirm['improved_pairs']}組・悪化{confirm['worsened_pairs']}組、正確McNemar p={confirm['mcnemar_two_sided']:.6f}、差の保守的95%区間は{confirm['conservative_difference_95'][0]*100:.1f}～{confirm['conservative_difference_95'][1]*100:.1f}ポイントでした。低予算の継続方策に限った現象ではありません。ただしこの追試も同じ1局面の固定初手比較で、人間の最善防御や別の対戦分布に対する保証ではありません。3kの探索用結果とは合算していません。</p>
<p>この改善は単なる僅差の入れ替わりではありません。変更が起きた探索乱数2026102303では、27k時点の訪問数がカビゴン26,916対スターミー84（99.7%対0.3%）、平均評価0.6345対0.5008でした。81kまで延長すると28,527対52,473、平均評価0.6330対0.6758になり逆転しました。追加の探索で、初期に低評価となっていた選択肢を見直した例です。UCBの適応的な訪問比を、そのまま「99.7%の確信度」と読むとこの改善を取り逃します。</p>
<p>再現した1組（seed2026101933）では、カビゴン先出しに相手がバトンタッチ→ガラガラのつるぎのまいで応じ、10ターン目に敗北。スターミー先出しではサンダースとの応酬の後にカビゴンが残り、13ターン目に勝利しました。相手の技選択も変わる継続対局の一例であり、唯一の原因・最善応手の読み切りとは言えません。</p>
<p>G1・G4の選出では3匹自体は同じで、先発が変わりました。勝ち数は増えていますが、6比較を考慮した検定では確証に達しません。G3のおんがえし／れいとうビームは探索を増やしても両方向に揺れ、継続対局は双方64勝でした。同じ結果の標本だけから、手の価値が完全に等しいとは証明できません。</p>
<p>したがって、追加探索には「稀に選ぶ劣る候補を減らす」実例があり、選出の先発精度も候補です。PPの消耗計画や滅びコンボ対策が改善の主因だという証拠は、この測定からは得られません。以前の756局は局面別の因果分解用ログではなく、今回の例をその6.48ポイントの内訳とは扱いません。</p>
<h2>3. PP耐久戦・滅びの歌へのソルバ</h2>
<h3>既存の実装と限界</h3>
<p>MCTSは木の最大先読み100ターン、展開先からの標準rolloutは8ターン、その後は静的評価です。PPの項は生存個体の残PP合計／最大PP合計に重み0.2を掛けています。この項だけでは「回復の最後の1PP」と「使わない技の1PP」の価値を区別しません。探索数を増やして標本誤差が減っても、短いrolloutや切断点の評価に由来する偏りは別に残り得ます。</p>
<p>滅びの歌には既に、交代可能なら期限に脱出する、捕まえた相手へ歌う、捕まった側が吹き飛ばし等で逃れるrollout処理があります。また、上下界を維持する汎用ソルバ、両者同時選択を扱う行列ゲーム、PP・残りターン数で処理順を付ける耐久戦分類も実装済みです。</p>
<p>ただし過去の汎用ソルバ混成案は、Web確認1,200局で<strong>45.75%（95%区間43.10～48.40%）</strong>、平均待ち時間+17.4%、最大Wasmメモリ+30.8%で不採用でした。公開情報の処理経路も変わる比較のため、悪化をソルバだけに帰属はできませんが、その案の再採用を支持する結果ではありません。</p>
<p>今回の新規終盤28局面を、各2秒・12,000ノード・最大100,000エンジン実行で再確認すると、値の幅0.02以内は<strong>8/28</strong>。残りは時間上限11、ノード上限9でした。回復ありの1対1として分類できたものは片側6、両側4ですが、主に早い決着の局面であり、PP切れまで粘る戦いの解決率を示す標本ではありません。局面値が狭くても、すべての初手の優劣が証明済みとは限りません。</p>
<h3>滅びの歌の再構成局面</h3>
{perish}
<p>4局面×4乱数×3予算。T26の脱出とT28の攻撃は予算を増やしても変わらず、T27の歌う／道連れの選択も各乱数内で変わりませんでした。T30では交代先が1系列だけ変化。過去の報告を再構成した入力であり、当時の厳密な乱数状態ではありません。ソルバは事前分布から作った単一仮説局面を解析し、全件[0,1]のままノード上限でした。この結果だけで、その局面が必敗／必勝とも、MCTSだけが原因とも言えません。</p>
<p>別の既存合成fixtureでは、相手の滅びが残り1で道連れを持つ時、エンジンの確率分岐モデル内で通常攻撃は得点上界0、ノーダメージで待つ手は得点下界約0.98065と識別できました。静的評価は0.30／0.38でも、帰結を見れば約0.98以上です。ただし27kのMCTSも両fixture・4乱数すべて正しく待ち、243kまで変わりません。これは帰結評価の実装可能性を示す対照で、新しい誤り修正ではありません。控えのコイキングを含む構成例なので実戦頻度の証拠にも使いません。</p>
<h3>次に実装するなら</h3>
<ol>
<li><strong>期限つき手順の判定を最優先にする。</strong>滅び残り1～3、拘束、交代・吠える・バトンタッチ等の逃走手段を入口にする。全合法応手と乱数分岐を調べ、「この相手は倒れる」「歌い手が退避できる」までの短い手順を証明する。残りの控えがいるなら撃破を試合の勝ち=1にせず、撃破・交代後の局面分布を評価する。分岐を探索し切れない部分は未確定として残す。</li>
<li><strong>PP耐久戦は、反復の検出と勝敗判定を分ける。</strong>HP・状態・場・選出がほぼ戻る反復を検出し、その間に減った攻撃PP、回復PP、回復アイテム、毒・眠り・能力段階を追う。回復1回に何回攻撃を消費させたか、先にどの有効技が尽きるかを計算する。総PP差は優勢の証明にしない。</li>
<li><strong>まず残り1対1・PP回復なしの領域へ限定する。</strong>既存のPPと残りターン数の単調な順序を利用し、資源が少ない状態から動的計画法で解く。重要PPが0になる境界、回復が間に合わなくなるHP境界を優先する。類似局面をまとめる際は、命中・急所・眠り・交代・悪あがき・PP回復で遷移が変わらないことを検証する。以前のPP圧縮試作は標本状態の削減が約2.36%に留まり、単なる圧縮の再導入では不十分です。</li>
<li><strong>完全な解より、具体的な戦略の保証を狙う。</strong>「この待機方策なら相手の全応手に勝てる」という下界や、コンボ防御に失敗するという上界を先に取る。全行動・全将来を一様に解く負担を避けられる可能性があります。有限資源や1001ターンの引き分けがあるため、見た目が同じでも無限の千日手とはみなしません。</li>
<li><strong>blindでは条件付きの保証として接続する。</strong>単一の仮説局面で解けても、情報集合全体の必勝ではありません。未観測技・持ち物・PPの候補を維持し、相手の秘密を知ったつもりの応手を混ぜない。MCTSの葉への条件付き評価と、公開情報だけで成立する確定手順を区別する。未解決なら通常探索へ戻し、判定の費用とメモリ上限を記録する。</li>
</ol>
<p>優先順位は、<strong>公開ビルドで81kの配分比較 → 実戦棋譜に基づく期限つきコンボ判定 → 限定したPP終盤の資源ソルバ</strong>です。汎用ソルバを常時ONにする案や、PP総量だけの新しい加点は今回の根拠では採用しません。報告された実戦の詰みや耐久負けそのものを同定するには、その棋譜と当時の観測情報が必要で、今回の既存再構成とは区別して追試します。</p>
<details><summary>検証・再現・解釈の範囲</summary><p>新規16局の全状態キー・合法手・抽出規則・終局を再検証。探索用継続768局と独立追試64局で不正手・1600判断打ち切りなし。滅び・道連れの既存回帰テスト4件も成功。同じ初手同士の4ペアは勝敗・手数が完全一致。各探索の訪問数合計・予算・乱数対応・全変更局面の追試網羅を検証しています。統計は64試行単位で、同じ局面の4探索を独立した実戦局面とは数えません。McNemarは2値の対応ある勝敗を使い、6比較を補正。差の区間は各勝率の97.5%正確二項区間の差による保守的95%区間です。区間は局面別で、6比較同時の保証ではありません。</p><p>旧24局コーパスは現エンジンで初期状態キーが一致せず採用を中止し、旧証明も転用していません。再構成相手チームは現在の検証に拒否されたため、検証を迂回せず単一belief仮説として別扱いにしました。失敗した試行・修正前ソース・データと実行ファイルのハッシュはプロジェクト内の findings/bot-strength-investigation-v1 に保存しています。</p><p><a href="summary.json">集計JSON</a> · <a href="reproduce.txt">再現手順と保存物</a></p></details>
<h2>アルゴリズムの参考資料</h2><p>MCTSに証明結果を戻す方法、determinizationの情報集合の扱い、同時選択を含む確率ゲームの上下界計算の一次資料です。適用方針はこのプロジェクトの実測とコードからの提案で、論文が本botの改善を保証するわけではありません。</p><ul>{refs}</ul>
</html>'''

if __name__=='__main__':
    j=collect();out=HERE/'report';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
    (out/'index.html').write_text(render(j))
    files=list((HERE/'probes').glob('*.json'))+list((HERE/'continuations').glob('*.json'))+list((HERE/'confirmation').glob('*.json'))+list((HERE/'solver').glob('*.json'))+list((HERE/'positions').glob('*.json'))+list((HERE/'tactics').glob('*.json'))
    (HERE/'result-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},indent=2)+'\n')
    print(json.dumps({'checks':j['checks'],'roots':j['roots'],'flips':len(j['flips']),'continuation_roots':len(j['continuations'])},ensure_ascii=False))
