#!/usr/bin/env python3
"""Verify focused proof-search runs and publish only bounded claims."""
from pathlib import Path
import hashlib,html,json,re
P=Path('findings/theoretical-focus-v2');EPS=1e-8
names=['g4-t10','g4-t53','loop4-t500','loop12-t500']
g4=[json.loads(z)for z in Path('findings/deployed-endurance-v1/games/g4.jsonl').read_text().splitlines()if json.loads(z).get('type')=='decision']
checks=len(g4)
for t in [4,12]:
 trace=json.loads(Path(f'findings/deployed-endurance-v1/continuations/g4-d12-baseline-{t}.json').read_text())['trace']
 checks+=next(i+1 for i,r in enumerate(trace)if r['turn']==500)
assert checks==1115
files=['baseline.jsonl','directed.jsonl','target.jsonl','counter-4.jsonl','counter-12.jsonl','resume-4.jsonl','resume-12.jsonl','fair.jsonl']
data={f:[json.loads(z)for z in (P/f).read_text().splitlines()]for f in files}
for f in ['baseline.jsonl','directed.jsonl','target.jsonl','fair.jsonl']:assert [r['name']for r in data[f]]==names
for f in files:
 if f not in ['baseline.jsonl','directed.jsonl','target.jsonl','fair.jsonl']:assert len(data[f])==1
config_files={f:('counter' if f.startswith('counter-') else 'resume' if f.startswith('resume-') else f.split('.')[0])+'-config.json' for f in files}
for f,rs in data.items():
 assert all(r['config']==json.loads((P/config_files[f]).read_text())for r in rs)
base={r['name']:r for r in data['baseline.jsonl']}
for rs in data.values():
 for r in rs:
  assert r['state_key']==base[r['name']]['state_key']
  assert r['total_runs']>0
  if 'value' in r:
   assert r['actions']==base[r['name']]['actions'] and r['selected']==base[r['name']]['selected']
   assert all(-EPS<=lo<=hi+EPS and hi<=1+EPS for lo,hi in r['cells']+[r['value']]+r['loss'])
  else:
   assert r['fixed_side']==1 and r['fixed_first']==r['free_side_first']=='switch 2'
   assert r['cell_p0'][0]<=r['cell_p0'][1]
chance=[json.loads(z)for z in (P/'chance.jsonl').read_text().splitlines()];assert [r['name']for r in chance]==names
for r in chance:
 assert r['state_key']==base[r['name']]['state_key']
 assert len(r['leaves'])==2 and abs(sum(l['probability']for l in r['leaves'])-1)<EPS
 assert len({l['key_after_clearing_only_quick_claw_roll']for l in r['leaves']})==1
 assert all(l['roll_observable'] for l in r['leaves'])
 assert {l['quick_claw_roll']:l['probability']for l in r['leaves']}=={False:196/256,True:60/256}
fair={r['name']:r for r in data['fair.jsonl']};target={r['name']:r for r in data['target.jsonl']}
resume={r['name']:r for f in ['resume-4.jsonl','resume-12.jsonl']for r in data[f]}
for name,r in resume.items():
 cps=r['checkpoints'];first=cps[0];assert first['total_runs']==target[name]['total_runs']==2000000
 assert first['nodes']==target[name]['nodes'] and first['closed']==target[name]['closed']
 prev=None
 for cp in cps:
  if prev:
   assert cp['total_runs']>prev['total_runs']
   assert cp['value'][0]>=prev['value'][0]-EPS and cp['value'][1]<=prev['value'][1]+EPS
   assert cp['loss'][0]>=prev['loss'][0]-EPS and cp['loss'][1]<=prev['loss'][1]+EPS
  prev=cp
 assert cps[-1]['total_runs']==r['total_runs']

def selected_diag(r):
 i,j=[r['actions'][s].index(r['selected'][s])for s in [0,1]]
 return r['diagnostics'][i*len(r['actions'][1])+j]
coverage=[]
for name in names:
 previous=resume.get(name,target[name]);before=selected_diag(previous);after=selected_diag(fair[name])
 assert abs(before['pending_mass'])<EPS and abs(after['pending_mass'])<EPS
 assert before['unique_children']==after['unique_children']==2
 minor=lambda d:next(c for c in d['largest_children']if abs(c['mass']-60/256)<EPS)
 assert minor(before)['runs']==0 and minor(after)['runs']>0
 raw=next(r for r in chance if r['name']==name)
 assert {c['key'] for c in before['largest_children']}=={'Exact('+l['state_key']+')' for l in raw['leaves']}
 coverage.append({'name':name,'before_total_work':previous['total_runs'],'before_minor_child_work':0,'after_minor_child_work':minor(after)['runs'],'after_child_uncertainty':after['child_uncertainty'],'fair_stop':fair[name]['stop']})
combined=[]
for name in names:
 rs=[r for f,rr in data.items()for r in rr if r['name']==name and 'value'in r]
 lo=max(r['value'][0] for r in rs);hi=min(r['value'][1]for r in rs)
 cells=[[max(r['cells'][k][0]for r in rs),min(r['cells'][k][1]for r in rs)]for k in range(len(rs[0]['cells']))]
 actions=rs[0]['actions'];nc=len(actions[1]);chosen=[actions[s].index(rs[0]['selected'][s])for s in [0,1]]
 # Fixed P2 is a legal opponent strategy. Its upper endpoint caps the
 # unrestricted P0 value of this joint first-action cell; its lower endpoint
 # is NOT a lower bound on unrestricted P0 value.
 for f in ['counter-4.jsonl','counter-12.jsonl']:
  for r in data[f]:
   if r['name']==name:cells[chosen[0]*nc+chosen[1]][1]=min(cells[chosen[0]*nc+chosen[1]][1],r['cell_p0'][1])
 losses=[]
 for s in [0,1]:
  row=[cells[chosen[s]*nc+j]for j in range(nc)] if s==0 else [[1-cells[i*nc+chosen[s]][1],1-cells[i*nc+chosen[s]][0]]for i in range(len(actions[0]))]
  pl,pu=min(v[0]for v in row),min(v[1]for v in row)
  vl,vu=(lo,hi)if s==0 else (1-hi,1-lo)
  losses.append([max(0,vl-pu-EPS),min(1,max(0,vu-pl+EPS))])
 combined.append({'name':name,'value_p0':[max(0,lo-EPS),min(1,hi+EPS)],'score_loss':losses,'cells_p0':cells})
assert all(r['value_p0']==[0,1] and r['score_loss']==[[0,1],[0,1]]for r in combined),'New evidence: rewrite the conclusion before delivery'
log=(P/'fixed-tests.log').read_text();assert re.search(r'test result: ok\. 6 passed; 0 failed;',log)
assert all(r['depth_limit_trials']==0 for rs in data.values()for r in rs if 'depth_limit_trials'in r)
summary={'conclusion':'No separation of current and alternative value bounds. No stronger action certified. This does not prove optimality.', 'roots':4,'solver_runs':sum(map(len,data.values())),'engine_executions':sum(r['total_runs']for rs in data.values()for r in rs),'combined':combined,'coverage':coverage,'resume':{n:r['checkpoints']for n,r in resume.items()},'maximum_compacted_solved_states':{n:r['closed']for n,r in resume.items()},'maximum_baseline_path_depth':max(len(r['depth_histogram'])-1 for r in data['baseline.jsonl']),'tests_passed':6,'model':'Full-information engine chance enumeration; optimal continuation unless fixed-counterpolicy is stated; score W=1,D=0.5,L=0; not a blind-policy win-rate experiment.'}
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
source=['crates/bot/examples/theoretical_focus.rs','crates/bot/examples/theoretical_headroom.rs','crates/bot/src/bounds.rs','crates/bot/src/bounds/loss_threshold.rs','crates/bot/src/bounds/correlated.rs','crates/engine/src/battle/enumerate.rs','crates/engine/src/battle/search.rs','crates/engine/src/battle/turn.rs','data/gen2stadium2.json','tools/theoretical-focus-report.py']
source += ['findings/deployed-endurance-v1/games/g4.jsonl']+[f'findings/deployed-endurance-v1/continuations/g4-d12-baseline-{t}.json'for t in [4,12]]
protocol={'base_commit':'32100309','roots':names,'input_parity':'Reconstruct from initial teams and recorded choices. Check 1115 historical states and original g4 PRNG states each invocation. Hidden Power display normalization inherited from the first study. Root keys and recorded actions must match between runs.','interpretation':'No new bot policy deployed or evaluated in live duels. Fixed-counterpolicy restricts only P2, so only its P0 upper endpoint is used as a full-game upper bound for the selected joint first actions.','schedules':{'baseline':'Previous 200000 work / 10000ms / 100000 nodes / depth24','directed':'Historical filename: probability-first eager expansion, safe Quick Claw quotient, depth1024; solve_policy_losses has no direction override, so material ordering is inactive here','target':'One-sided policy-loss threshold (1e-6), explicit bound direction, material descent; 2000000 work / 60000ms / 300000 nodes','resume':'Same graph reused up to five calls; 2000000 additional work per call; one cumulative node cap of 300000','fixed-counterpolicy':'P2 first switches; then priority Struggle, Surf, Thunder, Ice Beam, Rest, Switch2, with Rest<=90% and wake handling. P1 first action fixed to recorded switch; all later P1 choices remain free. Full-information certificate attempt only.','fair':'Same target configuration plus chance_fair_period=8; changes proof-search allocation only','chance-audit':'Complete first-step enumeration with recorded joint actions; coalesce equal state keys, inspect Quick Claw roll and compare keys after clearing only that flag'},'unused_config':'deep-config.json was not executed: diagnostics showed zero depth-limit hits.','reproduce':'cargo build --release -p nc2000-bot --example theoretical_focus; target/release/examples/theoretical_focus CONFIG.json NEW-OUTPUT.jsonl all|ROOT-NAME. Each output must be new. See *-config.json files.','limitations':['Floating-point numerical bounds, not formally verified real arithmetic','Independent uniform per-draw chance model, not enumeration of all PRNG seeds','Native engine has historical parity with pinned deployed WASM, not exhaustive equivalence','Known opponent sets and hidden state: not a blind information-set proof','Score is not pure win probability; all intervals remain unconstrained, so neither metric establishes improvement','Budgets are checked at work boundaries; node/time caps may be exceeded within one expansion','Runs overlap in wall-clock time: durations are not a controlled speed benchmark'],'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest()for f in source}}
(P/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
labels={'g4-t10':'回復・T10','g4-t53':'満タン回復・T53','loop4-t500':'交代反復・試行4 T500','loop12-t500':'交代反復・試行12 T500'}
tr=''.join(f'<tr><td>{labels[c["name"]]}</td><td>{c["before_total_work"]:,}</td><td>0</td><td>{c["after_minor_child_work"]:,}</td><td>0〜100%</td></tr>'for c in coverage)
cr=''.join(f'<tr><td>{labels[n]}</td><td>{r["total_runs"]:,}</td><td>{r["closed"]:,}</td><td>{r["stop"]}</td><td>0〜100%</td></tr>'for n,r in resume.items())
(P/'index.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NC2000：重要4局面の証明探索・続報</title><style>body{{max-width:62rem;margin:2rem auto;padding:0 1rem 3rem;background:#f1eee6;color:#302f29;font:17px/1.8 system-ui}}h1{{font-size:1.55rem}}h2{{font-size:1.15rem;margin-top:2rem}}a{{color:#4e6350}}.box{{padding:1rem;background:#e4e8dd;border-left:4px solid #708068}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}td,th{{padding:.5rem;border-bottom:1px solid #cac9bc;text-align:left}}.scroll{{overflow-x:auto}}</style>
<h1>重要4局面の証明探索・続報</h1><p>回復を選んだ2局面と、交代が反復した2局面について、現在手の上限と代替手の下限を分離できるか追試した。</p><div class="box"><strong>今回も、より勝率の高い手の存在は証明できなかった。</strong><p>4局面とも開始局面の得点率と現在手の改善余地は、依然として0〜100%／0〜100ポイントの範囲。現行botの最適性は未証明である。</p><p>新しく確認できたのは、証明用ソルバーが23.4375%の分岐を未展開に残していたこと。この偏りを解消する設定も試したが、開始局面の区間は縮まらなかった。</p></div>
<h2>探索を進めた範囲</h2><p>同じ4局面について、通常配分、確率優先、現在手の損失に集中する探索、相手を固定する上限計算、同じ探索グラフの継続利用、未展開の分岐にも定期的に配分する探索を行った。計{summary['solver_runs']}実行、合計{summary['engine_executions']:,}回のエンジン実行。これは対戦数・独立標本数ではなく、枝を列挙する計算量である。公開botの方策は変更していない。</p>
<h2>単純な深さ制限ではなかった</h2><p>元の設定は探索経路24段だったが、4局面の再測定では到達深さは最大{summary['maximum_baseline_path_depth']}段で、制限に当たった回数はゼロだった。上限を1,024段にした重点探索でも、深さ制限による打ち切りは起きていない。主な停止理由は計算時間・実行回数・保持局面数の上限だった。</p>
<h2>約23%の継続局面が未展開だった</h2><p>実際に双方が選んだ手の組合せについて、初手の乱数分岐は列挙済みだった。次の状態は、せんせいのツメの発動判定による2種類（非発動76.5625%、発動23.4375%）。このフラグだけを消すと状態キーは一致したが、次ターンの挙動に影響するため、そのまま同一局面として扱うことはできない。</p><p>従来の配分では76.5625%側の値が解けない間、23.4375%側の継続が未展開のまま残った。同じ分岐点への8回の訪問ごとに未解決の子を順番に選ぶ設定で、4局面ともこの側へ実行を配分できた。</p><div class="scroll"><table><tr><th>局面</th><th>従来側の総実行数</th><th>23.4375%側・従来</th><th>同・公平配分</th><th>開始局面の得点率</th></tr>{tr}</table></div><p>分岐側の実行数は、その継続局面に記録された累積値。他の経路からの到達を含みうる。公平配分は探索範囲を広げるため、保持局面数の上限に早く達する例もあり、証明が速くなるとは確認できなかった。</p>
<h2>解けた局面を再利用しても分離しなかった</h2><p>2百万実行ごとの探索を同じグラフ上で再開した。最初の2百万実行時点の局面数・解決済み数は、独立した前の実行と一致。以前の作業を捨てずに継続できていることを確認した。</p><div class="scroll"><table><tr><th>局面</th><th>累積実行数</th><th>圧縮保存された解決済み局面</th><th>停止理由</th><th>開始局面の得点率</th></tr>{cr}</table></div><p>解決済み局面が増えたことと、開始局面の評価が確定したことは別。未解決の応手・継続が残り、現在手と代替手の優劣は確定しなかった。「解決済みキャッシュがゼロ」は終局分岐ゼロを意味しない点も確認した。</p>
<h2>相手を固定した上限計算</h2><p>交代反復の2局面では、相手P2がまず交代し、その後は攻撃と条件付きのねむるを使う具体的な方策に固定した。P1は最初だけ記録された交代を選び、それ以降は全合法手から最善応答できる。相手の選択肢を制限したゲームのP1上限は、本来のゲームのP1上限として使える。一方、その下限を本来の下限として流用していない。</p><p>各60秒の計算で、この上限も1.0から下げられなかった。これは、その相手に必ず勝てる・負けるという意味ではなく、計算が未完了という意味である。詳細な相手方策は調査条件に保存した。</p>
<h2>結論と残る課題</h2><p>重点4局面の存在判定は未完了。今回の結果から、現在手が最適とも、良い代替手がないとも言えない。深さ上限、単純な計算量増加、初手分岐の偏りだけでは、この未解決を解消できなかった。</p><p>必要なのは、回復・交代・PP消耗が続く途中局面に、保証付きの継続価値を与えること。次に検討するのは、長い継続をまとめて扱う証明手法である。さらに予算を増やせば解けるかは未確認。未探索の値を評価関数で埋めても、それだけでは存在証明にならない。</p>
<h2>検証・解釈の範囲</h2><p>同じ初期チームと全着手から4局面を再構成し、各実行で1,115個の履歴盤面と元対局の乱数状態を照合した。初手の確率分布を別途完全列挙し、状態キー・発動フラグ・確率を検証。相手固定と初手固定の向き、および継続探索に関する既存テスト6件が通過した。</p><p>全情報・独立一様乱数モデルでの数値的な区間計算であり、形式検証済みの証明ではない。相手の未公開情報を知らないblind方策の証明でもない。得点は勝ち1・引き分け0.5で、純粋な勝率とは区別する。今回どちらについても改善の証拠は得られていない。公開WASMとの一致確認は保存履歴に対するもので、全反実仮想における同値性を証明したものではない。</p>
<p><a href="summary.json">集計</a> · <a href="protocol.json">調査条件・再現方法</a> · <a href="audit.json">検証結果</a> · <a href="fair.jsonl">公平配分の詳細</a> · <a href="chance.jsonl">初手分岐の完全列挙</a> · <a href="/nc2000-theoretical-headroom/index.html">前回の471局面調査</a></p></html>''')
tracked=source+[str(p)for p in P.glob('*-config.json')]+[str(P/f)for f in files+['chance.jsonl','summary.json','protocol.json','index.html','fixed-tests.log']]
audit={'passed':True,'root_key_and_action_parity':True,'four_chance_partitions_verified':True,'minor_branch_explored_in_all_fair_runs':True,'resume_first_checkpoint_reproduces_prior_run':True,'six_fixed_policy_tests_passed':True,'bounds_separated':False,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in tracked}}
(P/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items()if k in ['roots','solver_runs','engine_executions','maximum_baseline_path_depth','maximum_compacted_solved_states','tests_passed']},ensure_ascii=False,indent=2))
