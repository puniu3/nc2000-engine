#!/usr/bin/env python3
"""Audited, pair-clustered analysis of the fixed allocation screen."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {
    'preview81': '選出81k・対戦27k',
    'preview243': '選出243k・対戦27k',
    'battle81': '選出27k・対戦81k',
    'both81': '全体81k',
    'uniform30': '全体30k',
}


def read(path):
    return json.loads(path.read_text())


def selected(record, side):
    rows = [d for d in record['result']['decisions'] if d['phase'] == 'preview' and d['side'] == side]
    assert len(rows) == 1
    return tuple(map(int, re.findall(r'\d+', rows[0]['choice'])))


def audit_null():
    for pair in range(3):
        rows = [read(HERE / 'null' / f'{name}-p{pair:03}-s{side}.json')
                for name in ['null-a', 'null-b'] for side in range(2)]
        for r in rows:
            for d in r['result']['decisions']:
                del d['ns']
        assert all(r['result'] == rows[0]['result'] for r in rows)
        assert sum(r['a_score'] for r in rows) == 2


def analyze():
    p = read(HERE / 'protocol.json')
    plan = read(HERE / 'analysis-protocol.json')
    manifest = read(HERE / 'games/manifest.json')
    merge = read(HERE / 'merge.json') if (HERE / 'merge.json').exists() else None
    assert p == manifest['protocol']
    assert hashlib.sha256((HERE / 'protocol.json').read_bytes()).hexdigest() == manifest['protocol_sha256']
    assert hashlib.sha256((HERE / 'measured-source.rs').read_bytes()).hexdigest() == manifest['source_sha256']
    assert '1 passed; 0 failed' in (HERE / 'parity-test.log').read_text()
    audit_null()
    n = p['pairs']
    all_games = {}
    for q in p['profiles']:
        for pair in range(n):
            for side in range(2):
                r = read(HERE / 'games' / f'{q["id"]}-p{pair:03}-s{side}.json')
                if merge:
                    name=f'{q["id"]}-p{pair:03}-s{side}.json'
                    assert hashlib.sha256((HERE/'games'/name).read_bytes()).hexdigest()==merge['provenance'][name]['sha256']
                assert (r['profile'], r['pair'], r['a_side']) == (q['id'], pair, side)
                assert not r['result']['cap']
                assert r['a_score'] == (r['result']['p1_score'] if side == 0 else 1 - r['result']['p1_score'])
                for d in r['result']['decisions']:
                    config = q if d['side'] == side else p['opponent']
                    expected = 0 if d['phase'] == 'forced' else config[d['phase']]
                    assert d['iterations'] == expected
                    assert d['choices'] == 1 if d['phase'] == 'forced' else d['choices'] >= 2
                all_games[q['id'], pair, side] = r
    assert len(list((HERE / 'games').glob('*-p*-s*.json'))) == len(all_games)
    rng = np.random.default_rng(plan['bootstrap_seed'])
    draws = rng.integers(0, n, size=(plan['bootstrap_replicates'], n))
    ci = lambda values: np.quantile(np.asarray(values)[draws].mean(axis=1), [.025, .975]).tolist()
    pair_scores = {}
    summaries = []
    roots = []
    for q in p['profiles']:
        ident = q['id']
        rows = [all_games[ident, pair, side] for pair in range(n) for side in range(2)]
        scores = np.array([r['a_score'] for r in rows])
        pair_scores[ident] = scores.reshape(n, 2).mean(axis=1)
        iterations = []
        seconds = {}
        battle_calls = []
        phase_times = {}
        selections = {'same': 0, 'lead_only': 0, 'members_changed': 0, 'bench_order_only': 0}
        for r in rows:
            pair, side = r['pair'], r['a_side']
            own = [d for d in r['result']['decisions'] if d['side'] == side]
            iterations.append(sum(d['iterations'] for d in own))
            name=f'{ident}-p{pair:03}-s{side}.json'
            platform=merge['provenance'][name]['platform'] if merge else 'mac-arm64'
            seconds.setdefault(platform,[]).append(sum(d['ns'] for d in own) / 1e9)
            phase_times.setdefault(platform,{'preview':[],'battle':[]})
            battle_calls.append(sum(d['phase'] == 'battle' for d in own))
            for phase in phase_times[platform]:
                phase_times[platform][phase].extend(d['ns'] / 1e9 for d in own if d['phase'] == phase)
            before = selected(all_games['battle81', pair, side], side)
            # All baseline preview decisions must match, regardless of profile or assignment.
            for other in p['profiles']:
                assert selected(all_games[other['id'], pair, 1-side], side) == before
            after = selected(r, side)
            kind = ('same' if before == after else 'members_changed' if set(before) != set(after)
                    else 'lead_only' if before[0] != after[0] else 'bench_order_only')
            selections[kind] += 1
            roots.append(dict(profile=ident, pair=pair, side=side, baseline=before, chosen=after, change=kind))
            assert selected(all_games['preview81', pair, side], side) == selected(all_games['both81', pair, side], side)
        summaries.append(dict(id=ident, label=LABELS[ident], preview=q['preview'], battle=q['battle'],
            games=len(rows), wins=int(sum(scores == 1)), ties=int(sum(scores == .5)), losses=int(sum(scores == 0)),
            score=float(scores.mean()), ci95_cluster_bootstrap=ci(pair_scores[ident]),
            mean_iterations=float(np.mean(iterations)), median_iterations=float(np.median(iterations)),
            native_seconds_by_platform={key:dict(games=len(v),mean=float(np.mean(v))) for key,v in seconds.items()},
            mean_battle_calls=float(np.mean(battle_calls)), median_battle_calls=float(np.median(battle_calls)),
            mean_turns=float(np.mean([r['result']['turns'] for r in rows])),
            native_phase_timing={key:{phase: dict(n=len(v), mean=float(np.mean(v)), p95=float(np.quantile(v, .95))) for phase, v in phases.items()} for key,phases in phase_times.items()},
            selections=selections))
    contrasts = [dict(a=a, b=b, difference=float(np.mean(pair_scores[a]-pair_scores[b])),
                      ci95_cluster_bootstrap=ci(pair_scores[a]-pair_scores[b])) for a, b in plan['paired_contrasts']]
    baseline_calls = [sum(d['phase'] == 'battle' and d['side'] != r['a_side'] for d in r['result']['decisions']) for r in all_games.values()]
    common_b = float(np.mean(baseline_calls))
    for q in summaries:
        q['fixed_path_iteration_ratio'] = (q['preview']+q['battle']*common_b)/(27000*(1+common_b))
    preview_choices = [d['choices'] for r in all_games.values() for d in r['result']['decisions'] if d['phase'] == 'preview']
    catalog = {t['id']:t for t in read(HERE.parent.parent/'data/team-pool-v2/team-pool.json')['teams']}
    selection_comparison = []
    for schedule in manifest['schedules']:
        pair = schedule['pair']
        for side in range(2):
            team_id = schedule['teams'][side]
            species = catalog[team_id]['species']
            choices = {label:selected(all_games[ident,pair,side],side)
                       for label,ident in [('27k','battle81'),('81k','preview81'),('243k','preview243')]}
            selection_comparison.append(dict(pair=pair, side=side, team=team_id,
                opponent=schedule['teams'][1-side], choices=choices,
                species={k:[species[i-1] for i in indices] for k,indices in choices.items()}))
    return dict(schema='nc2000-budget-allocation-report-v1', protocol=p, analysis=plan,
                total_games=len(all_games), independent_pairs=n, arms=summaries, contrasts=contrasts,
                cost_reference_mean_battle_calls=common_b,
                preview_legal_actions=dict(min=min(preview_choices), max=max(preview_choices), median=float(np.median(preview_choices))),
                selection_roots=roots,
                selection_comparison=selection_comparison,
                preview81_to_243_changed=sum(r['choices']['81k'] != r['choices']['243k'] for r in selection_comparison),
                paired_scores={k:v.tolist() for k,v in pair_scores.items()},
                checks=dict(null_games=12, null_physical_trajectories_equal=True, unchanged_budget_parity=True,
                            caps=0, budget_assignments=True, baseline_preview_parity=True, preview81_both81_parity=True),
                execution_platforms=merge['platform_parity'] if merge else {'machine':'arm64'},
                input_hashes={k:v for k,v in manifest.items() if k.endswith('_sha256')})


def render(result=None):
    p = read(HERE / 'protocol.json')
    completed = len({f.name for folder in ['games','games-local','games-local-extra','games-local-recovery'] for f in (HERE/folder).glob('*-p*-s*.json')})
    results = f'<p class="notice">計測中：{completed} / 240 局。全局完了まで勝敗による候補選びは行いません。</p>'
    if result:
        rows = ''
        for q in result['arms']:
            lo, hi = q['ci95_cluster_bootstrap']
            changes = q['selections']
            rows += f'<tr><th>{q["label"]}</th><td>{q["wins"]} / {q["ties"]} / {q["losses"]}</td><td>{q["score"]:.1%}<small>{lo:.1%}～{hi:.1%}</small></td><td>{q["fixed_path_iteration_ratio"]:.2f}倍</td><td>{changes["lead_only"]} / {changes["members_changed"]} / {changes["bench_order_only"]}</td></tr>'
        contrast_rows = ''
        for c in result['contrasts']:
            lo, hi = c['ci95_cluster_bootstrap']
            contrast_rows += f'<tr><th>{LABELS[c["a"]]} − {LABELS[c["b"]]}</th><td>{100*c["difference"]:+.1f}pt</td><td>{100*lo:+.1f}～{100*hi:+.1f}pt</td></tr>'
        results = f'''<p class="notice">240局完了。各設定48局・24組。同一組の2局をまとめた95% bootstrap区間を表示しています。</p>
        <div class="scroll"><table><thead><tr><th>配分</th><th>勝 / 分 / 敗</th><th>対27k 得点率<small>95%区間</small></th><th>同じ判断回数なら<small>総探索量</small></th><th>選出変更 / 48<small>先発 / 3匹 / 控え順のみ</small></th></tr></thead><tbody>{rows}</tbody></table></div>
        <p>計算量の比較は、共通の対戦中判断回数 {result['cost_reference_mean_battle_calls']:.1f} 回に各設定を適用した算術値です。勝敗で試合の長さが変わる影響を除いています。実際の消費量・並列実行中の時間は <a href="summary.json">集計JSON</a> に含みます。</p>
        <h3>同じ組で比較した得点率の差</h3><div class="scroll"><table><thead><tr><th>比較</th><th>差</th><th>95%区間</th></tr></thead><tbody>{contrast_rows}</tbody></table></div>
        <p>区間は探索的な参考値です。5設定を同時に保証する区間ではなく、わずかな差の勝者や最適点を確定する検定ではありません。個々の対戦の勝敗は独立240標本として扱っていません。</p>'''
        species_ja = read(HERE.parent.parent/'data/i18n-ja.json')['species']
        translate = lambda name: species_ja.get(re.sub('[^a-z0-9]', '', name.lower()), name)
        selection_rows = ''.join('<tr><th>'+str(r['pair'])+' / '+str(r['side']+1)+'<small>'+html.escape(r['team'])+'</small></th>'+''.join('<td>'+html.escape(' → '.join(map(translate,r['species'][k])))+'</td>' for k in ['27k','81k','243k'])+'</tr>' for r in result['selection_comparison'])
        results += f'''<h3>具体的に変わった選出</h3><p>81kから243kでも {result['preview81_to_243_changed']} / 48 局面で選出順序が変わりました。判断が変わったことだけでは、強さが改善した・飽和したとは判定できません。以下は全48局面を組番号順に掲載しています。並びは先発、控え1、控え2です。</p><details><summary>全選出を比較</summary><div class="scroll"><table><thead><tr><th>組 / 座席</th><th>27k</th><th>81k</th><th>243k</th></tr></thead><tbody>{selection_rows}</tbody></table></div></details>'''
    verdict = (HERE / 'conclusion.html').read_text() if result and (HERE / 'conclusion.html').exists() else '<p>選出への追加計算は、1試合に1回だけなので低コストです。ただし、選出が重要であることと、27kから増やす限界効果が大きいことは別です。固定した5設定を比較して確かめています。</p>'
    return f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>探索予算の配分 — nc2000</title>
<style>:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f4f2eb;color:#292d28;font:16px/1.8 system-ui,sans-serif}}main{{max-width:1040px;margin:auto;padding:24px}}h1{{font-size:1.8rem;line-height:1.5}}h2{{font-size:1.35rem;margin-top:2.3em}}h3{{font-size:1.05rem}}a{{color:#526b47}}.notice{{border-left:4px solid #7e8867;padding:12px 16px;background:#e6e9df}}section{{background:#fffef9;padding:16px 24px;border-radius:8px;margin:18px 0}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #dcded2;white-space:nowrap}}small{{display:block;color:#5e625b;font-size:.8rem}}input{{accent-color:#647957;width:min(420px,100%)}}.bar{{height:16px;background:#889675;border-radius:3px;margin-bottom:10px}}.formula{{font-family:ui-monospace,monospace;overflow-wrap:anywhere}}footer{{margin:30px 0;color:#606459}}@media(max-width:600px){{main{{padding:14px}}section{{padding:14px}}h1{{font-size:1.5rem}}}}</style>
<main><p><a href="/">共有プレビュー</a> · <a href="/nc2000-strength/index.html">前回の調査</a></p><h1>選出に探索予算を寄せると、効率よく強くなるか</h1><p>探索回数だけを変える比較。ソルバ・評価関数・情報・探索方針は比較対象から外しています。</p>
<section><h2>現時点の判断</h2>{verdict}</section>
<section><h2>固定した比較の結果</h2>{results}</section>
<section><h2>1回の選出に多く使うと、総量はどのくらい増えるか</h2><p>対戦中の探索判断が <strong id="count">30</strong> 回ある場合。強制手には探索を割り当てません。</p><label>対戦中の判断回数 <input id="decisions" type="range" min="5" max="100" value="30" aria-label="対戦中の探索判断回数"></label><div id="costs"></div><p class="formula">総探索量 = 選出予算 + 対戦中の判断回数 × 対戦中予算</p><p>この図は回数の計算であり、勝率・待ち時間の予測ではありません。試合展開による判断回数の変化も含みません。</p></section>
<section><h2>公開版への適用で変わる点</h2><p>選出では強さを優先し、10秒程度まで許容するという利用者の希望を確認済みです。選出243kも実用候補に含めます。前回のMac Chromiumの選出1局面では、27kは0.60～0.61秒、81kは1.96～2.10秒、243kは6.65秒でした。iPad Safariの実測ではありません。</p><p>公開版は、人間が考えている間に最低予算27kを超えて、最大270kまで先読みします。選出に時間を使う利用者では、すでに81k以上を読んでいる可能性があります。選出の最低値だけ上げても、その利用者に追加の探索は発生しません。</p><p>現在の実装で27kを81kへ単純変更すると、上限も270kから810kへ増えます。利用者が指定した次の設計では、この伸び縮みを廃止し、先行して思考を開始したうえで固定回数で終了させます。今回の先読みなし比較は固定予算の比較ですが、現行公開版・対人戦の改善幅を直接示しません。</p></section>
<section><h2>調査後の実装方針（利用者指定）</h2><p>既知パーティの選出は事前計算してビルドに含め、未知パーティには効果の飽和を測定した探索数を使います。対戦中は2〜3段階の固定予算をプレイヤーが選びます。プレイヤーの選択中に計算する仕組みは維持し、早打ち・長考で探索回数が変わる仕組みは廃止します。同じ局面・乱数・選択した予算なら、入力タイミングにかかわらず探索回数と判断が一致することを確認条件にします。</p><p>今回の標本は既知カタログです。未知パーティの飽和点をこの結果から決めることはできません。旧選出テーブルの仕組みは残っていますが、現行カタログ用テーブルはなく、旧60候補の扱いも現行の順序付き選出に合わせて再検証が必要です。テーブル参照は公開情報から特定できる相手に限り、非公開の技・持ち物をキーとして漏らさない条件を維持します。対戦中243kの優位性も今回の比較対象には含みません。</p></section>
<section><h2>比較方法と確認</h2><p>製品カタログのdrawWeightに従ってチームを抽選。相手は全判断27k。24組のチーム・対戦乱数・座席ごとの探索乱数を共通にし、各設定で担当側を入れ替えた2局を実行します。全240局を事前固定し、設定と座席の実行順を混ぜています。</p><p>対戦の進行・探索乱数・観測を保持したまま、選出か対戦中かで反復数だけを変更。予算を変えない場合に既存agentと判断・乱数列が一致する試験を通過。さらに同じ設定同士12局で、時間を除く全状態・行動列が一致する対照を確認しました。反復数を変えると乱数の消費量も変わるため、同じ選出になった試合でも後の展開が完全一致するとは限りません。</p><p>96局完了時点で、利用者の依頼に従い残りを分割しました。最終的な割当はMac195局、Google Cloud Batch45局です。比較するチーム・乱数・予算・局数は固定したままです。異なるCPU間の確認用2局は得点に含めず、状態・行動・結果が一致することを確認して結果を統合しました。所要時間はCPU別に分けて記録しました。Cloudは準備時の2件の不備を修正し、最終実行では時間上限までに45局を完了しました。未完了3局は同じ条件でMacで補完しました。3回の起動を含む費用見積は約103円（クレジット適用前・請求額ではありません）。全ジョブのリソース削除を確認済みです。<a href="cloud-execution.json">実行記録</a>。</p><p>得点は勝ち1・引き分け0.5・負け0。同じ組の2局をクラスタとして10万回bootstrapし、設定間の差も組を対応させて計算します。公開WASMと今回のnativeは完全に同じ成果物ではありません。現行の公開設定は変更していません。</p><p><a href="protocol.json">固定プロトコル</a> · <a href="analysis-protocol.json">解析計画</a> · <a href="summary.json">全集計</a> · <a href="reproduce.txt">再現手順</a> · <a href="evidence.zip">生データと解析コード</a></p></section><footer>共有ハブの項目名：nc2000-budget。外部の公開サイトへの配信変更はありません。</footer></main>
<script>const profiles={json.dumps(p['profiles'])};const labels={json.dumps(LABELS, ensure_ascii=False)};const slider=document.querySelector('#decisions');function show(){{const d=Number(slider.value);document.querySelector('#count').textContent=d;document.querySelector('#costs').innerHTML=profiles.map(p=>{{const ratio=(p.preview+p.battle*d)/(27000*(d+1));return `<div>${{labels[p.id]}}：${{ratio.toFixed(2)}}倍（+${{((ratio-1)*100).toFixed(1)}}%）<div class="bar" style="width:${{ratio/3*100}}%"></div></div>`}}).join('')}}slider.addEventListener('input',show);show();</script></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    out = HERE / 'report'
    out.mkdir(exist_ok=True)
    result = None if args.partial else analyze()
    if result:
        (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    else:
        (out / 'summary.json').write_text(json.dumps({'status': 'running', 'scores_not_analyzed': True})+'\n')
    (out / 'index.html').write_text(render(result))
    for name in ['protocol.json', 'analysis-protocol.json', 'cloud-execution.json']:
        (out / name).write_bytes((HERE / name).read_bytes())
    (out / 'reproduce.txt').write_text('''From the nc2000-engine repository root (Rust 1.98.1):
cargo test --release -p nc2000-bot --example budget_allocation
cargo build --release -p nc2000-bot --example budget_allocation
target/release/examples/budget_allocation findings/budget-allocation-v1/null-protocol.json NEW_NULL_DIR
target/release/examples/budget_allocation findings/budget-allocation-v1/protocol.json NEW_GAME_DIR

Compare each newly produced game JSON to the same filename in the archived
null/ or merged games/ directory, deleting only result.decisions[*].ns from
both records. They must be identical. Do not replace the original evidence.
The current harness supports optional JOB_LIST and NC2000_BUDGET_THREADS;
these change scheduling only, as verified by shard and cross-platform canaries.

To regenerate the report from the original evidence:
python3 findings/budget-allocation-v1/report.py

Python dependency: numpy. The analyzer verifies original protocol/source hashes,
all 240 game files, each decision budget, null parity and preview parity. Merge
provenance contains per-game SHA256 hashes. Native concurrent elapsed times
are separated by CPU platform and must not be interpreted as browser latency.
''')
