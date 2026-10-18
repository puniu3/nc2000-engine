#!/usr/bin/env python3
"""Render only progress until all fixed-plan results have passed the audit."""
import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil

root = Path('findings/mixed-policy-strength-v1')
main = root / 'main'
plan = json.loads((main / 'plan.json').read_text())
completed = len(list((main / 'rows').glob('*.json')))
canary = json.loads((root / 'exploiter-canary.json').read_text()) if (root / 'exploiter-canary.json').exists() else None
parity_path = root / 'wasm-parity.json'
parity = json.loads(parity_path.read_text()) if parity_path.exists() else None
summary = json.loads((main / 'summary.json').read_text()) if (main / 'summary.json').exists() else None
pilot_path = root / 'pilot/summary.json'
pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else None
audit_path = root / 'final-audit.json'
audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
stamp = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M JST')
mode_name = {'legacy': '従来方式', 'visit-sample': '訪問割合で抽選', 'estimated-mix': '混合推定器'}
kind_name = {'direct': '従来方式と直接対決', 'oracle': '対象方策を読むexploiter', 'control': '方策を読まない対照'}
def esc(x): return html.escape(str(x))
def pct(x): return f'{100*x:.2f}%'
def pp(x): return f'{100*x:+.2f} pp'
def interval(x, signed=False):
    fmt = pp if signed else pct
    return f'{fmt(x["estimate"])} ［{fmt(x["interval"][0])}, {fmt(x["interval"][1])}］'

body = f'<p class="muted">{stamp} · nc2000-engine · 固定計画による測定</p><h1>混合方策は勝率と耐性を改善するか</h1>'
if summary is None:
    body += f'<div class="box"><strong>測定中：{completed} / {len(plan["jobs"])} 局完了</strong><p>途中の勝敗から結論を出さず、固定した全試合が揃ってから判定します。更新は手動の再読み込みで確認できます。</p></div>'
else:
    assert completed == len(plan['jobs']) == summary['games']
    body += f'<div class="box"><strong>{completed} 局の検証完了</strong><p>全局がエンジンの終局まで到達。未解決の打ち切りを引き分けへ置き換えていません。</p></div>'
    if audit:
        assert audit['status'] == 'pass' and audit['summary_sha256'] == hashlib.sha256((main/'summary.json').read_bytes()).hexdigest()
        visit = summary['cells']['direct-visit-sample']
        attack = summary['cells']['oracle-visit-sample']
        legacy = summary['cells']['oracle-legacy']
        direct_effect = summary['primary']['visit-sample']['direct_score_minus_half']
        resistance = summary['primary']['visit-sample']['oracle_score_minus_legacy']
        assert direct_effect['interval'][1] < 0 and resistance['interval'][0] <= 0 <= resistance['interval'][1]
        body += f'''<h2>結論：今回の2方式では改善を確認できなかった</h2>
<p><strong>訪問割合で抽選する方式は、直接対決で悪化しました。</strong> {visit['wins']}勝{visit['losses']}敗、勝率{pct(visit['pure_win']['estimate'])}。50%からの差は{interval(direct_effect,True)}（98.75%区間）です。</p>
<p><strong>exploiterへの耐性の改善は未確認です。</strong> 訪問割合方式は{attack['wins']}/{attack['games']}勝（{pct(attack['pure_win']['estimate'])}）、従来方式は{legacy['wins']}/{legacy['games']}勝（{pct(legacy['pure_win']['estimate'])}）。対応差は{interval(resistance,True)}（98.75%区間）で、改善とも悪化とも判定できません。</p>
<p>攻撃側の有効性も確認しました。対応する32組・64局では、方策を読む攻撃は、読まない対照より従来方式の勝率を18.75ポイント低下させました（補助比較の95%区間：3.13〜34.38ポイント）。ただし、訪問割合方式で攻撃効果が小さかったという点推定だけから、耐性が改善したとは結論できません。</p>
<p><strong>混合推定器は、実際の選択を変えませんでした。</strong> 直接対決は128勝128敗、exploiter戦は38勝90敗。対象botの11,067決定で混合を返したのは3回だけで、抽選結果はいずれも従来手と一致しました。残り11,064決定は従来方式へのフォールバックです。</p>
<p>全1,096局で引き分けは0でした。この測定では純勝率とスコアが一致します。既定方策の変更を支持する結果ではありません。</p>
<h2>なぜこうなるか／どこまで分かったか</h2>
<p>訪問割合は探索中に各手へ割り当てた回数です。その割合を抽選に使うだけでは、相手の応手を考えた均衡の実行確率にはなりません。今回、選出は172/256決定、対戦中は1,500/6,373決定で従来手から変わりました。この変更全体が直接対決を悪化させたことは測定できましたが、選出と対戦中の変更を同時に行っているため、悪化の寄与は分離できません。</p>
<p>混合推定器には別の制約があります。現行実装は選出では混合推定を使わず、対戦中も「混合の推定下限−最良純粋手の推定上限」が0.02を超える場合だけ採用します。対戦中の混合出力は直接対決で3/6,347決定、exploiter戦で0/2,739決定でした。棄却理由の内訳は記録していないため、証拠不足と推定利得不足の寄与は未確定です。この閾値も、非定常な探索継続値に基づくためゲーム全体の改善保証ではありません。</p>
<p>直接対決の混合推定器は全128組で左右交換した行動履歴が一致し、exploiter戦128局と対照64局でも従来方式と行動履歴が一致しました。差が0のbootstrap区間は、この標本で対応差がすべて0だった結果です。母集団での同等性や、十分に発動する混合方策の無効性を証明しません。</p>
<p>一般化できる教訓は、探索に割く割合・実際に使う確率分布・その先の継続方策を分けて評価する必要があることです。今回の結果は「同時手番でも混合は不要」という結論ではなく、訪問割合の転用と、ほとんど発動しない推定器だけでは改善を確認できなかったという結果です。</p>
<p><a href="final-audit.json">全件監査・元ファイル照合・行動履歴の一致検証</a></p>'''

body += '''<h2>比較条件</h2><p>現在の24チームを実際のdrawWeightで抽選し、同じチーム・対戦seedで左右を交換します。各方式は選出を含む全決定で81,000反復、UCB係数0.4。事前計算済み選出表を使わないライブ探索経路です。</p>
<table><tr><th>方式</th><th>実行する方策</th></tr><tr><td>従来方式</td><td>最多訪問の一手</td></tr><tr><td>訪問割合で抽選</td><td>探索の訪問割合を確率として保持</td></tr><tr><td>混合推定器</td><td>既存の組合せ評価から混合を求め、根拠不足なら従来方式へ戻る</td></tr></table>
<p>直接対決は各候補256局。exploiter戦は従来方式も含めて各128局。方策を読まない対照は各64局。従来方式同士の左右交換テスト8局を加え、計1,096局です。予備測定18局は本測定に混ぜません。</p>
<h2>exploiterの意味</h2><p>対象方策を、実際の対象botとは異なる3つのseedで推定し、現在の同時手番での応手を探索します。対象botが実際に抽選した手・内部seedは渡しません。対象botはblindですが、攻撃側には真の盤面を与える強いストレステストです。</p>
<p>攻撃側の探索予算は243,000反復。対照も同じ盤面情報・同じ探索予算を持ち、方策の推定だけを行いません。exploiterには追加で対象モデル3回分の計算があるため、総実行時間を揃えた比較ではありません。これは近似ベストレスポンスへの耐性であり、厳密な最悪応手やNash exploitabilityの証明ではありません。</p>
<h2>判定方法</h2><p>純勝率（勝ち数÷全対局数）と、引き分けを0.5とするスコアを分けます。左右交換した2局を一つの標本として、100,000回のブロックbootstrapを行います。主要4比較は各98.75%のbootstrap区間を使います（Bonferroniによる名目上の全体水準95%）。有限標本で厳密な被覆率を保証する区間ではありません。その他の95%区間は補助的な記述です。0をまたぐ差は「改善を確認できない」であり、同等性の証明ではありません。</p><p><a href="plan.json">実行前に固定した全条件・試合予定</a></p>'''
if parity:
    body += f'<p>実際のコンパイル済みWASMと評価用ネイティブbotは、{parity["decisions"]}決定で、選択手・分布・訪問数・平均値が一致しました。<a href="wasm-parity.json">一致検証</a></p>'
if pilot:
    assert pilot['pilot'] and pilot['games'] == 18 and pilot['all_terminal'] and pilot['null_control'] == 'pass'
    body += '<p>予備測定18局は完了し、全局の終局、探索予算、対象と攻撃側モデルのseed分離、左右交換の整合性が監査を通りました。予備測定は本測定の推定には含めません。</p>'
if canary:
    controls = sum(r['correct'] for r in canary['rows'] if not r['oracle'])
    body += f'<p>既知のカウンター局面で、方策を読むexploiterは正しい応手を{canary["oracle_correct"]}/{canary["oracle_trials"]}回選択し、読まない対照は{controls}/8回でした。方策の違いに応答できることを確認する正の対照であり、本測定の勝率推定には含めません。<a href="exploiter-canary.json">検証データ</a></p>'

if summary:
    body += '<h2>勝敗と純勝率</h2><div class="scroll"><table><tr><th>相手</th><th>対象方式</th><th>勝–分–敗</th><th>純勝率［95%区間］</th><th>スコア［95%区間］</th></tr>'
    for kind in ['direct', 'oracle', 'control']:
        for mode in ['legacy', 'visit-sample', 'estimated-mix']:
            if kind == 'direct' and mode == 'legacy': continue
            c = summary['cells'][f'{kind}-{mode}']
            body += f'<tr><td>{kind_name[kind]}</td><td>{mode_name[mode]}</td><td>{c["wins"]}–{c["draws"]}–{c["losses"]}<br><small>制限引分 {c["turn_limit_draws"]}</small></td><td>{interval(c["pure_win"])}</td><td>{interval(c["score"])}</td></tr>'
    body += '</table></div><p>「制限引分」はエンジンの1000ターン制限による引き分けで、分の内数です。勝率の改善と、引き分けを増やすことによるスコアの改善は区別します。</p><h2>主要な差：正なら改善</h2><table><tr><th>候補</th><th>直接対決：50%からの差［98.75%］</th><th>exploiter戦：従来方式からの差［98.75%］</th></tr>'
    for mode, r in summary['primary'].items():
        body += f'<tr><td>{mode_name[mode]}</td><td>{interval(r["direct_score_minus_half"],True)}</td><td>{interval(r["oracle_score_minus_legacy"],True)}</td></tr>'
    body += '</table><p>上表はスコア差です。混合推定器の0幅の区間は、全標本の対応差が0だったためです。母集団の同等性を保証しません。</p><h2>純勝率の差（補助比較）</h2><table><tr><th>候補</th><th>直接対決：候補の勝率−相手の勝率［95%］</th><th>exploiter戦：従来方式からの勝率差［95%］</th></tr>'
    for mode in ['visit-sample', 'estimated-mix']:
        direct_win = summary['secondary'][f'pure-win-direct-{mode}-minus-opponent']
        oracle_win = summary['secondary'][f'pure-win-oracle-{mode}-minus-legacy']
        body += f'<tr><td>{mode_name[mode]}</td><td>{interval(direct_win,True)}</td><td>{interval(oracle_win,True)}</td></tr>'
    body += '</table><p>直接対決では相手の勝ちが候補の負けなので、勝率差は（勝ち数−負け数）÷全対局数です。引き分けを除外した勝率ではありません。</p><h2>exploiterは実際に攻略できたか</h2><table><tr><th>対象方式</th><th>方策を読む攻撃による対象スコアの低下［95%］</th></tr>'
    for mode in ['legacy', 'visit-sample', 'estimated-mix']:
        effect = summary['secondary'][f'oracle-loss-vs-control-{mode}']
        body += f'<tr><td>{mode_name[mode]}</td><td>{interval(effect,True)}</td></tr>'
    body += '</table><p>同じ32組・64局の条件で、方策を読まない攻撃へのスコアから、読む攻撃へのスコアを引いた値です。正なら方策を読む攻撃の方が有効でした。0をまたぐ場合、攻撃側の検出力にも限界があり、耐性の証明にはなりません。補助比較は多重比較補正をしていません。</p><h2>混合は実際に使われたか</h2><table><tr><th>直接対決の対象方式</th><th>選出：混合決定／全決定</th><th>対戦中：混合決定／全決定</th><th>対戦中：従来手と異なる選択</th></tr>'
    for mode in ['visit-sample','estimated-mix']:
        stats = summary['cells'][f'direct-{mode}']['stats']
        p, b = stats['preview'], stats['battle']
        body += f'<tr><td>{mode_name[mode]}</td><td>{p["mixed"]}/{p["decisions"]}</td><td>{b["mixed"]}/{b["decisions"]}</td><td>{b["differs_from_legacy"]}</td></tr>'
    body += '</table><p><a href="summary.json">全結果・信頼区間・検証済み集計</a></p>'
body += '<p class="muted">この測定による既定方策の切り替えや本番公開は行いません。混合分布を返して抽選するAPIは実装済みです。</p>'
style = 'body{margin:0;background:#f6f3eb;color:#363c34;font:16px/1.85 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:24px}h1{font-size:1.7rem;line-height:1.5}h2{font-size:1.2rem;margin-top:2rem}a{color:#526b4c}.box{padding:18px;border:1px solid #c6cbbb;border-radius:8px}.muted{color:#62665e;font-size:.9rem}table{border-collapse:collapse;width:100%;font-size:.9rem}td,th{padding:10px;border-bottom:1px solid #d6d3c7;text-align:left;overflow-wrap:anywhere}th{background:#e9ebdf}.scroll{overflow-x:auto}.scroll table{min-width:800px}*{box-sizing:border-box}'
page = f'<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>混合方策の勝率とexploiter耐性</title><style>{style}</style><main>{body}</main></html>'
(root / 'index.html').write_text(page)
dest = Path('/tmp/nc2000-mixed-policy-strength-preview')
dest.mkdir(exist_ok=True)
shutil.copy2(root / 'index.html', dest / 'index.html')
shutil.copy2(main / 'plan.json', dest / 'plan.json')
for src in [root/'exploiter-canary.json', root/'wasm-parity.json', main/'summary.json', audit_path]:
    if src.exists(): shutil.copy2(src, dest/src.name)
print(json.dumps({'completed':completed, 'planned':len(plan['jobs']), 'final':summary is not None, 'previewDirectory':str(dest)}))
