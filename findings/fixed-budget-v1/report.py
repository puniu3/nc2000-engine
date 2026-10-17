"""Render the completed study as a self-contained, shareable HTML report."""
import html
from collections import Counter
import json
from pathlib import Path

P=Path(__file__).resolve().parent
repo=P.parent.parent
read=lambda p:json.loads(p.read_text())
pct=lambda x:f'{100*x:.1f}%'
pp=lambda x:f'{100*x:+.1f}pt'
s=read(P/'strength-result.json');u=read(P/'unknown-result.json')
changes=Counter((r['low_choice'].split()[0],r['high_choice'].split()[0]) for r in s['first_decision_changes'])
config=read(repo/'data/live-search.json');artifact=read(repo/'data/compiled-preview-v1.json')
coverage=read(P/'compiled-coverage.json');validation=read(P/'validation.json')
case=read(P/'selection-case-result.json');case_low=case['scores']['81k'];case_high=case['scores']['243k_and_above']
assert (243000 in config['battle'].values()) == s['add243'], 'Tier decision differs from registered strength result'
assert u['selected_budget'] == config['unknownPreview'], 'Unknown preview saturation/configuration still unresolved'
submitted=[f.parent for f in (P/'cloud').glob('*/execution.log') if '"phase": "submitted"' in f.read_text()]
assert all((f/'receipt.json').is_file() or (f/'cancellation-audit.json').is_file() for f in submitted), 'Cloud job still active or uncollected'
receipt_paths=[next((f for f in [folder/'receipt.json',folder/'cancellation-audit.json'] if f.exists()),None) for folder in sorted((P/'cloud').glob('*-run'))]
all_receipts=[read(f) for f in receipt_paths if f is not None]
receipts=[r for r in all_receipts if r.get('allowance_yen') is not None]
assert all(r.get('cleanup')=='not_started' and r.get('returncode') is None for r in all_receipts if r not in receipts), 'Unclassified preflight failure'
assert all(r.get('cleanup')=='verified' for r in receipts),'Cloud cleanup not complete'
assert s['checks']==dict(complete=True,no_caps=True,budgets=True,same_preview=True)
assert coverage['allLegal'] and coverage['uniqueKeys']==504
cost=sum(r.get('estimated_usage_yen',0) for r in receipts)
strength_verdict='243kを最上位の選択肢に追加' if s['add243'] else '243kの明確な優位は確認できず、選択肢への追加は見送り'
def unknown_row(r):
    guard_upper=max(g['upper_simultaneous'] for g in u['nonmonotonic_guard'] if g['low']==r['budget'])
    accepted=r['passes'] and guard_upper<=.03
    label='採用' if r['budget']==u['selected_budget'] else '適合' if accepted else '未確認'
    return f'<tr><td>{r["budget"]:,}</td><td>{pp(r["reference_gain"])}</td><td>{pp(r["ci95"][0])}〜{pp(r["ci95"][1])}</td><td>{pp(r["upper_simultaneous"])}</td><td>{pp(guard_upper)}</td><td>{label}</td></tr>'
unknown_rows=''.join(unknown_row(r) for r in u['results'])
timings=read(P/'browser-timing.json')
timing_rows=''.join(f'<tr><td>{b:,}</td><td>{min(r["ms"] for r in timings if r["budget"]==b)/1000:.1f}〜{max(r["ms"] for r in timings if r["budget"]==b)/1000:.1f}秒</td><td>{max(r["memoryBytes"] for r in timings if r["budget"]==b)/1048576:.0f} MiB</td></tr>' for b in sorted({r['budget'] for r in timings}))
battle_timings=read(P/'browser-battle-timing.json')
battle_timing_rows=''.join(f'<tr><td>{b:,}</td><td>{min(r["ms"] for r in battle_timings if r["budget"]==b)/1000:.1f}〜{max(r["ms"] for r in battle_timings if r["budget"]==b)/1000:.1f}秒</td><td>{max(r["memoryBytes"] for r in battle_timings if r["budget"]==b)/1048576:.0f} MiB</td></tr>' for b in sorted({r['budget'] for r in battle_timings}))
links=['strength-protocol.json','strength-analysis.json','strength-result.json','unknown-analysis.json','unknown-nonmonotonic-guard.json','unknown-result.json','validation.json','compiled-coverage.json','browser-timing.json','browser-battle-timing.json','selection-case-result.json','shared-preview-check.json']
link_html=' · '.join(f'<a href="{f}">{f}</a>' for f in links)
body=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NC2000 固定探索予算の調査</title>
<style>body{{background:#f6f3ec;color:#282b27;font:16px/1.8 system-ui,sans-serif;max-width:980px;margin:auto;padding:26px}}h1{{font-size:1.7rem}}h2{{font-size:1.25rem;margin-top:2rem}}a{{color:#385c42}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #c9c9bd}}.scroll{{overflow:auto}}.verdict{{border-left:4px solid #536c4c;padding:12px 18px;background:#eaece2}}small{{color:#565a52}}code{{overflow-wrap:anywhere}}</style>
<h1>NC2000 固定探索予算の調査</h1>
<p class="verdict">{strength_verdict}。通常対戦の探索予算は <strong>{'／'.join(f'{n:,}' for n in config['battle'].values())}</strong> 回。入力する速さで探索回数は変わりません。</p>
<h2>243k 対 81k：対戦中の探索回数だけを変更</h2>
<p>既知カタログ24パーティの既定の抽選重みで128条件を生成し、両サイドで対戦した256試合。243k側は<strong>{s['wins']}勝・{s['ties']}分・{s['losses']}敗</strong>、引き分けを0.5とした得点率は<strong>{pct(s['score'])}</strong>。条件単位の95%ブートストラップ区間は<strong>{pct(s['ci95'][0])}〜{pct(s['ci95'][1])}</strong>でした。</p>
<p>選出は両者243k、探索係数0.4、対戦相手の推定に使う事前分布、パーティ、乱数の割当を固定。ponderによる追加探索、選出表、ソルバは使っていません。サイドを入れ替えても各席の選出が同じであることと、全判断の探索回数を検査しました。試合の途中打ち切りはありません。</p>
<p>同じ盤面から判断が分かれる最初の地点を照合すると、128条件中{len(s["first_decision_changes"])}条件で選択が変わりました。81k→243kの方向で、技の変更{changes[("move","move")]}件、技から交代{changes[("move","switch")]}件、交代から技{changes[("switch","move")]}件、交代先の変更{changes[("switch","switch")]}件です。判断が変わること自体は、強さの改善を意味しません。個々の一手の因果効果は、この比較だけでは分離していません。</p>
<p>採用条件は計測前に「95%区間の下限が50%を上回る」と固定しました。追加を見送る場合も、243kと81kが同じ強さだと証明したわけではありません。</p>
<small>ネイティブ実行で探索アルゴリズムを比較。ブラウザ版は各判断の探索用乱数の渡し方が異なり、完成版では既知パーティの選出表も使うため、製品全体の勝率をそのまま表す実験ではありません。</small>
<h2>未知パーティの選出</h2>
<p>事前分布に一致しない合法なカスタムパーティ16組、各2探索シード。81k／243k／729k／2187kの選出を、同じ相手の選出・乱数・対戦中81kで各16回継続評価しました。選出が同じ条件は結果の差が厳密にゼロなので、試合を再計算していません。</p>
<div class="scroll"><table><tr><th>選出の探索回数</th><th>2187k側の得点差</th><th>差の95%区間</th><th>参照との補正上限</th><th>全上位との補正上限</th><th>判定</th></tr>{unknown_rows}</table></div>
<p>16パーティを独立単位として集計。参照2187kの上積みが3ポイント以内という基準を、3候補に対する補正を含めて検査します。さらに中間の回数も含め、より多い探索回数の上積みがすべて3ポイント以内となることを6対比較の補正付きで確認し、両基準を満たす最小の回数を選びます。採用設定は<strong>{config['unknownPreview']:,}回</strong>。これは試した参照値とパーティ生成方法に対する飽和判定で、未知のあらゆるパーティに対する保証ではありません。</p>
<h2>選出が変わる具体例</h2>
<p>結果を見る前に例示対象として固定した1条件では、81kの「エアームド・スイクン・カビゴン」が、243k以上では「ブラッキー・スイクン・カビゴン」に変わりました。同じ相手・乱数割当で16回ずつ継続すると、前者は{case_low['wins']}勝{case_low['ties']}分{case_low['losses']}敗、後者は{case_high['wins']}勝{case_high['ties']}分{case_high['losses']}敗、得点差は{pp(case['high_minus_low'])}でした。選出以降の探索は両者81kです。この1条件は判断の変化を説明する例で、一般的な強さの改善を示す独立した検定ではありません。</p>
<h2>既知パーティの事前計算</h2>
<p>カタログ24パーティ×公開ロスター21種類＝<strong>{artifact['matchups']}組</strong>を事前計算。各組を{artifact['iterations']:,}回×{len(artifact['sampleSeeds'])}探索シードで求め、選出をアプリに組み込みます。実行時は探索なしで選出できます。</p>
<p>照合に使うのは自分の詳細な構成と相手の公開種族・レベル・並び順で照合し、公開された性別・持ち物の有無の一致も確認します。相手の非公開の技や持ち物はキーに含めません。完全解や厳密な均衡ではなく、十分な探索を事前に実行した近似です。独自の事前分布、未対応パーティ、互換性のない入力ではライブ探索に戻ります。古い表や未完成の表では製品ビルドを止めます。</p>
<h2>待ち時間と検証</h2>
<div class="scroll"><table><tr><th>未知選出の回数</th><th>4局面の実測時間</th><th>最大WASMメモリ</th></tr>{timing_rows}</table></div>
<p>対戦中の固定した4局面でも計測しました。</p><div class="scroll"><table><tr><th>対戦中の回数</th><th>実測時間</th><th>最大WASMメモリ</th></tr>{battle_timing_rows}</table></div>
<p>計測は他の調査計算も動いているMacのChromium。iPad Safariの実測ではありません。プレイヤーが選択している間にも思考を進め、選択時点で未完了なら残りだけ待ちます。</p>
<p>早い入力／遅い入力／思考完了後の入力で判断が一致すること、複数ターンの乱数の継続、非公開情報を変えても選出表の照合が変わらないことを確認。実際のWASMでカタログ全576組の照合と合法性を確認しました。検証記録は下記から取得できます。</p>
<h2>Cloud利用</h2><p>計算ジョブ{len(receipts)}件、回収された利用見積もりの合計は約<strong>{cost:.0f}円</strong>（クレジット適用前・請求額ではありません）。時間枠を超えたジョブの完了分も回収し、未完了分だけを継続しました。担当条件を他の実行で完了した枠は停止し、その費用は記録された時間と単価から保守的に見積もっています。全ジョブのVM・ディスク・転送用オブジェクトの削除を確認済みです。</p>
<h2>フォーク向けインターフェース</h2><p><a href="preview-cache.html">選出キャッシュの生成・読み込みAPI</a>。保存と利用者間共有はフォーク側の責務です。</p>
<h2>再現用データ</h2><p><a href="../nc2000-budget/index.html">選出への予算配分の予備調査</a> · <a href="../nc2000-strength/index.html">27k→81kの強さと判断改善の調査</a></p><p>{link_html}</p><p><a href="evidence.zip">対戦ログ・計測条件・解析コード一式</a> · <a href="../nc2000-fixed/">対戦プレビュー</a></p>
</html>'''
(P/'index.html').write_text(body)
print(P/'index.html')
