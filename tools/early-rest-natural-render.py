import hashlib
import html
import json
import pathlib

out = pathlib.Path('findings/early-rest-natural-v1')
read = lambda name: json.loads((out / name).read_text())
inventory = read('inventory-audit.json')['rows']
unmatched = [r for r in inventory if r['fallback_at_preview']]
protocol = read('protocol.json')
count = sum(c['pairs'] for c in protocol['cases']) * 2
result = '<p>固定した52局の試験を実行中です。結果の判定と再生照合は完了後に行います。</p>'
links = ['protocol.json', 'analysis-plan.json', 'inventory-audit.json', 'source-sha256.json']
if (out / 'summary.json').exists():
    summary = read('summary.json')
    decision = read('decision.json')
    screen = summary['screen']
    rows = ''.join(f"<tr><td>{html.escape(c['case'])}</td><td>{c['pairs']}</td><td>{c['scores'][0]:g}</td><td>{c['scores'][1]:g}</td></tr>" for c in summary['cases'])
    p = decision['probability_worse']
    probability = f'{p:.2%}' if p is not None else '未算出（未決着あり）'
    result = f'''<p>旧設定{screen['scores'][0]:g}点、新設定{screen['scores'][1]:g}点／各24局。同じ乱数の組で改善{screen['better']}・悪化{screen['worse']}・同点{screen['same']}。未決着は旧設定{screen['caps_by_arm'][0]}局、新設定{screen['caps_by_arm'][1]}局です。</p>
<p>固定モデルでの悪化確率は{probability}。事前の取り消し基準は80%です。判定状態：{html.escape(decision['stage'])}。この小規模試験だけでは一般的な非劣性や人間上級者への優位は証明できません。</p>
<div class="scroll"><table><tr><th>相手／こちら</th><th>組</th><th>旧設定</th><th>新設定</th></tr>{rows}</table></div>
<p>{summary['replay']['games']}局・{summary['replay']['decisions']}判断を再生照合し、着手が分かれた{summary['reproduced_searches']}探索を再現しました。</p>'''
    links += ['summary.json', 'decision.json', 'replay.json', 'first-divergence.json', 'records.json.gz']
links_html = ' · '.join(f'<a href="{name}">{name}</a>' for name in links)
unmatched_html = '、'.join(html.escape(r['team']) for r in unmatched)
(out / 'index.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ねむる推定・通常対戦の追加検証</title>
<style>body{{font:17px/1.8 system-ui;background:#f7f5ef;color:#292820;max-width:1050px;margin:30px auto;padding:0 20px}}h1{{font-size:1.6em}}h2{{font-size:1.2em}}table{{border-collapse:collapse;width:100%;font-size:.9em}}td,th{{padding:9px;border-bottom:1px solid #ccc6b6;text-align:left}}.scroll{{overflow-x:auto}}a{{color:#4e643e}}</style>
<h1>選出・初手を強制しない追加検証</h1>{result}
<h2>既存チームの重複監査</h2>
<p>合法な人間作成の既存チーム{len(inventory)}件を、現在の推定処理で照合しました。{len(inventory)-len(unmatched)}件は見せ合い時点で候補があり、全技・持ち物にも整合する候補が残ります。候補が無い{len(unmatched)}件は{unmatched_html}です。</p>
<p>3件とも技・持ち物は既知編成と重複しますが、性別が異なるため編成全体の照合から外れます。元の編成を変更せずに未知編成の推定処理を試せます。ただし、未知の技構成や独立した人間対戦の検証にはなりません。</p>
<p>85件すべてで、見せ合い時の64標本は旧設定・新設定で一致しました。新設定は相手の登場後に作用するため、これだけで対戦中の不変性は保証しません。全技・持ち物の整合性は集合の照合であり、その公開順を実際の対戦で再現したものではありません。</p>
<h2>固定した条件</h2><p>相手3編成×こちら2編成×4組＝24組と、既知編成同士の対照2組、計{count}局。こちらの2編成はカタログの配布比重上位から元の並び順で選びました。旧設定EarlyPerishと新設定EarlyRestを、同じ相手EarlyPerish・同じ乱数で比較します。各判断27,000探索、c=0.4、上限300ターン。選出や初手の強制はありません。</p>
<p>6条件を等重みで集計し、各条件の対応勝敗に対する対称ディリクレ分布（主判定の事前質量1）で悪化確率80%以上なら暫定的なローカル採用を取り消します。事前質量0.5・4・16・64も併記。未決着は引き分けに置き換えず判定を未完了とします。従来の作成局面の試験と合算せず、結果による試行数の追加もしません。</p>
<p>{links_html}</p></html>''')
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name != 'verification.json'}
(out / 'verification.json').write_text(json.dumps({'artifact_sha256': hashes}, indent=2) + '\n')
