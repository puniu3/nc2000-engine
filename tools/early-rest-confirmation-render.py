import hashlib
import html
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
s = json.loads((out / 'summary.json').read_text())
d = json.loads((out / 'decision.json').read_text())
labels = {'snorlax-umbreon':'カビゴン / ブラッキー','snorlax-marowak':'カビゴン / ガラガラ','skarmory-marowak':'エアームド / ガラガラ','zapdos-snorlax':'サンダー / カビゴン','onpool-control':'既知編成の対照'}
rows = ''
for c in s['cases']:
    family = c['case'].removesuffix('-no-rest').removesuffix('-rest')
    label = labels.get(family,c['case'])
    rest = '有り' if c['actual_rest'] is True else '無し' if c['actual_rest'] is False else '—'
    rows += f"<tr><td>{html.escape(label)}</td><td>{rest}</td><td>{c['pairs']}</td><td>{c['scores'][0]:g}</td><td>{c['scores'][1]:g}</td></tr>"
primary = s['posterior'].get('1')
probability = f"{primary['probability_improved']:.1%}" if primary else '未算出'
stress = s['stress']
adoption = '<p>ローカルのソース設定へ採用しました。関連170テストと全ターゲットのコンパイルが通過。公開中の版は変更していません。<a href="adoption-verification.json">採用時の検証</a></p>' if d['stage'] == 'adopted_locally' else ''
if d['stage'] == 'adopted_locally':
    title = '独立追試の基準を通過：ローカル版に採用'
elif d['criterion_met']:
    title = '独立追試の基準を通過：採用前の確認中'
else:
    title = '独立追試の基準に届かず：採用見送り'
sensitivity = ''.join(f"<tr><td>{mass}</td><td>{v['probability_improved']:.1%}</td><td>{v['mean_delta']*100:+.2f}ポイント</td></tr>" for mass,v in s['posterior'].items() if isinstance(v,dict) and 'probability_improved' in v)
(out / 'index.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ねむる推定・独立確認試験</title>
<style>body{{font:17px/1.8 system-ui;background:#f7f5ef;color:#292820;max-width:1050px;margin:30px auto;padding:0 20px}}h1{{font-size:1.6em}}h2{{font-size:1.2em}}table{{border-collapse:collapse;width:100%;font-size:.9em}}td,th{{padding:9px;border-bottom:1px solid #ccc6b6;text-align:left}}.scroll{{overflow-x:auto}}a{{color:#4e643e}}</style>
<h1>{title}</h1>{adoption}<p>未公開のねむるを想定する試験版を、新しい乱数による64組の対戦で評価しました。通常版の得点は{stress['scores'][0]:g}、試験版は{stress['scores'][1]:g}。同じ乱数の組で改善{stress['better']}・悪化{stress['worse']}・同点{stress['same']}です。別途2組の既知編成対照も確認しました。</p>
<p>試験前に固定した集計モデルによる改善確率は<strong>{probability}</strong>。判定基準は80%です。予備試験の12勝対13勝は今回の判定に足していません。試験中の設定変更や、結果に応じた試行数の追加もありません。</p>
<div class="scroll"><table><tr><th>相手 / こちらの先頭</th><th>相手のねむる</th><th>組</th><th>通常版</th><th>試験版</th></tr>{rows}</table></div>
<p>ねむる有りでは{s['groups']['rest']['scores'][0]:g}対{s['groups']['rest']['scores'][1]:g}、無しでは{s['groups']['no_rest']['scores'][0]:g}対{s['groups']['no_rest']['scores'][1]:g}。上限到達は通常版{stress['caps_by_arm'][0]}局、試験版{stress['caps_by_arm'][1]}局で、未決着は引き分け扱いにせず別記しています。</p>
<h2>改善確率の意味と事前分布への感度</h2><p>8つの固定対戦条件を同じ重みで平均し、各条件の対応する勝敗に対して対称ディリクレ分布を使います。主判定の事前質量は各条件1。この固定した相手・条件に関する推定で、人間上級者への勝率や一般的な改善確率を表すものではありません。</p>
<table><tr><th>各条件の事前質量</th><th>改善確率</th><th>平均差</th></tr>{sensitivity}</table>
<h2>検証</h2><p>{s['replay']['games']}局・{s['replay']['decisions']}判断を再生し、合法手・局面・乱数・結果を照合しました。最初に着手が分かれた{s['reproduced_first_divergence_searches']}探索も同じコードで再現。比較元EarlyPerishの人間対局6,701件の推定が保持されていることも確認しています。</p>
<p><a href="decision.json">採否</a> · <a href="summary.json">全集計</a> · <a href="protocol.json">固定条件</a> · <a href="analysis-plan.json">判定計画</a> · <a href="first-divergence.json">着手分岐の照合</a> · <a href="records.json.gz">対戦・推定記録</a> · <a href="verification.json">検証記録</a></p></html>''')
hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='verification.json'}
(out / 'verification.json').write_text(json.dumps({'replay':s['replay'],'first_divergence_searches_reproduced':s['reproduced_first_divergence_searches'],'default_model_rows_reproduced':s['default_corpus_reproduced_labels'],'artifact_sha256':hashes},indent=2)+'\n')
