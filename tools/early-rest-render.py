import hashlib
import html
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
s = json.loads((out / 'summary.json').read_text())
rows = ''.join(f"<tr><td>{html.escape(c['case'])}</td><td>{c['pairs']}</td><td>{c['scores'][0]:g}</td><td>{c['scores'][1]:g}</td><td>{c['changed_pairs']}</td></tr>" for c in s['cases'])
primary = s['posterior'].get('1')
probability = f"{primary['probability_improved']:.1%}" if primary else '未算出（未決着あり）'
a,b = s['stress']['scores']
(out / 'index.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>未公開のねむるを想定する予備試験</title>
<style>body{{font:17px/1.8 system-ui;background:#f7f5ef;color:#292820;max-width:1050px;margin:30px auto;padding:0 20px}}h1{{font-size:1.6em}}h2{{font-size:1.2em}}table{{border-collapse:collapse;width:100%;font-size:.9em}}td,th{{padding:9px;border-bottom:1px solid #ccc6b6;text-align:left}}.scroll{{overflow-x:auto}}a{{color:#4e643e}}</style>
<h1>未公開のねむる：予備試験ではまだ採用できない</h1>
<p>人間の対局で多かった回復技の見落としを受け、未公開の「ねむる」だけを候補へ加える試験版を作りました。既存の事前分布から出現頻度を取り、公開済みの技・残りPP・個体値・能力値は保持。既存のほろび対策も維持します。通常版の既定設定は変更していません。</p>
<p>4種類の対戦状況で、相手がねむるを持つ／持たない構成を同数用意。各4組の共通乱数による比較では、通常版{a:g}勝／32局、試験版{b:g}勝／32局でした。同じ乱数の組で改善{s['stress']['better']}・悪化{s['stress']['worse']}・同点{s['stress']['same']}です。別途2組の既知編成対照は手順まで一致しました。</p>
<div class="scroll"><table><tr><th>対戦状況</th><th>組</th><th>通常版の得点</th><th>試験版の得点</th><th>着手変更</th></tr>{rows}</table></div>
<p>ねむる有りの相手では6勝対6勝、無しでは6勝対7勝。狙った回復対策の効果はまだ確認できません。固定した混合条件と弱い事前分布による改善確率は{probability}で、以前採用した80%基準には届きません。これは人間に勝てる確率でも、一般的な強さの改善確率でもありません。</p>
<h2>診断と検証</h2><p>初めて使われたねむるを64標本のどれかに含めた件数は123／436から332／436へ増加しました。ただし全ての初公開技で平均した1標本あたりの包含率は52.63%から52.42%へ微減。候補を増やすだけでは強さを示せません。</p>
<p>68局・{s['replay']['decisions']}判断の合法手、局面、乱数、結果を再生照合。最初に着手が分かれた{s['reproduced_first_divergence_searches']}探索も最終コードで再現しました。単体・局面テスト162件が通過し、通常版の人間対局6,701件の推定は元の結果と完全一致しています。</p>
<p>予備試験後、設定を変えず独立した乱数で64組の確認試験を固定しました。採否は確認試験単独で判定し、予備試験を足して基準を通すことはしません。</p>
<p><a href="summary.json">全集計・事前分布への感度</a> · <a href="protocol.json">対戦条件</a> · <a href="analysis-plan.json">集計計画</a> · <a href="first-divergence.json">着手分岐の再検証</a> · <a href="records.json.gz">対戦・推定記録</a> · <a href="verification.json">検証記録</a></p></html>''')
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='verification.json'}
source_paths=['crates/bot/src/belief.rs','crates/bot/examples/early_rest_duel.rs','crates/bot/examples/next_move_support.rs']
(out / 'verification.json').write_text(json.dumps({'tests':{'bot_unit':152,'position':10,'workspace_all_targets':'passed'},'default_model_rows_reproduced':6701,'first_divergence_searches_reproduced':s['reproduced_first_divergence_searches'],'final_source_sha256':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in source_paths},'artifact_sha256':hashes},indent=2)+'\n')
