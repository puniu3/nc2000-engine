import html
import json
from pathlib import Path

p=Path('findings/operational-loadout-draw-v1')
r=json.loads((p/'results.json').read_text());assert r['complete']
f=json.loads((p/'fork-summary.json').read_text())
labels={'adaptive':'初手ロック後は自由行動','stress':'固定の滅び手順','natural':'自由選出・自由行動','onpool':'登録カタログ対照','same':'同一方策対照'}
pp=lambda v:f'{v*100:+.1f}'
table=[];detail=[]
for cell in ['adaptive','stress','natural','onpool','same']:
    x=r['results'][cell];ca=x['comparisons']['c-a'];cb=x['comparisons']['c-b']
    table.append(f'<tr><td>{labels[cell]}</td><td>{x["triplets"]}</td>'+''.join(f'<td>{x["arms"][a]["wins"]}勝 / 上限{x["arms"][a]["caps"]}</td>' for a in 'abc')+f'<td>{pp(ca["paired_delta"])}pt<br>95%区間 {pp(ca["stratified_bootstrap95"][0])}〜{pp(ca["stratified_bootstrap95"][1])}</td><td>{pp(cb["paired_delta"])}pt</td></tr>')
    if cell in ['adaptive','stress','natural']:
        detail.append(f'<p>{labels[cell]}：現行から新候補で勝敗が改善した組は{ca["better_pairs"]}、悪化した組は{ca["worse_pairs"]}。終局した勝敗不一致ペアの両側正確検定 p={ca["terminal_exact_two_sided_p"]:.3f}。'+
                      '相手のみちづれ使用試合数は、現行・単一型・分布型の順に'+ ' / '.join(str(x['arms'][a]['opponent_bond_games']) for a in 'abc')+'。</p>')
primary=r['results']['adaptive']['comparisons']['c-a']
verdict='主試験で現行に対する改善を示した。通常対戦と単一型候補との差も含めて採否を判断する。' if primary['stratified_bootstrap95'][0]>0 else '主試験で勝率改善を確認できなかった。候補は本番に採用しない。'
moves={'move icebeam':'れいとうビーム','move curse':'のろい','move earthquake':'じしん','move rockslide':'いわなだれ','move crunch':'かみくだく','move dynamicpunch':'ばくれつパンチ','move fireblast':'だいもんじ','move swordsdance':'つるぎのまい','move hydropump':'ハイドロポンプ','move psychic':'サイコキネシス','move encore':'アンコール','move hiddenpowerrock':'めざめるパワー岩','move crosschop':'クロスチョップ','move doubleedge':'すてみタックル'}
ft=[]
for job in sorted(f,key=int):
    ft.append('<tr><td>'+job+'</td>'+''.join('<td>'+ '、'.join(f'{moves.get(m,m)} {n}回' for m,n in f[job][a]['choices'].items())+'</td>' for a in ['baseline','conditioned','draw'])+f'<td><a href="forks/{job}.jsonl">記録</a></td></tr>')
av=f['19']['draw']['reply_available_fraction']
report=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NC2000 — 型の分布を残す推定</title>
<style>body{{font:18px/1.75 system-ui,sans-serif;background:#f4f0e6;color:#28271f;max-width:1080px;margin:32px auto;padding:0 20px}}h1,h2{{line-height:1.35}}h2{{margin-top:2em}}a{{color:#684928}}table{{border-collapse:collapse;width:100%;font-size:16px}}th,td{{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #c7c2b6}}.scroll{{overflow-x:auto}}.lead{{font-size:22px}}strong{{color:#3b4931}}code{{overflow-wrap:anywhere}}</style>
<p><a href="/nc2000-operational-combo-validation/">前回の勝敗検証</a>・<a href="/nc2000-operational-combo-validation/fork-analysis.html">分岐の原因調査</a></p>
<h1>型の分布を残す推定</h1><p class="lead">{verdict}</p><p>本番の既定方策は変更していない。以下の「現行」はLayered、「単一型」は前回のRevealedLoadout、「分布型」は今回のRevealedLoadoutDraw。</p>
<h2>実装した変更</h2><p>未登録パーティ相手に技または元の道具が判明したら、同種族の整合する完成済みの型を元データの重みで抽選する。技・道具・能力値を一緒に取り出すので、別々の型の技を混ぜない。公開された技・道具・PP・盤面情報を優先し、観測レベルで覚えられない技を含む型を除外する。</p>
<p>初見の種族、登録パーティ相手、整合する型がない場合は現行の経路を保つ。抽選候補は観測の更新時に構築して再利用し、探索の各反復で引き直す。種族間の道具や選出の相関は追加していない。</p>
<h2>終局までの比較</h2><div class="scroll"><table><tr><th>条件</th><th>組数</th><th>現行</th><th>単一型</th><th>分布型</th><th>分布型−現行</th><th>分布型−単一型</th></tr>{''.join(table)}</table></div>
{''.join(detail)}
<p>主試験は64組。ムウマ先発と最初の黒い眼差しだけを固定し、その後は同じ通常botがみちづれ・滅びの歌・交代などを自由に選ぶ。相手は自分の入力や未公開の型を知ることはできない。従来の固定手順32組、通常対戦32組を別条件として測った。</p>
<p>3方策で自分のパーティ、相手のパーティ、陣営、対戦乱数、探索乱数を対応させた。双方27,000反復、UCB c=0.4。同一方策の対照のみ3,000反復。相手は前回と同じ8つの合法な構成で、実際の非公開チームの復元ではない。</p>
<p>区間は相手構成ごとに層別した対応差のbootstrap。少数の勝敗不一致しかない場合は区間だけを過信せず、正確検定も併記した。上限は500ターンで、長い試合を除外していない。上限到達は表に分離し、集計JSONには最悪・最良の差の範囲も保存している。この試験は対人勝率の推定ではない。</p>
<h2>前回の11局面はどう変わったか</h2><p>過去の分岐局面を固定し、各8〜9探索乱数で比較した。事例19の探索内で滅びの歌を相手が使用可能だった割合は{av.get('move perishsong',0)*100:.1f}%、みちづれは{av.get('move destinybond',0)*100:.1f}%。歌の見落としを解消しつつ、みちづれの確定扱いは解消している。</p>
<p>ポリゴン2は攻撃を維持し、ゴローニャはじしんへ戻る。一方でバンギラスの待ち手とガラガラのつるぎのまいは残る。みちづれを約半分しか持たないと想定しても待ち手を選ぶ局面はあり、着手の見た目を改善目標にしてはいけない。</p>
<div class="scroll"><table><tr><th>事例</th><th>現行</th><th>単一型</th><th>分布型</th><th>根拠</th></tr>{''.join(ft)}</table></div>
<p>273回の再探索で、現行と単一型の元の着手・訪問数・評価平均が再現することも確認した。これらは前回の勝敗が変わった局面を選んだ原因調査であり、新しい独立した勝率データではない。</p>
<h2>検査と再現</h2><p>bot単体136件、belief統合6件、blind統合2件が通過。滅びの歌の保持、実際の型の組合せと抽選比率、追加観測・道具による絞り込み、未公開情報を書き換えた場合の独立性、未観測経路の乱数まで含む一致を検査した。全workspace・全targetの型検査も通過。</p>
<p>再測定は <code>tools/operational-loadout-draw</code>。前回結果を上書きせず、新しい作業ディレクトリに3方策の全試験を保存する。</p>
<p><a href="protocol.json">事前の条件</a>・<a href="results.json">全集計</a>・<a href="fork-summary.json">局面別集計</a>・<a href="adaptive.jsonl">初手ロック後自由行動</a>・<a href="stress.jsonl">固定手順</a>・<a href="natural.jsonl">通常対戦</a>・<a href="onpool.jsonl">登録対照</a>・<a href="same.jsonl">同一方策対照</a>・<a href="checks.json">検査記録</a></p></html>'''
(p/'index.html').write_text(report)
