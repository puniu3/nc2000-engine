import html
import json
from pathlib import Path

p = Path('findings/operational-retain-perish-v1')
r = json.loads((p / 'results.json').read_text())
assert r['complete']
forks = json.loads((p / 'fork-summary.json').read_text())
history = [json.loads(line) for line in (p / 'history.jsonl').read_text().splitlines()]
labels = {'adaptive': '初手ロック後は自由行動', 'stress': '固定の滅び手順', 'natural': '自由選出・自由行動',
          'onpool': '登録カタログ対照', 'same': '同一方策対照'}
pp = lambda x: f'{100 * x:+.1f}'
table = []
details = []
for cell in ['adaptive', 'stress', 'natural', 'onpool', 'same']:
    x = r['results'][cell]
    table.append(f'<tr><td>{labels[cell]}</td><td>{x["pairs"]}</td>' +
                 ''.join(f'<td>{x[a]["wins"]}勝<br>上限{x[a]["capped"]}</td>' for a in ['a', 'b']) +
                 f'<td>{pp(x["paired_delta"])}pt<br>95%区間 {pp(x["stratified_bootstrap95"][0])}〜{pp(x["stratified_bootstrap95"][1])}</td>' +
                 f'<td>{x["better_pairs"]} / {x["worse_pairs"]}</td><td>{x["terminal_exact_two_sided_p"]:.3f}</td></tr>')
    details.append(f'<tr><td>{labels[cell]}</td>' + ''.join(
        f'<td>{x[a]["exposed_games"]}試合 / {x[a]["exposed_decisions"]}判断<br>歌を落とした判断 {x[a]["missing_perish_decisions"]}</td>'
        for a in ['a', 'b']) + f'<td>{x["same_trajectory_pairs"]}/{x["pairs"]}</td></tr>')

primary = r['results']['adaptive']
verdict = ('主試験の事前基準を通過。副試験と合わせて採否を判断する。' if r['primary_passed'] else
           '滅びの歌の見落としは解消したが、勝率改善の事前基準を通過しなかった。本番採用は見送る。')
moves = {'move icebeam': 'れいとうビーム', 'move curse': 'のろい', 'move earthquake': 'じしん',
         'move rockslide': 'いわなだれ', 'move crunch': 'かみくだく', 'move fireblast': 'だいもんじ',
         'move swordsdance': 'つるぎのまい', 'move hydropump': 'ハイドロポンプ', 'move psychic': 'サイコキネシス',
         'move hiddenpowerrock': 'めざめるパワー岩'}
ft = []
for job, variants in forks.items():
    ft.append(f'<tr><td>{job}</td>' + ''.join('<td>' + '、'.join(
        f'{html.escape(moves.get(move, move))} {count}回' for move, count in variants[arm]['choices'].items()) + '</td>'
        for arm in ['baseline', 'retain-perish']) + f'<td><a href="forks/{job}.jsonl">記録</a></td></tr>')
changed = [x for x in history if x['changed_samples']]
searches = sum(v['searches'] for variants in forks.values() for v in variants.values())
natural = r['results']['natural']
caps = sum(x[a]['capped'] for x in r['results'].values() for a in ['a', 'b'])
ordinary = (f'通常戦は{natural["same_trajectory_pairs"]}/{natural["pairs"]}組で対戦履歴まで一致した。'
            f'対象局面の出現は現行{natural["a"]["exposed_games"]}試合、候補{natural["b"]["exposed_games"]}試合。'
            '対象が出なかった場合の一致は、影響範囲が狭いことの確認であり、実戦勝率の向上を示すものではない。')
report = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NC2000 — 滅びの歌だけを保持する補完</title>
<style>body{{font:18px/1.75 system-ui,sans-serif;background:#f4f0e6;color:#28271f;max-width:1040px;margin:32px auto;padding:0 20px}}h1,h2{{line-height:1.35}}h2{{margin-top:2em}}a{{color:#684928}}table{{border-collapse:collapse;width:100%;font-size:16px}}th,td{{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #c7c2b6}}.scroll{{overflow-x:auto}}.lead{{font-size:22px}}code{{overflow-wrap:anywhere}}</style>
<p><a href="/">プレビューハブ</a>・<a href="/nc2000-loadout-draw/">前回の型抽選試験</a>・<a href="/nc2000-operational-combo-validation/fork-analysis.html">分岐の原因調査</a></p>
<h1>滅びの歌だけを保持する補完</h1><p class="lead">{verdict}</p>
<p>本番の既定方策はLayeredのまま。候補RetainPerishを評価専用で実装し、新しい対戦乱数で140組・280戦を比較した。</p>
<h2>今回の変更範囲</h2><p>未登録パーティの補完で、黒い眼差しが公開済み・公開技が4つ未満・選ばれた補完元に合法な滅びの歌がある、という条件を満たし、通常の4技への切り詰めで歌が消える場合だけ、最後の未公開技を歌に置き換える。補完元の選択、道具、能力値、性別、残りの技順は現行を保つ。</p>
<div class="scroll"><table><tr><th>公開情報</th><th>現行の相手想定</th><th>候補の相手想定</th></tr><tr><td>黒い眼差し</td><td>黒い眼差し / どくどく / あやしいひかり / いたみわけ</td><td>黒い眼差し / どくどく / あやしいひかり / 滅びの歌</td></tr></table></div>
<p>道連れの推定を追加する変更や、完成した型全体を抽選する変更は含まない。4技が公開済みなら公開情報をそのまま優先する。</p>
<h2>終局までの比較</h2><div class="scroll"><table><tr><th>条件</th><th>組数</th><th>現行</th><th>限定修正</th><th>勝率差</th><th>改善 / 悪化</th><th>正確検定p</th></tr>{''.join(table)}</table></div>
<p>主試験は初手ロック後自由行動の64組。ムウマ先発と最初の黒い眼差しだけを固定し、その後の滅びの歌・道連れ・交代などは相手botが自由に選ぶ。相手の道連れは現行側{primary['a']['opponent_bond_games']}/64試合、候補側{primary['b']['opponent_bond_games']}/64試合で実際に使われた。</p>
<p>双方27,000探索反復、UCB c=0.4。同一方策対照のみ3,000反復。パーティ・陣営・対戦乱数・双方の探索乱数を対応させ、相手8構成ごとに層別した対応差のbootstrapを10,000回実行した。事前の主試験合格条件は95%区間の下限が0より大きく、勝敗不一致ペアの両側正確検定pが0.05未満。さらに副試験の未解消の退行がないことを求める。</p>
<p>500ターン上限の到達は{caps}戦。長い試合を除外していない。上限に達した場合は便宜上0.5点とし、最悪・最良の差の範囲も集計JSONへ保存する。相手は実戦で観測された2つの見せ合い構成と4つのムウマ型を組み合わせた合法な8構成で、実際の非公開チームの復元ではない。対人勝率への換算はできない。</p>
<h2>狙った見落としに届いたか</h2><div class="scroll"><table><tr><th>条件</th><th>現行の対象局面</th><th>候補の対象局面</th><th>履歴まで一致</th></tr>{''.join(details)}</table></div>
<p>対象局面は、相手の場のムウマが黒い眼差しを公開し、まだ滅びの歌を公開していない補完局面。両方策は途中から別の対戦経過になるため、対象判断数が異なる。{ordinary}</p>
<h2>実戦ログへの適用範囲</h2><p>既存の実戦再構成{len(history)}局面を各64サンプルで照合した。現行の技補完は元の記録とすべて一致。候補で変わったのは、黒い眼差しの後に歌を見落としていた{len(changed)}局面だけで、全サンプルで歌を保持した。他の{len(history) - len(changed)}局面は乱数消費を含めた状態が一致した。変わった局面も、相手の技以外の状態が一致することを検査した。</p>
<p>これは後に特定の技を使った試合を抽出した診断であり、一般的な遭遇率や誤り率ではない。元の190対戦は実効的な対戦相手が約2人に偏り、自分の未公開構成には再構成上の推定が含まれる。</p>
<h2>判断が割れた11局面</h2><p>前回の分岐局面を固定し、各8〜9乱数・計{searches}探索で比較した。ポリゴン2の事例5・21・29では候補が全24回れいとうビームを選ぶ。道連れ・道具・能力値をまとめて変えなくてもこの変化は起きる。</p>
<p>ゴローニャの事例35は全8回じしんを維持する。一方、HP1のムウマに対するガラガラの事例25は全8回つるぎのまい。道連れを新しく想定しなくても待つため、この手を単純に「道連れの過大推定」とは説明できない。着手の見た目は採否の基準にせず、上の終局結果で判断する。</p>
<div class="scroll"><table><tr><th>事例</th><th>現行</th><th>限定修正</th><th>根拠</th></tr>{''.join(ft)}</table></div>
<p>現行の元の着手・訪問数・評価平均は11局面とも完全再現。これらは仮説の元になった選択済みの局面であり、独立した勝率データに数えない。</p>
<h2>検査・再現</h2><p>bot単体138件、belief統合6件、blind統合2件が通過。全workspace・全targetの型検査も通過。新規テストでは公開技と道具の保持、対象外の一致、技以外の状態と乱数消費の一致、未公開情報を書き換えても推定が変わらないことを確認した。</p>
<p>終局試験の再実行は <code>tools/operational-retain-perish</code>。各試合・各方策の結果を別々に保存し、組み合わせ前に入力メタデータの一致を検査する。乱数、入力のハッシュ、実行バイナリのハッシュを記録した。</p>
<p><a href="protocol.json">事前条件</a>・<a href="execution.json">実行情報</a>・<a href="results.json">全集計</a>・<a href="fork-summary.json">局面別集計</a>・<a href="history.jsonl">実戦局面への適用</a>・<a href="adaptive.jsonl">初手ロック後自由行動</a>・<a href="stress.jsonl">固定手順</a>・<a href="natural.jsonl">通常戦</a>・<a href="onpool.jsonl">登録対照</a>・<a href="same.jsonl">同一方策対照</a>・<a href="checks.json">検査記録</a></p></html>'''
(p / 'index.html').write_text(report)
