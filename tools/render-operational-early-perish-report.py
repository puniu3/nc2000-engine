import html
import json
from pathlib import Path

p = Path('findings/operational-early-perish-v1')
r = json.loads((p / 'results.json').read_text())
assert r['complete']
protocol = json.loads((p / 'protocol.json').read_text())
gains = json.loads(Path('findings/operational-retain-perish-v1/gain-review/summary.json').read_text())
labels = {'adaptive': '初手ロック後は自由行動', 'stress': '固定の滅び手順',
          'natural': '自由選出・自由行動', 'onpool': '登録カタログ対照', 'same': '同一方策対照'}
pp = lambda n: f'{100 * n:+.1f}'
rows = []
exposures = []
for cell in ['adaptive', 'stress', 'natural', 'onpool', 'same']:
    x = r['results'][cell]
    rows.append(f'<tr><td>{labels[cell]}</td><td>{x["triplets"]}</td>' +
                ''.join(f'<td>{x["arms"][a]["wins"]}勝<br>上限{x["arms"][a]["caps"]}</td>' for a in 'abc') +
                ''.join(f'<td>{pp(x["comparisons"][key]["paired_delta"])}pt<br>' +
                        '〜'.join(pp(n) for n in x['comparisons'][key]['stratified_bootstrap95']) + '</td>'
                        for key in ['b-a', 'c-a', 'c-b']) + '</tr>')
    exposures.append(f'<tr><td>{labels[cell]}</td>' + ''.join(
        f'<td>{x["arms"][a]["early_exposed"]}判断<br>{x["arms"][a]["early_combo_samples"]} / {64 * x["arms"][a]["early_exposed"]}標本</td>'
        for a in 'abc') + '</tr>')
primary = r['results']['adaptive']['comparisons']
kit_names = {'1': '眠る・守る', '2': '守る・道連れ', '3': '怪しい光・守る', '4': '電磁砲・道連れ'}
opponent_rows = ''.join('<tr><td>' + ('ヌオー入り' if key.startswith('quagsire') else 'キングドラ入り') +
                        f' / {kit_names[key[-1]]}</td><td>{group["pairs"]}</td>' +
                        ''.join(f'<td>{pp(primary[c]["by_opponent"][key]["paired_delta"])}pt</td>'
                                for c in ['b-a', 'c-a', 'c-b']) + '</tr>'
                        for key, group in sorted(primary['c-a']['by_opponent'].items()))
stats = ''.join(f'<tr><td>{name}</td><td>{primary[key]["better_pairs"]} / {primary[key]["worse_pairs"]}</td>'
                f'<td>{primary[key]["terminal_exact_two_sided_p"]:.4f}</td><td>{primary[key].get("holm_two_comparisons_p", "—")}</td></tr>'
                for key, name in [('b-a', '限定修正 − 現行'), ('c-a', '先回り版 − 現行'), ('c-b', '先回り版 − 限定修正')])
moves = {'move rockslide': 'いわなだれ', 'move earthquake': 'じしん', 'move fireblast': 'だいもんじ',
         'move curse': 'のろい', 'move icebeam': 'れいとうビーム', 'move surf': 'なみのり',
         'move hydropump': 'ハイドロポンプ', 'move psychic': 'サイコキネシス', 'move spikes': 'まきびし',
         'switch 2': '交代2', 'switch 3': '交代3'}
gain_rows = ''.join(f'<tr><td>{c["job"]} / T{c["turn"]}</td><td>' +
                    ' → '.join(html.escape(moves.get(m, m)) for m in c['original_choices']) + '</td>' +
                    ''.join('<td>' + '、'.join(f'{moves.get(m, m)} {n}回' for m, n in c['choices'][a].items()) + '</td>'
                            for a in ['baseline', 'retain-perish']) +
                    f'<td>{c["identical_roots"]}/{c["paired_seeds"]}</td></tr>' for c in gains['cases'])
decision = json.loads((p / 'assessment.json').read_text())
total = sum(x['triplets'] * 3 for x in r['results'].values())
caps = sum(a['caps'] for x in r['results'].values() for a in x['arms'].values())
report = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NC2000 — ロックを初手から想定する</title>
<style>body{{font:18px/1.75 system-ui,sans-serif;background:#f4f0e6;color:#28271f;max-width:1120px;margin:32px auto;padding:0 20px}}h1,h2{{line-height:1.35}}h2{{margin-top:2em}}a{{color:#684928}}table{{border-collapse:collapse;width:100%;font-size:16px}}th,td{{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #c7c2b6}}.scroll{{overflow-x:auto}}.lead{{font-size:22px}}code{{overflow-wrap:anywhere}}</style>
<p><a href="/">プレビューハブ</a>・<a href="/nc2000-retain-perish/">前回の限定修正試験</a></p>
<p>このページは採用判断前の試験記録。<a href="/nc2000-perish-bayes/">後続のベイズ判断と採用状況</a>を参照。</p>
<h1>ロックを初手から想定する</h1><p class="lead">{html.escape(decision['verdict'])}</p>
<p>{html.escape(decision['interpretation'])}</p>
<h2>何を調整したか</h2>
<p>限定修正B（RetainPerish）は、黒い眼差しが見えてから滅びの歌の欠落を補う。先回り版C（EarlyPerish）は、ムウマが場に出た時点から未公開のロックも想定する。現行A（Layered）を含めた3方式を比較した。本番の既定方式は変更していない。</p>
<p>凍結済みの事前データでは、ムウマ28構成のうち23構成が黒い眼差しと滅びの歌を持つ。重みの合計は15.499999 / 20.499999、約75.6%。しかし個別の最頻構成には黒い眼差しがなく、現行は初手でロックを想定しない。Cは、公開技と両立する構成の重みから組み合わせの確率を計算し、探索ごとに技だけを補完する。</p>
<p>公開済みの技を必ず保持し、道具・能力値・性別を変えない。未登場の見せ合い段階、倒れた相手、組み合わせの支持がない場合は抽選しない。道具や性別で確率を条件付けた完全な型推定ではなく、既存の技補完へ狭く追加する候補である。</p>
<p>実戦経路の確認で、初登場を観測してもキャッシュ更新番号が進まず初手では候補が働かない不具合も発見した。全方式で観測の更新を修正し、独立した動作確認でA/Bの履歴・全カウンターの一致とCの発動を確認してから本試験を開始した。</p>
<h2>新しい乱数での終局比較</h2>
<div class="scroll"><table><tr><th>条件</th><th>組</th><th>A 現行</th><th>B 限定修正</th><th>C 先回り版</th><th>B−A / 95%区間</th><th>C−A / 95%区間</th><th>C−B / 95%区間</th></tr>{''.join(rows)}</table></div>
<p>主試験128組では、相手ムウマの先発と初手の黒い眼差しだけを固定し、その後は相手botが滅びの歌・道連れ・交代を自由に選ぶ。双方27,000反復、UCB c=0.4。同一方策対照だけ3,000反復。合計{total}戦、500ターンの上限到達は{caps}戦。長い試合を除外していない。</p>
<div class="scroll"><table><tr><th>主試験の比較</th><th>改善 / 悪化</th><th>両側正確検定p</th><th>現行に対する2比較のHolm補正p</th></tr>{stats}</table></div>
<p>相手8構成ごとの層別bootstrapを10,000回実施。事前基準は主試験の95%区間の下限が正、かつ現行に対する2候補のHolm補正pが0.05未満。そのうえで副試験の未解消の退行がないことを求める。C−Bは先回りの寄与を分ける診断として報告する。旧64組は合算していない。</p>
<p>対戦相手は過去の実戦で見た2つの見せ合い構成と4つのムウマ型を組み合わせた8構成。固定bot相手の条件付き試験であり、対人の総合勝率に換算できない。通常戦で対象が出なければ、差がない結果は適用範囲の確認にとどまる。</p>
<div class="scroll"><table><tr><th>相手構成 / ロック・歌以外の2技</th><th>組</th><th>B−A</th><th>C−A</th><th>C−B</th></tr>{opponent_rows}</table></div>
<p>各構成の組数は少なく、内訳は診断用。よかった構成だけを選び直して合格判定には使わない。</p>
<p>キングドラ入り・守る＋道連れ型では、CはAより6勝少なかった。全6悪化例の初分岐を調べ、うち5・21はムウマとの相打ち後まで同じ公開進行だった。この2組・4戦を全履歴まで再現し、最初の分岐を同じ盤面・同じ探索乱数で各9回比較すると、着手・訪問数・評価平均まで全18組で一致した。この2敗をその局面の共通の判断退行とは説明できない。</p>
<p>45・109は、限定修正でも生じるポリゴン2の「のろい→れいとうビーム」の分岐。旧試験の改善例と同じ変更が、別の対戦では敗北につながった。53・77はより後の分岐で、今回そこまでの同乱数再探索はしていない。6敗はすべて集計に残し、後付けの条件分岐で候補を採用し直さない。<a href="cluster-review.json">6例の記録と36回の再探索集計</a></p>
<h2>先回りの想定は使われたか</h2>
<div class="scroll"><table><tr><th>条件</th><th>A 現行</th><th>B 限定修正</th><th>C 先回り版</th></tr>{''.join(exposures)}</table></div>
<p>場のムウマが黒い眼差しをまだ見せていない判断ごとに、探索とは別の固定乱数で64標本を取り、黒い眼差し＋滅びの歌の同時補完を数えた。判断数や対戦経過は方式ごとに変わる。これは発動の確認用で、標本を合算して独立した確率推定や強さの証明には使わない。</p>
<h2>旧試験で改善した6組も同じ方法で調査</h2>
<p>前回の悪化4組に続き、改善した6組・12戦を履歴と全カウンターまで再現した。最初に手が割れた同じ盤面を、探索乱数を揃えて計102回再探索した。</p>
<div class="scroll"><table><tr><th>組 / ターン</th><th>元の分岐</th><th>現行の再探索</th><th>限定修正の再探索</th><th>探索結果が完全一致</th></tr>{gain_rows}</table></div>
<p>3・20は乱数を揃えると訪問数・評価平均まで一致する。13のポリゴン2と52のパルシェンは、限定修正で8乱数すべてれいとうビームを選ぶ。単なる乱数ずれだけでなく、歌の保持が判断を変える例がある。ただし、攻撃へ変わったこと自体を最善手の証明にはしない。</p>
<p>改善6組と悪化4組はすべて旧試験の勝敗に残した。都合のよい分岐だけを新たな勝率根拠として数えていない。<a href="/nc2000-retain-perish/gain-review/summary.json">改善例の診断集計</a>・<a href="/nc2000-retain-perish/loss-review/summary.json">悪化例の診断集計</a></p>
<h2>検証と再現</h2><p>bot単体141件、belief統合6件、blind統合2件が通過。全workspace・全targetの型検査も通過。初登場での発動、重み付き抽選の比率、公開技・技以外の状態の保持、対象外と倒れた相手での乱数消費の一致を検査した。</p>
<p>再実行は <code>tools/operational-early-perish</code>。事前条件、入力ハッシュ、実行バイナリのハッシュ、各試合の対戦・探索乱数を保存している。</p>
<p><a href="protocol.json">事前条件</a>・<a href="execution.json">実行情報</a>・<a href="results.json">全集計</a>・<a href="assessment.json">判断</a>・<a href="checks.json">検査記録</a>・<a href="adaptive.jsonl">主試験</a>・<a href="stress.jsonl">固定手順</a>・<a href="natural.jsonl">通常戦</a>・<a href="onpool.jsonl">登録対照</a>・<a href="same.jsonl">同一方策対照</a></p></html>'''
(p / 'index.html').write_text(report)
