#!/usr/bin/env python3
"""Render only shareable experiment summaries into the preview directory."""
import html
import hashlib
import json
from pathlib import Path
from datetime import datetime

OUT = Path(__file__).resolve().parent
summary = json.loads((OUT / "br-control-summary.json").read_text())
assert summary["complete"]
report = OUT / "report"
report.mkdir(exist_ok=True)


def pct(x):
    return f"{100*x:.1f}%"


def pp(x):
    return f"{100*x:+.1f}"


rows = []
for r in summary["results"]:
    target = "完全情報の相手" if r["target"].startswith("skuct") else "非公開情報の相手"
    lo, hi = r["paired_ci95"]
    rows.append(f"""<tr><th>{target}<small>{html.escape(r['target'])}</small></th>
<td>{pct(r['oracle_score'])}</td><td>{pct(r['control_score'])}</td>
<td>{pp(r['paired_delta'])} pp<small>95%区間 {pp(lo)} ～ {pp(hi)} pp</small></td></tr>""")

historical = []
for name, label in [
    ("scale-3k-vs-1k", "探索 3,000 対 1,000"),
    ("scale-9k-vs-3k", "探索 9,000 対 3,000"),
    ("scale-27k-vs-9k", "探索 27,000 対 9,000"),
    ("info-open-vs-blind-inpool-9k", "既知の構築：相手の技・持ち物を知る効果（9k）"),
    ("info-open-vs-blind-offpool-27k", "事前分布にない構築：同じ情報を知る効果（27k）"),
]:
    r = json.loads((OUT / "duels" / (name + ".jsonl")).read_text())["result"]
    historical.append(f"<tr><th>{label}</th><td>{r['games']}</td><td>{pct(r['score'])} ± {100*r['ci95']:.1f} pp</td></tr>")

resolved = any(r["paired_ci95"][0] > 0 for r in summary["results"])
headline = ("相手戦略を参照する効果を一部の対戦条件で検出" if resolved
            else "相手戦略を参照する追加効果は、今回の標本では確定できない")
scale_panel = ""
scale_history = ""
scale_path = OUT.parent / "headroom-scale-v1/summary.json"
if scale_path.exists():
    scale = json.loads(scale_path.read_text())
    headline = "強さの改善余地を、同じ情報条件で直接検証中"
    if scale["positive"]:
        headline = "計算量を増やすことで、強さが伸びる余地を確認"
    measurement = f"検査済みの全対戦は {scale['games']} 戦。確認用のブロックを計算中です。"
    if scale["analysis_games"]:
        interval = scale["betting_interval"]
        measurement = (f"確認解析に使う {scale['analysis_games']} 戦。8.1万回側の勝点率 {pct(scale['score'])}。"
                       f"事前登録した区間：{pct(interval[0])} ～ {pct(interval[1])}。")
    phase = "直接対戦を実行中"
    decision = "まず512戦、登録上限2,048戦。途中確認に対応した区間（alpha 0.025）の下限が50%を超えた完成ブロックで改善ありと判定します。差が未確定でも『改善余地なし』とは判断しません。"
    cost = "総思考時間は同じではありません。"
    if scale["positive"]:
        audit_path = OUT.parent / "headroom-scale-v1/final-audit.json"
        audit = json.loads(audit_path.read_text())
        assert audit["status"] == "pass" and audit["stopped"]
        assert audit["summary_sha256"] == hashlib.sha256(scale_path.read_bytes()).hexdigest()
        phase = "結論：強さの改善余地は残っている"
        measurement += f" {audit['wins']}勝{audit['losses']}敗、引き分け{audit['ties']}戦。"
        decision = (f"事前登録した逐次区間（alpha 0.025）の下限が50%を初めて超えたブロックで停止しました。"
                    f"実行した比較は{audit['executed_comparison_games']}戦で、探索用128戦と各確認ブロックの最初のペアを除く{audit['confirmation_games']}戦が結論の根拠です。打ち切り・無効対戦は0でした。")
        cost = (f"探索量は3倍。全比較896戦の平均選択時間は、8.1万回側{audit['timing']['a']['mean_ms']/1000:.2f}秒、"
                f"2.7万回側{audit['timing']['b']['mean_ms']/1000:.2f}秒（約{audit['mean_choose_time_ratio']:.1f}倍）でした。"
                "これは10並列で負荷をかけたMac上の測定で、iPadでの応答時間ではありません。")
        history_rows = "".join(
            f"<tr><td>{c['analysis_games']}</td><td>{pct(c['score'])}</td><td>{pct(c['betting_interval'][0])} ～ {pct(c['betting_interval'][1])}</td></tr>"
            for c in scale['confirmation_checkpoints'])
        scale_history = f"""<h2>完成ブロックごとの確認結果</h2>
<div class="scroll"><table><thead><tr><th>確認用対戦数</th><th>勝点率</th><th>登録した逐次区間</th></tr></thead>
<tbody>{history_rows}</tbody></table></div>
<p>確認手順と集計コードは確認用の最初の結果が出る前に固定しました。389個の入力のハッシュ、全896戦の元データ、除外規則を検査し、別計算でも区間と最初の基準通過を再現しました。</p>
<p>この結果は、現在のローカル実装がカタログ一様抽選の条件で計算量により伸びる証拠です。
同じ応答時間での改善、人間の自作構築に対する改善幅、理論上の最適戦略までの残り距離は未測定です。</p>"""
        (report / "final-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    scale_panel = f"""<section class="note"><h2 style="margin-top:0">{phase}：8.1万回 対 2.7万回</h2>
<p>同じ探索方式・同じ非公開情報の条件で、探索量だけを3倍にしています。{measurement}</p>
<p>{decision}</p>
<p>最初の128戦は70勝58敗（54.7%）でしたが、これは探索的結果です。
乱数の監査で各ブロック最初の1戦が両側同じシードになることが判明したため、
後続ブロックの集計を見る前に、該当ペアを除く確認手順を固定しました。
確認用は最大1,890戦で、元の2,048戦上限は増やしていません。</p>
<p>クラウドの認証期限切れによりMacの10スレッドで実行。カタログ一様抽選、ポンダリングなし。
{cost}</p><small>このページの更新：{datetime.now().astimezone().isoformat(timespec='seconds')}</small></section>"""
    (report / "scale-summary.json").write_text(json.dumps(scale, indent=2) + "\n")
page = f"""<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NC2000 — 強さの伸びしろ実験</title>
<style>
:root{{color-scheme:light}}body{{margin:0;background:#f5f3ed;color:#292c27;font:16px/1.8 system-ui,sans-serif}}
main{{max-width:920px;padding:30px 22px 70px;margin:auto}}h1{{font-size:1.8rem;line-height:1.5}}h2{{font-size:1.25rem;margin-top:2.5em}}
.note{{padding:16px 20px;border-left:4px solid #73836d;background:#e9ede5}}small{{display:block;color:#5b6157;font-size:.8rem}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:.95rem}}th,td{{padding:12px 10px;border-bottom:1px solid #cccfc5;text-align:left}}
td{{white-space:nowrap}}a{{color:#466044}}code{{font-size:.9em}}.meta{{color:#62675d}}
</style><main>
<p class="meta">NC2000 / headroom-probe-v1 / 2026-10-16 再開</p>
<h1>{headline}</h1>
{scale_panel}
{scale_history}
<p>中断していた「現在のAIにどれだけ伸びしろがあるか」の調査を引き継ぎ、最優先の対照実験を完了しました。
各条件100戦×4組と同一AIの確認8戦、計408戦です。</p>
<div class="note">以前の攻略用AIの勝点率65%・71%には、計算量や情報量の差も含まれていました。
今回は攻略用AIの探索・評価・状態技と交代の処理をそろえ、相手戦略を参照する処理だけを外した対照と比較しています。</div>
<h2>相手戦略の参照で、どれだけ増えたか</h2>
<div class="scroll"><table><thead><tr><th>対戦相手</th><th>参照あり</th><th>参照なし</th><th>同じ対戦条件での差</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p>勝点率は勝ち1・引き分け0.5・負け0。各組100戦＝P1/P2を入れ替えた50ペア。
差の区間は同じペア同士の差から計算した通常の95%近似区間で、多重比較の補正はありません。
これは探索的な計測であり、製品への採用判定ではありません。</p>
<p>両方の攻略側は完全情報で9,000回探索します。相手は3,000回・探索係数0.4です。
「参照あり」はさらに相手AIを最大3回実行するため、総思考時間は同じではありません。
この差は理論上の最適戦略からの距離や、27,000回で動く製品の弱点の大きさを証明するものではありません。</p>
<h2>検証したこと</h2>
<ul><li>過去の攻略側200戦を新しい実行ファイルで再現。全対戦の勝敗と総ターン数が一致しました。</li>
<li>同一AIの対照8戦は勝点率50%、P1/P2ペアの不一致0でした。</li>
<li>相手戦略の参照回数を検査するテスト、対戦設定と成果物の検査を通過。打ち切り・無効対戦は0でした。</li>
<li>既存の製品設定と公開サイトは変更していません。</li></ul>
<h2>引き継いだ既存の証拠</h2>
<div class="scroll"><table><thead><tr><th>比較</th><th>対戦数</th><th>前者の勝点率・95%幅</th></tr></thead>
<tbody>{''.join(historical)}</tbody></table></div>
<p>過去の9k対3k・27k対9kの比較は点推定ではプラスですが、区間は50%を含みます。これらだけでは飽和とも、改善確定とも言えません。
事前分布にない構築では情報を与えた差が大きく、今後調べる候補です。ただし相手の正確な構築を知る実験なので、
推測の改善で回収できる量はまだ測っていません。対象はランダム生成した120構築で、
半数の60構築は全員が持ち物なしです。この結果を人間の構築に対する改善幅には換算できません。</p>
<p>カタログの実験は24構築を一様に抽選しています。製品の重み付き抽選とは分布が違います。
選出のノイズ推定も既存の探索統計からの近似で、実際に探索を繰り返したときの不一致率ではありません。</p>
<h2>今後調べるなら</h2>
<ol><li>同じ応答時間で今回の改善をどれだけ回収できるか。</li>
<li>事前分布にない構築で、実装可能な推測の改善がどれだけ回収できるか。</li>
<li>選出だけ探索量を増やす効果があるか。</li></ol>
<p><a href="br-control-summary.json">今回の集計データ（JSON）</a></p>
<p><a href="scale-summary.json">探索量の比較結果（JSON）</a> · <a href="final-audit.json">最終監査（JSON）</a></p>
</main></html>"""
(report / "index.html").write_text(page)
(report / "br-control-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(report / "index.html")
