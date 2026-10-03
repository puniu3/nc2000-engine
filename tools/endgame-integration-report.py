import gzip
import hashlib
import html
import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
source = root / "tmp/endgame-integration-v1"
destination = root / "findings/endgame-integration-v1"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


summaries = {}
for name in ("audit", "audit-budget-fix"):
    path = source / f"{name}.jsonl"
    metadata, *rows = read_rows(path)
    assert len(rows) == metadata["jobs"] == 144
    assert len({row["job"] for row in rows}) == 144
    assert all(row["legal"] for row in rows)
    groups = []
    for information in ("full", "public-protocol"):
        for fraction in (0, 0.05, 0.1, 0.2):
            group = [r for r in rows if r["information"] == information and r["solver_fraction"] == fraction]
            assert len(group) == 18
            groups.append({
                "information": information, "solver_fraction": fraction, "decisions": len(group),
                "source_games": len({r["game"] for r in group}),
                "calls": sum(r["stats"]["calls"] for r in group),
                "replacements": sum(r["stats"]["value_replacements"] for r in group),
                "decisions_with_replacements": sum(r["stats"]["value_replacements"] > 0 for r in group),
                "root_overrides": sum(r["stats"]["root_overrides"] for r in group),
                "solver_ms": sum(r["stats"]["solver_ms"] for r in group),
                "max_decision_ms": max(r["elapsed_ms"] for r in group),
                "peak_solver_entries": max(r["stats"]["peak_entries"] for r in group),
                "max_probe_ms": max(r["stats"].get("max_probe_ms", 0) for r in group) if name != "audit" else None,
                "within_share_plus_one_probe": all(r["stats"]["solver_ms"] <= r["budget_ms"] * fraction + r["stats"]["max_probe_ms"] + 0.1 for r in group) if name != "audit" else None,
            })
    summaries[name] = {"revision": metadata["revision"], "dirty": metadata["dirty"], "binary_sha256": metadata["binary_sha256"], "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "groups": groups}
assert all(g["within_share_plus_one_probe"] for g in summaries["audit-budget-fix"]["groups"])
(destination / "audit-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
archive = {}
for name in ("strong.jsonl", "strong-measured.jsonl"):
    archive[f"inputs/{name}"] = (root / "tmp/selfplay-endgames-v1" / name).read_text()
for path in sorted(source.rglob("*")):
    if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
        archive[str(path.relative_to(source))] = path.read_text()
with gzip.GzipFile(filename=str(destination / "integration-measurements.json.gz"), mode="wb", mtime=0) as out:
    out.write(json.dumps(archive).encode())
classification = json.loads((destination / "unresolved-classification.json").read_text())
table = "".join(
    f'<tr><td>{html.escape(g["information"])}</td><td>{g["solver_fraction"]:.0%}</td><td>{g["calls"]}</td><td>{g["replacements"]} / {g["decisions_with_replacements"]}局面</td><td>{g["root_overrides"]}</td><td>{g["solver_ms"]/1000:.2f}</td><td>{g["max_decision_ms"]:.0f}</td></tr>'
    for g in summaries["audit-budget-fix"]["groups"]
)
development = json.loads((destination / "native-development.json").read_text())
assert development["complete"]
development_source = root / "tmp/endgame-development-v1"
development_records = {}
for path in sorted(development_source.rglob("*")):
    if path.is_file() and path.suffix in (".json", ".jsonl", ".log"):
        development_records[str(path.relative_to(development_source))] = path.read_text()
for name, arm in development["arms"].items():
    games = [text for path, text in development_records.items() if path.startswith(f"{name}/pair-")]
    assert len(games) == arm["completed_games"] == 100
    assert all(json.loads(text.splitlines()[-1])["type"] == "result" for text in games)
with gzip.GzipFile(filename=str(destination / "native-development-records.json.gz"), mode="wb", mtime=0) as out:
    out.write(json.dumps(development_records).encode())
development_table = "".join(
    f'<tr><td>{arm["protocol"]["candidate"]["solver_fraction"]:.0%}</td><td>{arm["completed_games"]}</td><td>{arm["score"]:.1%}</td><td>{arm["ci95"][0]:.1%}–{arm["ci95"][1]:.1%}</td><td>{arm["difference_vs_public_off"]["mean"]:+.1%}</td><td>{arm["metrics"]["candidate"]["p99_ms"]:.1f}</td></tr>'
    for arm in development["arms"].values()
)
selected = development["arms"][development["selected_for_confirmation"]]
holdout = json.loads((destination / "holdout-evaluation.json").read_text())
assert holdout["complete"]
holdout_table = "".join(
    f'<tr><td>{information}</td><td>{result["score"]:.2%}</td><td>{result["ci95"][0]:.2%}–{result["ci95"][1]:.2%}</td><td>{result["candidate"]["replacements"]} / {result["candidate"]["decisions_with_replacements"]}判断</td><td>{result["candidate"]["unresolved_calls"]/result["candidate"]["calls"]:.2%}</td><td>{result["candidate"]["p99_ms"]:.1f} / {result["baseline"]["p99_ms"]:.1f}</td></tr>'
    for information, result in holdout["results"].items()
)
page = f'''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NC2000 終盤ソルバー接続の検証</title>
<style>body{{margin:2rem auto;padding:0 1rem;max-width:68rem;background:#eee9df;color:#302c25;font:17px/1.7 system-ui}}h1{{font-size:1.6rem}}h2{{font-size:1.15rem;margin-top:2rem}}table{{border-collapse:collapse;width:100%;font-size:.85rem}}th,td{{padding:.45rem;border-bottom:1px solid #c8c0b3;text-align:left}}a{{color:#6b4826}}.scroll{{overflow-x:auto}}</style>
<h1>NC2000 終盤ソルバー接続の検証</h1>
<p>証明付き混合戦略を全情報botの実際の着手へ、証明幅0.02以内の局面値をblind探索の葉へ接続しました。Webの通常対局・局面解析・分岐対局でも試験経路を動かしています。<strong>強化の採否は未判定です。公開botの既定設定は変更していません。</strong></p>
<h2>何が判断に使われるか</h2>
<p>全情報では最悪損失0.02以内の証明付き混合戦略から一度だけ着手を抽選します。blindでは、プレイヤーに届く公開情報と自身のチームから作った仮説局面だけを解析します。その値はその仮説に条件付けられた値であり、隠れた真の状態や情報集合全体の解とは扱いません。blindの実戦ルートを全情報戦略へ直接切り替える処理はありません。</p>
<h2>10秒の接続監査</h2>
<p>過去の開発用コーパスから各残存数区分の先頭3局面、行動可能な18視点を使用。全情報／公開情報と0／5／10／20%配分を組み合わせた144件です。同一対局からの局面は独立標本ではありません。以下は強さの比較ではなく、実際に値や戦略が使われたかの監査です。</p>
<div class="scroll"><table><tr><th>情報</th><th>配分</th><th>solve呼出</th><th>葉の置換回数 / 発動局面</th><th>戦略採用</th><th>solve計・秒</th><th>最長判断・ms</th></tr>{table}</table></div>
<p>初回監査では予算消化後もキャッシュ照会を繰り返し、5%指定でも数秒を費やす不具合を検出しました。予算到達時に照会を止め、回帰テストと同一プロトコルの再監査を実施。全件で配分＋最後の1バッチ以内でした。外部時計を各バッチ間で確認するため厳密な時間上限ではなく、最長バッチは{max(g['max_probe_ms'] for g in summaries['audit-budget-fix']['groups']):.0f}msでした。Webの実機応答時間の証明には使いません。</p>
<h2>未解決42局面の分類</h2>
<p>時間上限28件、メモリ上限14件。1対1が8件、1対2が7件、2対1が10件、2対2が17件。40件で生存ポケモンにPPの残る回復・吸収技がありました。これは最初に到達した停止条件の分類で、上限を増やせば解けるという推定ではありません。道具による回復はこの分類に含めていません。</p>
<h2>対戦評価</h2>
<p>nativeの1手10秒で、凍結済みの開発400局が完了しました。公開情報経路の0／5／10／20%各100局を、既定のraw blind経路と比較。実際のdrawWeightとv3 priorを使い、同じチーム・乱数の対局でbotの席を交替しました。0%はソルバーを除いた対照です。同一bot同士の8局は全ペア0.5で対局経路も一致し、意図的に探索を空にしたランダム方策との16局は探索側が0.75でした。</p>
<div class="scroll"><table><tr><th>配分</th><th>局数</th><th>対raw得点</th><th>ペア単位95%区間</th><th>公開情報OFFとの差</th><th>判断p99・ms</th></tr>{development_table}</table></div>
<p>事前の選択規則により{selected['protocol']['candidate']['solver_fraction']:.0%}配分を確認候補へ固定しました。全4群で無効手・配分＋最後の1バッチを超える予算違反・キャッシュ上限違反・エンジン打ち切りは0でした。10%と公開情報OFFとの差は9.0ポイント、共有50ペアを単位とした95%区間は0.06～17.94ポイントです。複数候補から選んだ開発結果であり、強化の確証や公開の根拠には使いません。プロセスRSS最大は0／5／10／20%で454,496／623,744／661,456／725,184KiB。両席とアロケータ保持を含み、候補単体のメモリ差ではありません。<a href="native-development.json">全4群の開発結果と選択記録</a></p>
<p><a href="native-development-records.json.gz">開発400局の全判断・対局結果・実行条件・RSS記録</a></p>
<p>固定した10%候補で、別シード1,200局のWeb確認戦と独立終盤評価を開始しました。採用条件は本番条件で得点0.52以上、対局ペア単位95%区間の下端0.50超、時間・メモリ・無効手・打ち切りの回帰なし。呼出数や全情報版の改善だけでは採用しません。</p>
<p>確認戦はWebの公開中botを相手に新しい600ペア・1,200局。Webのキャッシュは資源予備測定に基づき256エントリ、待ち時間は32局の時間対照と事前の固定式から796msへ固定しました。nativeの12,000エントリとは異なる資源条件であり、nativeの得点を移し替えません。結果を見た再調整や標本追加はしません。<a href="confirmation-design.json">事前固定条件と採否基準</a> ／ <a href="web-confirmation-protocol.json">実行中のWeb確認条件</a></p>
<h2>独立終盤の評価</h2>
<p>独立コーパスの全288継続対局が完了しました。別シードの24自己対戦中、6局には抽出対象の終盤がなく、18局から72局面を使用。全情報・公開情報それぞれでON／OFFを比較し、両席を評価しました。判断10秒、参照解析60秒、配分10%、キャッシュ12,000エントリ。元対局を等重みの18統計単位とし、同じ対局からの複数局面を独立標本とは数えません。公開情報の履歴は元の棋譜から復元しました。<a href="holdout-evaluation-protocol.json">独立終盤の実行条件</a></p>
<div class="scroll"><table><tr><th>情報</th><th>ONの対OFF得点</th><th>元対局単位95%区間</th><th>葉の置換 / 発動判断</th><th>未解決呼出率</th><th>判断p99・ms ON / OFF</th></tr>{holdout_table}</table></div>
<p>両方の得点区間が50%をまたぎ、独立終盤での改善は確認できませんでした。全情報の初手115視点では、証明付きの誤り修正は0、既に正しかった方策の維持は33、未解決比較は82でした。参照解析72件中19件が幅0.02以内、53件は外部時間上限で終了。未解決や混合方策の測定不能な損失を0とは扱いません。全情報の戦略採用は継続対局全体で77回、公開情報では0回です。</p>
<p>全288局で無効手・エンジン打ち切り・キャッシュ上限違反・配分＋最後の1バッチを超える予算違反は0。ソルバー時間の合計は全情報621.8秒、公開情報706.9秒、最長の不可分バッチは55.8／125.3msでした。参照解析は最後のバッチを含め最大60,370.6ms。呼出や置換が発生しても、この標本では勝率改善や誤り修正の証明には結びついていません。Web確認戦の採否は全1,200局の完了後に別途判断します。</p>
<p><a href="holdout-evaluation.json">独立終盤の全結果と方策損失分類</a> ／ <a href="holdout-cost-audit.json">予算・上限・参照解析の監査</a> ／ <a href="holdout-evaluation-records.json.gz">全288継続対局と72参照解析の記録</a></p>
<p><a href="holdout-integrity.json">独立コーパスの指紋と再生検証</a> ／ <a href="holdout-harness-calibration.json">継続対局の対照検証</a> ／ <a href="holdout-generation-protocol.json">生成条件</a></p>
<h2>Webの待ち時間</h2>
<p>このMacのChromiumで、公開中と同じ27,000試行・通常のBotWorkerによる8局を測定しました。強制手以外の617判断で、要求から結果までの平均は890ms、中央値922ms、99パーセンタイル1,452msでした。既存の試験用既定値2.3秒をこの環境の公平な比較に流用しません。測定平均を切り捨てた889msの時間対照では、公開情報OFFが既定botより14.5%多く時間を使ったため、そのまま強さを比較しません。時間比だけから776msへ補正した別シード16局では、逆に8.8%短くなりました。両試行の待ち時間だけを合算し、要求処理の実測上乗せを引いた814msで新しい32局を検証。強制手以外の要求時間は候補平均818.4ms、既定bot平均800.7msで、候補が2.2%長くなりました。事前に定めた式から確認戦は796msへ固定。実際の確認戦でも時間回帰がないことを別途確認します。勝敗による時間調整はしていません。iPad Safariの性能を示す数字ではありません。</p>
<p>同じブラウザ対戦経路の対照では、同一設定8局の全ペアが0.5で着手列も一致。中身を空にしたランダム方策との16局は探索側0.9375でした。候補の強化成績とは別の検証です。以後は各workerのWasm確保メモリも測定し、未計測値を0として扱いません。</p>
<p><a href="web-calibration.json">Webの時間・比較器の対照</a> ／ <a href="web-calibration-records.json.gz">Web測定記録</a> ／ <a href="web-budget-control.json">初回時間対照の結果</a> ／ <a href="web-budget-control-v2.json">2回目の時間対照結果</a> ／ <a href="web-budget-control-v3.json">814ms・32局の検証結果</a> ／ <a href="web-confirmation-budget.json">796msの確定根拠</a> ／ <a href="web-budget-control-records.json.gz">全時間対照の記録</a></p>
<h2>Webのメモリ予備測定</h2>
<p>同じチーム・乱数の1ペアを5／10／20%で測定しました。12,000エントリ上限では候補workerのWasm確保メモリの最大値が29.3／60.8／110.2MB（対照10.8／11.2／13.2MB）。256エントリ上限での再測定では15.7／21.5／21.6MB（対照14.4／12.0／10.6MB）でした。ここでのMBは10⁶バイトで、プロセス全体のRSSではありません。対局経路も変わるため、この差を同一局面の厳密な費用差とは扱いません。</p>
<p>256エントリ版の6局は全て合法手・可視ページ・配分＋最後の1バッチ以内で、キャッシュは256以下でした。ただし証明済みの葉の置換は0回。大きい上限でも0／1／0回です。小さい上限はメモリ負担を抑えましたが、強さやメモリ回帰なしの証明ではありません。配分は凍結済みnative開発計画に従って選び、これらの予備対局の勝敗では選んでいません。Web確認戦では事前に固定した256エントリ上限を使用しています。</p>
<p><a href="web-memory-research-cap.json">12,000エントリの測定</a> ／ <a href="web-memory-bounded.json">256エントリの測定</a> ／ <a href="web-memory-records.json.gz">全メモリ予備測定の記録</a></p>
<p><a href="audit-summary.json">監査集計</a> ／ <a href="unresolved-classification.json">42局面の分類</a> ／ <a href="calibration.json">比較器の対照</a> ／ <a href="development-10-protocol.json">凍結済み対戦条件</a> ／ <a href="integration-measurements.json.gz">監査・対照の測定記録</a></p>
</html>'''
(destination / "index.html").write_text(page)
print(destination / "index.html")
