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
<p>nativeの1手10秒で400局の開発計画を凍結しました。公開情報経路の0／5／10／20%各100局を、既定のraw blind経路と比較します。実際のdrawWeightとv3 priorを使い、同じチーム・乱数の対局でbotの席を交替。0%はソルバーを除いた対照です。同一bot同士の8局は全ペア0.5で対局経路も一致し、意図的に探索を空にしたランダム方策との16局は探索側が0.75でした。</p>
<p>開発結果で一案を固定した後、別シード1,200局の確認戦、独立終盤、Web条件の評価が必要です。採用条件は本番条件で得点0.52以上、対局ペア単位95%区間の下端0.50超、時間・メモリ・無効手・打ち切りの回帰なし。呼出数や全情報版の改善だけでは採用しません。</p>
<h2>独立終盤の準備</h2>
<p>別シードで24局の自己対戦を生成し、各状態キー・合法な棋譜再生・抽出位置・終局結果の一致を確認しました。候補の選択が終わるまで、このコーパスでの候補評価は封印します。継続対局の比較器は別のスモークデータで検証し、同一設定同士は全情報・公開情報とも同じ対局経路とペア得点0.5を再現しました。複数の終盤を取った元対局は1つの統計単位として扱います。</p>
<p><a href="holdout-integrity.json">独立コーパスの指紋と再生検証</a> ／ <a href="holdout-harness-calibration.json">継続対局の対照検証</a> ／ <a href="holdout-generation-protocol.json">生成条件</a></p>
<h2>Webの待ち時間</h2>
<p>このMacのChromiumで、公開中と同じ27,000試行・通常のBotWorkerによる8局を測定しました。強制手以外の617判断で、要求から結果までの平均は890ms、中央値922ms、99パーセンタイル1,452msでした。既存の試験用既定値2.3秒をこの環境の公平な比較に流用しません。測定平均を切り捨てた889msの時間対照では、公開情報OFFが既定botより14.5%多く時間を使ったため、そのまま強さを比較しません。時間比だけから776msへ補正した別シード16局では、逆に8.8%短くなりました。両試行の待ち時間だけを合算し、要求処理の実測上乗せを引いた814msで、新しい32局を検証中です。勝敗による時間調整はしていません。iPad Safariの性能を示す数字ではありません。</p>
<p>同じブラウザ対戦経路の対照では、同一設定8局の全ペアが0.5で着手列も一致。中身を空にしたランダム方策との16局は探索側0.9375でした。候補の強化成績とは別の検証です。以後は各workerのWasm確保メモリも測定し、未計測値を0として扱いません。</p>
<p><a href="web-calibration.json">Webの時間・比較器の対照</a> ／ <a href="web-calibration-records.json.gz">Web測定記録</a> ／ <a href="web-budget-control.json">初回時間対照の結果</a> ／ <a href="web-budget-control-v2.json">2回目の時間対照結果</a> ／ <a href="web-budget-control-v3-protocol.json">814msの検証条件</a></p>
<h2>Webのメモリ予備測定</h2>
<p>同じチーム・乱数の1ペアを5／10／20%で測定しました。12,000エントリ上限では候補workerのWasm確保メモリの最大値が29.3／60.8／110.2MB（対照10.8／11.2／13.2MB）。256エントリ上限での再測定では15.7／21.5／21.6MB（対照14.4／12.0／10.6MB）でした。ここでのMBは10⁶バイトで、プロセス全体のRSSではありません。対局経路も変わるため、この差を同一局面の厳密な費用差とは扱いません。</p>
<p>256エントリ版の6局は全て合法手・可視ページ・配分＋最後の1バッチ以内で、キャッシュは256以下でした。ただし証明済みの葉の置換は0回。大きい上限でも0／1／0回です。小さい上限はメモリ負担を抑えましたが、強さやメモリ回帰なしの証明ではありません。配分の選択は引き続き凍結済みnative開発計画に従い、これらの予備対局の勝敗では選びません。Web確認戦の資源条件は開始前に別途固定します。</p>
<p><a href="web-memory-research-cap.json">12,000エントリの測定</a> ／ <a href="web-memory-bounded.json">256エントリの測定</a> ／ <a href="web-memory-records.json.gz">全メモリ予備測定の記録</a></p>
<p><a href="audit-summary.json">監査集計</a> ／ <a href="unresolved-classification.json">42局面の分類</a> ／ <a href="calibration.json">比較器の対照</a> ／ <a href="development-10-protocol.json">凍結済み対戦条件</a> ／ <a href="integration-measurements.json.gz">監査・対照の測定記録</a></p>
</html>'''
(destination / "index.html").write_text(page)
print(destination / "index.html")
