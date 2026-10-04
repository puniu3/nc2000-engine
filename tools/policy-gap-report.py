import collections
import gzip
import hashlib
import html
import json
import math
import pathlib
import statistics
import sys


def verify_assessment(matrix, side, assessment):
    if assessment is None:
        return
    rows, cols = map(len, matrix["actions"])
    p = assessment["policy"]
    assert len(p) == (rows if side == 0 else cols)
    assert all(math.isfinite(x) and x >= 0 for x in p)
    assert abs(math.fsum(p) - 1) < 1e-8
    replies = []
    for reply in range(cols if side == 0 else rows):
        lower, upper = [], []
        for action, probability in enumerate(p):
            cell = matrix["cells"][action * cols + reply if side == 0 else reply * cols + action]
            lo, hi = (cell["lo"], cell["hi"]) if side == 0 else (1 - cell["hi"], 1 - cell["lo"])
            assert -1e-8 <= lo <= hi + 1e-8 <= 1 + 2e-8
            lower.append(probability * lo)
            upper.append(probability * hi)
        replies.append((math.fsum(lower), math.fsum(upper)))
    floor, ceiling = min(x[0] for x in replies), min(x[1] for x in replies)
    assert assessment["guarantee"]["lo"] <= floor + 1e-8
    assert assessment["guarantee"]["hi"] >= ceiling - 1e-8
    lo, hi = matrix["value"]["lo"], matrix["value"]["hi"]
    if side:
        lo, hi = 1 - hi, 1 - lo
    expected = (0, 0) if len(p) == 1 else (max(0, lo - ceiling), max(0, hi - floor))
    assert assessment["loss"]["lo"] <= expected[0] + 1e-8
    assert assessment["loss"]["hi"] >= expected[1] - 1e-8


def summarize(rows, arm):
    groups = collections.defaultdict(list)
    counts = collections.Counter()
    for row in rows:
        value = row.get(arm)
        if value is None:
            counts["missing"] += 1
            groups[row["game"]].append({"lo": 0.0, "hi": 1.0})
            continue
        loss = value["loss"]
        assert -1e-9 <= loss["lo"] <= loss["hi"] + 1e-9 <= 1 + 2e-9
        groups[row["game"]].append(loss)
        counts["near_optimal" if loss["hi"] <= .0200000001 else "proven_loss" if loss["lo"] > .0200000001 else "unresolved"] += 1
    if not groups:
        return {"counts": dict(counts), "source_games": 0, "mean_loss_bounds": None}
    interval = {key: statistics.mean(statistics.mean(x[key] for x in losses) for losses in groups.values()) for key in ("lo", "hi")}
    worst = {key: max(loss[key] for losses in groups.values() for loss in losses) for key in ("lo", "hi")}
    return {"counts": dict(counts), "source_games": len(groups), "mean_loss_bounds": interval, "max_loss_bounds": worst}


def report(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    positions = [json.loads(p.read_text()) for p in sorted(directory.glob("game-*.json"))]
    assert len(positions) == manifest["expected_positions"], "incomplete run"
    assert len({(r["game"], r["decision"]) for r in positions}) == len(positions)
    for position in positions:
        matrix = position["reference"]
        assert len(matrix["cells"]) == len(matrix["actions"][0]) * len(matrix["actions"][1])
        for endpoint in position["endpoints"]:
            for arm in ("archived", "uniform", "max_damage", "oracle"):
                verify_assessment(matrix, endpoint["side"], endpoint[arm])
            for entry in endpoint["pure_actions"]:
                verify_assessment(matrix, endpoint["side"], entry)
            for entry in endpoint.get("budget_ladder", []):
                verify_assessment(matrix, endpoint["side"], entry["empirical_mixture"])
    endpoints = [{"game": position["game"], "decision": position["decision"], **endpoint} for position in positions for endpoint in position["endpoints"]]
    eligible = [r for r in endpoints if r["has_good_action"] and r["has_bad_action"]]
    arms = ["archived", "uniform", "max_damage", "oracle"]
    subsets = {
        "all": endpoints,
        "nonforced": [r for r in endpoints if not r["forced"]],
        "discriminating": eligible,
        "all_actions_good": [r for r in endpoints if r["all_actions_good"]],
    }
    ladder = {}
    for budget in manifest["protocol"]["iterations"]:
        rows = []
        for row in eligible:
            entry = next((r for r in row.get("budget_ladder", []) if r["iterations"] == budget), None)
            if entry:
                rows.append({**row, "current": entry["empirical_mixture"]})
        ladder[str(budget)] = summarize(rows, "current")
    failures = [{"game":r["game"],"decision":r["decision"],"side":r["side"],"choice":r["archived_choice"],
                 "loss":r["archived"]["loss"],"actions":r["actions"],"pure_losses":[a["loss"] for a in r["pure_actions"]]}
                for r in endpoints if r["archived"]["loss"]["lo"] > .0200000001]
    return {
        "schema": "nc2000-policy-gap-report-v1",
        "complete": True,
        "independent_matrix_interval_checks": "passed",
        "positions": len(positions),
        "endpoints": len(endpoints),
        "source_games_with_samples": len({r["game"] for r in endpoints}),
        "source_games_without_samples": manifest["source_games"] - len({r["game"] for r in endpoints}),
        "forced": sum(r["forced"] for r in endpoints),
        "all_actions_good": sum(r["all_actions_good"] for r in endpoints),
        "discriminating_endpoints": len(eligible),
        "reference_stops": dict(collections.Counter(r["reference"]["stop"] for r in positions)),
        "reference_seconds": sum(r["reference"]["elapsed_ms"] for r in positions) / 1000,
        "reference_runs": sum(r["reference"]["runs"] for r in positions),
        "subsets": {name: {arm: summarize(rows, arm) for arm in arms} for name, rows in subsets.items()},
        "budget_ladder_empirical_only": ladder,
        "proven_archived_losses": sorted(failures, key=lambda r: -r["loss"]["lo"]),
        "manifest": manifest,
        "interpretation": "Intervals bound one-decision full-information loss on this fixed corpus, with optimal continuation. Means weight source games equally within each subset. These are solver bounds, not population confidence intervals. Budget ladders describe empirical mixtures of four search seeds on oracle-selected discriminating endpoints; finite-seed sampling error is not included. Unresolved cases remain in all-case mean bounds. No full-game, blind, whole-policy or deployment claim.",
    }, positions


def percent(bounds):
    if bounds is None:
        return "未測定"
    return f'{100*bounds["lo"]:.2f}–{100*bounds["hi"]:.2f}'


def document(result):
    labels = {"archived": "既存10秒botの保存済み着手", "uniform": "合法手を一様ランダム", "max_damage": "探索なし・最大ダメージ", "oracle": "ソルバの混合方策"}
    tables = []
    for subset, title in [("all", f'全{result["endpoints"]}判断（未解決を含む）'), ("discriminating", "良い手・悪い手の両方を証明できた判断")]:
        rows = []
        for arm, summary in result["subsets"][subset].items():
            c = summary["counts"]
            rows.append(f'<tr><td>{labels[arm]}</td><td>{c.get("near_optimal",0)}</td><td>{c.get("proven_loss",0)}</td><td>{c.get("unresolved",0)+c.get("missing",0)}</td><td>{percent(summary["mean_loss_bounds"])}</td></tr>')
        tables.append(f'<h2>{title}</h2><div class="scroll"><table><thead><tr><th>方策</th><th>損失≤2pt</th><th>損失&gt;2pt</th><th>未解決</th><th>平均損失の保証区間（pt）</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    ladder = ''.join(f'<tr><td>{int(b):,}</td><td>{percent(s["mean_loss_bounds"])}</td><td>{s["counts"].get("near_optimal",0)}</td><td>{s["counts"].get("proven_loss",0)}</td></tr>' for b,s in result["budget_ladder_empirical_only"].items())
    failures = ''.join(f'<tr><td>{r["game"]} / {r["decision"]} / {r["side"]}</td><td>{html.escape(r["choice"])}</td><td>{percent(r["loss"])}</td></tr>' for r in result["proven_archived_losses"])
    all_counts = result["subsets"]["all"]["archived"]["counts"]
    return f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>nc2000 終盤の最適値との差</title>
<style>body{{font:17px/1.8 system-ui,sans-serif;background:#f7f5ef;color:#292820;max-width:1050px;margin:32px auto;padding:0 20px}}h1{{font-size:1.7em}}h2{{font-size:1.2em;margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:.9em}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ccc6b6}}th{{background:#eee9dc}}.scroll{{overflow-x:auto}}.lead{{padding:18px;background:#e7ede2;border-left:5px solid #536c44}}a{{color:#4e643e}}small{{color:#625f55}}</style>
<h1>終盤で、botは最適値からどれだけ落とすか</h1>
<p class="lead">保存済み{result["endpoints"]}判断のうち、{all_counts.get("near_optimal",0)}件は最適値から2ポイント以内、{all_counts.get("proven_loss",0)}件は2ポイント超の損失を証明。{all_counts.get("unresolved",0)}件は判定不能。</p>
<p>測定対象は全情報の一判断。自分の着手分布に対して相手が最善応手を選び、その後は双方最適に指す場合の、勝ち1・引き分け0.5・負け0の期待得点の損失です。保存済み着手は実際に選ばれた一手を固定した評価で、当時の乱数全体の方策への最適応手ではありません。対戦全体の「理論限界の何％」ではありません。</p>
<p>{result["positions"]}局面・{result["source_games_with_samples"]}元対局。標本のない元対局は{result["source_games_without_samples"]}局。強制手は{result["forced"]}判断、全合法手が2ポイント以内の判断は{result["all_actions_good"]}件。良い手と悪い手を区別できたのは{result["discriminating_endpoints"]}判断です。</p>
{''.join(tables)}
<p>平均は元対局ごとに同じ重み。区間はソルバが保証する数値範囲であり、母集団の95%信頼区間ではありません。解けなかった局面も平均区間に含めています。ソルバの方策も未解決なら天井達成とは数えません。</p>
<h2>現在の探索予算を増やすと変わるか</h2>
<p>上の「良い手・悪い手の両方を証明できた判断」だけで、現行の全情報SkuctSearch（c=0.4）を4乱数seedで計測。各予算は同じ探索の続きです。表は4seedの着手頻度から作った経験分布の損失で、無限seedでの実際の方策との差は保証していません。</p>
<div class="scroll"><table><tr><th>探索回数</th><th>経験分布の平均損失区間（pt）</th><th>≤2pt</th><th>&gt;2pt</th></tr>{ladder}</table></div>
<h2>損失が証明された保存済み着手</h2><div class="scroll"><table><tr><th>元対局 / 判断番号 / side</th><th>着手</th><th>損失区間（pt）</th></tr>{failures or '<tr><td colspan="3">該当なし</td></tr>'}</table></div>
<h2>再現と計測範囲</h2><p>局面価値が収束しても、着手損失の区間が広ければ計測を続けます。未評価の応手と乱数枝は区間のまま保持。混合方策にはその分布に対する一つの応手を使い、抽選結果を見てから相手が手を変える評価にはしません。小さなゲームの全列挙と独立な行列計算で検証しています。</p>
<p>1局面の参照予算は{result["manifest"]["protocol"]["reference_ms"]/1000:g}秒、ノード上限{result["manifest"]["protocol"]["node_budget"]:,}。終了理由：{html.escape(json.dumps(result["reference_stops"],ensure_ascii=False))}。</p>
<p><a href="summary.json">集計JSON</a> · <a href="records.json.gz">全局面の証明区間・着手記録</a> · <a href="protocol.json">固定した計測条件</a></p><small>公開botの設定は変更していません。2026-10-05</small></html>'''


if __name__ == "__main__":
    directory, output = map(pathlib.Path, sys.argv[1:])
    result, records = report(directory)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == result["manifest"]["protocol"]
    else:
        protocol_path.write_text(json.dumps(result["manifest"]["protocol"], indent=2) + "\n")
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
    result["records_sha256"] = hashlib.sha256(payload).hexdigest()
    with (output / "records.json.gz").open("wb") as file:
        with gzip.GzipFile(fileobj=file, mode="wb", mtime=0) as compressed:
            compressed.write(payload)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "index.html").write_text(document(result))
    print(json.dumps({k: result[k] for k in ("positions", "endpoints", "subsets", "budget_ladder_empirical_only")}, indent=2))
