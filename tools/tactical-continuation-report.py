import collections
import gzip
import hashlib
import html
import json
import pathlib
import sys
import subprocess


def read(path):
    return json.loads(path.read_text())


def report(directory, output):
    manifest = read(directory / "manifest.json")
    protocol = manifest["protocol"]
    rows = [read(p) for p in sorted(directory.glob("*-t*-a*.json"))]
    roots = [read(p) for p in sorted(directory.glob("*-roots.json"))]
    assert len(rows) == sum(len(case["arms"]) for case in protocol["cases"]) * protocol["trials"]
    assert len(roots) == len(protocol["cases"])
    verification = subprocess.run(["target/release/examples/tactical_continuation_verify", str(directory)], check=True, capture_output=True, text=True)
    replay = json.loads(verification.stdout)
    assert replay["verified_games"] == len(rows)
    cases = []
    for case in protocol["cases"]:
        own = [row for row in rows if row["case"] == case["id"]]
        pairs = []
        for trial in range(protocol["trials"]):
            arms = {r["arm"]: r for r in own if r["trial"] == trial}
            assert set(arms) == set(case["arms"])
            assert len({r["battle_seed"] for r in arms.values()}) == 1
            assert len({r["decisions"][0]["key"] for r in arms.values()}) == 1
            assert len({r["decisions"][0]["choices"][1-case["side"]] for r in arms.values()}) == 1
            for action, row in arms.items():
                assert row["decisions"][0]["choices"][case["side"]] == action
                assert row["score"] in (None, 0, .5, 1)
            a, b = [arms[action]["score"] for action in case["arms"]]
            pairs.append({"trial": trial, "baseline": a, "alternative": b, "delta": None if a is None or b is None else b-a})
        root = next(r for r in roots if r["case"] == case["id"])
        ladder = {}
        for budget in protocol["root_budgets"]:
            rr = [r for r in root["roots"] if r.get("iterations") == budget]
            assert len(rr) == protocol["root_seeds"]
            ladder[str(budget)] = dict(collections.Counter(r["choice"] for r in rr))
        diffs = [p["delta"] for p in pairs]
        bounds = {"lo": sum(-1 if d is None else d for d in diffs)/len(diffs), "hi": sum(1 if d is None else d for d in diffs)/len(diffs)}
        cases.append({"id": case["id"], "arms": case["arms"], "pairs": pairs,
                      "arm_scores": {arm: [r["score"] for r in own if r["arm"] == arm] for arm in case["arms"]},
                      "paired_delta_bounds": bounds, "root_ladder": ladder,
                      "archived_reproduced_choice": root["roots"][0]["choice"]})
    result = {"schema": "nc2000-tactical-continuation-report-v1", "cases": cases, "games": len(rows),
              "terminal_games": sum(r["terminal"] for r in rows), "manifest": manifest, "replay_verification": replay,
              "checks": {"paired_future_seeds": True, "paired_root_state_keys": True, "same_uninformed_first_reply": True, "forced_first_actions": True, "complete_trials_and_root_seeds": True},
              "limits": ["Eight fresh paired trials per selected state are exploratory; no confidence or noninferiority claim.",
                         "Delta bounds only retain censored outcomes; they are not population confidence intervals.",
                         "Subsequent blind policies are fixed at 27,000 iterations, not optimal continuations.",
                         "Source games and cases were selected before these trial outcomes, but after reviewing the exposure games.",
                         "The actual hidden opponent team is fixed across trials; this is conditional on that team's truth."]}
    payload = json.dumps({"games": rows, "roots": roots}, ensure_ascii=False).encode()
    result["records_sha256"] = hashlib.sha256(payload).hexdigest()
    (output / "continuation-records.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    (output / "continuation-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    labels = {"alakazam-recover": "回復 / 攻撃", "baton-recipient": "ラプラス / ガラガラ", "encore-opportunity": "攻撃 / アンコール"}
    trial_count = protocol["trials"]
    score_text = lambda c: " / ".join(f"{sum(v for v in scores if v is not None):g}点（{len(scores)}局、未終局{scores.count(None)}）" for scores in c["arm_scores"].values())
    table = "".join(f'<tr><td>{labels.get(c["id"], c["id"])}</td><td>{score_text(c)}</td><td>{c["paired_delta_bounds"]["lo"]:+.3f}〜{c["paired_delta_bounds"]["hi"]:+.3f}</td></tr>' for c in cases)
    details = "".join(f'<p>{c["id"]}：{html.escape(str(c["root_ladder"]))}</p>' for c in cases)
    verdict = "この小標本では、代案が保存手を上回った局面はありません。最適性や優劣の確定はできません。" if all(c["paired_delta_bounds"]["hi"] <= 0 for c in cases) else "代案が上回る候補がありますが、独立した追加試験で確認が必要です。"
    appendix = f'''<section id="continuations"><h2>候補手を変えた継続対戦</h2><p><strong>{verdict}</strong></p>
<p>保存時と同じ観測履歴・探索seedから元の着手を再現した後、未来の対戦乱数を振り直し、最初の手だけを変えて双方の通常blind探索で終局まで続けました。各局面{trial_count}組、{protocol["iterations"]:,}探索。相手の実際の選出・セットは元対局に固定しています。</p>
<p>表の各局面で、保存手と代案を比較。全{len(rows)}対局中{result["terminal_games"]}対局が終局しました。{trial_count}組は探索用の小標本で、優劣確定や非劣性の証明ではありません。表の差は代案−保存手です。</p>
<div class="scroll"><table><tr><th>局面</th><th>保存手 / 代案の得点</th><th>平均得点差の観測範囲</th></tr>{table}</table></div>
<p>別seedの探索予算ラダーは以下の通り。探索自身の好みであり、手の正しさの証明ではありません。</p>{details}
<p><a href="continuation-summary.json">継続対戦集計</a> · <a href="continuation-records.json.gz">全試行・探索分布</a> · <a href="continuation-protocol.json">継続対戦条件</a></p></section>'''
    page = output / "index.html"
    document = page.read_text()
    if '<section id="continuations">' in document:
        start = document.index('<section id="continuations">')
        end = document.index('</section>', start) + len('</section>')
        document = document[:start] + document[end:]
    page.write_text(document.replace('</h1>', '</h1>' + appendix, 1))
    return result


if __name__ == "__main__":
    directory, output = map(pathlib.Path, sys.argv[1:])
    result = report(directory, output)
    print(json.dumps({key: result[key] for key in ["games", "terminal_games", "cases"]}, indent=2))
