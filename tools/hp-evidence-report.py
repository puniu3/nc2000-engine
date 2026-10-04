import collections
import gzip
import hashlib
import html
import json
import pathlib
import sys
import subprocess
import tempfile


def report(run, output):
    output.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads(pathlib.Path("findings/hp-evidence-v1/calibration.json").read_text())
    with tempfile.TemporaryDirectory(prefix="hp-calibration-", dir=run) as temporary:
        temp = pathlib.Path(temporary)
        for name, fixture in fixtures.items():
            (temp / (name + ".raw.log")).write_text(fixture["log"])
        subprocess.run(["python3", "tools/hp-evidence-census.py", str(temp), str(temp / "out")], check=True, capture_output=True)
        cases = json.loads((temp / "out/cases.json").read_text())
        for name, fixture in fixtures.items():
            actual = [c["allowed_types"] for c in cases if pathlib.Path(c["log"]).name == name + ".raw.log"]
            assert actual == fixture["expected_allowed_types"], name
    rows = [json.loads(line) for line in (run / "models.jsonl").read_text().splitlines()]
    roots = [json.loads(line) for line in (run / "probe.jsonl").read_text().splitlines()]
    census = json.loads((run / "census-summary.json").read_text())
    assert len(rows) == census["next_turn_cases"]
    valid = [r for r in rows if "skip" not in r]
    bad = [r for r in valid if r["contradictions"]]
    for row in valid:
        assert sum(row["model_types"].values()) == row["samples"]
        allowed = set(row["case"]["evidence"][0]["compatible_types"])
        for evidence in row["case"]["evidence"][1:]:
            allowed &= set(evidence["compatible_types"])
        assert allowed == set(row["case"]["allowed_types"])
        assert row["contradictions"] == sum(n for t, n in row["model_types"].items() if t not in allowed)
    assert all(r["contradictions"] == r["samples"] for r in bad)
    (output / "probe-protocol.json").write_bytes((run / "probe-protocol.json").read_bytes())
    probes = []
    for battle in sorted({r["battle"] for r in roots}):
        own = [r for r in roots if r["battle"] == battle]
        if any("skip" in r for r in own):
            probes.append({"battle": battle, "skip": own[0]["skip"]})
            continue
        seeds = sorted({r["seed"] for r in own})
        assert len(seeds) == 4 and len(own) == 8
        pairs = []
        for seed in seeds:
            arms = {r["arm"]: r for r in own if r["seed"] == seed}
            assert set(arms) == {"baseline", "known_type"}
            assert all(sum(a["visits"] for a in r["actions"]) == 27000 for r in arms.values())
            pairs.append({"seed": seed, **{arm: r["best"] for arm, r in arms.items()}})
        probes.append({"battle": battle, "changed": sum(p["baseline"] != p["known_type"] for p in pairs), "pairs": pairs,
                       "choices": {arm: dict(collections.Counter(r["best"] for r in own if r["arm"] == arm)) for arm in ["baseline", "known_type"]}})
    summary = {"schema": "nc2000-hp-evidence-v1", "census": census, "reconstructed": len(valid),
               "skips": dict(collections.Counter(r["skip"] for r in rows if "skip" in r)),
               "contradictory_cases": len(bad), "contradictory_games": len({r["case"]["battle"] for r in bad}),
               "singleton_contradictions": sum(len(r["case"]["allowed_types"]) == 1 for r in bad),
               "probes": probes, "product_changed": False, "parser_controls": {name: True for name in fixtures},
               "limits": ["Cases are selected by observed informative Hidden Power uses, not a representative population.",
                          "Reconstruction imputes unrevealed own sets and sometimes own selections; probes recover selection from full own-side log.",
                          "Effective type constraints use past public messages only and exclude games with type changes, Transform, or Foresight.",
                          "Sampling contradiction and changed choices do not establish a win-rate gain or action optimality.",
                          "Typed probe retains the assumed 70 power; actual Hidden Power power remains unknown."]}
    payload = json.dumps({"models": rows, "roots": roots}, ensure_ascii=False).encode()
    summary["records_sha256"] = hashlib.sha256(payload).hexdigest()
    (output / "records.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    labels = {10: "プテラ→ハガネール：岩", 346: "ハクリュー→ハピナス：格闘", 395: "フリーザー→カビゴン：格闘", 559: "ウインディ→カビゴン：格闘"}
    table = "".join(f'<tr><td>{labels.get(p["battle"], p["battle"])}</td><td>{html.escape(str(p.get("choices", {}).get("baseline", p.get("skip"))))}</td><td>{html.escape(str(p.get("choices", {}).get("known_type", "")))}</td><td>{p.get("changed", "—")}/4</td></tr>' for p in probes)
    cases = "".join(f'<tr><td>{r["case"]["battle"]} / T{r["case"]["turn"]}</td><td>{r["case"]["foe"]}</td><td>{html.escape(str(r["case"]["allowed_types"]))}</td><td>{html.escape(str(r["model_types"]))}</td></tr>' for r in bad)
    (output / "index.html").write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>めざめるパワーの観測情報</title>
<style>body{{font:17px/1.8 system-ui;background:#f7f5ef;color:#292820;max-width:1050px;margin:30px auto;padding:0 20px}}h1{{font-size:1.6em}}h2{{font-size:1.2em}}table{{border-collapse:collapse;width:100%;font-size:.9em}}td,th{{padding:10px;border-bottom:1px solid #ccc6b6;text-align:left}}.scroll{{overflow-x:auto}}a{{color:#4e643e}}</style>
<h1>公開された相性を、モデルが使えていない</h1>
<p>人間同士の570局を調査。「めざめるパワー」の効果抜群・いまひとつという表示からタイプ候補を絞った後、次の判断を{len(valid)}件再構成しました。{len(bad)}件（{summary["contradictory_games"]}局）で、補完した64標本すべてが既知の相性と矛盾しました。再構成できなかった{len(rows)-len(valid)}件は分母から除外しています。</p>
<p>ノーマル単タイプへの効果抜群なら格闘と分かりますが、補完は悪のままでした。このようにタイプが一意に決まる4局面で、公開情報を明示的に渡して判断を比較しました。27,000探索×4seed、各seedで通常／訂正の2通りです。</p>
<h2>判断への影響</h2><div class="scroll"><table><tr><th>過去の観測から確定したタイプ</th><th>通常</th><th>タイプ訂正</th><th>着手変更</th></tr>{table}</table></div>
<p>ハピナスとカビゴンでは4/4seedで判断が変わりました。これはモデルの欠陥と判断への影響の証拠で、勝率改善の証明ではありません。bot本体は未変更です。</p>
<h2>条件と限界</h2><p>タイプ変化・変身・見破るを含む3局を除外。相手のタイプ制約は判断時点より前の公開ログのみから導出しています。自分の未公開セットは資料から補完しているため、実対局の完全な再現ではありません。対戦相手の真のめざめるパワー威力も不明で、比較は補完された威力70を両側で保っています。</p>
<p>過去の相性表示を満たすモデルを作り、独立した継続対戦で改善を確認するのが次の工程です。</p>
<details><summary>矛盾した全{len(bad)}局面</summary><div class="scroll"><table><tr><th>匿名局番号 / 判断</th><th>相手</th><th>公開情報と両立するタイプ</th><th>モデル64標本</th></tr>{cases}</table></div></details>
<p><a href="summary.json">集計・限界</a> · <a href="records.json.gz">匿名化した観測制約と探索結果</a> · <a href="probe-protocol.json">固定した探索条件</a></p></html>''')
    return summary


if __name__ == "__main__":
    result = report(*map(pathlib.Path, sys.argv[1:]))
    print(json.dumps({k: result[k] for k in ["reconstructed", "contradictory_cases", "singleton_contradictions", "probes"]}, ensure_ascii=False, indent=2))
