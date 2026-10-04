import collections
import gzip
import hashlib
import html
import importlib.util
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAMILIES = {
    "healing": {"rest", "recover", "milkdrink", "softboiled", "moonlight", "synthesis", "morningsun", "painsplit"},
    "setup": {"curse", "swordsdance", "bellydrum", "agility", "growth", "amnesia", "doubleteam", "minimize"},
    "explosion": {"explosion", "selfdestruct"},
    "perish": {"perishsong"},
    "trap": {"meanlook", "spiderweb"},
    "bond": {"destinybond"},
    "encore": {"encore"},
    "retaliation": {"counter", "mirrorcoat"},
    "toxic": {"toxic"},
    "batonpass": {"batonpass"},
    "spikes": {"spikes"},
    "screens": {"reflect", "lightscreen"},
    "substitute": {"substitute"},
}
LABELS = dict(healing="回復", setup="積み", explosion="爆発", perish="滅び", trap="黒い眼差し等", bond="道連れ", encore="アンコール", retaliation="カウンター等", toxic="毒々", batonpass="バトン", spikes="まきびし", screens="壁", substitute="身代わり", replacement="死に出し")


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def key(row):
    return row["game"], row["decision"]


def normalize(move):
    return "".join(c.lower() for c in move if c.isalnum())


def intersect(a, b):
    result = {"lo": max(a["lo"], b["lo"]), "hi": min(a["hi"], b["hi"])}
    assert result["lo"] <= result["hi"], (a, b)
    return result


def assess(matrix, side, policy):
    rows, cols = map(len, matrix["actions"])
    assert len(policy) == (rows if side == 0 else cols)
    assert min(policy) >= 0 and abs(sum(policy) - 1) < 1e-9
    replies = []
    for reply in range(cols if side == 0 else rows):
        values = []
        for action, probability in enumerate(policy):
            cell = matrix["cells"][action * cols + reply if side == 0 else reply * cols + action]
            interval = (cell["lo"], cell["hi"]) if side == 0 else (1 - cell["hi"], 1 - cell["lo"])
            values.append(tuple(probability * x for x in interval))
        replies.append(tuple(sum(v[i] for v in values) for i in (0, 1)))
    guarantee = {"lo": min(v[0] for v in replies), "hi": min(v[1] for v in replies)}
    lo, hi = matrix["value"]["lo"], matrix["value"]["hi"]
    if side:
        lo, hi = 1 - hi, 1 - lo
    loss = {"lo": max(0, lo - guarantee["hi"] - 1e-10), "hi": min(1, max(0, hi - guarantee["lo"]) + 1e-10)}
    if len(policy) == 1:
        loss = {"lo": 0, "hi": 0}
    return {"policy": policy, "guarantee": guarantee, "loss": loss}


def verdict(loss):
    return "near_optimal" if loss["hi"] <= .0200000001 else "proven_loss" if loss["lo"] > .0200000001 else "unresolved"


def calibrate_assessor():
    matching = {"actions": [[0, 1], [0, 1]], "value": {"lo": .5, "hi": .5}, "cells": [{"lo": x, "hi": x} for x in [1, 0, 0, 1]]}
    for side in (0, 1):
        assert assess(matching, side, [.5, .5])["loss"]["hi"] < 1e-8
        loss = assess(matching, side, [1, 0])["loss"]
        assert abs(loss["lo"] - .5) < 1e-8 and abs(loss["hi"] - .5) < 1e-8
    dominant = {**matching, "value": {"lo": 1, "hi": 1}, "cells": [{"lo": x, "hi": x} for x in [1, 1, 0, 0]]}
    for policy, expected in [([1, 0], 0), ([.5, .5], .5), ([0, 1], 1)]:
        loss = assess(dominant, 0, policy)["loss"]
        assert abs(loss["lo"] - expected) < 1e-8 and abs(loss["hi"] - expected) < 1e-8
    unknown = {**matching, "value": {"lo": 0, "hi": 1}, "cells": [{"lo": 0, "hi": 1}] * 4}
    assert assess(unknown, 0, [.5, .5])["loss"] == {"lo": 0, "hi": 1}
    forced = {"actions": [[0], [0, 1]], "value": {"lo": 0, "hi": 1}, "cells": [{"lo": 0, "hi": 1}] * 2}
    assert assess(forced, 0, [1])["loss"] == {"lo": 0, "hi": 0}
    assert assess(forced, 1, [.5, .5])["loss"] == {"lo": 0, "hi": 1}


def tags(snapshot):
    available = {m["id"] for side in snapshot["sides"] for p in side if not p["fainted"] for m in p["moves"] if m["pp"] > 0}
    result = [name for name, moves in FAMILIES.items() if available & moves]
    if snapshot["essence"]["requestState"] == "switch":
        result.append("replacement")
    return result


def build(scratch, output):
    calibrate_assessor()
    source = ROOT / "findings/policy-gap-v1"
    original = json.loads(gzip.decompress((source / "records.json.gz").read_bytes()))
    states = lines(scratch / "states.jsonl")
    snapshots = {key(r): r for r in states}
    assert len(snapshots) == len(states) == len(original) == 72
    protocol = read(output / "protocol.json")
    notes = read(output / "case-notes.json")
    selected = {(r["game"], r["decision"], r["side"]) for r in protocol["selection"]}
    assert selected == {(r["game"], r["decision"], r["side"]) for r in notes}
    probe_manifest = read(scratch / "search/manifest.json")
    assert probe_manifest["protocol"] == protocol
    probes = [read(p) for p in sorted((scratch / "search").glob("game-*.json"))]
    assert len(probes) == probe_manifest["expected_states"]
    probes = {key(r["snapshot"]): r for r in probes}
    assert set(probes) == {(g, d) for g, d, s in selected}
    reference_manifest = read(scratch / "reference/manifest.json")
    refined = [read(p) for p in sorted((scratch / "reference").glob("game-*.json"))]
    assert len(refined) == reference_manifest["expected_positions"] == 6
    refinement = {key(r): r for r in refined}
    assert len(refinement) == 6
    checker_spec = importlib.util.spec_from_file_location("policy_gap_report", ROOT / "tools/policy-gap-report.py")
    checker = importlib.util.module_from_spec(checker_spec)
    checker_spec.loader.exec_module(checker)
    for row in original + refined:
        for endpoint in row["endpoints"]:
            for arm in ("archived", "uniform", "max_damage", "oracle"):
                checker.verify_assessment(row["reference"], endpoint["side"], endpoint[arm])
            for pure in endpoint["pure_actions"]:
                checker.verify_assessment(row["reference"], endpoint["side"], pure)
    endpoints = []
    for row in original:
        snapshot = snapshots[key(row)]
        assert row["state_key"] == snapshot["state_key"]
        assert row["alive"] == [sum(not p["fainted"] for p in side) for side in snapshot["sides"]]
        matrix = row["reference"]
        if key(row) in refinement:
            fresh = refinement[key(row)]
            assert fresh["state_key"] == row["state_key"]
            assert fresh["reference"]["actions"] == matrix["actions"]
            matrix = {**matrix, "value": intersect(matrix["value"], fresh["reference"]["value"]), "cells": [intersect(a, b) for a, b in zip(matrix["cells"], fresh["reference"]["cells"]) ]}
        for endpoint in row["endpoints"]:
            side = endpoint["side"]
            assert endpoint["actions"] == snapshot["actions"][side]
            updated = assess(matrix, side, endpoint["archived"]["policy"])
            entry = {"game": row["game"], "decision": row["decision"], "side": side, "alive": row["alive"], "turn": row["turn"],
                     "forced": endpoint["forced"], "tags": tags(snapshot), "stop": row["reference"]["stop"],
                     "actions": endpoint["actions"], "archived_choice": endpoint["archived_choice"], "archived_iterations": endpoint["archived_iterations"],
                     "old_loss": endpoint["archived"]["loss"], "refined": key(row) in refinement, "updated": updated,
                     "controls": {arm: assess(matrix, side, endpoint[arm]["policy"]) for arm in ("uniform", "max_damage")},
                     "pure_actions": [assess(matrix, side, p["policy"]) for p in endpoint["pure_actions"]], "ladder": []}
            if (row["game"], row["decision"], side) in selected:
                probe = probes[key(row)]
                assert probe["snapshot"]["state_key"] == row["state_key"]
                assert len(probe["search_records"]) == len(protocol["iterations"]) * protocol["search_seeds"]
                for budget in protocol["iterations"]:
                    records = [r for r in probe["search_records"] if r["iterations"] == budget]
                    assert {r["seed_index"] for r in records} == set(range(protocol["search_seeds"]))
                    counts = collections.Counter(r["sides"][side]["choice"] for r in records)
                    policy = [counts[a] / len(records) for a in endpoint["actions"]]
                    top_shares = []
                    flat = 0
                    for r in records:
                        search = r["sides"][side]
                        assert search["actions"] == endpoint["actions"]
                        assert sum(search["visits"]) == budget
                        top_shares.append(max(search["visits"]) / budget)
                        flat += max(search["means"]) - min(search["means"]) < 1e-9
                    entry["ladder"].append({"iterations": budget, "choices": dict(counts), "empirical_mixture": assess(matrix, side, policy),
                                            "top_visit_share_range": [min(top_shares), max(top_shares)], "flat_roots": flat,
                                            "elapsed_ms_median": statistics.median(r["elapsed_ms"] for r in records)})
                by_seed = collections.defaultdict(dict)
                for r in probe["search_records"]:
                    by_seed[r["seed_index"]][r["iterations"]] = r["sides"][side]["choice"]
                entry["low_high_choice_changes"] = sum(p[protocol["iterations"][0]] != p[protocol["iterations"][-1]] for p in by_seed.values())
            endpoints.append(entry)
    unresolved = [r for r in endpoints if verdict(r["old_loss"]) == "unresolved"]
    assert len(endpoints) == 115 and len(unresolved) == 59
    cohorts = {}
    for name, rows in [("unresolved", unresolved), ("resolved_nonforced", [r for r in endpoints if not r["forced"] and verdict(r["old_loss"]) == "near_optimal"])]:
        cohorts[name] = {"endpoints": len(rows), "states": len({key(r) for r in rows}), "games": len({r["game"] for r in rows}),
                         "tags": dict(collections.Counter(t for r in rows for t in r["tags"])),
                         "alive": dict(collections.Counter("v".join(map(str, r["alive"])) for r in rows)),
                         "stops": dict(collections.Counter(r["stop"] for r in rows))}
    coverage = lines(scratch / "coverage.jsonl")
    corpus = [json.loads(line) for line in gzip.decompress((source / "corpus.jsonl.gz").read_bytes()).decode().splitlines()]
    assert {r["game"] for r in coverage} == {r["game"] for r in corpus} and len(coverage) == 24
    assert all(len(side) == 3 for r in coverage for side in r["picked"])
    tactic_coverage = {}
    for family, moves in FAMILIES.items():
        tactic_coverage[family] = {
            "registered_games": [g["game"] for g in corpus if any(moves & {normalize(m) for m in p["moves"]} for side in g["teams"] for p in side)],
            "picked_games": [g["game"] for g in coverage if any(moves & set(p["moves"]) for side in g["picked"] for p in side)],
            "chosen_games": [g["game"] for g in coverage if any(c["action"].split()[1] in moves for c in g["chosen_moves"])],
            "live_sample_states": [list(key(s)) for s in states if family in tags(s)],
        }
    selected_rows = [r for r in endpoints if r["ladder"]]
    summary = {"schema": "nc2000-headroom-cases-report-v1", "cohorts": cohorts, "selected_endpoints": len(selected_rows),
               "selected_games": len({r["game"] for r in selected_rows}), "tactic_coverage": tactic_coverage,
               "updated_archived_counts": dict(collections.Counter(verdict(r["updated"]["loss"]) for r in endpoints)),
               "refinement_transitions": [{"game": r["game"], "decision": r["decision"], "side": r["side"], "old": r["old_loss"], "new": r["updated"]["loss"]} for r in unresolved if r["refined"]],
               "proven_archived_losses": [r for r in endpoints if verdict(r["updated"]["loss"]) == "proven_loss"],
               "selected_high_budget_counts": dict(collections.Counter(verdict(r["ladder"][-1]["empirical_mixture"]["loss"]) for r in selected_rows)),
               "low_high_choice_changes": sum(r["low_high_choice_changes"] for r in selected_rows), "paired_seed_decisions": len(selected_rows) * protocol["search_seeds"],
               "refinement_seconds": sum(r["reference"]["elapsed_ms"] for r in refined) / 1000,
               "refinement_stops": dict(collections.Counter(r["reference"]["stop"] for r in refined)),
               "controls": {arm: dict(collections.Counter(verdict(r["controls"][arm]["loss"]) for r in endpoints)) for arm in ("uniform", "max_damage")},
               "checks": {"all_replayed_keys_match": True, "all_reference_assessments_independently_checked": True,
                          "reference_intersections_nonempty": True, "all_probe_actions_and_visits_checked": True,
                          "known_matrix_ceiling_uniform_ablated_controls": True,
                          "hidden_mixture_and_unknown_forced_controls": True},
               "limits": ["Exploratory, selected cases; no population frequency or overall strength estimate.", "Full-information one-decision oracle; subsequent optimal play is assumed.", "Four-seed empirical mixtures omit sampling uncertainty about the bot policy.", "Search root means, action changes, and the presence of a move are not proof of a policy loss.", "Concurrent elapsed times are not device performance or fixed 10/60-second comparisons.", "Coverage records chosen moves, including moves that may fail; it is not successful execution frequency."],
               "probe_manifest": probe_manifest, "reference_manifest": reference_manifest}
    archive = {"snapshots": states, "coverage": coverage, "probes": list(probes.values()), "references": refined, "endpoints": endpoints}
    payload = json.dumps(archive, ensure_ascii=False, sort_keys=True).encode()
    summary["records_sha256"] = hashlib.sha256(payload).hexdigest()
    summary["inputs_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source / "records.json.gz", source / "corpus.jsonl.gz", output / "protocol.json", output / "case-notes.json", output / "reference-protocol.json"]}
    (output / "records.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary, endpoints, notes, snapshots


def percent(b):
    return f'{100*b["lo"]:.2f}–{100*b["hi"]:.2f}pt'


def document(result, endpoints, notes, snapshots):
    esc = html.escape
    by_key = {(r["game"], r["decision"], r["side"]): r for r in endpoints}
    cases = []
    for note in notes:
        row = by_key[note["game"], note["decision"], note["side"]]
        snapshot = snapshots[key(row)]
        ladder = "".join(f'<tr><td>{r["iterations"]:,}</td><td>{esc(str(r["choices"]))}</td><td>{percent(r["empirical_mixture"]["loss"])}</td><td>{r["top_visit_share_range"][0]:.1%}–{r["top_visit_share_range"][1]:.1%}</td></tr>' for r in row["ladder"])
        mons = []
        for side, party in enumerate(snapshot["sides"]):
            for p in party:
                if p["fainted"]:
                    continue
                moves = " / ".join(f'{m["id"]} {m["pp"]}/{m["maxpp"]}' for m in p["moves"])
                boosts = " ".join(f'{name}{value:+}' for name, value in zip(["攻", "防", "特攻", "特防", "速", "命中", "回避"], p["boosts"]) if value)
                mons.append(f'<li>P{side+1} {"場" if p["active"] else "控え"}：{esc(p["species"])} Lv{p["level"]} {p["hp"]}/{p["maxhp"]} {esc(p["status"])} {esc(p["item"] or "")} {boosts}<br><small>{esc(moves)}</small></li>')
        cases.append(f'<details><summary>G{row["game"]} D{row["decision"]} P{row["side"]+1} — {esc(note["label"])} / 保存手 {esc(row["archived_choice"])} / 損失 {percent(row["updated"]["loss"])}</summary><p>{esc(note["observation"])}</p><p>{esc(note["question"])}</p><ul>{"".join(mons)}</ul><div class="scroll"><table><tr><th>探索回数</th><th>4seedの着手</th><th>経験混合の損失区間</th><th>各探索の最大visit比率</th></tr>{ladder}</table></div></details>')
    cohort = result["cohorts"]
    tag_rows = "".join(f'<tr><td>{LABELS[t]}</td><td>{cohort["unresolved"]["tags"].get(t,0)}/59</td><td>{cohort["resolved_nonforced"]["tags"].get(t,0)}/34</td></tr>' for t in ["healing", "setup", "explosion", "replacement", "substitute"])
    coverage_rows = "".join(f'<tr><td>{LABELS[t]}</td><td>{len(r["registered_games"])}</td><td>{len(r["picked_games"])}</td><td>{len(r["chosen_games"])}</td><td>{len(r["live_sample_states"])}</td></tr>' for t, r in result["tactic_coverage"].items() if t in ["perish", "trap", "encore", "retaliation", "toxic", "batonpass", "spikes"])
    refine_rows = "".join(f'<tr><td>G{r["game"]} D{r["decision"]} P{r["side"]+1}</td><td>{percent(r["old"])}</td><td>{percent(r["new"])}</td></tr>' for r in result["refinement_transitions"])
    c = result["updated_archived_counts"]
    return f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>nc2000 強化余地・未解決局面の監査</title>
<style>body{{font:17px/1.8 system-ui,sans-serif;background:#f7f5ef;color:#292820;max-width:1050px;margin:32px auto;padding:0 20px}}h1{{font-size:1.7em}}h2{{font-size:1.2em;margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:.9em}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ccc6b6}}th{{background:#eee9dc}}.scroll{{overflow-x:auto}}.lead{{padding:18px;background:#e7ede2;border-left:5px solid #536c44}}a{{color:#4e643e}}small{{color:#625f55}}details{{border-top:1px solid #ccc6b6;padding:14px 0}}summary{{cursor:pointer}}</style>
<h1>強化余地：未解決59判断の中身</h1>
<p class="lead">未解決59判断は36局面・15対局。33判断が2対2で、この標本の2対2は一つも近最適と判定できていません。回復技の有無だけでは、解決済みとの違いを説明できません。</p>
<p>凍結した24自己対戦から72終盤・115判断を抽出した前回監査を再調査。全36局面を読み、15対局を含む20判断を詳しく確認しました。強化余地の探索用標本であり、実戦頻度を推定する無作為標本ではありません。</p>
<h2>参照解が届かない領域</h2><p>未解決59判断のうちノード上限46、時間上限11、実行回数上限2。1対1の未解決は2判断、2対2は33判断です。以下は生存個体がPPを残して技を持つことを数え、実際にその技を使ったことや誤りの原因を意味しません。</p>
<table><tr><th>局面の特徴</th><th>未解決</th><th>解決済み・非強制</th></tr>{tag_rows}</table>
<h2>追加参照：6局面</h2><p>1局面180秒・60万ノード・800万エンジン実行の上限で再計算し、前回と今回の保証区間の共通部分を採用。各局面の経過時間の合計は{result["refinement_seconds"]:.0f}秒（並行実行）。全115判断の保存手は、近最適{c.get("near_optimal",0)}、損失を証明{c.get("proven_loss",0)}、未解決{c.get("unresolved",0)}。近最適は損失上限2pt以内。</p>
<div class="scroll"><table><tr><th>判断</th><th>前回の損失区間</th><th>追加後の損失区間</th></tr>{refine_rows}</table></div>
<p>損失は完全情報で最初の着手分布を固定し、その後は双方が最適に指す場合の期待得点差です。勝ち1・引き分け0.5・負け0。保証区間は統計的信頼区間ではなく、未知の枝を含むソルバの上下界です。</p>
<h2>標本から抜けている戦術</h2><p>登録チームに技がある対局数→実際の選出3匹に含まれる対局数→着手された対局数→生存して技PPが残る終盤標本数。最初の3列の分母は24対局、最後は72局面。着手には失敗した技も含みます。</p>
<div class="scroll"><table><tr><th>戦術</th><th>登録</th><th>選出</th><th>着手</th><th>終盤標本</th></tr>{coverage_rows}</table></div>
<p>滅び・アンコール・毒々・バトンは、この24対局では選出段階で露出が消えています。この終盤監査では、それらへの対応力を判断できません。Perishは別の実戦再構成・改善試験があるため、その成果を再実験せず参照します。人工的な待機・道連れ反例も既存ですが、自然な対局の出現頻度を保証しません。</p>
<h2>20判断の具体例</h2><p>各判断で独立4seed、27,000→108,000→432,000回の同一探索を継続。低予算と高予算の着手が変わったのは{result["low_high_choice_changes"]}/{result["paired_seed_decisions"]}。着手変更は改善の証拠ではありません。最大visit比率は各探索の集中度で、実際の着手確率ではありません。経験混合は4seedの最終着手から計算し、seed母集団への誤差保証はありません。時刻予算10秒/60秒の比較でもありません。</p>
{''.join(cases)}
<h2>次に投資する対象</h2><p>2対2の未知区間を誤り扱いして評価関数を調整する根拠はありません。まず、登録チームを保ったままアンコール・毒々・バトンの使い手を選出に含め、合法な手順で対戦する小規模な戦術別コーパスを作るのが未被覆部分への直接の測定になります。勝ち筋が具体的に再現したケースに限り、独立seedと対戦全体の継続で改善量を確認します。</p>
<p>完全情報の参照解が届く終盤と、blindでの未知セット対応は別の測定対象です。現在の記録から全体の最適性や上級者に対する勝率は主張しません。製品botは変更していません。</p>
<p><a href="summary.json">集計と検証JSON</a> · <a href="records.json.gz">再構成局面・探索・追加参照</a> · <a href="protocol.json">探索条件</a> · <a href="reference-protocol.json">追加参照条件</a> · <a href="case-notes.json">20判断の観察</a></p><small>2026-10-05</small></html>'''


if __name__ == "__main__":
    scratch, output = [pathlib.Path(p).resolve() for p in sys.argv[1:]]
    result, endpoints, notes, snapshots = build(scratch, output)
    (output / "index.html").write_text(document(result, endpoints, notes, snapshots))
    print(json.dumps({k: result[k] for k in ["cohorts", "updated_archived_counts", "selected_high_budget_counts", "low_high_choice_changes", "refinement_stops", "refinement_transitions"]}, indent=2))
