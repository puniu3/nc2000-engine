import collections
import gzip
import hashlib
import html
import json
import pathlib
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def corpus(games, iterations):
    result = []
    for game in games:
        decisions = []
        classes = collections.defaultdict(list)
        for row in game["records"]:
            decisions.append({"key": row["state_key"], "choices": [row["actions"][s].index(row["choices"][s]) if row["choices"][s] else None for s in (0, 1)]})
            alive = tuple(side["pokemonLeft"] for side in row["state"]["sides"])
            if all(1 <= n <= 2 for n in alive):
                classes[alive].append(row["decision"])
        samples = sorted({i for group in classes.values() for i in [group[0], group[-1]]})
        result.append({"schema": "nc2000-selfplay-endgames-v1", "game": len(result), "seed": game["battle_seed"], "teams": game["teams"],
                       "iterations": iterations, "decisions": decisions, "samples": samples, "outcome": game["outcome"]})
    return result


def summarize_game(game):
    side = game["source_side"]
    relevant = {"encore": {"encore"}, "toxic": {"toxic"}, "batonpass": {"substitute", "batonpass"}}[game["family"]]
    result = {key: game[key] for key in ["block", "family", "opponent", "source_team", "source_side", "arm", "source_score", "terminal", "turns", "seconds"]}
    result["source_moves"] = dict(collections.Counter(r["choices"][side] for r in game["records"] if r["choices"][side] and r["choices"][side].startswith("move ")))
    result["overrides"] = [{"decision": r["decision"], "turn": r["state"]["turn"], "ordinary": r["ordinary"][side], "chosen": r["choices"][side]} for r in game["records"] if r["override_reason"] == "public_state_script" and r["ordinary"][side] != r["choices"][side]]
    events = []
    exposure = []
    for row in game["records"]:
        if not row["choices"][side] or row["choices"][side].startswith("team "):
            continue
        own = next((p for p in row["state"]["sides"][side]["pokemon"] if p["active"] and not p["fainted"]), None)
        if own and relevant & {m["id"] for m in own["moves"] if m["pp"] > 0}:
            model = row["context"][1-side]
            if model:
                exposure.append({"decision": row["decision"], "turn": row["state"]["turn"], "species": own["species"], "fallback": model["fallback"],
                                 "samples": len(model["models"]), "with_tactic": sum(game["family"] in m["moves"] for m in model["models"])})
        for line in row["log"]:
            if ("|-start|" in line and "|Encore" in line) or ("|-status|" in line and "|tox" in line) or ("|switch|" in line and "[from] Baton Pass" in line):
                event = {"decision": row["decision"], "turn": row["state"]["turn"], "line": line}
                if "[from] Baton Pass" in line:
                    event["line"] = "|".join(line.split("|")[:4]) + "|[from] Baton Pass"
                if event not in events:
                    events.append(event)
    result["tactic_events"] = events
    result["model_exposure"] = exposure
    return result


def report(directory, output):
    manifest = read(directory / "manifest.json")
    games = [read(p) for p in sorted(directory.glob("game-*.json"))]
    assert len(games) == manifest["expected_games"]
    grouped = collections.defaultdict(dict)
    for game in games:
        assert game["arm"] not in grouped[game["block"]]
        grouped[game["block"]][game["arm"]] = game
    paired = []
    for block, arms in grouped.items():
        assert set(arms) == {"ordinary", "specialist"}
        a, b = arms["ordinary"], arms["specialist"]
        for key in ["source_side", "family", "opponent", "teams", "battle_seed", "agent_seeds"]:
            assert a[key] == b[key]
        divergence = None
        for ra, rb in zip(a["records"], b["records"]):
            assert ra["state_key"] == rb["state_key"]
            assert ra["ordinary"] == rb["ordinary"]
            assert ra["context"] == rb["context"]
            if ra["choices"] != rb["choices"]:
                assert rb["override_reason"] == "public_state_script"
                assert ra["choices"][1-a["source_side"]] == rb["choices"][1-a["source_side"]]
                divergence = {"decision": ra["decision"], "turn": ra["state"]["turn"], "ordinary": ra["choices"], "specialist": rb["choices"]}
                break
        if divergence is None:
            assert len(a["records"]) == len(b["records"]) and a["source_score"] == b["source_score"]
        paired.append({"block": block, "family": a["family"], "opponent": a["opponent"], "ordinary": a["source_score"], "specialist": b["source_score"],
                       "delta": b["source_score"] - a["source_score"] if a["terminal"] and b["terminal"] else None, "first_divergence": divergence})
    output.mkdir(exist_ok=True, parents=True)
    records = corpus(games, manifest["protocol"]["iterations"])
    corpus_path = directory / "corpus.jsonl"
    payload = "".join(json.dumps(row) + "\n" for row in records).encode()
    corpus_path.write_bytes(payload)
    verify = subprocess.run(["target/release/examples/selfplay_endgames", "verify", str(corpus_path)], check=True, capture_output=True, text=True)
    summaries = [summarize_game(g) for g in games]
    summary = {"schema": "nc2000-tactical-exposure-report-v1", "games": len(games), "pairs": paired, "summaries": summaries,
               "terminal_games": sum(g["terminal"] for g in games), "capped_games": sum(not g["terminal"] for g in games),
               "replay_verification": json.loads(verify.stdout), "paired_prefix_verification": True,
               "manifest": manifest, "corpus_sha256": hashlib.sha256(payload).hexdigest(),
               "limits": [f"Exploratory {len(paired)} blocks on forced source leads, not a representative population.",
                          "Specialist is a simple public-state script, not an optimal opponent or a stronger oracle.",
                          "Same-seed outcomes nominate cases; fresh continuation seeds are required for a gain claim.",
                          "Native BlindAgent with truth-backed observer, not an end-to-end deployed ProtocolAgent test.",
                          "27,000 fixed iterations are not a 10-second native or 60-second oracle comparison."]}
    (output / "corpus.jsonl.gz").write_bytes(gzip.compress(payload, mtime=0))
    data = json.dumps(games, ensure_ascii=False).encode()
    (output / "records.json.gz").write_bytes(gzip.compress(data, mtime=0))
    summary["records_sha256"] = hashlib.sha256(data).hexdigest()
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary, games


def document(summary, games):
    labels = {"encore": "アンコール", "toxic": "毒々", "batonpass": "身代わり→バトン"}
    pairs = "".join(f'<tr><td>{labels[p["family"]]}</td><td>{p["opponent"]}</td><td>{p["ordinary"]}</td><td>{p["specialist"]}</td><td>{"同一" if p["first_divergence"] is None else "T"+str(p["first_divergence"]["turn"])}</td></tr>' for p in summary["pairs"])
    details = []
    prefix = summary["manifest"]["protocol"].get("prefix", [])
    followup = '<p><a href="encore/">合法な手順でアンコールの使用機会を作った追加試験</a></p>' if not prefix else ""
    prefix_note = "" if not prefix else f'<p>この追加試験は合法な手順で使用機会を作ったものです。両者共通の固定手順：{html.escape(str(prefix))}。HP・PP・状態を直接書き換えていません。自然な出現頻度は測っていません。</p>'
    for game, row in zip(games, summary["summaries"]):
        side = game["source_side"]
        log = []
        for r in game["records"]:
            if r["choices"][side] and r["choices"][side].startswith("team "):
                continue
            active = [next((p for p in s["pokemon"] if p["active"] and not p["fainted"]), None) for s in r["state"]["sides"]]
            board = " / ".join("-" if p is None else f'{p["species"]} {p["hp"]}/{p["maxhp"]} {p["status"]}' for p in active)
            changed = r["ordinary"][side] != r["choices"][side]
            log.append(f'<tr><td>{r["decision"]} / T{r["state"]["turn"]}</td><td>{html.escape(board)}</td><td>{html.escape(str(r["choices"]))}</td><td>{html.escape(str(r["ordinary"][side])) if changed else ""}</td></tr>')
        details.append(f'<details><summary>{labels[row["family"]]} / {row["opponent"]} / {row["arm"]} / 得点 {row["source_score"]}</summary><p>戦術側はP{side+1}、{row["source_team"]}。通常判断からの上書き{len(row["overrides"])}回。</p><div class="scroll"><table><tr><th>判断 / ターン</th><th>P1 / P2</th><th>実際の着手</th><th>上書き前の判断</th></tr>{"".join(log)}</table></div></details>')
    return f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>nc2000 戦術を選出させた対戦監査</title>
<style>body{{font:17px/1.8 system-ui,sans-serif;background:#f7f5ef;color:#292820;max-width:1100px;margin:32px auto;padding:0 20px}}h1{{font-size:1.7em}}h2{{font-size:1.2em;margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:.9em}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ccc6b6}}th{{background:#eee9dc}}.scroll{{overflow-x:auto}}a{{color:#4e643e}}details{{border-top:1px solid #ccc6b6;padding:14px 0}}summary{{cursor:pointer}}</style>
<h1>戦術を実際に選出させるとどうなるか</h1><p>元の24自己対戦で一度も選出されなかったアンコール・毒々・バトンの使い手を、既存構築のまま先発に固定。{len(summary["pairs"])}条件×通常判断と簡単な戦術手順の2通り、合計{summary["games"]}対局です。</p>{prefix_note}
<p>双方blind、c=0.4、27,000探索。技・持ち物・数値は変更していません。通常判断と戦術手順は同じ対戦乱数・探索乱数・選出で開始し、最初の上書きまで局面・通常判断・信念の記録が完全一致することを確認。戦術手順は公開された相手の状態だけを使います。</p>
<p>アンコールは相手が積み・回復などを使った後、毒々は状態異常のない相手、バトンは身代わりが残る時に使います。これらは最適解ではありません。初回試験の勝敗差は具体例を探すためのもので、一般的な強さや搾取可能性の証明ではありません。</p>
<h2>同じ条件での比較</h2><p>得点は戦術側の勝ち1・引き分け0.5・負け0。上限打ち切りはnullのまま保持。終局{summary["terminal_games"]}局、打ち切り{summary["capped_games"]}局。</p><table><tr><th>戦術</th><th>相手構築</th><th>通常判断</th><th>戦術手順</th><th>最初の相違</th></tr>{pairs}</table>
<h2>具体的な対局</h2>{''.join(details)}
{followup}
<p><a href="summary.json">集計・対照・再生検証</a> · <a href="protocol.json">固定した試験条件</a> · <a href="records.json.gz">全対局・観測・補完セット</a> · <a href="corpus.jsonl.gz">再生可能な棋譜</a></p></html>'''


if __name__ == "__main__":
    directory, output = map(pathlib.Path, sys.argv[1:])
    summary, games = report(directory, output)
    (output / "index.html").write_text(document(summary, games))
    print(json.dumps({k: summary[k] for k in ["games", "terminal_games", "capped_games", "pairs", "replay_verification"]}, ensure_ascii=False, indent=2))
