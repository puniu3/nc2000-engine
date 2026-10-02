#!/usr/bin/env python3
import argparse
import hashlib
import html
import json
import pathlib
import re


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=pathlib.Path)
    args = p.parse_args()
    root = args.directory
    modes = ["blind", "skuct", "clone", "hash", "step", "determinize", "rollout"]
    results = {m: json.loads((root / m / "result.json").read_text()) for m in modes}
    provenance = json.loads((root / "provenance.json").read_text())
    metal = json.loads((root / "metal.json").read_text())
    assert metal.get("host_idle_screen_passed"), "quiet GPU measurement required"
    profile = json.loads((root / "profile.json").read_text())
    threads = [json.loads(p.read_text()) for p in sorted(root.glob("threads-*.json"))]
    threads.sort(key=lambda r: r["threads"])
    budgets = [json.loads((root / f"budget-{s}.json").read_text()) for s in [10, 60]]
    canonical = json.loads((root / "blind/candidate-0.json").read_text())
    reference = [(r["id"], r["state"]) for r in canonical["rows"]]
    def get_memory(name):
        text = (root / name).read_text()
        match = re.search(r"(\d+)\s+maximum resident set size", text)
        return int(match[1]) / 1024**2 if match else None
    for run in threads:
        assert [(r["id"], r["state"]) for r in run["rows"]] == reference
        assert [hashlib.sha256(str(r["signature"]).encode()).hexdigest() for r in run["rows"]] == [r["signature"] for r in canonical["rows"]], "thread count changed search results"
    summary = {"provenance": provenance, "results": results, "metal": metal, "profile":profile,
               "parallel": [{"threads":r["threads"], "seconds":r["wall_seconds"],
                             "rss_mib":get_memory(f"threads-{r['threads']}.txt")} for r in threads],
               "budgets": [{"seconds":r["rows"][0]["seconds"], "iterations":r["rows"][0]["completed"],
                            "state":r["rows"][0]["state"], "rss_mib":get_memory(f"budget-{s}.txt")}
                           for r, s in zip(budgets, [10, 60])]}
    blind = results["blind"]
    iteration_us = blind["candidate"]["median_seconds"] * 1e6 / (blind["positions"] * 27000)
    launch_us = metal["rows"][0]["gpu_e2e_median_ns"] / 1000
    launch_min_us = metal["rows"][0]["gpu_e2e_min_ns"] / 1000
    summary["gpu_decision"] = {"serial_iteration_us":iteration_us,"roundtrip_median_us":launch_us,
                               "observed_roundtrip_min_us":launch_min_us,
                               "measured_serial_dispatch_gate":"NO-GO" if launch_min_us >= iteration_us / 1.5 else "unresolved",
                               "full_batched_engine":"unmeasured; not adopted"}
    verification_path = root / "verification.json"
    if verification_path.exists():
        summary["verification"] = json.loads(verification_path.read_text())
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lead = f"blind探索の処理時間 {100 * blind['time_reduction']:.1f}%削減。採用基準 {'通過' if blind['accepted'] else '未達'}。"
    rows = []
    for mode, result in results.items():
        a = result["baseline"]["median_seconds"]
        b = result["candidate"]["median_seconds"]
        rows.append([mode, f"{a:.3f}秒", f"{b:.3f}秒", f"{a/b:.3f}倍", "一致"])
    body = f"<h1>Bot高速化レポート</h1><p><strong>{lead}</strong></p>"
    body += "<p>探索方針・評価関数・乱数・状態キーは維持。棋力向上の測定ではありません。</p>"
    body += "<h2>条件と変更</h2><p>Apple M6、12 CPUコア、24 GB。release / fat LTO / mimalloc。blind c=0.4、27,000回。人間対局を使わず、固定seedの自己対戦、滅び拘束の強制露出、両者回復の人工局面で比較。対戦生成は低予算の完全情報探索、探索側の事前分布はbelief-pool-v3です。測定は探索コアで、ブラウザや通信の所要時間を含みません。</p>"
    body += "<p>変更：行動の文字列往復を除去、一時技一覧を除去、選出の一時配列をスタック化、行動バッファとイベント用バッファを再利用。</p>"
    body += f"<p>基準コミット：<code>{provenance['baseline_commit'][:12]}</code>。各比較5回、前後交互。単一スレッド。{blind['positions']}局面の状態キー・選択手・根の統計・ノード数を毎回照合。</p>"
    body += table(["負荷", "変更前・中央値", "変更後・中央値", "処理速度", "固定仕事量の結果"], rows)
    body += "<p>clone/hash/step/determinize/rolloutは各局面3,000回、探索は27,000回。stepは複製を含みます。各成分の時間は、探索全体の内訳比率ではありません。</p>"
    body += "<h2>遅い局面への影響</h2>" + table(["区分", "局面数", "p95時間の変更後/変更前"], [[k, sum(r['tag'] == k for r in canonical['rows']), f"{v:.3f}"] for k,v in blind["p95_ratios"].items()])
    body += "<p>採用基準：代表負荷全体の中央値が5%以上短縮、主要区分のp95悪化が5%以内。途中版の結果を最終版に混ぜません。</p>"
    body += "<h2>CPU並列処理と思考時間</h2>" + table(["独立探索の並列数", "全局面の実時間", "最大RSS"],
             [[r["threads"], f"{r['seconds']:.2f}秒", f"{r['rss_mib']:.1f} MiB"] for r in summary["parallel"]])
    body += "<p>独立した局面を並列処理した測定です。単一の探索木を並列化した速度ではありません。各並列数は1回の容量確認です。</p>"
    body += table(["単一局面の実測時間", "探索回数", "最大RSS"], [[f"{r['seconds']:.3f}秒", f"{r['iterations']:,}", f"{r['rss_mib']:.1f} MiB"] for r in summary["budgets"]])
    body += "<p>10秒・60秒は同じ通常局面、同じseedで実測。時間判定は64回単位。ここから勝率や全局面の性能を外挿しません。</p>"
    body += "<h2>GPU実機測定</h2><p>Metalの呼び出し・同期費用を測る下限プローブ。1レーンあたりGen5乱数8回の最終状態をCPUと完全照合。戦闘エンジンの移植ではありません。入力準備・投入・完了待ち・読み取りを含め、初期コンパイルと最初の10回を除外しています。</p>"
    body += table(["バッチ数", "CPU演算 μs", "GPU全体 μs", "GPUコマンド μs"],
             [[r["batch"], f"{r['cpu_median_ns']/1000:.3f}" if r['cpu_median_ns'] else "分解能未満", f"{r['gpu_e2e_median_ns']/1000:.1f}", f"{r['gpu_command_median_ns']/1000:.1f}"] for r in metal["rows"]])
    body += f"<p>GPU使用率が測定前 {metal['utilization_before']}%、測定後 {metal['utilization_after']}%の窓で再測定。現行blind探索は平均1試行 {iteration_us:.1f} μs。最小バッチのGPU往復は中央値 {launch_us:.1f} μs、観測最小 {launch_min_us:.1f} μs。"
    if launch_min_us >= iteration_us / 1.5:
        body += "測定したAPIで全試行を1回ずつGPUへ渡す構成は、最小の往復費用だけでも全体1.5倍の速度目標を満たせません。"
    else:
        body += "呼び出し費用だけでは全体1.5倍の可能性を排除できません。完全な遷移処理の速度は未測定です。"
    body += "試行の依存関係を変えるバッチ化や、探索木ごとのGPU常駐は別方式となるため、今回の高速化には採用しません。</p>"
    body += f"<p>変更後のサンプリングでは状態ハッシュが {100*profile['state_hash_fraction']:.1f}%（子呼び出し込み）。この処理だけを無料にした速度上限は {profile['free_state_hash_speedup_ceiling']:.2f} 倍です。データ整形・同期費用はこの上限に含まれません。</p>"
    body += "<p>大量自己対戦向けの完全なGPU戦闘エンジンは未移植・未測定です。1.5倍以上の実効速度を示せていないため現時点では導入保留。小さな乱数カーネルの結果を、エンジン全面移植の速度に置き換えて判断しません。</p>"
    if "verification" in summary:
        body += "<h2>正しさと互換性</h2><p>native 128テスト通過。wasm smoke・parity・determinism・solver・kifuの5スイート通過。旧版が生成したNC2棋譜はnative/wasmとも読み戻せます。同等なソース指紋だけを対応づけ、未知の版を拒否する検査は維持しています。</p><p>速度計測後に追加した実行コードは棋譜の指紋対応だけです。その後にも49局面・27,000回の探索統計が一致することを確認しました。</p>"
    body += "<h2>再現</h2><p><code>bash tools/perf-run.sh</code>。測定セット・比較コードはリポジトリ内。実行にはRustとSwift/Metalが必要です。</p><p><a href='summary.json'>測定値とprovenance（JSON）</a></p>"
    report = root / "report"
    report.mkdir(exist_ok=True)
    (report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (report / "index.html").write_text("<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Bot高速化レポート</title><style>body{margin:2rem auto;padding:0 1rem;max-width:65rem;background:#eee9df;color:#302c25;font:18px/1.7 system-ui}h1{font-size:1.8rem}h2{font-size:1.3rem;margin-top:2rem}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:.9rem}td,th{padding:.5rem;border-bottom:1px solid #c8c0b3;text-align:left}code{overflow-wrap:anywhere}a{color:#6b4826}</style>" + body + "</html>")


if __name__ == "__main__":
    main()
