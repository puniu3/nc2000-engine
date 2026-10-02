import collections
import gzip
import hashlib
import html
import json
import pathlib
import statistics
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)


def rows(name):
    return [json.loads(line) for line in (source / name).read_text().splitlines()]


def read(name):
    return json.loads((source / name).read_text())


def corpus_summary(name, measurements):
    games = rows(name)
    result = rows(measurements)
    expected = {(g['game'], d) for g in games for d in g['samples']}
    assert {(r['game'], r['decision']) for r in result} == expected
    errors = []
    audited = 0
    for r in result:
        for side in (0, 1):
            chosen = next((a for a in r['policies'][side] if a['chosen']), None)
            if chosen is None:
                continue
            action = next((a for a in r['certified_actions'][side] if a['action'] == chosen['action']), None)
            if action is None:
                continue
            audited += 1
            lower = r['lo'] if side == 0 else 1 - r['hi']
            gap = lower - action['hi']
            if gap > .01:
                errors.append({'game': r['game'], 'decision': r['decision'], 'side': side, 'gap': gap})
    return {
        'games': len(games), 'games_without_small_endgames': sum(not g['samples'] for g in games),
        'capped_games': sum(g['outcome'] == 'None' for g in games), 'positions': len(result),
        'solved': sum(r['hi'] - r['lo'] <= .0200000001 for r in result),
        'stop_counts': dict(collections.Counter(r['stop'] for r in result)),
        'classes': dict(collections.Counter('v'.join(map(str, r['alive'])) for r in result)),
        'seconds': sum(r['seconds'] for r in result),
        'median_width': statistics.median(r['hi'] - r['lo'] for r in result),
        'audited_pure_choices': audited, 'proven_choice_errors_above_001': errors,
    }


def duel(name):
    result = read(name)
    result['ci_low'] = result['score'] - result['ci95']
    result['ci_high'] = result['score'] + result['ci95']
    result['time_ratio'] = result['a_ms_per_move'] / result['b_ms_per_move']
    result['promoted'] = result['ci_low'] > .5 and result['time_ratio'] <= 1.05
    return result


timing = read('solved-timing.json')
medians = {arm: statistics.median(sum(r['seconds'] for r in run['rows']) for run in timing if run['arm'] == arm)
           for arm in ('baseline', 'candidate')}
for repeat in range(7):
    pair = sorted((run for run in timing if run['repeat'] == repeat), key=lambda r: r['arm'])
    assert [(r['key'], r['lo'], r['hi']) for r in pair[0]['rows']] == [(r['key'], r['lo'], r['hi']) for r in pair[1]['rows']]

summary = {
    'baseline_commit': 'ab58b38bd3fd8cd0a23dda80eb77a925009d02e0',
    'solver_commit': '7c60376',
    'deployment': 'not_deployed_no_promoted_strength_candidate',
    'discovery': corpus_summary('discovery.jsonl', 'instrumented.jsonl'),
    'strong': corpus_summary('strong.jsonl', 'strong-measured.jsonl'),
    'trace_reuse': {'positions': 6, 'alternating_pairs': 7, 'median_seconds': medians,
                    'time_reduction': 1 - medians['candidate'] / medians['baseline'], 'identical_certified_values': True},
    'forced': duel('forced-discovery.json'),
    'root_pruning': duel('prune-discovery.json'),
    'null_controls': {'forced': read('forced-null.json'), 'root_pruning': read('prune-null.json')},
    'validation': read('validation.json'),
    'scope': {
        'source': 'Uniform sampling from the 24-party shipped catalog; blind selfplay, frozen v3 belief pool; no human matches.',
        'samples': 'First and last decision in each alive-count class with at most two living party members per side.',
        'strong_limits': '27000 selfplay/search iterations, c=0.4; solver 10 seconds checked between trials, 120000 live nodes, 1000000 engine runs, width target 0.02; chronological graph reuse.',
        'oracle': 'Full-information numerical minimax brackets; comparison search is perfect-information skuct. A missing certified error does not establish optimal play or blind-policy correctness.',
        'duels': 'Two independently seeded 200-game discovery tests at 3000 iterations, side-swapped CRN pairs, uniform catalog, blind agents; no post-hoc pooling. Confirmation at 27000 only if the preregistered discovery gate passes.',
        'earlier_warm_cold': 'The earlier 1-second warm/cold pilot excludes reroot maintenance from seconds and stores cumulative graph stats; strong-measured includes maintenance and per-call counter deltas.',
        'fingerprints': '128-bit solver state fingerprints and floating-point probability/LP bounds retain the existing solver certification model.',
    },
    'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        pathlib.Path('data/gen2stadium2.json'), pathlib.Path('data/team-pool-v2/team-pool.json'),
        pathlib.Path('data/belief-pool-v3/belief-pool.json'), source / 'discovery.jsonl', source / 'strong.jsonl']},
}
assert all(not summary[name]['promoted'] for name in ('forced', 'root_pruning')), 'A passing discovery requires confirmation before final reporting'
for result in summary['null_controls'].values():
    assert result['score'] == .5 and all(score == .5 for score in result['pair_scores'])
(destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
archive = {p.name: p.read_text() for p in sorted(source.iterdir()) if p.suffix in ('.json', '.jsonl', '.log', '.rs', '.patch')}
with gzip.GzipFile(filename=str(destination / 'measurements.json.gz'), mode='wb', mtime=0) as out:
    out.write(json.dumps(archive).encode())

s = summary['strong']
t = summary['trace_reuse']
comparisons = ''.join(f'<tr><th>{title}</th><td>{r["wins"]}勝 {r["losses"]}敗</td><td>{r["score"]:.1%} [{r["ci_low"]:.1%}, {r["ci_high"]:.1%}]</td><td>{r["time_ratio"]:.3f}</td><td>不採用</td></tr>'
                      for title, r in [('強制局面を通過する展開', summary['forced']), ('最終除外手を探索中にも除外', summary['root_pruning'])])
stop_rows = ''.join(f'<tr><th>{html.escape(k)}</th><td>{v}</td></tr>' for k, v in s['stop_counts'].items())
page = f'''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NC2000 終盤解析と強化候補の検証</title>
<style>body{{margin:2rem auto;padding:0 1rem;max-width:64rem;background:#eee9df;color:#302c25;font:18px/1.7 system-ui}}h1{{font-size:1.7rem}}h2{{font-size:1.2rem;margin-top:2rem}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{padding:.5rem;border-bottom:1px solid #c8c0b3;text-align:left}}a{{color:#6b4826}}.scroll{{overflow-x:auto}}</style>
<h1>NC2000 終盤解析と強化候補の検証</h1>
<p>強化候補2案は対戦で有意な改善を示さず、デプロイしていません。自己対戦からの終盤抽出、証明付きの着手評価、解析の再利用と高速化を実装し、回帰検証を完了しました。今回の候補に対する採否であり、今後の強化が不可能という結論ではありません。</p>
<h2>同程度の計算時間での対戦比較</h2>
<p>各200局、3,000試行、c=0.4、blind同士。チームと乱数を合わせて左右を入れ替え、対ごとの得点から95%区間を計算しました。採用条件は事前に固定し、通過した場合だけ27,000試行で別シードの確認戦に進む設計です。両案とも確認戦への条件を満たしませんでした。</p>
<div class="scroll"><table><tr><th>候補</th><th>勝敗</th><th>勝率 [95%区間]</th><th>平均思考時間比</th><th>採否</th></tr>{comparisons}</table></div>
<p>同一設定同士の各20局では全対の得点が0.5になり、比較器の対照検証を通過しました。対戦の勝率は、以下の解析速度とは別の測定です。</p>
<h2>自己対戦からの終盤</h2>
<p>予備調査は1,000試行の12局・54局面。本測定は現在の24パーティから一様に選んだ27,000試行の16局で、双方の残りが2体以下になった各段階の最初と最後を保存しました。人間の対局は使用していません。{s['games_without_small_endgames']}局はこの条件に到達せず、対象は{s['positions']}局面です。</p>
<p>各局面の解析上限は10秒、実行100万回、保持12万局面。時間上限は一括展開の区切りで検査するため、1処理分の超過があります。幅0.02以内まで解けたのは{s['solved']}/{s['positions']}局面。残りは広い区間のままで、勝率の中点を正解として扱っていません。</p>
<table><tr><th>停止理由</th><th>局面数</th></tr>{stop_rows}</table>
<p>証明区間で比較できた純粋着手は{s['audited_pure_choices']}件で、最善値より0.01以上悪いと証明できた探索の選択は{len(s['proven_choice_errors_above_001'])}件でした。比較対象は全情報を渡したskuct探索です。未解決区間が多いため、0件でも最適性やblind botの正しさを証明したことにはなりません。</p>
<h2>解析器の改良</h2>
<p>乱数履歴を調べるために既に実行した枝を再実行せず、その結果を使います。解ける6局面で実行順を交互にした7往復の比較では、証明値を全て一致させて合計時間の中央値が{medians['baseline']:.3f}→{medians['candidate']:.3f}秒、{t['time_reduction']:.1%}短縮しました。この6局面の測定を未解決終盤全体への速度保証には使っていません。</p>
<p>次の局面から到達できる探索結果を保持し、不要になった祖先を削除します。1秒の予備測定では3局面で追加のエンジン実行が不要になりました。行列の区間が不変のときの再計算も省きます。ターン上限までの残り時間で勝率が変わる反例があるため、絶対ターンを消す同一視は行っていません。</p>
<h2>回帰検証と再現</h2>
<p>native {summary['validation']['native']['passed']}テスト、Wasm {len(summary['validation']['wasm'])}スイートを通過。独立した全列挙との後続状態・確率の一致、実行回数の上限、再利用前後の証明値、時間切れからの再開、ターン上限の反例を確認しました。凍結済みの対局記録の互換性も維持しています。</p>
<p>自己対戦と本測定は <code>bash tools/endgame-run.sh</code> で再実行できます。却下した強化案の差分、対戦結果、コーパスと計測値は測定記録に保存しています。公開中のbotは変更していません。</p>
<p><a href="summary.json">集計と測定条件</a> ／ <a href="measurements.json.gz">全測定記録・候補差分（JSON gzip）</a></p>
</html>'''
(destination / 'index.html').write_text(page)
print(json.dumps({'strong': summary['strong'], 'forced': summary['forced']['score'], 'pruning': summary['root_pruning']['score'], 'time_reduction': t['time_reduction']}, indent=2))
