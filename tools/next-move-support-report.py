import collections
import gzip
import hashlib
import html
import json
import pathlib
import sys


def read(path):
    return json.loads(path.read_text())


def summarize(rows):
    n = len(rows)
    return {'labels': n, 'games': len({r['battle'] for r in rows}), 'arms': [
        {'policy': rows[0]['arms'][i]['policy'] if rows else '',
         'sampled_support': sum(r['arms'][i]['hits'] > 0 for r in rows),
         'sampled_support_fraction': sum(r['arms'][i]['hits'] > 0 for r in rows) / n if n else None,
         'mean_sample_inclusion': sum(r['arms'][i]['hits'] / r['arms'][i]['samples'] for r in rows) / n if n else None}
        for i in range(3)]}


def main(run, out):
    manifest = read(run / 'manifest.json')
    assert not manifest['smoke'] and manifest['files'] == 570
    protocol = manifest['protocol']
    counts = read(run / 'counts.json')
    records = [json.loads(s) for s in (run / 'rows.jsonl').read_text().splitlines()]
    rows = [r for r in records if 'skip' not in r]
    assert counts['novel_labels'] == len(rows)
    assert counts['reconstruction_failed'] == len(records) - len(rows)
    assert counts['included_game'] + counts['excluded_copy_game'] == 570
    assert counts['direct_move_decisions'] == sum(counts[k] for k in ['already_revealed', 'already_used', 'novel_labels', 'reconstruction_failed', 'synthetic_move'])
    def plain(key):
        return 'hiddenpower' if key.startswith('hiddenpower') else key
    seen = set()
    for r in rows:
        known = {plain(m) for m in r['revealed']}
        assert len(known) < 4 and r['move'] not in known
        label = (r['battle'], r['side'], r['species'], r['move'])
        assert label not in seen
        seen.add(label)
        assert r['revealed_only_control'] == 0 and r['label_added_control'] == 64
        assert [a['policy'] for a in r['arms']] == protocol['policies']
        for arm in r['arms']:
            assert arm['samples'] == protocol['samples'] == 64
            assert sum(k['count'] for k in arm['kits']) == 64
            assert arm['hits'] == sum(k['count'] for k in arm['kits'] if r['move'] in {plain(m) for m in k['moves']})
            assert all(known <= {plain(m) for m in k['moves']} and len(k['moves']) <= 4 for k in arm['kits'])
    summary = summarize(rows)
    summary.update({'schema': protocol['schema'], 'counts': counts,
        'by_revealed_count': {str(n): summarize([r for r in rows if len(r['revealed']) == n]) for n in range(4)},
        'by_source': {source: summarize([r for r in rows if r['arms'][0]['source'] == source]) for source in sorted({r['arms'][0]['source'] for r in rows})},
        'by_move': {move: summarize([r for r in rows if r['move'] == move]) for move in sorted({r['move'] for r in rows})},
        'loadout_draw_support_gained': sum(r['arms'][0]['hits'] == 0 and r['arms'][2]['hits'] > 0 for r in rows),
        'loadout_draw_support_lost': sum(r['arms'][0]['hits'] > 0 and r['arms'][2]['hits'] == 0 for r in rows),
        'product_changed': False,
        'limitations': ['Only positive, newly selected move labels. Not P(carries), a probability calibration test, or a strength score.',
            '64-sample absence is not proof of zero distribution support. Samples within a decision and decisions within games are dependent.',
            'Existing priors may share sources with these human games. No held-out training-distribution claim.',
            'Selection excludes entire games with copying mechanics and only includes ordinary direct move decisions.',
            'Own submitted sets are reconstructed with full own-side evidence; opponent knowledge is prefix-only.',
            'Future actual moves label predictions, never enter opponent model construction. Never-used moves cannot be treated as absent.',
            'Larger candidate unions can improve this diagnostic without stronger decisions. Existing whole-loadout sampler did not pass strength evaluation.']})
    out.mkdir(parents=True, exist_ok=True)
    frozen_protocol = pathlib.Path('findings/next-move-support-v1/protocol.json')
    assert read(frozen_protocol) == protocol
    (out / 'protocol.json').write_bytes(frozen_protocol.read_bytes())
    for name in ['manifest.json', 'counts.json']:
        (out / name).write_bytes((run / name).read_bytes())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'records.json.gz').write_bytes(gzip.compress(json.dumps(records, separators=(',', ':')).encode(), mtime=0))
    largest = sorted(summary['by_move'].items(), key=lambda item: -(item[1]['labels'] - item[1]['arms'][0]['sampled_support']))[:25]
    table = ''.join(f"<tr><td>{html.escape(move)}</td><td>{v['labels']}</td><td>{v['labels'] - v['arms'][0]['sampled_support']}</td><td>{v['labels'] - v['arms'][2]['sampled_support']}</td></tr>" for move, v in largest)
    arms = summary['arms']
    (out / 'index.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>初公開技の事前候補診断</title>
<style>body{{font:17px/1.8 system-ui;background:#f7f5ef;color:#292820;max-width:1000px;margin:30px auto;padding:0 20px}}h1{{font-size:1.6em}}h2{{font-size:1.2em}}table{{border-collapse:collapse;width:100%;font-size:.9em}}td,th{{padding:9px;border-bottom:1px solid #ccc6b6;text-align:left}}a{{color:#4e643e}}.scroll{{overflow-x:auto}}</style>
<h1>初めて使われる技を、事前に候補へ入れていたか</h1>
<p>570局のうちコピー・変身技を含む10局を除外。560局で、そのターン以前には未公開だった通常選択技6,701件を再構成しました。相手の推定はターン開始時までの情報だけで作り、その後に使われた技を答えとして照合。再構成できなかった9件は別記しています。</p>
<p>通常版では{arms[0]['sampled_support']:,}件（{arms[0]['sampled_support_fraction']:.1%}）、既存の構成抽選版では{arms[2]['sampled_support']:,}件（{arms[2]['sampled_support_fraction']:.1%}）で、64標本のどれかに答えの技が含まれていました。ほろび対策を外した対照は{arms[1]['sampled_support_fraction']:.1%}でした。</p>
<p><strong>これは強さの点数ではありません。</strong>広く候補を挙げれば上がります。構成抽選版は以前の対戦試験で通常対戦の改善を示しておらず、今回も採用しません。64標本内の包含率と、勝ちやすい判断は区別します。</p>
<h2>見落としが多かった技</h2><div class="scroll"><table><tr><th>技ID</th><th>初公開件数</th><th>通常版の64標本に無し</th><th>構成抽選版の64標本に無し</th></tr>{table}</table></div>
<p>ねむるは436件中313件を通常版が候補に含めていませんでした。そのうち156件はカビゴンです。じこさいせいは74件中61件。回復技を持たない単一構成への偏りが、次に切り分ける対象です。回復候補だけを増やして強くなるかは未測定で、未公開技をすべて増やす提案ではありません。</p>
<h2>解釈の範囲</h2><p>使われた技という正例しかありません。使わなかった技が不所持とは言えず、技所持確率の較正や推定精度全体は測れていません。試料と既存事前分布の元資料には重複の可能性があり、学習資料から独立した評価とも扱いません。標本数内で出なかったことは、確率が厳密に0という意味ではありません。</p>
<p>全行について公開済み技の保持、4枠上限、包含数の再計算、初公開ラベルの重複なしを検証。公開済み技だけの対照は0件、答えを追加する検査用対照は全件を包含しました。これらは集計器の動作確認で、強さ指標の較正ではありません。bot本体の変更はありません。</p>
<p><a href="summary.json">全集計</a> · <a href="protocol.json">測定条件</a> · <a href="records.json.gz">匿名化した各判断の推定</a> · <a href="verification.json">検証</a></p></html>''')
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name != 'verification.json'}
    (out / 'verification.json').write_text(json.dumps({'novel_labels': len(rows), 'reconstructed_games': len({r['battle'] for r in rows}), 'sampled_mons': len(rows) * 64 * 3,
        'checks': ['all labels absent from prefix reveals', 'one label per game/side/species/move', 'reveals retained in every sampled kit', 'four-slot cap', 'sample and inclusion counts recomputed', 'revealed-only floor and label-added ceiling'], 'artifacts_sha256': hashes}, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['labels', 'games', 'arms', 'loadout_draw_support_gained', 'loadout_draw_support_lost']}, indent=2))


if __name__ == '__main__':
    main(*map(pathlib.Path, sys.argv[1:]))
