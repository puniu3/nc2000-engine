import collections
import gzip
import hashlib
import html
import json
import pathlib
import sys


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(run, out):
    old = json.loads(gzip.decompress(pathlib.Path('findings/hp-evidence-v1/records.json.gz').read_bytes()))
    models = lines(run / 'release-models.jsonl')
    roots = lines(run / 'release-probe.jsonl')
    assert [r for r in roots if r['arm'] == 'baseline'] == [r for r in old['roots'] if r['arm'] == 'baseline']
    assert lines(run / 'default-models.jsonl') == old['models']
    assert lines(run / 'default-probe.jsonl') == old['roots']
    valid = [r for r in models if 'skip' not in r]
    assert len(models) == len(old['models']) == 127
    assert len(valid) == 101
    assert [r['case'] for r in models] == [r['case'] for r in old['models']]
    for r in valid:
        assert sum(r['model_types'].values()) == r['samples'] == 64
        assert r['contradictions'] == sum(n for t, n in r['model_types'].items() if t not in r['case']['allowed_types']) == 0
    probe_rows = []
    for battle in sorted({r['battle'] for r in roots}):
        group = [r for r in roots if r['battle'] == battle]
        assert len(group) == 8
        paired = collections.defaultdict(dict)
        for r in group:
            assert sum(a['visits'] for a in r['actions']) == 27000
            paired[r['seed']][r['arm']] = r['best']
        assert len(paired) == 4 and all(set(p) == {'baseline', 'inferred'} for p in paired.values())
        probe_rows.append({'battle': battle, 'changed': sum(p['baseline'] != p['inferred'] for p in paired.values()),
            'choices': {arm: dict(collections.Counter(r['best'] for r in group if r['arm'] == arm)) for arm in ['baseline', 'inferred']}})
    manifest = read(run / 'duel/manifest.json')
    protocol = manifest['protocol']
    protocol_bytes = (run / 'duel-protocol.json').read_bytes()
    assert hashlib.sha256(protocol_bytes).hexdigest() == manifest['protocol_sha256']
    verified = read(run / 'duel-verification.json')
    assert verified['games'] == 2 * sum(c['pairs'] for c in protocol['cases'])
    games, pairs, cases = [], [], []
    for case in protocol['cases']:
        cp = []
        for trial in range(case['pairs']):
            arms = [read(run / 'duel' / f"{case['id']}-t{trial:03}-a{arm}.json") for arm in range(2)]
            assert arms[0]['battle_seed'] == arms[1]['battle_seed']
            assert arms[0]['agent_seeds'] == arms[1]['agent_seeds']
            first = None
            for i, (a, b) in enumerate(zip(arms[0]['decisions'], arms[1]['decisions'])):
                if first is None:
                    assert a['key'] == b['key'] and a['search_seeds'] == b['search_seeds'] and a['hp_masks'] == b['hp_masks']
                    if a['choices'] != b['choices']:
                        assert any(m != 65535 for m in a['hp_masks'])
                        first = i
            if first is None:
                assert arms[0]['decisions'] == arms[1]['decisions']
                assert arms[0]['score'] == arms[1]['score']
            exposed = [any(any(m != 65535 for m in r['hp_masks']) for r in a['decisions']) for a in arms]
            scores = [a['score'] for a in arms]
            pair = {'case': case['id'], 'trial': trial, 'scores': scores, 'delta': None if None in scores else scores[1] - scores[0],
                'first_choice_difference': first, 'exposed': exposed}
            cp.append(pair)
            pairs.append(pair)
            games.extend(arms)
        def tally(arm):
            return dict(collections.Counter('cap' if p['scores'][arm] is None else str(p['scores'][arm]) for p in cp))
        cases.append({'case': case['id'], 'pairs': len(cp), 'baseline': tally(0), 'inferred': tally(1),
            'changed_pairs': sum(p['first_choice_difference'] is not None for p in cp),
            'exposed_pairs': sum(any(p['exposed']) for p in cp),
            'paired_delta_sum': sum(p['delta'] for p in cp if p['delta'] is not None),
            'complete_pairs': sum(p['delta'] is not None for p in cp)})
    stress = [p for p in pairs if p['case'] != 'onpool-control']
    complete = [p for p in stress if p['delta'] is not None]
    evidence = read(run / 'reveal-gate.json')
    assert evidence['violations'] == 0 and evidence['checks']['total'] == 4316484
    divergences = read(run / 'first-divergence.json')
    missing_moves = sum(any(set(r['true_foe_moves']) - set(kit.split(':')[0].split(',')) for kit in r['models'][1]['kits']) for r in divergences)
    summary = {'adoption': 'research-only; automatic inference disabled by default with hp-type-evidence Cargo feature', 'schema': 'nc2000-hp-evidence-fix-v1', 'first_divergence_missing_true_move': {'cases': len(divergences), 'with_missing_moves': missing_moves, 'causal_claim': False}, 'default_model_and_root_outputs_exact': True, 'reconstructed': len(valid), 'unreconstructed': len(models) - len(valid),
        'contradictory_before': sum(r.get('contradictions', 0) > 0 for r in old['models']), 'contradictory_after': 0,
        'baseline_roots_exact': 16, 'probes': probe_rows, 'duel_cases': cases,
        'stress_pairs': len(stress), 'complete_stress_pairs': len(complete),
        'stress_scores': [sum(p['scores'][a] for p in complete) for a in range(2)],
        'paired_mean_delta': sum(p['delta'] for p in complete) / len(complete) if complete else None,
        'paired_improved': sum(p['delta'] > 0 for p in complete), 'paired_worsened': sum(p['delta'] < 0 for p in complete),
        'paired_tied': sum(p['delta'] == 0 for p in complete),
        'replay_verification': verified, 'public_reveal_gate': evidence,
        'limitations': ['Constructed exposures, not historical full-team reconstructions or a prevalence estimate.',
            'Exploratory pilot; small paired sample, common corrected opponent, no general strength or human-superiority claim.',
            'Caps remain unknown. Complete-pair mean is conditional when any pair is capped.',
            'DV projection changes type, power and correlated stats coherently; not a type-only intervention.',
            'Only informative super-effective/resisted events are consumed. Ambiguous compatible prior types are retained.',
            'Known target type changes/copy mechanics are conservatively excluded; neutral and immune inference are unimplemented.',
            'Reveal gate uses its legacy no-prior corpus path. 165 HP-percent drift diagnostics are present.']}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'duel-protocol.json').write_bytes(protocol_bytes)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for source, dest in [(run / 'duel/manifest.json', 'duel-manifest.json'), (run / 'duel-verification.json', 'duel-verification.json')]:
        (out / dest).write_bytes(source.read_bytes())
    (out / 'first-divergence.json').write_bytes((run / 'first-divergence.json').read_bytes())
    (out / 'records.json.gz').write_bytes(gzip.compress(json.dumps({'models': models, 'roots': roots, 'pairs': pairs, 'games': games}, separators=(',', ':')).encode(), mtime=0))
    table = ''.join(f"<tr><td>{html.escape(c['case'])}</td><td>{c['pairs']}</td><td>{html.escape(str(c['baseline']))}</td><td>{html.escape(str(c['inferred']))}</td><td>{c['changed_pairs']}</td></tr>" for c in cases)
    (out / 'index.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>めざめるパワー推定の修正と対戦試験</title>
<style>body{{font:17px/1.8 system-ui;background:#f7f5ef;color:#292820;max-width:1050px;margin:30px auto;padding:0 20px}}h1{{font-size:1.6em}}h2{{font-size:1.2em}}table{{border-collapse:collapse;width:100%;font-size:.9em}}td,th{{padding:10px;border-bottom:1px solid #ccc6b6;text-align:left}}.scroll{{overflow-x:auto}}a{{color:#4e643e}}</style>
<h1>公開情報の推定修正：通常版への採用は保留</h1><p>予備対戦では旧版相当16勝、試験版13勝。改善を確認できなかったため、自動推定は研究用の切替（通常は無効）に隔離しました。</p><p>効果抜群・いまひとつの表示から、相手のめざめるパワーの候補タイプを絞る修正です。人間同士の570局から再構成できた101判断では、過去の公開情報と矛盾する推定が35件から0件に減りました。各判断64標本。再構成できなかった26件は除外しています。</p>
<p>個体値・能力値・技威力も整合させます。タイプが複数残る場合は両立する既存候補を残し、矛盾する補完だけを修正します。相手の本当の未公開セットを参照せず、位置データの保存・復元でも制約を維持します。</p>
<h2>52局の予備対戦</h2><p>3種の構成した対戦状況×8組と、既知の型を使う対照2組。各組で同じ対戦・探索乱数を使い、比較対象の側だけ公開タイプ制約を無視／利用します。相手は共通の修正版。27,000探索、c=0.4。表の1.0は勝ち、0.0は負け、0.5は引き分け、capは上限到達です。</p>
<div class="scroll"><table><tr><th>構成</th><th>組</th><th>旧版相当</th><th>修正版</th><th>着手が変わった組</th></tr>{table}</table></div>
<p>対照を除いた完了{len(complete)}/{len(stress)}組の得点は、旧版相当{summary['stress_scores'][0]:g}、修正版{summary['stress_scores'][1]:g}。同じ乱数の組で改善{summary['paired_improved']}・悪化{summary['paired_worsened']}・同点{summary['paired_tied']}です。少数の予備試験であり、一般的な勝率改善や人間上級者への優越は確認していません。</p>
<h2>悪化の原因はまだ未確定</h2><p>着手が変わった{len(divergences)}組の最初の分岐を再構成すると、{missing_moves}組で相手の実際の技構成の一部が補完に含まれていませんでした。これは構成した対戦状況での診断で、敗因と証明したものではありません。フリーザーの補完にゴッドバード・ふぶきが入り、実際のねむる・ねごとを欠く例などがあります。</p><h2>検証</h2><p>保存した{verified['games']}局・{verified['decisions']}判断を再生し、合法手・局面キー・乱数・観測制約・勝敗を照合しました。旧版相当の過去局面16探索は着手・訪問数・評価値まで保存結果と一致。公開情報の回帰検証は約432万チェックで違反0件でした。ただしHP割合のずれを示す診断165件は検出されています。</p>
<p>切替を無効にした通常設定では、過去の推定127件と探索32件が元の保存結果と完全一致しました。自動タイプ推定は既定では動きません。</p><p>公開相性と矛盾する推定は修正できましたが、この試験版を強化とは扱いません。最適戦略との差の大きさや、人間にどこまで勝てるかは別途測定が必要です。</p>
<p><a href="summary.json">集計と限界</a> · <a href="duel-protocol.json">固定した対戦条件</a> · <a href="duel-verification.json">再生検証</a> · <a href="records.json.gz">匿名局面と構成対戦の記録</a> · <a href="first-divergence.json">最初の着手分岐と推定技構成</a></p></html>''')
    source_hashes = read(run / 'measured-source-sha256.json')
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.name != 'verification.json' and f.is_file()}
    (out / 'verification.json').write_text(json.dumps({'checks': {'model_contradictions': 0, 'baseline_roots_exact': 16, 'default_models_exact': 127, 'default_roots_exact': 32, 'first_divergence_after_public_evidence': True}, 'measured_source_sha256': source_hashes, 'artifacts_sha256': hashes}, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['stress_pairs', 'stress_scores', 'paired_mean_delta', 'paired_improved', 'paired_worsened', 'duel_cases']}, indent=2))


if __name__ == '__main__':
    main(*map(pathlib.Path, sys.argv[1:]))
