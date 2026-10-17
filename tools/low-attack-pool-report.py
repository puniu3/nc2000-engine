#!/usr/bin/env python3
"""Render only the low-Attack proposal's public report and download files.

Usage: python3 tools/low-attack-pool-report.py AUDIT_DIR NEW_PREVIEW_DIR
"""
import html
import json
import pathlib
import shutil
import sys


def main():
    audit_dir, preview_dir = map(pathlib.Path, sys.argv[1:])
    report = json.loads((audit_dir / 'audit.json').read_text())
    preview_dir.mkdir(parents=True, exist_ok=False)
    for name in ('audit.json', 'candidate-pool.json'):
        shutil.copyfile(audit_dir / name, preview_dir / name)
    names = {'Venusaur': 'フシギバナ', 'Gengar': 'ゲンガー', 'Starmie': 'スターミー',
             'Umbreon': 'ブラッキー', 'Suicune': 'スイクン', 'Alakazam': 'フーディン',
             'Jolteon': 'サンダース', 'Lapras': 'ラプラス', 'Jynx': 'ルージュラ',
             'Vaporeon': 'シャワーズ'}
    escape = lambda value: html.escape(str(value))

    def damage_range(row, key):
        values = [d[key] for d in row['damage']]
        return str(min(values)) if min(values) == max(values) else f'{min(values)}–{max(values)}'

    rows = []
    for r in report['rows']:
        moves = '<br>'.join(escape(d['move']) for d in r['damage'])
        rows.append(
            f'<tr><td><code>{escape(r["baseId"])}</code><br>枠{r["slot"]}</td>'
            f'<td>{escape(names.get(r["species"], r["species"]))}<br>'
            f'Lv{r["level"]} / {escape(r["gender"])} / HP {r["hp"]}</td>'
            f'<td>{r["attackDvBefore"]} → {r["attackDvAfter"]}</td>'
            f'<td>{r["attackBefore"]} → {r["attackAfter"]}</td>'
            f'<td>{damage_range(r, "neutralBefore")} → {damage_range(r, "neutralAfter")}</td>'
            f'<td>{damage_range(r, "swaggerBefore")} → {damage_range(r, "swaggerAfter")}</td>'
            f'<td><details><summary>技構成</summary>{moves}</details></td></tr>')
    reductions = [d['neutralBefore'] - d['neutralAfter'] for r in report['rows'] for d in r['damage']]
    checked = report['checks']['selfHitComparisons']
    source = 'https://github.com/smogon/pokemon-showdown/blob/79c04dcc4c055d190a7aba724ec83733ee3f7b38/data/mods/gen2/conditions.ts'
    document = f'''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>NC2000 攻撃最小化の調整候補</title>
<style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#f5f2eb;color:#292924;font:16px/1.75 system-ui,sans-serif}}
main{{max-width:1150px;margin:auto;padding:32px 20px 64px}}h1{{font-size:1.65rem;line-height:1.4}}h2{{font-size:1.15rem;margin-top:2em}}
a{{color:#425d39}}.lead{{font-size:1.1rem}}.note{{background:#e9e5db;padding:16px 20px;border-left:4px solid #6d7858}}
.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:.91rem}}th,td{{padding:12px 10px;border-bottom:1px solid #d3d0c5;text-align:left;vertical-align:top}}
th{{background:#e9e5db;white-space:nowrap}}td:nth-child(n+3):nth-child(-n+6){{white-space:nowrap}}code{{font-size:.84em;overflow-wrap:anywhere}}
summary{{cursor:pointer}}.small{{font-size:.88rem;color:#55564c}}.downloads{{display:flex;gap:16px;flex-wrap:wrap}}.downloads a{{padding:10px 14px;background:#e2e7d9;border-radius:5px}}
</style><main>
<p class="small">NC2000 ENGINE · 独立監査 / 2026-10-17</p>
<h1>物理技を持たない個体の攻撃を下げる調整候補</h1>
<p class="lead">現行{report['teams']}パーティのうち{report['changedTeams']}パーティ・{report['changedSets']}枠が対象。
攻撃努力値を0、攻撃DVをHP・性別・色違い条件が変わらない最小値にすると、
通常状態の混乱自傷は1回あたり{min(reductions)}〜{max(reductions)}軽減しました。</p>
<p class="note">既存のプール・起動中のアプリには適用していません。候補の勝率は未測定です。
「わるあがき」の与ダメージも下がるため、すべての局面で有利とは言えません。</p>
<h2>調整条件</h2>
<ul><li>めざめるパワー、物理分類の技、他の技を取得する「ゆびをふる・ものまね・オウムがえし・へんしん」がある個体は除外。
物理／特殊は第2世代のデータで判定します。「おいうち」「どろぼう」は特殊です。</li>
<li>HP、性別、色違いのDV条件、攻撃以外の能力、技・持ち物・レベル・抽選重みを維持します。</li>
<li>最大HPの個体では攻撃DVの奇数を維持。♂フシギバナ・♂サンダース・♂シャワーズ・♂ブラッキーはDV3が下限です。
♀や性別不明の今回の対象はDV1になります。</li>
<li>対象のうち{report['eligibleAlreadyAtMinimumDv']}枠は既にこの条件で最小DVでした。攻撃努力値を下げる余地が残っていました。</li></ul>
<h2>対象一覧</h2>
<p class="small">全対象で攻撃努力値255 → 0。攻撃は実数値。自傷はHP減少量で、乱数幅ではありません。
選んだ技のタイプと持ち物によって値が分かれる場合は最小〜最大を表示。
「いばる後」は攻撃+2段階・防御±0、回復などのターン終了効果を加える前の値です。</p>
<div class="table"><table><thead><tr><th>元パーティ / 枠</th><th>個体</th><th>攻撃DV</th><th>攻撃</th><th>通常自傷</th><th>いばる後の自傷</th><th>技</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<h2>検証結果と限界</h2>
<p>現行ネイティブエンジンの混乱自傷処理を使い、各対象の全技で攻撃・防御それぞれ−6〜+6段階を比較。
{checked:,}組で自傷の増加は0でした。元と候補の全{report['teams']}パーティを組み込みバリデータで検証し、
HP・性別・攻撃以外の実数値が一致することも確認しています。</p>
<p>この比較はやけど・天候・壁なしの状態です。バリデータは単独技の習得可否を確認しますが、
イベント・タマゴ技の全組合せを保証しません。今回は技構成を変更していません。
PP枯渇戦での「わるあがき」の損失と、実際に混乱する頻度を含めた勝率差は別途評価が必要です。</p>
<p>候補の変更パーティIDには <code>-low-atk</code> を追加しました。
既存の選定スコアは元構成の記録として <code>baselineScore</code> に退避し、候補の測定値と区別しています。
既存の対戦相手事前分布・Nash解・コンパイル済み選出表は書き換えていません。
本採用時は新しい入力として選出表との整合性を確認する必要があります。</p>
<div class="downloads"><a href="candidate-pool.json" download>候補プール JSON</a><a href="audit.json" download>全検証結果 JSON</a></div>
<p class="small">入力: <code>{escape(report['sourcePool'])}</code><br>
SHA-256: <code>{escape(report['sourcePoolSha256'])}</code><br>
比較対象: <a href="{source}">参照実装の第2世代混乱処理</a>。
再現: <code>cargo run --locked --offline -j 1 --target-dir tmp/low-attack-audit-target -p nc2000-engine --example low_attack_pool_audit -- data/team-pool-v2/team-pool.json NEW_OUTPUT_DIRECTORY</code></p>
</main></html>'''
    (preview_dir / 'index.html').write_text(document)
    print(f'{preview_dir}/index.html')


if __name__ == '__main__':
    main()
