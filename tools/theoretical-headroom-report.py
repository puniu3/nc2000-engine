#!/usr/bin/env python3
"""Audit root certificates and render the bounded theoretical-headroom search."""
import collections, hashlib, html, json
from pathlib import Path
P=Path('findings/theoretical-headroom-v1')
rows=[json.loads(line) for f in ['screen-v6.jsonl','wide.jsonl'] for line in (P/f).read_text().splitlines()]
assert len({(r['id'],r['decision']) for r in rows})==len(rows)
focus=[json.loads(line) for line in (P/'focus.jsonl').read_text().splitlines()]
assert len(focus)==4
focus_by_key={(r['id'],r['decision']):r for r in focus}
for i,r in enumerate(rows):
    if (r['id'],r['decision']) in focus_by_key:rows[i]=focus_by_key[(r['id'],r['decision'])]
counts=collections.Counter(); witnesses=[]; records=[]
PAD=1e-8

def interval_matrix(r,s,kind):
    nc=len(r['actions'][1]); mat=[]
    for a in range(len(r['actions'][s])):
        row=[]
        for j in range(len(r['actions'][1-s])):
            k=a*nc+j if s==0 else j*nc+a
            if kind=='win':
                c=r['cells'][k];v=[c['win'][s],c['win'][s]+c['unknown']]
            else:
                c=r['bounds']['cells'][k];v=c if s==0 else [1-c[1],1-c[0]]
            assert -PAD<=v[0]<=v[1]+PAD and v[1]<=1+PAD
            row.append(v)
        mat.append(row)
    return mat

def elementary(m,selected):
    # Any row guarantees its minimum lower endpoint. A single opposing column
    # caps EVERY mixture by its largest upper endpoint. No matrix LP required.
    chosen_lo=min(v[0] for v in m[selected]);chosen_hi=min(v[1] for v in m[selected])
    best_lo=max(min(v[0] for v in row)for row in m)
    best_hi=min(max(row[j][1] for row in m)for j in range(len(m[0])))
    return {'selected':[max(0,chosen_lo-PAD),min(1,chosen_hi+PAD)],'best':[max(0,best_lo-PAD),min(1,best_hi+PAD)],'improvement':[max(0,best_lo-chosen_hi-2*PAD),min(1,max(0,best_hi-chosen_lo+2*PAD))]}
# Positive control: recognizes a strictly superior action, and leaves an
# entirely unresolved game unresolved. These also guard action orientation.
assert elementary([[[0,0],[0,0]],[[1,1],[1,1]]],0)['improvement'][0]>.999
assert elementary([[[0,1],[0,1]],[[0,1],[0,1]]],0)['improvement']==[0,1]
for r in rows:
    counts['roots']+=1
    counts['original_roots' if len(r['id'])==2 else 'continuation_roots']+=1
    assert len(r['cells'])==len(r['bounds']['cells'])==len(r['actions'][0])*len(r['actions'][1])
    for c in r['cells']:assert abs(sum(c['win'])+c['tie']+c['unknown']-1)<1e-8
    for s in [0,1]:
        counts['side_decisions']+=1
        chosen=r['selected'][s];assert chosen is not None
        forced=len(r['actions'][s])==1
        counts['forced' if forced else 'multiple_actions']+=1
        win=elementary(interval_matrix(r,s,'win'),chosen)
        score=elementary(interval_matrix(r,s,'score'),chosen)
        # If score >= L then P(win) >= max(0,2L-1), since a draw scores 1/2.
        # If optimal score <= U then optimal P(win) <= U. This yields a safe,
        # potentially very loose, pure-win headroom bound without equating the metrics.
        from_score=min(1,max(0,score['best'][1]-max(0,2*score['selected'][0]-1)+2*PAD))
        win_upper=min(win['improvement'][1],from_score)
        if not forced:
            counts['win_headroom_at_most_0_2pp' if win_upper<=.002+1e-6 else 'win_unresolved']+=1
            counts['score_headroom_at_most_0_1pp' if score['improvement'][1]<=.001+1e-6 else 'score_unresolved']+=1
        rec={'id':r['id'],'decision':r['decision'],'turn':r['turn'],'side':s,'action':r['actions'][s][chosen], 'forced':forced,'win_one_step':win,'score':score,'win_improvement_upper':win_upper,'solver_loss':r['bounds']['losses'][s],'stop':r['bounds']['stop']}
        records.append(rec)
        if win['improvement'][0]>PAD or score['improvement'][0]>PAD or rec['solver_loss'][0]>PAD:witnesses.append(rec)
summary={'counts':dict(counts),'certified_inferior_current_actions':witnesses,'records':records,'state_parity':{'states':6082,'original_prng_states':356,'normalization':'Hidden Power typed ID/name versus generic ID/name only; no numeric state fields normalized','scope':'sampled historical state parity, not exhaustive engine equivalence'},'model':'full-information stochastic game; independent uniform chance draws, floating-point arithmetic with padding. Selected first action fixed, subsequent play optimal. Opponent may choose any legal reply.','focus_roots':4,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('crates/bot/examples/theoretical_headroom.rs'),Path('crates/bot/src/bounds.rs'),Path('crates/engine/src/battle/enumerate.rs'),Path('data/gen2stadium2.json')]} }
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
interesting=[x for x in records if (x['id']=='g4' and x['turn'] in [10,53]) or (x['turn']==500)]
tr=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['id'],r['turn'],r['side']+1,r['action'],f"{r['score']['improvement'][0]*100:.2f}–{r['score']['improvement'][1]*100:.2f}",r['stop']])+'</tr>' for r in interesting)
assert not witnesses, 'Rewrite conclusion around discovered witnesses before publishing'
text=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NC2000：より良い選択肢の存在を調べる</title><style>
body{{max-width:62rem;margin:2rem auto;padding:0 1rem 3rem;background:#f1eee6;color:#302f29;font:17px/1.8 system-ui}}h1{{font-size:1.55rem}}h2{{font-size:1.15rem;margin-top:2rem}}a{{color:#4e6350}}.box{{padding:1rem;background:#e4e8dd;border-left:4px solid #708068}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}td,th{{padding:.5rem;border-bottom:1px solid #cac9bc;text-align:left}}.scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere}}</style>
<h1>より良い選択肢は存在するか</h1><p>2026-10-17開始の調査。対象は前回保存した公開版「強」（81,000回探索）の対局。方策を変更する対戦実験ではなく、記録された選択の理論的な改善余地を調べた。</p>
<div class="box"><strong>現在より勝率の高い手がある、という証拠は今回も得られなかった。ただし「存在しない」とも結論できない。</strong><p>終盤で絞れたのは、主にすでに勝敗が決まりかけた局面。問題の回復・PP消費・交代反復がある局面は、依然として区間が広く、優劣未判定だった。</p></div>
<h2>調べた範囲</h2><ul><li>元の8対局の通常の着手局面すべてと、前回の32継続対局から終盤・交代反復の代表局面を抽出。合計{counts['roots']}局面、両側{counts['side_decisions']}判断。独立した{counts['roots']}対局という意味ではない。</li><li>選択肢が複数ある判断は{counts['multiple_actions']}件。選択肢が一つしかない{counts['forced']}件は、改善できないという成果から除外した。</li><li>複数手から選べるうち、全情報モデルで勝率の改善上限を0.2ポイント以下に抑えられた判断は{counts['win_headroom_at_most_0_2pp']}件。残り{counts['win_unresolved']}件は未判定。</li><li>勝ち1・引き分け0.5の得点では、改善上限0.1ポイント以下が{counts['score_headroom_at_most_0_1pp']}件。これをそのまま勝率と読み替えていない。</li></ul>
<h2>何を「より良い」としたか</h2><p>最初の自分の一手を固定し、その後は双方が最善に指す全情報ゲームとして比較する。相手の初手も固定せず、合法な全応手を含める。代替手の保証下限が現在の手の上限を超えた場合、改善の証拠とする。混合戦略による改善は既存の区間ソルバーでも検査した。</p><p>乱数分岐を列挙し、未展開の枝を都合のよい評価値で埋めず、0〜1の未知区間に残した。1局面あたりの追加ソルバー予算は20,000回のエンジン実行・約1秒・10,000ノード。これとは別に初手行列の各セルを最大4,096実行で列挙した。回復を選んだ2局面と交代反復中の2局面は、200,000実行・約10秒・100,000ノードで再確認し、以下では追加計算後の区間を使用した。時間上限は処理単位の境界で確認するため、厳密な壁時計上限ではない。</p><p>初手行列から、現在の手の最低保証と、すべての混合戦略に対する上限を別のPython計算でも検査した。得点下限Lから勝率下限 max(0, 2L−1)、最適得点上限Uから最適勝率上限Uを得る保守的な変換を用いた。勝率改善の正の下限は得られなかった。</p>
<h2>肝心な局面はまだ解けていない</h2><div class="scroll"><table><tr><th>記録</th><th>ターン</th><th>側</th><th>現在の手</th><th>得点の改善余地（ポイント）</th><th>停止理由</th></tr>{tr}</table></div>
<p>区間が0〜100ポイントなら、「何も改善できない」と「大幅に改善できる」を区別できていない。区間の中点を勝率とせず、未判定とする。交代が長く続く見た目だけを、誤手の根拠にはできない。</p>
<h2>この結果で分かること・分からないこと</h2><p>今回の否定的な結果は、現行botが最適という証拠ではない。容易な終盤を解く能力と、PPを消耗しながら回復・交代する中盤を解く能力には大きな差がある。また、初手後の最善継続を仮定した評価であり、現行botが実際に続けて指す方策の勝率を保証するものではない。</p><p>相手の未公開情報も与えた実際の一世界を解いている。同じ観測と両立する全候補をまとめたblind方策の証明ではなく、相手を現在のbotに固定した最良応答の証明でもない。したがって、この結果だけで強化版を実装・採用する根拠はない。</p><p>次に狙う対象は、回復・交代反復の実局面を少数固定し、現在手の上限と代替手の下限を分離する証明探索。まず相手の応手ごとの区間を詰め、全情報で差が見つかってから、その差が未公開情報なしでも成立するかを検証する。方策変更の大量対戦へ戻る前に、この段階を通す必要がある。</p>
<h2>検証と再現性</h2><p>公開WASMの保存済み6,082盤面を手元のエンジンで再生し、元8対局では356乱数状態も照合した。Hidden Powerの型付きID・表示名と汎用ID・表示名だけを正規化し、HP・PP・状態など数値は変更していない。棋譜形式に互換性がないため、初期チームと全着手から再構成した。これは観測した履歴の一致であって、全反実仮想におけるエンジン同値性の証明ではない。</p><p>確率モデルは各乱数抽選を独立一様とする既存エンジンの列挙モデル。実PRNGの全シードを全列挙した確率ではない。計算は浮動小数点で、判定には余裕幅を設けた。数学的な形式検証済み証明書ではなく、このモデルと実装を前提とする数値的な保証である。初手列挙の陽性対照として、g0・20ターン目のスイクンをHP1に変更した人工局面では、わざと選んだ「ねごと」より「なみのり」の保証勝率が100ポイント高いと判定できた。この人工局面は471局面と現行botの誤手件数には含めていない。</p><p><a href="summary.json">集計・局面別区間（JSON）</a> · <a href="protocol.json">調査条件</a> · <a href="screen.jsonl">終盤の行列</a> · <a href="wide.jsonl">中盤の行列</a> · <a href="focus.jsonl">重点4局面の行列</a> · <a href="/nc2000-endurance/index.html">前回の耐久戦レポート</a></p></html>'''
(P/'index.html').write_text(text)
print(json.dumps({'counts':dict(counts),'witnesses':len(witnesses)},ensure_ascii=False,indent=2))
