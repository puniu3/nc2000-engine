#!/usr/bin/env python3
"""Standalone research figure derived solely from the production trace."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];src=root/'findings/deployed-endurance-v1';out=root/'previews/deployed-endurance-v1'
rows=[json.loads(l) for l in (src/'games/g4.jsonl').read_text().splitlines()];ds=[r for r in rows if r['type']=='decision' and r['turn']>0]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','figure.facecolor':'#f6f4ee','axes.facecolor':'#f6f4ee'})
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained');colors=['#345c45','#8b4d29']
for side,color in enumerate(colors):
 t=[];hp=[];heal=[];damage=[];markers=[]
 for r in ds:
  p=next(p for p in r['state']['sides'][side]['party'] if p['species']=='Blissey');moves={m['id']:m['pp'] for m in p['moves']};t.append(r['turn']);hp.append(100*p['hp']/p['maxhp']);heal.append(moves['softboiled']);damage.append(moves['thunder']+moves['icebeam'])
  if dict(r['picks']).get(side)=='move softboiled' and p['hp']==p['maxhp'] and p['status']!='frz':markers.append(r['turn'])
 terminal=json.loads((src/'games/g4.json').read_text());p=next(p for p in terminal['state']['sides'][side]['party'] if p['species']=='Blissey');m={v['id']:v['pp'] for v in p['moves']};t.append(129.5);hp.append(100*p['hp']/p['maxhp']);heal.append(m['softboiled']);damage.append(m['thunder']+m['icebeam'])
 for ax,y in zip(axes,[hp,heal,damage]):ax.step(t,y,where='post',color=color,label=f'P{side+1}',linewidth=1.7)
 axes[0].scatter(markers,[100]*len(markers),color=color,marker='x',s=45,zorder=4)
for ax,label in zip(axes,['Blissey HP (%)','Soft-Boiled PP','Thunder + Ice Beam PP']):
 ax.set_ylabel(label);ax.grid(axis='y',color='#d7d8d1',linewidth=.7);ax.axvspan(19,32,color='#999999',alpha=.14);ax.set_xlim(1,129.5);ax.set_ylim(bottom=0)
axes[0].set_ylim(0,108);axes[0].legend(loc='lower right',frameon=False,ncol=2);axes[1].set_yticks([0,4,8,12,16]);axes[2].set_xlabel('Turn (last point: after T129)');fig.suptitle('Production strong bot: 129-turn mirror (g4)',fontsize=15)
fig.text(.5,-.015,'Shaded: P1 Blissey frozen. ×: Soft-Boiled chosen at full HP. Final winner: P1.',ha='center',fontsize=10)
for suffix in ['svg','png']:fig.savefig(out/f'g4-resources.{suffix}',dpi=150,bbox_inches='tight')
print(out/'g4-resources.svg')
