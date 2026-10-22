"""Plot audited development results; measured fit times are not official scores."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT/'analysis.json').read_text())
rows = [r for r in report['all_prefix_candidates'] if r['eligible']]
fig, axes = plt.subplots(1,2,figsize=(11.4,4.7),layout='constrained')
colors = {'initial':'#6b7280','unregularized':'#2563eb','regularized':'#d97706'}
for row in rows:
    name = row['arm']
    kind = 'initial' if not name.startswith('noguard') else ('unregularized' if name=='noguard7200' else 'regularized')
    axes[0].scatter(row['mean_full_call_seconds'],row['accuracy_pct'],color=colors[kind],s=25,alpha=.8)
best = report.get('selected')
if best:
    axes[0].scatter(best['mean_full_call_seconds'],best['accuracy_pct'],facecolor='none',edgecolor='#111827',s=120,linewidth=1.5)
for kind,label in [('initial','Original fallback rule'),('unregularized','Fallback removed, no added regularization'),('regularized','Fallback removed, regularized')]:
    axes[0].scatter([],[],color=colors[kind],s=28,label=label)
axes[0].axhline(98.6,color='#b91c1c',linestyle='--',linewidth=1,label='Proposed 98.6% target')
axes[0].set(xscale='log',xlabel='Cumulative member fit + inference (seconds)',ylabel='Development accuracy (%)',title='Nineteen methods and ensemble prefixes')
axes[0].legend(loc='lower right',fontsize=7.5,frameon=False)
steps = [('baseline450',450),('long1800',1800),('long7200',7200),('long28800',28800)]
for k,color in [(1,'#2563eb'),(4,'#d97706'),(8,'#0f766e')]:
    points = [r for name,step in steps for r in rows if r['arm']==name and r['members']==k]
    xs = [dict(steps)[r['arm']] for r in points]
    ys = [r['accuracy_pct'] for r in points]
    axes[1].plot(xs,ys,marker='o',color=color,label=f'{k} model'+('s' if k>1 else ''))
    for x,r in zip(xs,points):
        axes[1].scatter([x]*3,[p['accuracy_pct'] for p in r['rows']],marker='_',s=45,color=color,alpha=.45)
axes[1].axhline(98.6,color='#b91c1c',linestyle='--',linewidth=1)
axes[1].set(xscale='log',xlabel='Training updates per model',ylabel='Development accuracy (%)',title='Observed training-duration comparison')
axes[1].set_xticks([450,1800,7200,28800],['450','1,800','7,200','28,800'])
axes[1].legend(loc='lower right',fontsize=8,frameon=False)
for ax in axes:
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('D6 accuracy calibration on the released 60-dimensional MNIST task',fontsize=13)
fig.supxlabel('Three 10,000-query draws per point; initial coordinate drift is disclosed in the report.\nDevelopment selection requires fresh confirmation.',fontsize=9)
fig.savefig(ROOT/'development-frontier.png',dpi=180)
fig.savefig(ROOT/'development-frontier.pdf')
