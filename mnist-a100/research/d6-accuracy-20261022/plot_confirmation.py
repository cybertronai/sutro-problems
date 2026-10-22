"""Plot only the complete, valid frozen fresh-draw confirmation audit."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/'confirmation-analysis.json').read_text())
assert data['all_shards_finalized'] and data['all_arms_valid'] and not data['global_errors']
roles = data['role_to_algorithm']
arms = {arm['name']:arm for arm in data['arms']}
ordered = [('baseline450','Original 450-update model','#6b7280'),
           ('matched_unregularized_control','Long training, no added L2','#2563eb'),
           ('selected','Long training + L2','#15803d')]
fig,axes = plt.subplots(1,2,figsize=(11.4,4.6),layout='constrained')
for index,(role,label,color) in enumerate(ordered):
    arm = arms[roles[role]]
    ci = arm['accuracy_draw_interval']
    mean = ci['mean'];low,high = ci['approximate_95pct_t_interval']
    axes[0].errorbar(mean,index,xerr=[[mean-low],[high-mean]],fmt='o',color=color,capsize=5,markersize=7)
    axes[0].annotate(f'{mean:.3f}%  ·  {arm["mean_cumulative_algorithm_seconds"]:.1f}s',
                     (mean,index),xytext=(0,13),textcoords='offset points',ha='center',fontsize=9,color=color)
axes[0].set_yticks(range(3),[label for _,label,_ in ordered])
axes[0].axvline(98.6,color='#b91c1c',linestyle='--',linewidth=1,label='Telegram 98.6% reference')
axes[0].set(xlabel='Fresh-draw accuracy (%)',title='Frozen recipes on 11 fresh draws',ylim=(-.5,2.6))
axes[0].legend(loc='lower right',fontsize=8,frameon=False)
comparison = next(c for c in data['comparisons'] if c['left_role']=='selected' and c['right_role']=='matched_unregularized_control')
differences = [d['accuracy_difference_pp'] for d in comparison['paired_difference_by_draw']]
ci = comparison['paired_accuracy_difference_pp_interval'];mean=ci['mean'];low,high=ci['approximate_95pct_t_interval']
axes[1].axhline(0,color='#6b7280',linewidth=1)
axes[1].axhspan(low,high,color='#15803d',alpha=.12,label='95% interval for mean gain')
axes[1].axhline(mean,color='#15803d',linestyle='--',linewidth=1.3,label=f'Mean gain {mean:+.3f} pp')
axes[1].scatter(np.arange(1,12),differences,s=40,color='#2563eb',label='Paired difference per draw')
axes[1].set(xticks=np.arange(1,12),xlabel='Fresh draw',ylabel='L2 minus matched control (percentage points)',title='Added L2 at the same training budget')
axes[1].legend(loc='best',fontsize=8,frameon=False)
for ax in axes:
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('D6 confirmation on released 60-dimensional MNIST features',fontsize=13)
fig.supxlabel('Intervals describe draw-level means conditional on the fixed pool. Research timing excludes warmup and evidence serialization.',fontsize=8.5)
fig.savefig(ROOT/'confirmation-results.png',dpi=180)
fig.savefig(ROOT/'confirmation-results.pdf')
