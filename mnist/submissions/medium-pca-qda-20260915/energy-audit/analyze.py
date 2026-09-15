"""Recompute both meters from raw records and plot the power trace."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--results',default='results')
args = parser.parse_args()
suffix = '' if args.results=='results' else '-'+args.results.removeprefix('results-')
original = json.loads((ROOT/args.results/'original.json').read_text())
independent = json.loads((ROOT/args.results/'independent.json').read_text())
published = json.loads((ROOT/'published-gpu-results.json').read_text())
assert original['measurement_valid'] and independent['passed']
assert original['validation']['gpu_agrees_with_frozen_cpu_predictions'] == 110000
assert sum(row['matches'] for row in independent['validation']) == 110000
assert not independent['monitor_errors']

def near(a,b):
    assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-8),(a,b)

for row in original['rounds']:
    duration = row['measurement_end']['time_s']-row['measurement_start']['time_s']
    energy = row['measurement_end']['energy_mj']-row['measurement_start']['energy_mj']
    idle = []
    for name in ('idle_before','idle_after'):
        record = row[name]
        power = (record['end']['energy_mj']-record['start']['energy_mj'])/1000/(record['end']['time_s']-record['start']['time_s'])
        near(power,record['power_w'])
        idle.append(power)
    near(energy/row['repeats'], row['gross_mj'])
    near((energy-statistics.mean(idle)*duration*1000)/row['repeats'],row['adjusted_mj'])

for field, raw in (('gross_mj_median', 'gross_mj'), ('adjusted_mj_median', 'adjusted_mj'),
                   ('cuda_ms_median', 'cuda_ms'), ('wall_ms_median', 'wall_ms')):
    near(original['summary'][field], statistics.median(r[raw] for r in original['rounds']))

samples = sorted(independent['samples'],key=lambda r:r['t'])
t = np.array([r['t'] for r in samples])
p = np.array([r['power_w'] for r in samples])
for row in independent['intervals']:
    a,b = row['start']['t'],row['end']['t']
    inside = (t>a)&(t<b)
    ts = np.r_[a,t[inside],b]
    energy = float(np.trapezoid(np.interp(ts,t,p),ts))
    near(energy,row['integrated_power_j'])
    near((row['end']['energy_mj']-row['start']['energy_mj'])/1000,row['counter_j'])
    near(b-a, row['seconds'])
    for meter in ('counter', 'integrated_power'):
        near(row[meter+'_j']/(b-a), row[meter+'_w'])

intervals_by_name = {r['name']: r for r in independent['intervals']}
for index, row in enumerate(independent['comparisons']):
    before = intervals_by_name[f'{index}-idle-before']
    middle = intervals_by_name[f"{index}-{row['kind']}"]
    after = intervals_by_name[f'{index}-idle-after']
    assert row['repeats'] == middle['repeats']
    denominator = middle['repeats'] or 6000
    assert row['nominal_tasks'] == denominator
    near(row['wall_ms_per_task'], middle['seconds']*1000/denominator)
    if middle['repeats']:
        near(row['cuda_ms_per_task'], middle['cuda_ms'])
    for meter in ('counter', 'integrated_power'):
        baseline = statistics.mean((before[meter+'_w'], after[meter+'_w']))
        near(row[f'{meter}_gross_mj_per_task'], middle[meter+'_j']*1000/denominator)
        near(row[f'{meter}_idle_adjusted_mj_per_task'],
             (middle[meter+'_j']-baseline*middle['seconds'])*1000/denominator)

active = [r for r in independent['comparisons'] if r['kind']=='active']
shams = [r for r in independent['comparisons'] if r['kind']=='sham']
summary = {'hardware':independent['hardware'], 'software':independent['software'],
           'all_110000_predictions_match':True,'raw_measurement_arithmetic_verified':True,
           'original_protocol':original['summary'],
           'independent':{},'idle_only_shams':shams,
           'peak_allocated_bytes':independent['peak_allocated_bytes'],
           'power_sample_count':len(samples),'temperature_c_range':[min(r['temperature_c'] for r in samples),max(r['temperature_c'] for r in samples)],
           'sm_clock_mhz_values':sorted(set(r['sm_clock_mhz'] for r in samples)),
           'counter_vs_power_integration_relative_errors':[r['integrated_power_j']/r['counter_j']-1 for r in independent['intervals']],
           'caveat':'Both meters share NVML/hardware telemetry; no independent external wattmeter. Same data are replayed; transfers and preprocessing excluded. One measured board, paired idle controls, no fleet-wide confidence claim.'}
for method in ('counter','integrated_power'):
    summary['independent'][method]={}
    for kind in ('gross','idle_adjusted'):
        values=[r[f'{method}_{kind}_mj_per_task'] for r in active]
        summary['independent'][method][kind]={'values_mj':values,'median_mj':statistics.median(values),'mean_mj':statistics.mean(values),'sample_sd_mj':statistics.stdev(values)}
summary['independent']['cuda_ms_values']=[r['cuda_ms_per_task'] for r in active]
summary['independent']['wall_ms_values']=[r['wall_ms_per_task'] for r in active]
trimmed = []
by_name = {r['name']:r for r in independent['intervals']}
for index in (0,2,4):
    middle=by_name[f'{index}-active']
    baseline=[]
    for position in ('before','after'):
        row=by_name[f'{index}-idle-{position}']
        a,b=row['start']['t']+3,row['end']['t']
        ts=np.r_[a,t[(t>a)&(t<b)],b]
        baseline.append(float(np.trapezoid(np.interp(ts,t,p),ts))/(b-a))
    baseline_w=statistics.mean(baseline)
    trimmed.append({'round':index,'baseline_w':baseline_w,
                    'active_w':middle['integrated_power_w'],
                    'idle_adjusted_mj_per_task':(middle['integrated_power_j']-baseline_w*middle['seconds'])*1000/middle['repeats']})
summary['trimmed_idle_sensitivity']={'method':'Post-hoc sensitivity analysis: discard the first 3 seconds of each 10-second idle window to remove the observed power-state tail; integrate remaining sampled power. Same entire active windows retained.',
                                    'rounds':trimmed,'median_mj':statistics.median(r['idle_adjusted_mj_per_task'] for r in trimmed)}
(ROOT/('summary'+suffix+'.json')).write_text(json.dumps(summary,indent=2)+'\n')

fig=plt.figure(figsize=(12,8),layout='constrained')
grid=fig.add_gridspec(2,2,height_ratios=[1.2,1])
ax=fig.add_subplot(grid[0,:])
ax.plot(t-t[0],p,lw=.7,color='#2563eb',label='Sampled board power')
for row in independent['intervals']:
    name=row['name']
    if name.endswith('-active') or name.endswith('-sham'):
        color='#10b981' if name.endswith('-active') else '#94a3b8'
        ax.axvspan(row['start']['t']-t[0],row['end']['t']-t[0],color=color,alpha=.15)
ax.set(xlabel='Seconds',ylabel='Board power (W)',title='Fresh A100 measurement: green = training + prediction; gray = idle-only control')
ax.grid(alpha=.2)
ax=fig.add_subplot(grid[1,:])
values=[[r['adjusted_mj'] for r in original['rounds']],
        [r['counter_idle_adjusted_mj_per_task'] for r in active],
        [r['integrated_power_idle_adjusted_mj_per_task'] for r in active],
        [r['counter_idle_adjusted_mj_per_task'] for r in shams]]
for i,vs in enumerate(values):
    ax.scatter(np.full(len(vs),i),vs,s=30)
    ax.hlines(statistics.median(vs),i-.2,i+.2,color='black',lw=2)
ax.axhline(published['summary']['adjusted_mj_median'],color='#64748b',ls='--',label='Published net 3.8 mJ')
ax.axhline(0,color='black',lw=.5)
ax.set(xticks=range(4),xticklabels=['Original\nprotocol','Long\ncounter','Long\npower','Idle-only\n/ 6,000'],ylabel='Idle-subtracted energy (mJ/task)',title='Energy above idle')
ax.legend(fontsize=8)
ax.grid(axis='y',alpha=.2)
fig.suptitle(independent['hardware']['name'],fontsize=14)
fig.savefig(ROOT/('energy-audit'+suffix+'.png'),dpi=180)
print(json.dumps(summary,indent=2))
