"""Verify frozen official records using only the Python standard library."""
from pathlib import Path
import hashlib,json,statistics,ast,math
root=Path(__file__).resolve().parent
# Read timing-floor constants directly from the checked-out scorer without importing Torch.
constants={}
for node in ast.parse((root.parents[1]/'mnist.py').read_text()).body:
 if isinstance(node,ast.Assign):
  for target in node.targets:
   if isinstance(target,ast.Tuple) and all(isinstance(t,ast.Name) for t in target.elts):
    names=[t.id for t in target.elts]
    if 'FLOOR_SLACK_MS' in names:constants.update(zip(names,ast.literal_eval(node.value)))
   if isinstance(target,ast.Name) and target.id in ('FLOOR_SLACK_MS','FLOOR_SLACK_FRACTION'):
    constants[target.id]=ast.literal_eval(node.value)
results={}
for name,base in [('submitted',root),('comparison2500',root/'comparison-2500')]:
 sha=hashlib.sha256((base/'submission.py').read_bytes()).hexdigest();rows=[];uuids=[]
 for i in range(1,4):
  outer=json.loads((base/f'official/run-{i}.json').read_text());r=outer['record']
  assert outer['returncode']==0 and r['sandboxed'] and r['band_bp']==190
  assert r['source_sha256']==sha and not r['problems'] and not r['energy']['problems']
  assert len(r['calls'])==15
  groups=[[c for c in r['calls'] if c['dataset']=='mnist'],[c for c in r['calls'] if c['dataset']!='mnist']]
  assert [len(g) for g in groups]==[11,4]
  correct=sum(c['correct'] for c in groups[0]);total=sum(c['total'] for c in groups[0]);assert total==110000 and correct>=107910
  def group_time(g):
   reported=statistics.mean(c['ms'] for c in g)
   floor=statistics.mean(c['parent_ms']-c['overhead_ms'] for c in g)-constants['FLOOR_SLACK_MS']-constants['FLOOR_SLACK_FRACTION']*reported
   return max(reported,floor)
  assert math.isclose(max(map(group_time,groups)),r['ranked_ms'],abs_tol=1e-6)
  assert r['energy']['mj_per_call']>0
  uuids.append(r['energy']['device']['uuid'])
  rows.append(dict(run=i,error_pct=100*(total-correct)/total,correct=correct,total=total,ranked_ms=r['ranked_ms'],energy_mj=r['energy']['mj_per_call']))
 assert len(set(uuids))==3
 results[name]=dict(sha256=sha,passes=3,runs=rows,median_ms=statistics.median(r['ranked_ms'] for r in rows),median_energy_mj=statistics.median(r['energy_mj'] for r in rows),pooled_error_pct=100*sum(r['total']-r['correct'] for r in rows)/330000)
print(json.dumps(results,indent=2))
