"""Score the candidate with bounded histogram memory and exact integer sums.

The cost rules, address placement and affine address histograms are unchanged
from the shared scorer. Optimizations are checked against it on small programs.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, sys, time
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'shared'))
import affine
import score as S


def initialization(self):
    initialized = np.zeros(self.words + 1, dtype=np.bool_)
    count = visits = 0
    def walk(body, env):
        nonlocal count, visits
        for node in body:
            if count == self.words: return
            visits += 1
            if 'loop' in node:
                inner = node['body']
                if len(inner) == 1 and inner[0].get('op') in ('set','recv'):
                    dst = inner[0]['dst']
                    coeff = dst['coefficients']
                    if coeff.get(node['loop']) == 1 and set(coeff) <= {node['loop']}:
                        first = self.regions[dst['region']][0] + dst['offset'] + node['start']
                        end = first + node['count']
                        count += end-first-int(initialized[first:end].sum())
                        initialized[first:end] = True
                        visits += node['count']
                        continue
                for value in range(node['start'], node['start']+node['count']):
                    walk(inner, {**env,node['loop']:value})
                    if count == self.words: return
            else:
                for src in node.get('src',[]):
                    if not initialized[self.address(src,env)]:
                        raise ValueError('Uninitialized source: '+str(src))
                if 'dst' in node:
                    address = self.address(node['dst'],env)
                    if not initialized[address]:
                        initialized[address]=True; count+=1
    walk(self.document['body'],{})
    self.initialization_visits=visits
    self.initialized_words=count


def histograms(program):
    counts={name:np.zeros(program.words+1,dtype=np.int64)
            for name in ('reads','writes','input_destinations','output_sources')}
    def charge(name,operand,scope):
        first,h=program.histogram(operand,scope)
        counts[name][first:first+len(h)]+=h
    for node,scope,mult in program.leaves:
        if not mult: continue
        if node['op']=='recv': charge('input_destinations',node['dst'],scope)
        elif node['op']=='send': charge('output_sources',node['src'][0],scope)
        else:
            for src in node.get('src',[]): charge('reads',src,scope)
            charge('writes',node['dst'],scope)
    return counts,len(program.leaves)


def exact_dot(counts,costs):
    total=0
    for first in range(0,len(counts),1000000):
        a=counts[first:first+1000000]; b=costs[first:first+1000000]
        if int(a.max(initial=0))*int(b.max(initial=0))*len(a)<2**63:
            total+=int(np.dot(a,b))
        else:
            total+=sum(int(x)*int(y) for x,y in zip(a,b) if x)
    return total


def optimized():
    affine.Program._validate_initialization=initialization
    S.histogram_counts=histograms
    S.exact_dot=exact_dot


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('program',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--compare-original',action='store_true')
    a=p.parse_args()
    document=json.loads(gzip.decompress(a.program.read_bytes()))
    baseline=S.score(document) if a.compare_original else None
    optimized(); result=S.score(document)
    if baseline:
        fields=('energy_fj','cycles','energy_mj','time_ms','total_executed_instructions',
                'peak_allocated_scratch_bytes','memory_tiles','components','port_counts')
        for field in fields:
            if field in baseline: assert baseline[field]==result[field],field
        result['optimized_scorer_matches_original']=True
    result['program_metadata']=document['metadata']
    result['scorer_optimizations']=['Vectorized proof of contiguous source-free initialization.',
        'Uncached affine histograms to bound host memory.',
        'Chunked integer dot products with checked int64 overflow bounds.']
    result['numeric_execution_checked_by_scorer']=False
    result['source_sha256']={str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [Path(__file__),HERE/'shared/score.py',HERE/'shared/affine.py',HERE/'build_candidate.py']}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('energy_mj','time_ms','energy_fj','cycles',
        'total_executed_instructions','peak_allocated_scratch_bytes','time_to_score_seconds')}),flush=True)
