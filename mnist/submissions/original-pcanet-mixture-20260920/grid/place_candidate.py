"""Choose region placement from access density only; no input data or labels.

Logical instructions and region sizes stay identical. Thus this permutation
changes physical costs while preserving all program values and output words.
"""
from pathlib import Path
from fractions import Fraction
import gzip,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'shared')]
from score_candidate import optimized,histograms
from affine import Program


if __name__=='__main__':
    source,target=map(Path,sys.argv[1:3])
    doc=json.loads(gzip.decompress(source.read_bytes()))
    optimized();program=Program(doc);counts,_=histograms(program)
    density={}
    for name,(base,words) in program.regions.items():
        accesses=int(counts['reads'][base:base+words].sum())+int(counts['writes'][base:base+words].sum())
        density[name]=Fraction(accesses,words)
    order=sorted(density,key=lambda name:(-density[name],name))
    original_regions=doc['regions']
    byname={r['name']:r for r in original_regions}
    doc['regions']=[byname[name] for name in order]
    assert sorted(original_regions,key=lambda x:x['name'])==sorted(doc['regions'],key=lambda x:x['name'])
    bodysha=hashlib.sha256(json.dumps(doc['body'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    proof={'kind':'Region placement permutation; logical instruction stream and per-region sizes unchanged',
           'original_program_sha256':hashlib.sha256(gzip.decompress(source.read_bytes())).hexdigest(),
           'logical_body_sha256':bodysha,'original_regions':original_regions,'placed_regions':doc['regions'],
           'selection':'Descending exact rational ordinary-read/write accesses per word, then region name; no dataset or predictions used.',
           'densities':{n:str(density[n]) for n in order}}
    encoded=json.dumps(doc,sort_keys=True,separators=(',',':')).encode()
    target.write_bytes(gzip.compress(encoded,mtime=0))
    proof['placed_program_sha256']=hashlib.sha256(encoded).hexdigest()
    target.with_suffix('.placement.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'placed_program_sha256':proof['placed_program_sha256'],'first_regions':order[:12]}),flush=True)
