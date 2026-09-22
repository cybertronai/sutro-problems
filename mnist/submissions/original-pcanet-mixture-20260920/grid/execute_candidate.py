"""Numerical execution of the grid document, with no test-label input.

This is the generic instruction compiler from the CG submission. Consecutive
constant stores are emitted as an ordered loop over program literals to keep
compiler memory bounded; they remain charged as individual set instructions.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, re, subprocess, sys, time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'shared'))
import il_compiler as C
from score_candidate import optimized


def compact_literals(source):
    lines=source.splitlines(); out=[]; n=0; table=0
    pat=re.compile(r'^(\s*)m\[(\d+)\] = (\d+)U;$')
    while n<len(lines):
        first=pat.match(lines[n]); end=n+1
        if first:
            while end<len(lines):
                nxt=pat.match(lines[end])
                if not nxt or nxt[1]!=first[1] or int(nxt[2])!=int(first[2])+end-n: break
                end+=1
        if first and end-n>=8:
            vals=[pat.match(line)[3]+'U' for line in lines[n:end]]
            pad=first[1]
            out.append(pad+'{')
            out.append(pad+f'  static const uint32_t literals_{table}[] = {{'+','.join(vals)+'};')
            out.append(pad+f'  for (uint64_t lit=0; lit<{len(vals)}ULL; ++lit) m[{first[2]}ULL+lit]=literals_{table}[lit];')
            out.append(pad+'}'); table+=1; n=end
        else:
            line=lines[n]
            loop=re.match(r'^  for \(int64_t (v\d+) =',line)
            if loop:
                out.append('  std::fprintf(stderr, "begin '+loop[1]+'\\n"); std::fflush(stderr);')
            out.append(line); n+=1
    return '#include <cstdio>\n'+'\n'.join(out)+'\n'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('program',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--raw',type=Path)
    p.add_argument('--data-module',type=Path)
    p.add_argument('--fixture',action='store_true')
    p.add_argument('--compile-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    doc=json.loads(gzip.decompress(a.program.read_bytes()))
    optimized()
    source=compact_literals(C.emit_cpp(doc))
    src=a.output/'program.cpp';src.write_text(source)
    library=a.output/'program.so'
    command=['g++',*C.FLAGS,str(src),'-o',str(library)]
    subprocess.run(command,check=True)
    mapping={};counter=0
    def walk(body,depth=0):
        nonlocal counter
        for node in body:
            if 'loop' in node:
                key=f'v{counter}';counter+=1
                if depth==0:mapping[key]=node['loop']
                walk(node['body'],depth+1)
    walk(doc['body'])
    (a.output/'loop-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    provenance={'program_sha256':C.document_digest(doc),'command':command,
        'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
        'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
        'literal_compaction':'Consecutive constant stores only; same destination/value/order.',
        'executor_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'build.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if a.compile_only:return
    cfg=doc['metadata']['config'];N,Q,side=cfg['N'],cfg['Q'],cfg['side']
    if a.fixture:
        rng=np.random.default_rng(20260921)
        train=rng.uniform(0,1,(N,side,side)).astype(np.float32)
        query=rng.uniform(0,1,(Q,side,side)).astype(np.float32)
        labels=(np.arange(N)%10).astype(np.uint32)
    else:
        assert (N,Q,side)==(60000,10000,28)
        assert a.raw and a.data_module
        sys.path.insert(0,str(a.data_module))
        from data import official
        # official() loads the canonical files, including test labels for its
        # manifest; the model tape contains no test labels and no scoring uses them.
        data,manifest=official(a.raw)
        (a.output/'data-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        train,labels,query=data[:3]
        labels=np.asarray(labels,dtype=np.uint32)
    tape=np.concatenate([train.reshape(-1).view(np.uint32),query.reshape(-1).view(np.uint32),labels])
    np.save(a.output/'input.npy',tape,allow_pickle=False)
    print('compiled; executing',len(tape),'input words',flush=True)
    result=C.run_shared(library,tape,C.document_digest(doc))
    np.save(a.output/'predictions.npy',result.pop('output'),allow_pickle=False)
    memory=result.pop('memory')
    from affine import Program
    program=Program(doc)
    regions={}
    for name,(base,size) in program.regions.items():
        values=memory[base:base+size]
        regions[name]={'words':size,'sha256':hashlib.sha256(values.tobytes()).hexdigest(),
                       'nonfinite_as_fp32':int((~np.isfinite(values.view(np.float32))).sum())}
    result['regions']=regions
    np.save(a.output/'memory.npy',memory,allow_pickle=False)
    (a.output/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='regions'}),flush=True)


if __name__=='__main__':main()
