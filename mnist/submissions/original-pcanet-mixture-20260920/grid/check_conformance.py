"""Recheck scalar/parallel/interpreter parity and seeded rank selection."""
from pathlib import Path
import argparse,gzip,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'shared'),str(HERE/'imported')]
import il_compiler as C
import pcanet_ir as P
from affine import Program
import parallel_compiler
import build_seeded


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--torch-seeds',action='store_true',help='Also compare against installed PyTorch CPU randperm')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    d=json.loads(gzip.decompress((HERE/'conformance/program.json.gz').read_bytes()))
    serial=C.compile_shared(d,a.output/'serial')
    parallel_compiler.install()
    parallel=C.compile_shared(d,a.output/'parallel')
    program=Program(d);fixtures=[]
    for name,directory in [('dense',HERE/'conformance'),('sparse_imbalanced',HERE/'conformance/sparse')]:
        tape=np.load(directory/'input.npy',allow_pickle=False)
        expected=np.load(directory/'memory.npy',allow_pickle=False)
        s=C.run_shared(serial,tape);q=C.run_shared(parallel,tape)
        assert np.array_equal(s['memory'],q['memory']) and np.array_equal(s['memory'],expected),name
        assert np.array_equal(s['output'],q['output']),name
        out,regions=P.execute(d,tape,return_memory=True)
        assert np.array_equal(np.asarray(out,dtype=np.uint32),s['output']),name
        for region,values in regions.items():
            base,size=program.regions[region]
            assert np.array_equal(values.view(np.uint32),s['memory'][base:base+size]),(name,region)
            assert np.isfinite(values).all(),(name,region)
        fixtures.append({'name':name,'all_words_match':program.words,'predictions_match':len(out)})
    C.emit_cpp=parallel_compiler.ORIGINAL
    rng_doc=build_seeded.rng_document()
    rng_library=C.compile_shared(rng_doc,a.output/'rng')
    if a.torch_seeds:import torch
    cases=0
    for n in [8,9,10,11,12,15,17,31,127,500,5421,6000,6742,60000]:
        result=C.run_shared(rng_library,np.full(10,n,dtype=np.float32).view(np.uint32))
        actual=result['output'].view(np.float32).astype(np.int64).reshape(10,8)
        for c in range(10):
            words=np.random.RandomState(900+c).randint(0,2**32,8,dtype=np.uint32)
            reference=list(range(n))
            for i,w in enumerate(words):
                j=i+int(w)%(n-i);reference[i],reference[j]=reference[j],reference[i]
            assert np.array_equal(actual[c],reference[:8]),(n,c)
            if a.torch_seeds:
                expected=torch.randperm(n,generator=torch.Generator().manual_seed(900+c))[:8].numpy()
                assert np.array_equal(actual[c],expected),(n,c)
            cases+=1
    report={'passed':True,'fixtures':fixtures,'rng_cases':cases,'torch_checked':a.torch_seeds}
    (a.output/'conformance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
