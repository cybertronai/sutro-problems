"""Compare numerical specialization stages with generic compiled IL memory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from executor import run

parser=argparse.ArgumentParser()
parser.add_argument("fixture",type=Path)
parser.add_argument("--affine-dir",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.affine_dir))
from affine import Program
doc=json.loads((args.fixture/"program.json").read_text())
program=Program(doc)
memory_path=args.fixture/"compiled/memory.npy"
if not memory_path.exists():memory_path=args.fixture/"memory.compiled.npy"
memory=np.load(memory_path,allow_pickle=False)
tape=np.load(args.fixture/"tape.npy",allow_pickle=False)
meta=doc["metadata"]
n,qn,F,T=(meta[k] for k in ("N","Q","F","T"))
def region(name):
    base,length=program.regions[name]
    return memory[base:base+length]
snapshot={}
x=tape[:n*81].view(np.float32).reshape(n,81)
y=tape[n*81:n*81+n].astype(np.int64)
q=tape[n*81+n:].view(np.float32).reshape(qn,81)
start=time.perf_counter()
pred,scores=run(x,y,q,region("w").view(np.float32),region("bias").view(np.float32),
    iterations=T,threads=16,snapshot=lambda name,array:snapshot.__setitem__(name,array.copy()))
elapsed=time.perf_counter()-start
comparisons={}
for name,array in snapshot.items():
    target="sc2" if name=="sc2_normalized" else name
    if target not in program.regions:
        continue
    observed=array.reshape(-1).view(np.uint32)
    expected=region(target)
    different=np.flatnonzero(observed!=expected)
    comparisons[target]={"words":len(observed),"different_words":len(different),
        "sha256":hashlib.sha256(observed.tobytes()).hexdigest()}
    if len(different):
        j=int(different[0]); comparisons[target].update(first_difference=j,
            expected_word=int(expected[j]),observed_word=int(observed[j]),
            expected_float=str(expected.view(np.float32)[j]),observed_float=str(observed.view(np.float32)[j]))
output_path=args.fixture/"compiled/output.npy"
if not output_path.exists():output_path=args.fixture/"output.compiled.npy"
expected_out=np.load(output_path,allow_pickle=False)
comparisons["out"]={"words":len(pred),"different_words":int(np.count_nonzero(pred!=expected_out))}
result={"all_compared_regions_bitwise_equal":all(v["different_words"]==0 for v in comparisons.values()),
    "shape":meta,"elapsed_seconds":elapsed,"predictions":pred.tolist(),"regions":comparisons}
args.output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
raise SystemExit(0 if result["all_compared_regions_bitwise_equal"] else 1)
