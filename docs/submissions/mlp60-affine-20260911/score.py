"""Generate and exactly score the complete fixed MLP program without expansion."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parents[1]/'experiments'/'accuracy-il-20260911'
sys.path.insert(0, str(EXPERIMENT))
from mlp_il import build_mlp
from il import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    document=build_mlp(32,300,0.2,101)
    samples=[score(document) for _ in range(5)]
    prior=json.loads((EXPERIMENT/'h32-e300-lr0.2.score.json').read_text())
    exact_fields=['program_sha256','instructions','total_instructions','charged_reads','charged_writes',
                  'time_ticks_0_2_ps','energy_fj','area_um2_occupied_cells']
    assert all(sample[key]==prior[key] for sample in samples for key in exact_fields)
    result=samples[0]
    result['time_to_score_samples_seconds']=[sample['time_to_score_seconds'] for sample in samples]
    result['time_to_score_seconds']=statistics.median(result['time_to_score_samples_seconds'])
    result['time_to_score_statistic']='median of five complete scoring calls'
    result['area_mm2_occupied_cells']=result['area_um2_occupied_cells']/1e6
    result['time_ms']=result['time_ticks_0_2_ps']/5e9
    result['energy_mj']=result['energy_fj']/1e12
    result['matches_frozen_study_exactly']=True
    result['software']={'python':sys.version,'numpy':np.__version__,'platform':platform.platform()}
    result['source_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [Path(__file__),EXPERIMENT/'mlp_il.py',EXPERIMENT/'il.py']}
    args.output.mkdir(parents=True,exist_ok=True)
    program=args.output/'program.il.json'
    program.write_text(json.dumps(document,indent=2)+'\n')
    result['program_file_bytes']=program.stat().st_size
    (args.output/'model-score.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
