"""Build and score a configured medium MLP through exact affine-v4 aggregation.

The scorer is the unchanged generic prototype from the frozen small study.
Numerical training and accuracy verification are separate from this command.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import numpy as np
from model_ir import build_mlp, EXPERIMENT
from il import score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    for name in ('width','epochs','seed','features','n-train','n-test','batch'):
        parser.add_argument('--'+name,type=int)
    parser.add_argument('--learning-rate',type=float)
    parser.add_argument('--buffer-all-queries',action='store_true',help='Legacy comparison only: retain every query')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    if args.repeats<1:
        parser.error('At least one scoring call is required')
    configured=json.loads(args.config.read_text()) if args.config else {}
    if 'configuration' in configured:
        configured=configured['configuration']
    aliases={'width':'hidden_width','batch':'batch_size','n_train':'train_examples','n_test':'test_examples'}
    defaults={'seed':101,'features':81,'n_train':6000,'n_test':6000,'batch':30,'learning_rate':.05}
    options={}
    for name in ('width','epochs','learning_rate','seed','features','n_train','n_test','batch'):
        value=getattr(args,name)
        if value is None:
            value=configured.get(name,configured.get(aliases.get(name,''),defaults.get(name)))
        if value is None:
            parser.error(f'{name} must be supplied by --config or an explicit option')
        options[name]=value
    options['stream_queries']=False if args.buffer_all_queries else configured.get('stream_queries',True)
    document=build_mlp(**options)
    results=[score(document) for _ in range(args.repeats)]
    fields=('program_sha256','instructions','total_instructions','charged_reads','charged_writes',
            'time_ticks_0_2_ps','energy_fj','area_um2_occupied_cells','peak_initialized_scratch_words')
    assert all(result[key]==results[0][key] for result in results for key in fields)
    result=results[0]
    result['configuration']=options
    result['time_to_score_samples_seconds']=[item['time_to_score_seconds'] for item in results]
    result['time_to_score_seconds']=statistics.median(result['time_to_score_samples_seconds'])
    result['time_to_score_statistic']=f'median of {args.repeats} complete static scoring calls'
    result['time_ms']=result['time_ticks_0_2_ps']/5e9
    result['energy_mj']=result['energy_fj']/1e12
    result['area_mm2_occupied_cells']=result['area_um2_occupied_cells']/1e6
    result['source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (Path(__file__),HERE/'model_ir.py',EXPERIMENT/'il.py')}
    result['software']={'python':sys.version,'numpy':np.__version__,'platform':platform.platform()}
    result['allocation_policy']='One query next to hot scalars; receive/normalize/infer/send after training' if options['stream_queries'] else 'Legacy full query buffer'
    result['scorer_limits_unchanged']=True
    result['classification_checked_by_this_command']=False
    args.output.mkdir(parents=True,exist_ok=True)
    program=args.output/'program.il.json'
    program.write_text(json.dumps(document,indent=2)+'\n')
    result['program_file_bytes']=program.stat().st_size
    result['program_file_sha256']=hashlib.sha256(program.read_bytes()).hexdigest()
    (args.output/'model-score.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
