"""Reproduce v4 model costs; no learning, GPU jobs or test labels."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import panels
from learner import config

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=('energy','baseline','no_slowdown'),default='energy')
    args = parser.parse_args()
    document = panels.build(None if args.variant=='baseline' else config(args.variant))
    measured = panels.il.score(document)
    samples = []
    for _ in range(5):
        start = time.perf_counter(); panels.il.score(document); samples.append(time.perf_counter()-start)
    saved_name = 'optimized' if args.variant=='energy' else args.variant
    saved = json.loads((HERE/'costs.json').read_text())[saved_name]
    for key in ('energy_fj','time_ticks_0_2_ps','peak_initialized_scratch_words','instructions','charged_reads','charged_writes'):
        assert measured[key]==saved[key],key
    args.output.mkdir(parents=True,exist_ok=True)
    program = json.dumps(document,separators=(',',':'))+'\n'
    result = {**measured,'energy_mj':measured['energy_fj']/1e12,'time_ms':measured['time_ticks_0_2_ps']*2e-10,
        'area_mm2':measured['peak_initialized_scratch_words']/1e6,
        'time_to_score_wall_samples':samples,'time_to_score_wall_median':statistics.median(samples),
        'file_sha256':hashlib.sha256(program.encode()).hexdigest()}
    (args.output/'program.il.json').write_text(program)
    (args.output/'model-score.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
