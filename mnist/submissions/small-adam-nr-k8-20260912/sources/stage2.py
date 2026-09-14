"""Stage-2 selection: top configs on 10 additional disjoint pilot seeds."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import sweep

HERE = Path(__file__).resolve().parent
STAGE2_SEEDS = list(range(20261006, 20261016))
CONFIGS = [
    (80, 400, 25, 0.2), (96, 400, 50, 0.2), (128, 400, 50, 0.2),
    (80, 300, 25, 0.2), (96, 400, 50, 0.4), (128, 300, 50, 0.2),
    (64, 200, 25, 0.2), (64, 150, 25, 0.2), (32, 300, 25, 0.2),
]


def evaluate(config):
    accs = [sweep.train_eval(config, seed) for seed in STAGE2_SEEDS]
    return config, accs


if __name__ == '__main__':
    stage1 = {(r['config']['width'], r['config']['epochs'], r['config']['batch'], r['config']['lr']): r
              for r in json.loads((HERE / 'sweep.json').read_text())}
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(evaluate, CONFIGS))
    rows = []
    for config, accs in results:
        combined = stage1[config]['pilot_accs'] + accs
        row = {'config': dict(zip(('width', 'epochs', 'batch', 'lr'), config)),
               'combined_mean': float(np.mean(combined)), 'combined_sd': float(np.std(combined, ddof=1)),
               'stage2_mean': float(np.mean(accs)), 'n': len(combined),
               'energy_mj': stage1[config]['energy_mj'], 'time_ms': stage1[config]['time_ms']}
        rows.append(row)
    rows.sort(key=lambda r: -r['combined_mean'])
    (HERE / 'stage2.json').write_text(json.dumps({'seeds': STAGE2_SEEDS, 'rows': rows}, indent=2) + '\n')
    print(f"{'w':>4}{'ep':>5}{'b':>4}{'lr':>5}{'combined':>10}{'sd':>7}{'stage2':>8}{'mJ':>8}{'ms':>7}")
    for r in rows:
        c = r['config']
        print(f"{c['width']:>4}{c['epochs']:>5}{c['batch']:>4}{c['lr']:>5}{r['combined_mean']*100:>10.2f}{r['combined_sd']*100:>7.2f}{r['stage2_mean']*100:>8.2f}{r['energy_mj']:>8.3f}{r['time_ms']:>7.0f}")
