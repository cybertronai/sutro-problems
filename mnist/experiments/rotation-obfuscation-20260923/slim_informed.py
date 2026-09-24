#!/usr/bin/env python
"""Move the 81x81 recovery maps out of results/informed-*.json into results/informed-maps-s<seed>.npz
so the JSON stays small enough to commit.  figures.py and stage_smuggle read the npz when the
JSON no longer carries 'H'."""
import glob, json
import numpy as np
import common
for seed in (1, 2, 3):
    maps = {}
    for f in sorted(glob.glob(str(common.HERE / 'results' / f'informed-*-s{seed}.json'))):
        rec = json.load(open(f))
        if 'variant' not in rec:
            continue
        for name, entry in rec['inits'].items():
            for tag in ('init', 'refined'):
                if 'H' in entry[tag]:
                    maps[f"{rec['variant']}|{name}|{tag}|H"] = np.asarray(entry.pop('H') if False else entry[tag].pop('H'), np.float32)
                    maps[f"{rec['variant']}|{name}|{tag}|offset"] = np.asarray(entry[tag].pop('offset'), np.float32)
        common.jdump(f, rec)
    if maps:
        np.savez_compressed(common.HERE / 'results' / f'informed-maps-s{seed}.npz', **maps)
        print('seed', seed, len(maps), 'arrays moved')
