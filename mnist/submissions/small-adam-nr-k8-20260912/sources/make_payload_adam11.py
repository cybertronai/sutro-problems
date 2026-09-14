"""GPU payload: 11 frozen draws + expected predictions + draw-0 mutation cases."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adam_evidence_v2 as ev

HERE = Path(__file__).resolve().parent
E = HERE / 'evidence-adam-v2'


def main():
    frozen = np.load(HERE / 'generated/adam-expected.npz')
    existing = {k: frozen[k] for k in frozen.files}
    payload = {}
    expected = dict(existing)
    for index, seed in enumerate(ev.SEEDS):
        train, test, x, q, target, train_px, test_px, train_lab = ev.build(seed)
        f32 = np.float32
        payload[f'x{index}'] = ev.ds.area_resize(train_px.astype(f32) / f32(255), 3).reshape(1000, 9)
        payload[f'q{index}'] = ev.ds.area_resize(test_px.astype(f32) / f32(255), 3).reshape(1000, 9)
        payload[f'y{index}'] = train_lab.astype(np.int32)
        expected[f'pred{index}'] = np.load(E / 'predictions' / f'draw-{index:02d}.npy', allow_pickle=False)
    base = np.load(HERE / 'generated/adam-payload.npz')
    for key in ('train_images', 'test_images', 'train_labels', 'mutated_queries', 'initial', 'c1', 'c2'):
        payload[key] = base[key]
    np.savez(HERE / 'generated/adam11-payload.npz', **payload)
    np.savez(HERE / 'generated/adam11-expected.npz', **expected)
    print('wrote', {k: v.shape for k, v in payload.items() if k.startswith(('x', 'q', 'y'))})


if __name__ == '__main__':
    main()
