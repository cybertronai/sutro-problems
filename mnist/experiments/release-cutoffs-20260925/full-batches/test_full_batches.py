"""full_batches=True must change nothing where no minibatch is short, and must change the fit where one is.

    /tmp/penv/bin/python -m pytest -q test_full_batches.py      # CPU, about 20 s
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import neural  # noqa: E402
import study  # noqa: E402

CONFIG = json.loads((ROOT / 'selection.json').read_text())['candidates'][0]['config']


def logits(n, full):
    arrays = study.job_arrays(study.DEV_SEEDS[0], n)
    config = dict(CONFIG, target_steps=12, full_batches=full)
    out = neural.fit_predict(arrays['train_z'], arrays['train_y'], arrays['query_z'], config, seed=11,
                             device='cpu', deadline_unix=time.time() + 600)
    return np.asarray(out['logits'])


def test_the_frozen_recipe_is_the_default():
    assert neural.LADDER_DEFAULTS['full_batches'] is False
    assert CONFIG['full_batches'] is True


@pytest.mark.parametrize('n', [100, 1000])
def test_bit_identical_when_no_minibatch_is_short(n):
    assert np.array_equal(logits(n, False), logits(n, True))


def test_changes_the_fit_when_the_last_minibatch_is_short():
    assert not np.array_equal(logits(316, False), logits(316, True))
