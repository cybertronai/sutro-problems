import pytest

import sweep_with_dally as sweep


def test_candidates_honors_exact_count():
    assert list(sweep.candidates(5)) == [
        (0, 2), (0, 3), (0, 4), (1, 2), (1, 3),
    ]
    with pytest.raises(ValueError, match="positive"):
        list(sweep.candidates(0))


def test_rank_by_cap_never_compares_different_caps():
    ranked = sweep.rank_by_cap([
        (90, 0, 3),
        (70, 2, 2),
        (80, 1, 3),
        (100, 0, 2),
    ])
    assert ranked == {
        2: [(70, 2), (100, 0)],
        3: [(80, 1), (90, 0)],
    }
