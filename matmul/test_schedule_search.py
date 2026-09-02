"""Focused regression tests for the 4x4 schedule-search harness."""
from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

import matmul
from matmul import schedule_search


@pytest.mark.parametrize(
    "tree,order,expected_cost",
    [
        ("left", "batched", 1_258),
        ("right", "batched", 1_258),
        ("balanced", "batched", 1_325),
        ("left", "pipelined", 1_309),
    ],
)
def test_recycling_releases_every_consumed_temporary(
    tree: str,
    order: str,
    expected_cost: int,
) -> None:
    rng = random.Random(0)
    ops, outputs = schedule_search.build_schedule(tree, order, True, rng)
    ir = schedule_search.to_ir(ops, outputs)

    assert schedule_search.score(ops, outputs) == expected_cost
    assert matmul.score_4x4(ir) == expected_cost

    unrecycled_ops, unrecycled_outputs = schedule_search.build_schedule(
        tree, order, False, random.Random(0)
    )
    assert expected_cost < schedule_search.score(
        unrecycled_ops, unrecycled_outputs
    )


def test_pipelined_order_has_one_supported_tree_shape() -> None:
    with pytest.raises(ValueError, match="left-linear"):
        schedule_search.build_schedule(
            "balanced", "pipelined", True, random.Random(0)
        )


def test_direct_script_execution_uses_the_scorer(
    tmp_path: Path,
) -> None:
    assert schedule_search.__file__ is not None
    result = subprocess.run(
        [sys.executable, schedule_search.__file__, "1"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    ir = (tmp_path / "best_schedule.ir").read_text()
    assert matmul.score_4x4(ir) == 1_258
    assert "symbolically verified IR" in result.stdout


def test_main_symbolically_verifies_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "best_schedule.ir"
    verified: list[str] = []

    def verify(ir: str) -> int:
        assert not output.exists()
        verified.append(ir)
        return matmul.score_4x4(ir)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["schedule_search.py", "1"])
    monkeypatch.setattr(schedule_search, "score_4x4", verify)

    schedule_search.main()

    assert verified == [output.read_text()]


def test_main_rejects_a_cost_mismatch_before_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["schedule_search.py", "1"])
    monkeypatch.setattr(schedule_search, "score_4x4", lambda ir: -1)

    with pytest.raises(RuntimeError, match="cost mismatch"):
        schedule_search.main()

    assert not (tmp_path / "best_schedule.ir").exists()
