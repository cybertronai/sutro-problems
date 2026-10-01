"""Verify the five changed sources and one official run per category (stdlib only)."""

import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
sys.path.insert(0, str(BASE))
import mnist

CONTRACTS = {
    1: ("mlp_d1.py", {"STEPS": 200}),
    2: ("mlp_d2.py", {"K": 8, "TARGET_STEPS": 500, "EMA": 0.992}),
    3: ("ladder_d3.py", {"STEPS": 1100}),
    4: ("ladder_d4.py", {"STEPS": 2200}),
    5: ("ladder_d5.py", {"STEPS": 8400}),
}
TF32 = {"torch.backends.cuda.matmul.allow_tf32", "torch.backends.cudnn.allow_tf32"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_tree(source):
    tree = ast.parse(source)
    # Only documentation, the entry-point wrapper and the upstream CLI are packaging.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        or isinstance(node, ast.FunctionDef) and node.name == "custom_kernel"
        or isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
        or isinstance(node, ast.Import) and [a.name for a in node.names] == ["mnist"]
    )]
    return tree


def constants(tree, replacements=None):
    """Read literal module constants; optionally replace exactly the named constants."""
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            pairs = [(target, node.value)]
        elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
            pairs = list(zip(target.elts, node.value.elts))
        else:
            continue
        for name, value in pairs:
            if not isinstance(name, ast.Name) or not isinstance(value, ast.Constant):
                continue
            found[name.id] = value.value
            if replacements is not None and name.id in replacements:
                value.value = replacements[name.id]
    if replacements is not None:
        assert replacements.keys() <= found.keys(), (replacements, found)
    return found


def check_changes(source, upstream, difficulty):
    """Permit only the documented compute reductions and D1 optimizer/TF32 changes."""
    source_text = source.read_text()
    submitted = canonical_tree(source_text)
    expected = canonical_tree(upstream.read_text())
    changes = CONTRACTS[difficulty][1]
    values = constants(submitted)
    assert all(values.get(name) == value for name, value in changes.items()), values
    assert values["BATCH"] == (512 if difficulty <= 2 else 1000)
    before = constants(expected, changes)
    assert all(before[name] != value for name, value in changes.items()), before

    wrapper = next(n for n in ast.parse(source_text).body
                   if isinstance(n, ast.FunctionDef) and n.name == "custom_kernel")
    target = "mlp" if difficulty <= 2 else "ladder"
    expected_wrapper = ast.parse(
        f"def custom_kernel(train_x, train_y, test_x):\n"
        f"    return {target}(train_x, train_y, test_x)\n"
    ).body[0]
    assert ast.dump(wrapper) == ast.dump(expected_wrapper), "entry-point wrapper changed"

    if difficulty == 1:
        switches = [n for n in submitted.body if isinstance(n, ast.Assign)
                    and len(n.targets) == 1 and ast.unparse(n.targets[0]) in TF32]
        assert len(switches) == 2
        assert {ast.unparse(n.targets[0]) for n in switches} == TF32
        assert all(isinstance(n.value, ast.Constant) and n.value.value is True for n in switches)
        submitted.body = [n for n in submitted.body if n not in switches]
        optimizers = [n for n in ast.walk(submitted) if isinstance(n, ast.Call)
                      and ast.unparse(n.func) == "torch.optim.AdamW"]
        assert len(optimizers) == 1
        fused = [kw for kw in optimizers[0].keywords if kw.arg == "fused"]
        assert len(fused) == 1
        assert ast.dump(fused[0].value) == ast.dump(ast.parse(
            'train_x.device.type == "cuda"', mode="eval").body)
        optimizers[0].keywords.remove(fused[0])

    # This includes every Ladder learning function: only its step constant may change.
    assert ast.dump(submitted) == ast.dump(expected), f"D{difficulty}: undocumented source change"


def check_run(entry, source, run):
    difficulty = entry["difficulty"]
    path = HERE / f"evidence/d{difficulty}/run-{run}.json"
    result = json.loads(path.read_text())
    assert result["returncode"] == 0, result.get("stderr", result)
    record = result["record"]
    assert record["version"] == mnist.VERSION
    assert record["source_sha256"] == entry["sha256"]
    assert record["file"] == source.name
    assert record["function"] == entry["function"]
    assert record["sandboxed"] is True
    assert "A100" in record["device"] and "80GB" in record["device"]
    assert record["band_bp"] == mnist.band_bp(difficulty)
    calls = [SimpleNamespace(**call) for call in record["calls"]]
    assert len(calls) == 15
    assert sum(c.dataset == "mnist" for c in calls) == 11
    assert record["holdout"] in mnist.FOREIGN
    assert all(c.dataset in ("mnist", record["holdout"]) for c in calls)
    assert all(c.total == 10000 and 0 <= c.correct <= c.total for c in calls)
    assert all(0 < c.ms <= mnist.MAX_CALL_MS for c in calls)
    assert all(math.isfinite(c.parent_ms) and math.isfinite(c.overhead_ms) for c in calls)
    problems, score, _ = mnist.judge(calls, record["band_bp"])
    assert not problems and not record["problems"], problems
    assert math.isclose(score, record["ranked_ms"], rel_tol=1e-12)
    energy = record.get("energy") or {}
    assert not energy.get("problems"), energy
    assert energy.get("mj_per_call", 0) > 0, energy
    rows = [c for c in calls if c.dataset == "mnist"]
    checked_energy = mnist.energy_summary(
        energy["windows"], energy["device"], record["band_bp"],
        mnist._median([c.ms for c in rows]),
    )
    assert not checked_energy["problems"], checked_energy["problems"]
    assert math.isclose(checked_energy["mj_per_call"], energy["mj_per_call"], rel_tol=1e-12)
    error = 100 * (1 - sum(c.correct for c in rows) / sum(c.total for c in rows))
    print(f"D{difficulty} run {run}: PASS, error {error:.4f}%, "
          f"{score:.3f} ms, {energy['mj_per_call']:.1f} mJ")


def main():
    manifest = json.loads((HERE / "evidence/provenance.json").read_text())
    assert digest(BASE / "mnist.py") == manifest["scorer_sha256"]
    assert sorted(e["difficulty"] for e in manifest["entries"]) == list(CONTRACTS)
    for entry in manifest["entries"]:
        difficulty = entry["difficulty"]
        assert entry["file"] == CONTRACTS[difficulty][0]
        assert entry["function"] == "custom_kernel"
        source = HERE / entry["file"]
        upstream = BASE / entry["upstream_file"]
        assert digest(source) == entry["sha256"]
        assert digest(upstream) == entry["upstream_sha256"]
        assert source.stat().st_size == entry["bytes"] <= mnist.MAX_SOURCE_BYTES
        check_changes(source, upstream, difficulty)
        mnist.check_source(source.read_bytes(), entry["function"], source.name)
        assert mnist.review_flags(source.read_bytes()) == entry["review_flags"]
        check_run(entry, source, 1)
    print("Verified all five sources and five full sandboxed A100 runs (75 timed calls).")


if __name__ == "__main__":
    main()
