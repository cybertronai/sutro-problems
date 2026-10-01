"""Check provenance and recompute verdicts from the saved official runs (stdlib only)."""

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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def learning_ast(source):
    tree = ast.parse(source)
    # The wrappers, local launcher, and its scorer import are packaging only.
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name == "custom_kernel"
        or isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
        or isinstance(node, ast.Import) and [a.name for a in node.names] == ["mnist"]
    )]
    return ast.dump(tree, include_attributes=False)


def main():
    manifest = json.loads((HERE / "evidence/provenance.json").read_text())
    assert digest(BASE / "mnist.py") == manifest["scorer_sha256"]
    for entry in manifest["entries"]:
        d = entry["difficulty"]
        source = HERE / entry["file"]
        upstream = BASE / entry["upstream_file"]
        assert digest(source) == entry["sha256"]
        assert digest(upstream) == entry["upstream_sha256"]
        assert learning_ast(source.read_text()) == learning_ast(upstream.read_text())
        mnist.check_source(source.read_bytes(), entry["function"], source.name)
        result = json.loads((HERE / f"evidence/d{d}/run-1.json").read_text())
        assert result["returncode"] == 0, result["stderr"]
        record = result["record"]
        assert record["version"] == mnist.VERSION
        assert record["source_sha256"] == entry["sha256"]
        assert record["file"] == source.name
        assert record["function"] == entry["function"]
        assert record["sandboxed"] is True
        assert "A100" in record["device"] and "80GB" in record["device"]
        assert record["band_bp"] == mnist.band_bp(d)
        calls = [SimpleNamespace(**call) for call in record["calls"]]
        assert len(calls) == 15
        assert sum(c.dataset == "mnist" for c in calls) == 11
        assert record["holdout"] in mnist.FOREIGN
        assert all(c.dataset in ("mnist", record["holdout"]) for c in calls)
        assert all(c.total == 10000 and c.ms <= mnist.MAX_CALL_MS for c in calls)
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
        print(f"D{d}: PASS, error {error:.4f}%, {score:.3f} ms, {energy['mj_per_call']:.1f} mJ")


if __name__ == "__main__":
    main()
