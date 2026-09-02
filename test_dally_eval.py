import json

import pytest

import dally_eval


def _fake_binary(tmp_path):
    binary = tmp_path / "dally-eval"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "assert sys.argv[1:] == ['verify']\n"
        "text = sys.stdin.read()\n"
        "if text == 'bad':\n"
        "    print('malformed IR', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "print(json.dumps({'cost': 123, 'ops': 1, 'inputs': 1, 'outputs': 1}))\n"
    )
    binary.chmod(0o755)
    return binary


def test_static_cost_uses_verify_stdin_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("DALLY_EVAL_BIN", str(_fake_binary(tmp_path)))
    assert dally_eval.static_cost("valid IR") == 123


def test_static_cost_surfaces_cli_rejection(tmp_path, monkeypatch):
    monkeypatch.setenv("DALLY_EVAL_BIN", str(_fake_binary(tmp_path)))
    with pytest.raises(ValueError, match="malformed IR"):
        dally_eval.static_cost("bad")


def test_static_cost_returns_none_without_binary(monkeypatch):
    monkeypatch.setenv("DALLY_EVAL_BIN", "/missing/dally-eval")
    monkeypatch.setattr(dally_eval.shutil, "which", lambda _: None)
    assert dally_eval.static_cost("anything") is None


@pytest.mark.parametrize("output", ["not json", json.dumps({"ops": 1})])
def test_parse_cost_rejects_invalid_cli_output(output):
    with pytest.raises(ValueError):
        dally_eval._parse_cost(output)
