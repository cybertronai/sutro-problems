#!/usr/bin/env python3
"""End-to-end submission runner for the wikitext energy benchmark.

What it does:
  1. Build a Docker image with the user's submission baked in.
  2. Push it to GHCR (you must already be logged in to ghcr.io).
  3. Provision a Lambda On-Demand A100 SXM4 instance.
  4. Cloud-init pulls + runs the image; entrypoint.sh writes
     /results/result.json on success or /results/FAIL on any failure.
  5. Orchestrator opens one blocking SSH command that waits for either
     sentinel, prints the result, then terminates the instance (always,
     in a finally block).
  6. Saves the result JSON to submissions/ and appends a row to the
     Record History table in README.md.

Setup (once):
  - Lambda Cloud account; SSH key registered on cloud.lambda.ai with the
    matching private key loaded in your local ssh-agent.
  - GHCR auth on this machine:
      gh auth token | docker login ghcr.io -u <github_username> --password-stdin
  - Environment:
      export LAMBDA_API_KEY=...        # cloud.lambda.ai → API keys
      export GHCR_USER=<github_username>

Usage:
  python3 submit.py path/to/my_submission.py --config small
  python3 submit.py example_submission.py --config tiny --yes
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import task  # task-pinned constants — single source of truth

LAMBDA_API = "https://cloud.lambda.ai/api/v1"
INSTANCE_TYPE = task.INSTANCE_TYPE

# (est_instance_minutes, est_cost_usd_at_$1.79/hr) — informational, drives
# only the wall-clock timeout. These cover the **whole instance lifetime**
# (boot + image pull + NVML probe + data fetch + train + eval), which is
# what Lambda actually bills for. Per-config training-only wall-clocks in
# RUNBOOK.md are smaller; the gap is ~5–7 min of overhead per submission.
COST_ESTIMATES = {
    "tiny":  (10, 0.30),
    "small": (50, 1.50),
    "gpt2":  (310, 9.30),
}

HERE = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from the nearest .env on the way up to the
    repo root. Doesn't overwrite anything already set in os.environ —
    so explicit `export FOO=bar` still wins.
    """
    for d in (HERE, *HERE.parents):
        env = d / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
            return


_load_dotenv()


# ---------------------------------------------------------------------------
# Lambda Cloud REST helpers
# ---------------------------------------------------------------------------

def _api_auth_header() -> str:
    api_key = os.environ.get("LAMBDA_API_KEY")
    if not api_key:
        sys.exit("LAMBDA_API_KEY not set in environment")
    return f"Bearer {api_key}"


def lambda_request(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{LAMBDA_API}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": _api_auth_header(),
            "Content-Type": "application/json",
            # Cloudflare in front of cloud.lambda.ai blocks the default
            # Python-urllib UA with a 403; pass an explicit one.
            "User-Agent": "wikitext-submit/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"Lambda API {method} {path} failed: {e.code} {e.read().decode()}")


def find_available_region() -> str:
    info = lambda_request("GET", "/instance-types")["data"]
    if INSTANCE_TYPE not in info:
        sys.exit(f"Lambda no longer offers {INSTANCE_TYPE}; check API")
    regions = info[INSTANCE_TYPE]["regions_with_capacity_available"]
    if not regions:
        sys.exit(f"no Lambda capacity for {INSTANCE_TYPE} in any region; "
                 f"try later or fall back to RunPod (see RUNBOOK.md)")
    return regions[0]["name"]


def pick_ssh_key(name: str | None) -> str:
    keys = lambda_request("GET", "/ssh-keys")["data"]
    if not keys:
        sys.exit("no SSH keys registered with Lambda; add one in the dashboard first")
    if name:
        for k in keys:
            if k["name"] == name:
                return name
        sys.exit(f"SSH key {name!r} not found among: {[k['name'] for k in keys]}")
    return keys[0]["name"]


def list_running_wikitext_instances() -> list[dict]:
    """Return only instances whose name matches the exact pattern
    ``submit.py`` creates (``wikitext-<10-digit-unix-timestamp>``).

    Other ``wikitext-…``-prefixed instances (e.g. ``wikitext-baseline``,
    ``wikitext-timing-…``) belong to the user's own work and must not
    be flagged as "leaked" — that's how a previous run accidentally
    terminated active user instances.
    """
    import re
    pat = re.compile(r"^wikitext-\d{10}$")
    insts = lambda_request("GET", "/instances")["data"]
    return [i for i in insts if pat.match(i.get("name") or "")]


# ---------------------------------------------------------------------------
# Image build + push
# ---------------------------------------------------------------------------

HARNESS_FILES = (
    "wikitext.py",
    "baseline_ngram.py",
    "baseline_transformer.py",
    "run_eval.py",
    "verify_nvml.py",
    "fetch_data.py",
    "task.py",
    "entrypoint.sh",
    "Dockerfile",
)


def check_ghcr_login() -> None:
    """Verify the local Docker config has a ghcr.io entry before the
    slow build+push step. Saves a multi-minute round-trip when the
    user hasn't run ``docker login ghcr.io`` yet.
    """
    cfg_path = Path.home() / ".docker" / "config.json"
    msg = (
        "GHCR auth not found. Run:\n"
        "  gh auth token | docker login ghcr.io -u $GHCR_USER --password-stdin"
    )
    if not cfg_path.exists():
        sys.exit(msg)
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        sys.exit(msg)
    if "ghcr.io" not in (cfg.get("auths") or {}):
        sys.exit(msg)


def build_and_push_image(submission_path: Path, repo_owner: str) -> str:
    sub_name = submission_path.stem.replace("_", "-")
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "--short=10", "HEAD"], cwd=HERE
    ).decode().strip()
    tag = f"ghcr.io/{repo_owner.lower()}/wikitext-bench:{sub_name}-{git_sha}"

    print(f"[build] {tag}")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for f in HARNESS_FILES:
            shutil.copy(HERE / f, td_path / f)
        shutil.copy(submission_path, td_path / "submission.py")
        subprocess.run(["docker", "build", "-t", tag, str(td_path)], check=True)

    print(f"[push] {tag}")
    subprocess.run(["docker", "push", tag], check=True)
    return tag


# ---------------------------------------------------------------------------
# Launch + wait + terminate
# ---------------------------------------------------------------------------

CLOUD_INIT_TEMPLATE = """#cloud-config
write_files:
  - path: /opt/run.sh
    permissions: '0755'
    content: |
      #!/bin/bash
      set -e
      mkdir -p /results
      chmod 0777 /results
      while ! docker info >/dev/null 2>&1; do sleep 2; done
      docker pull {image}
      docker run --rm --gpus all -v /results:/results {image} >>/results/docker.log 2>&1 || true
runcmd:
  - /opt/run.sh
"""


def launch_instance(image_tag: str, ssh_key: str, region: str) -> str:
    body = {
        "region_name": region,
        "instance_type_name": INSTANCE_TYPE,
        "ssh_key_names": [ssh_key],
        "user_data": CLOUD_INIT_TEMPLATE.format(image=image_tag),
        "name": f"wikitext-{int(time.time())}",
    }
    print(f"[launch] {INSTANCE_TYPE} in {region}")
    resp = lambda_request("POST", "/instance-operations/launch", body)
    return resp["data"]["instance_ids"][0]


def wait_for_active(instance_id: str, timeout_s: int = 600) -> str:
    print(f"[wait] instance {instance_id} → active")
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        info = lambda_request("GET", f"/instances/{instance_id}")["data"]
        if info["status"] == "active" and info.get("ip"):
            print(f"  active at {info['ip']}")
            return info["ip"]
        time.sleep(10)
    raise TimeoutError(f"instance {instance_id} never became active")


SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
]


def wait_for_ssh(ip: str, timeout_s: int = 300) -> None:
    print(f"[wait] ssh {ip}")
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        r = subprocess.run(
            ["ssh", *SSH_OPTS, "-o", "ConnectTimeout=5", f"ubuntu@{ip}", "true"],
            capture_output=True,
        )
        if r.returncode == 0:
            return
        time.sleep(5)
    raise TimeoutError(f"ssh to {ip} never became reachable")


def wait_for_result(ip: str, timeout_s: int) -> dict:
    """Block on a single SSH connection until /results/result.json or
    /results/FAIL appears, then print + parse the result JSON."""
    print(f"[wait] result (timeout {timeout_s // 60} min)")
    remote_cmd = (
        "until [ -f /results/result.json ] || [ -f /results/FAIL ]; do sleep 10; done; "
        "if [ -f /results/result.json ]; then cat /results/result.json; "
        "else echo __FAIL__; tail -100 /results/error.log 2>/dev/null; "
        "tail -50 /results/docker.log 2>/dev/null; fi"
    )
    r = subprocess.run(
        ["ssh", *SSH_OPTS, "-o", "ServerAliveInterval=30", f"ubuntu@{ip}", remote_cmd],
        capture_output=True, text=True, timeout=timeout_s,
    )
    if r.stdout.startswith("__FAIL__"):
        raise RuntimeError(f"container failed:\n{r.stdout}")
    return json.loads(r.stdout)


def terminate_instance(instance_id: str) -> None:
    print(f"[terminate] {instance_id}")
    lambda_request(
        "POST",
        "/instance-operations/terminate",
        {"instance_ids": [instance_id]},
    )


def scp_artifacts(ip: str, sub_name: str, date: str) -> None:
    """Pull /results/run.log + /results/nvml.json from the Lambda host
    into submissions/ with the names submissions/README.md documents.

    Failures are logged but non-fatal — result.json already has the
    canonical (energy, accuracy) tuple; the log + NVML evidence files
    are supporting artifacts.
    """
    out_dir = HERE / "submissions"
    out_dir.mkdir(exist_ok=True)
    pulls = [
        ("/results/run.log", f"{sub_name}_{date}.log"),
        ("/results/nvml.json", f"{sub_name}_{date}.nvml.json"),
    ]
    for remote, local_name in pulls:
        local = out_dir / local_name
        r = subprocess.run(
            ["scp", *SSH_OPTS, f"ubuntu@{ip}:{remote}", str(local)],
            capture_output=True,
        )
        if r.returncode == 0 and local.exists():
            print(f"[scp] {remote} → submissions/{local_name}")
        else:
            print(f"[scp] WARN: could not pull {remote} "
                  f"({r.stderr.decode().strip() or 'unknown error'})")


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(result: dict, submission_path: Path) -> Path:
    out_dir = HERE / "submissions"
    out_dir.mkdir(exist_ok=True)
    sub_name = submission_path.stem
    date = result["date_utc"][:10]
    out_path = out_dir / f"{sub_name}_{date}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return out_path


def append_record(result: dict, json_relpath: str) -> None:
    """Append one row to the Record History table in README.md.

    Replaces the placeholder dash row if present, otherwise appends.
    """
    readme = HERE / "README.md"
    text = readme.read_text()
    energy = result.get("training_energy_J")
    energy_cell = f"{energy:>10,.0f}" if energy is not None else "         —"
    row = (
        f"| {result['date_utc'][:10]} "
        f"| {energy_cell} "
        f"| {result['test_char_accuracy']:.4f} "
        f"| {result['submission']} "
        f"| [json]({json_relpath}) "
        f"| @you |\n"
    )
    placeholder = "| —    |          — |        — | —      | —          | —           |\n"
    if placeholder in text:
        text = text.replace(placeholder, row, 1)
    else:
        text = text.rstrip() + "\n" + row
    readme.write_text(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("submission", type=Path,
                   help="Python file exposing train(train_text, valid_text=None) -> CharModel")
    p.add_argument("--config", choices=list(COST_ESTIMATES), default="small",
                   help="Cost-estimate preset (used for the wall-clock timeout). "
                        "Submission code is not bound by it.")
    p.add_argument("--ssh-key", default=None,
                   help="Lambda SSH key name (defaults to first registered)")
    p.add_argument("--yes", action="store_true",
                   help="Skip cost confirmation prompt")
    args = p.parse_args()

    if not args.submission.exists():
        sys.exit(f"submission file not found: {args.submission}")
    repo_owner = os.environ.get("GHCR_USER")
    if not repo_owner:
        sys.exit("GHCR_USER not set in environment")

    est_min, est_cost = COST_ESTIMATES[args.config]
    print(f"╭─ Lambda A100 wikitext submission ────────────")
    print(f"│  submission:    {args.submission}")
    print(f"│  config:        {args.config}")
    print(f"│  est. instance: ~{est_min} min  (boot + train + eval)")
    print(f"│  est. cost:     ~${est_cost:.2f}")
    print(f"╰───────────────────────────────────────────────")

    # Surface and offer to clean any leaked instances from prior runs
    # before launching a new one.
    leaked = list_running_wikitext_instances()
    if leaked:
        print(f"⚠ {len(leaked)} wikitext instance(s) currently running:")
        for i in leaked:
            print(f"    {i['id']}  {i.get('name')}  {i.get('ip')}")
        if input("terminate them now? [Y/n] ").strip().lower() != "n":
            for i in leaked:
                terminate_instance(i["id"])

    if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    check_ghcr_login()
    ssh_key = pick_ssh_key(args.ssh_key)
    region = find_available_region()
    image_tag = build_and_push_image(args.submission, repo_owner)

    instance_id = launch_instance(image_tag, ssh_key, region)
    try:
        ip = wait_for_active(instance_id, timeout_s=600)
        wait_for_ssh(ip, timeout_s=300)
        result = wait_for_result(ip, timeout_s=int(est_min * 60 * 2.5))
        scp_artifacts(ip, args.submission.stem, result["date_utc"][:10])
    finally:
        terminate_instance(instance_id)

    out_path = save_result(result, args.submission)
    append_record(result, json_relpath=f"submissions/{out_path.name}")
    print(f"[done] {out_path}")
    print(f"       energy = {result['training_energy_J']:,.0f} J")
    print(f"       acc    = {result['test_char_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
