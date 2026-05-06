"""Reference submission runner.

Trains one of the baselines on WikiText-103 train split (energy
measured), then evaluates greedy char-acc on the test split (energy
not measured, per the v0 design — training-only accounting).

Example::

    python3 run_eval.py --data-dir /data/wikitext-103 --baseline ngram --n 5
    python3 run_eval.py --data-dir /data/wikitext-103 --baseline transformer

The output line is the (energy, accuracy) pair the runner uses to
populate the record table.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from wikitext import (
    BudgetExceededError,
    CharModel,
    EnergyMeter,
    evaluate,
    load_wikitext103,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory containing wiki.{train,valid,test}.raw")
    p.add_argument("--baseline", choices=["ngram", "transformer"],
                   default="ngram")
    p.add_argument("--n", type=int, default=5,
                   help="(ngram) order")
    p.add_argument("--config", choices=["tiny", "small", "gpt2"], default="tiny",
                   help="(transformer) model size preset")
    p.add_argument("--n-steps", type=int, default=2000,
                   help="(transformer) training steps")
    p.add_argument("--batch-size", type=int, default=64,
                   help="(transformer) batch size")
    p.add_argument("--peak-lr", type=float, default=3e-4,
                   help="(transformer) peak learning rate")
    p.add_argument("--max-test-chars", type=int, default=60_000,
                   help="Test-stream length. The runner sets this from "
                        "task.TEST_CHARS (currently 60,000); local dev "
                        "may override. Pass 0 to score the full 1.3M-char "
                        "test split.")
    p.add_argument("--save-model", type=Path, default=None,
                   help="(transformer) Save trained model to this path "
                        "after training, before eval — lets you re-run "
                        "eval without retraining if eval crashes.")
    p.add_argument("--submission", type=Path, default=None,
                   help="Path to a Python file exposing "
                        "`train(train_text, valid_text=None) -> CharModel`. "
                        "Overrides --baseline.")
    p.add_argument("--results-json", type=Path, default=None,
                   help="Write the final (energy, accuracy, gpu, …) tuple "
                        "as a JSON object to this path. Used by submit.py.")
    p.add_argument("--e-max-joules", type=float, default=None,
                   help="Training energy budget in joules. The runner sets "
                        "this from task.E_MAX_JOULES (a leaderboard rule); "
                        "submissions cannot vary it. If set, an NVML "
                        "watchdog kills the training run when the running "
                        "net energy crosses this threshold; the submission "
                        "is reported as DISQUALIFIED. No-op on hosts without "
                        "NVML.")
    args = p.parse_args()

    print(f"loading WikiText-103 from {args.data_dir} ...")
    train_text = load_wikitext103(args.data_dir, "train")
    test_text = load_wikitext103(args.data_dir, "test")
    if args.max_test_chars:
        test_text = test_text[: args.max_test_chars]
    print(f"  train chars: {len(train_text):,}")
    print(f"  test  chars: {len(test_text):,}")

    meter = EnergyMeter(e_max_joules=args.e_max_joules)
    if not meter.available:
        print("WARNING: NVML energy counter not available on this host; "
              "energy will not be measured. Submissions must run on a "
              "host with NVML access (Lambda On-Demand A100, etc.).")
        if args.e_max_joules is not None:
            print("WARNING: --e-max-joules requires NVML; killswitch is a "
                  "no-op on this host.")
    elif args.e_max_joules is not None:
        print(f"energy budget: {args.e_max_joules:,.0f} J  "
              f"(watchdog poll {meter.poll_interval_s:.2f}s)")

    m = None
    submission_name = "unknown"
    try:
        if args.submission is not None:
            submission_name = args.submission.stem
            spec = importlib.util.spec_from_file_location("user_submission", args.submission)
            if spec is None or spec.loader is None:
                sys.exit(f"could not import submission file: {args.submission}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            train_fn = getattr(mod, "train", None)
            if not callable(train_fn):
                sys.exit(f"submission must define train(train_text, valid_text=None) -> "
                         f"CharModel: {args.submission}")
            # Pass valid_text only if the user's signature accepts it.
            accepts_valid = "valid_text" in inspect.signature(train_fn).parameters
            valid_text = load_wikitext103(args.data_dir, "valid") if accepts_valid else None
            print(f"training submission {args.submission} ...")
            with meter.measure() as m:
                model = train_fn(train_text, valid_text=valid_text) if accepts_valid \
                    else train_fn(train_text)
            if not isinstance(model, CharModel):
                sys.exit(f"submission's train() returned {type(model).__name__}, "
                         f"expected a CharModel subclass")
        elif args.baseline == "ngram":
            from baseline_ngram import NGramModel
            submission_name = f"baseline_ngram_n{args.n}"
            model = NGramModel(n=args.n)
            print(f"training n-gram (n={args.n}) ...")
            with meter.measure() as m:
                model.train(train_text)
        else:
            from baseline_transformer import TransformerModel, train_transformer
            submission_name = f"baseline_transformer_{args.config}"
            valid_text = load_wikitext103(args.data_dir, "valid")
            print(f"training transformer config={args.config} ({args.n_steps} steps) ...")
            with meter.measure() as m:
                trained = train_transformer(
                    train_text,
                    config=args.config,
                    valid_text=valid_text,
                    batch_size=args.batch_size,
                    n_steps=args.n_steps,
                    peak_lr=args.peak_lr,
                )
            if args.save_model is not None:
                import torch
                torch.save({
                    "state_dict": trained.state_dict(),
                    "config": args.config,
                }, args.save_model)
                print(f"saved trained model to {args.save_model}")
            model = TransformerModel(trained)
    except BudgetExceededError as e:
        print("---")
        print(f"DISQUALIFIED: {e}")
        print(f"submission         : {submission_name}")
        if m is not None:
            print(f"training duration  : {m.duration_s:.1f}s")
            if m.energy_joules is not None:
                print(f"training energy (J): {m.energy_joules:,.1f}  (at kill)")
        if args.results_json is not None:
            payload = {
                "submission": submission_name,
                "disqualified": True,
                "reason": "energy_budget_exceeded",
                "e_max_joules": args.e_max_joules,
                "training_energy_J": m.energy_joules if m is not None else None,
                "training_duration_s": m.duration_s if m is not None else None,
                "gpu_name": _gpu_name(),
                "date_utc": datetime.now(timezone.utc)
                    .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            }
            args.results_json.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"wrote {args.results_json}")
        sys.exit(2)

    print(f"training: {m}")
    print(f"evaluating on test split ...")
    progress_every = max(1, len(test_text) // 50)
    result = evaluate(model, test_text, progress_every=progress_every)
    print(result)

    print("---")
    print(f"submission         : {submission_name}")
    if m.energy_joules is not None:
        print(f"training energy (J): {m.energy_joules:,.1f}")
    else:
        print("training energy (J): NOT MEASURED")
    print(f"training duration  : {m.duration_s:.1f}s")
    print(f"test char-accuracy : {result.accuracy:.4f}")
    print(f"test chars         : {result.n_chars:,}")

    if args.results_json is not None:
        payload = {
            "submission": submission_name,
            "training_energy_J": m.energy_joules,
            "training_duration_s": m.duration_s,
            "test_char_accuracy": result.accuracy,
            "test_chars": result.n_chars,
            "gpu_name": _gpu_name(),
            "date_utc": datetime.now(timezone.utc)
                .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
        args.results_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.results_json}")


def _gpu_name() -> str | None:
    try:
        import pynvml  # type: ignore[import-not-found]
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        return n.decode() if isinstance(n, bytes) else n
    except Exception:
        return None


if __name__ == "__main__":
    main()
