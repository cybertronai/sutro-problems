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
from pathlib import Path

from wikitext import EnergyMeter, evaluate, load_wikitext103


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
                   help="Test-stream length. Default 60,000 chars is the "
                        "standard eval — finishes in ~2 min on A100, gives "
                        "±0.4–1.3pp 95%% CI on accuracy. Pass 0 to score "
                        "the full 1.3M-char test split.")
    p.add_argument("--save-model", type=Path, default=None,
                   help="(transformer) Save trained model to this path "
                        "after training, before eval — lets you re-run "
                        "eval without retraining if eval crashes.")
    args = p.parse_args()

    print(f"loading WikiText-103 from {args.data_dir} ...")
    train_text = load_wikitext103(args.data_dir, "train")
    test_text = load_wikitext103(args.data_dir, "test")
    if args.max_test_chars:
        test_text = test_text[: args.max_test_chars]
    print(f"  train chars: {len(train_text):,}")
    print(f"  test  chars: {len(test_text):,}")

    meter = EnergyMeter()
    if not meter.available:
        print("WARNING: NVML energy counter not available on this host; "
              "energy will not be measured. Submissions must run on a "
              "host with NVML access (Lambda On-Demand A100, etc.).")

    if args.baseline == "ngram":
        from baseline_ngram import NGramModel
        model = NGramModel(n=args.n)
        print(f"training n-gram (n={args.n}) ...")
        with meter.measure() as m:
            model.train(train_text)
    else:
        from baseline_transformer import TransformerModel, train_transformer
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

    print(f"training: {m}")
    print(f"evaluating on test split ...")
    progress_every = max(1, len(test_text) // 50)
    result = evaluate(model, test_text, progress_every=progress_every)
    print(result)

    print("---")
    print(f"baseline           : {args.baseline}")
    if m.energy_joules is not None:
        print(f"training energy (J): {m.energy_joules:,.1f}")
    else:
        print("training energy (J): NOT MEASURED")
    print(f"training duration  : {m.duration_s:.1f}s")
    print(f"test char-accuracy : {result.accuracy:.4f}")
    print(f"test chars         : {result.n_chars:,}")


if __name__ == "__main__":
    main()
