"""Example wikitext submission: 5-gram char model.

A submission is any Python file that exposes::

    def train(train_text: str, valid_text: str | None = None) -> CharModel: ...

The runner energy-meters the call to ``train()`` and then evaluates the
returned ``CharModel`` on the held-out test split.

This example wraps the existing 5-gram baseline so a smoke run finishes
in seconds. Copy this file, swap in your own model, and ship it via::

    python3 submit.py path/to/your_submission.py
"""
from baseline_ngram import NGramModel
from wikitext import CharModel


def train(train_text: str, valid_text: str | None = None) -> CharModel:
    m = NGramModel(n=5)
    m.train(train_text)
    return m
