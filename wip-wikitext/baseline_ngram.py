"""Char-level n-gram baseline with stupid-backoff smoothing.

Implements ``CharModel`` so it plugs into the standard streaming
evaluator. Training is a single pass over the input string; eval is
``O(n)`` per character (lookup + dict normalize at the longest matching
context).

Stupid backoff (Brants et al. 2007): if the longest context has any
matching counts, use them; otherwise drop one char and recurse;
unigram floor.

Not Kneser-Ney — KN is the better baseline numerically but
significantly more code. Stupid backoff is one of the simplest
defensible smoothings and good enough as a v0 reference.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from wikitext import CharModel


class NGramModel(CharModel):

    def __init__(self, n: int = 5):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        # counts[order][prefix_str] = Counter({next_char: count})
        # order ranges 1..n; counts[1][""] is unigram counts (treated as
        # always-matching context).
        self.counts: list[defaultdict[str, Counter[str]]] = [
            defaultdict(Counter) for _ in range(n + 1)
        ]
        self.unigram: Counter[str] = Counter()
        self._context: str = ""

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str) -> None:
        """Single-pass count update from ``text``."""
        self.unigram.update(text)
        for order in range(1, self.n + 1):
            for i in range(len(text) - order):
                prefix = text[i : i + order]
                next_char = text[i + order]
                self.counts[order][prefix][next_char] += 1

    # ------------------------------------------------------------------
    # CharModel
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._context = ""

    def predict(self) -> dict[str, float]:
        for order in range(min(self.n, len(self._context)), 0, -1):
            prefix = self._context[-order:]
            ctr = self.counts[order].get(prefix)
            if ctr:
                total = sum(ctr.values())
                return {c: cnt / total for c, cnt in ctr.items()}
        total = sum(self.unigram.values())
        if total == 0:
            return {}
        return {c: cnt / total for c, cnt in self.unigram.items()}

    def observe(self, char: str) -> None:
        # Keep the rolling context bounded to the longest order we use.
        if self.n == 0:
            return
        ctx = self._context + char
        if len(ctx) > self.n:
            ctx = ctx[-self.n :]
        self._context = ctx
