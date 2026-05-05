"""Char-level GPT-2-style transformer baseline.

A small decoder-only transformer trained from scratch on bytes. Plugs
into ``CharModel`` via a per-instance KV-cache so streaming eval is
``O(1)`` marginal per character (one new query attends against the
cached keys/values).

PyTorch is an optional dependency — importing this module raises
``ImportError`` on hosts without ``torch``. The n-gram baseline is the
torch-free fallback for development on CPU-only machines.

GPT-2 conventions adopted:

* Pre-norm transformer blocks (LN before attn / MLP).
* Tied input + output embeddings (weight sharing).
* AdamW with betas ``(0.9, 0.95)`` and weight decay ``0.1``.
* Linear warmup then cosine decay to ``0.1 × peak_lr``.
* Dropout in residual paths and on embeddings.

Three named configs (call ``CONFIGS[name]`` to expand):

| name  | layers | d_model | heads | params  | indicative train cost |
|-------|--------|---------|-------|---------|-----------------------|
| tiny  | 4      | 128     | 4     | ~0.6 M  | minutes on A100       |
| small | 6      | 256     | 8     | ~5 M    | ~30 min on A100       |
| gpt2  | 12     | 384     | 6     | ~22 M   | ~3-6 hr on A100       |

These are not Pareto-optimal — they're starting points to anchor the
record table.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class Config:
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int = 512
    dropout: float = 0.1


CONFIGS: dict[str, Config] = {
    "tiny":  Config(d_model=128, n_layers=4,  n_heads=4),
    "small": Config(d_model=256, n_layers=6,  n_heads=8),
    "gpt2":  Config(d_model=384, n_layers=12, n_heads=6, max_len=1024),
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        if k_cache is not None and v_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        is_causal = (k_cache is None) and T > 1
        attn_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal, dropout_p=attn_p,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_drop(self.out(out)), k, v


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, k, v = self.attn(self.ln1(x), k_cache, v_cache)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, k, v


class CharTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, 4 * d_model, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie input and output embeddings (GPT-2 convention).
        self.head.weight = self.tok_emb.weight
        self.apply(_init_weights)

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        T = x.shape[1]
        offset = 0 if kv_caches is None else kv_caches[0][0].shape[2]
        if offset + T > self.max_len:
            raise RuntimeError(
                f"context overflow: offset={offset} + T={T} > max_len={self.max_len}"
            )
        pos = torch.arange(offset, offset + T, device=x.device)
        h = self.emb_drop(self.tok_emb(x) + self.pos_emb(pos))
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            kc, vc = (None, None) if kv_caches is None else kv_caches[i]
            h, k, v = cast("Block", block)(h, kc, vc)
            new_caches.append((k, v))
        h = self.ln_f(h)
        return self.head(h), new_caches


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _lr_at(step: int, *, peak_lr: float, warmup: int, total: int,
           min_lr_frac: float = 0.1) -> float:
    """Linear warmup → cosine decay to ``min_lr_frac × peak_lr``."""
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)


def train_transformer(
    text: str,
    *,
    config: Config | str = "tiny",
    valid_text: str | None = None,
    batch_size: int = 64,
    n_steps: int = 2000,
    peak_lr: float = 3e-4,
    warmup_steps: int = 200,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    device: str | None = None,
    dtype: str = "bf16",
    log_every: int = 200,
    valid_every: int = 1000,
) -> CharTransformer:
    """Train a CharTransformer on ``text`` (UTF-8 bytes).

    Uses bf16 autocast on CUDA by default; falls back to fp32 on CPU.
    """
    cfg = config if isinstance(config, Config) else CONFIGS[config]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device == "cuda" and dtype == "bf16")

    train_ids = torch.tensor(
        list(text.encode("utf-8")), dtype=torch.long, device=device,
    )
    if train_ids.numel() < cfg.max_len + 1:
        raise ValueError(f"need at least {cfg.max_len+1} bytes; got {train_ids.numel()}")
    valid_ids = None
    if valid_text:
        valid_ids = torch.tensor(
            list(valid_text.encode("utf-8")), dtype=torch.long, device=device,
        )

    model = CharTransformer(
        vocab_size=256,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
    ).to(device)

    # Weight-decay only on 2-D params (linear weights, embeddings),
    # not on biases or LayerNorm — GPT-2 convention.
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr, betas=(0.9, 0.95),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params  config={cfg}")

    model.train()
    t0 = time.monotonic()
    for step in range(n_steps):
        lr = _lr_at(step, peak_lr=peak_lr, warmup=warmup_steps, total=n_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        idx = torch.randint(
            0, train_ids.numel() - cfg.max_len - 1, (batch_size,), device=device,
        )
        x = torch.stack([train_ids[i : i + cfg.max_len] for i in idx])
        y = torch.stack([train_ids[i + 1 : i + 1 + cfg.max_len] for i in idx])

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if log_every and (step % log_every == 0 or step == n_steps - 1):
            elapsed = time.monotonic() - t0
            print(f"step {step:6d}  loss {loss.item():.4f}  "
                  f"lr {lr:.2e}  elapsed {elapsed:.0f}s")

        if valid_ids is not None and valid_every and step > 0 and step % valid_every == 0:
            v = _validate(model, valid_ids, cfg.max_len, batch_size=batch_size,
                          n_batches=20, device=device, use_amp=use_amp)
            print(f"   validation loss: {v:.4f}")
            model.train()
    return model


@torch.no_grad()
def _validate(model: CharTransformer, ids: torch.Tensor, max_len: int,
              *, batch_size: int, n_batches: int,
              device: str, use_amp: bool) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        idx = torch.randint(0, ids.numel() - max_len - 1, (batch_size,), device=device)
        x = torch.stack([ids[i : i + max_len] for i in idx])
        y = torch.stack([ids[i + 1 : i + 1 + max_len] for i in idx])
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


# ---------------------------------------------------------------------------
# CharModel adapter (KV-cached streaming)
# ---------------------------------------------------------------------------

class TransformerModel(CharModel):
    """KV-cached streaming wrapper around a trained ``CharTransformer``."""

    def __init__(self, model: CharTransformer, device: str | None = None):
        self.model = model
        self.device = device or next(model.parameters()).device.type
        self.model.eval()
        self._kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._next_logits: torch.Tensor | None = None

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        # Seed with a single zero byte so the first predict() has a
        # valid distribution. The zero byte is conventionally treated
        # as a stream-start sentinel; submissions can override.
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        logits, self._kv = self.model(x, None)
        self._next_logits = logits[0, -1]

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        if self._next_logits is None:
            raise RuntimeError("predict() called before reset()")
        self._maybe_trim_cache()
        probs = F.softmax(self._next_logits.float(), dim=-1)
        out: dict[str, float] = {}
        for byte_id, p in enumerate(probs.tolist()):
            try:
                ch = bytes([byte_id]).decode("utf-8")
            except UnicodeDecodeError:
                continue
            out[ch] = p
        return out

    @torch.no_grad()
    def observe(self, char: str) -> None:
        if self._kv is None:
            raise RuntimeError("observe() called before reset()")
        for byte in char.encode("utf-8"):
            self._maybe_trim_cache()
            x = torch.tensor([[byte]], dtype=torch.long, device=self.device)
            logits, self._kv = self.model(x, self._kv)
            self._next_logits = logits[0, -1]

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        cur = self._kv[0][0].shape[2]
        if cur < self.model.max_len:
            return
        keep = self.model.max_len - 1
        self._kv = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in self._kv]
