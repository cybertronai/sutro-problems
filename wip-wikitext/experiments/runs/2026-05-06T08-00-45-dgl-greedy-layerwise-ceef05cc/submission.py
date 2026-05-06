"""DGL (Decoupled Greedy Layer-wise) char-LM for WikiText-103.

Each transformer block is trained *one at a time* with its own local
linear LM head:

  for L in range(n_layers):
      with no_grad: h = embed(x); for prev in 0..L-1: h = block_prev(h)
      h = h.detach()                # no autograd link to prev blocks
      h = block_L(h)                # ONLY this block keeps activations
      logits = aux_head_L(ln_L(h))
      loss = ce(logits, next_char)
      loss.backward()               # backward graph spans 1 block + head
      adamw.step()

At any moment the autograd graph contains a single block's activations
plus the local head. There is no full-network forward cache, no
end-to-end backward pass, and the inter-layer gradient signal is
*absent by construction* (the `.detach()` cuts it). This satisfies
the leaderboard's "no backprop over a multi-layer graph" rule.

Final inference uses a normal KV-cached forward through all trained
blocks; the deepest aux head's weight is copied into the model's
final head before returning the streaming wrapper.

Reference: Belilovsky et al. 2019, "Greedy Layerwise Learning Can
Scale to ImageNet" — DGL within ~1 pp of full backprop on ImageNet
with a similar architecture.
"""
from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from wikitext import CharModel


# ---------------------------------------------------------------------------
# RoPE (matches baseline_transformer.py — same cache trim semantics)
# ---------------------------------------------------------------------------

def _rope_cos_sin(d_head, offset, T, *, device, base=10000.0):
    inv_freq = 1.0 / (base ** (
        torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head
    ))
    pos = torch.arange(offset, offset + T, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_rope(x, cos, sin):
    x_e, x_o = x[..., 0::2], x[..., 1::2]
    c = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)
    s = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    rot_e = x_e * c - x_o * s
    rot_o = x_e * s + x_o * c
    return torch.stack((rot_e, rot_o), dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, cos, sin, k_cache=None, v_cache=None):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if k_cache is not None and v_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        is_causal = (k_cache is None) and T > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out), k, v


class Block(nn.Module):
    """Pre-norm transformer block with ReLU² MLP (modded-nanogpt's
    free swap-in for GELU)."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x, cos, sin, k_cache=None, v_cache=None):
        h, k, v = self.attn(self.ln1(x), cos, sin, k_cache, v_cache)
        x = x + h
        u = F.relu(self.fc1(self.ln2(x)))
        x = x + self.fc2(u * u)  # ReLU²
        return x, k, v


# Logit softcap (Gemma-2 style). Applied identically at train and eval
# so the streaming distribution matches the training distribution.
SOFTCAP = 30.0


def _softcap(z):
    return SOFTCAP * torch.tanh(z / SOFTCAP)


class DGLNet(nn.Module):
    """Final inference network. After DGL training, `head` and `ln_f`
    are populated from the last layer's aux head/LN."""

    def __init__(self, vocab_size=256, d_model=384, n_layers=5,
                 n_heads=6, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_head = d_model // n_heads
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, kv_caches=None, *, position=0):
        T = x.shape[1]
        h = self.tok_emb(x)
        cos, sin = _rope_cos_sin(self.d_head, position, T, device=x.device)
        new_caches = []
        for i, block in enumerate(self.blocks):
            kc, vc = (None, None) if kv_caches is None else kv_caches[i]
            h, k, v = block(h, cos, sin, kc, vc)
            new_caches.append((k, v))
        return _softcap(self.head(self.ln_f(h))), new_caches


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


def _lr_at(step, *, peak_lr, warmup, total, min_lr_frac=0.1):
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)


# ---------------------------------------------------------------------------
# DGL training
# ---------------------------------------------------------------------------

def train(train_text, valid_text=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    # Architecture
    vocab_size = 256
    d_model = 384
    n_layers = 5
    n_heads = 6
    max_len = 512

    # Per-layer training schedule. Calibrated against the modded baseline
    # (55 kJ for 6L full-backprop, 3000 steps, batch=64, seq=512 → ≈1 J
    # per layer-forward-pass-equivalent on the pinned A100). DGL step at
    # layer L costs (L+1) frozen-fwds + 1 fwd-with-grad + 1 bwd ≈ (L+3)
    # layer-fwd-equivalents. Sum over 5 layers, n steps each:
    # 25·n layer-fwd-eqs vs modded's 18·3000 = 54k. Picking n=3000 gives
    # 75k layer-fwd-eqs ≈ 76 kJ projected — fits comfortably under 100 kJ
    # with ~24 kJ safety margin.
    batch_size = 64
    n_steps_per_layer = 3000
    peak_lr = 5e-4
    warmup_steps = 200
    weight_decay = 0.1
    grad_clip = 1.0

    train_ids = torch.tensor(
        list(train_text.encode("utf-8")), dtype=torch.long, device=device,
    )
    if train_ids.numel() < max_len + 1:
        raise ValueError(f"need at least {max_len+1} bytes; got {train_ids.numel()}")

    model = DGLNet(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=max_len,
    ).to(device)
    model.apply(_init_weights)

    # Per-layer auxiliary heads (linear LM head + LayerNorm) used only
    # during DGL training. After training, the deepest layer's aux LN +
    # head are copied into model.ln_f / model.head for inference.
    aux_lns = nn.ModuleList(
        [nn.LayerNorm(d_model).to(device) for _ in range(n_layers)]
    )
    aux_heads = nn.ModuleList(
        [nn.Linear(d_model, vocab_size, bias=False).to(device)
         for _ in range(n_layers)]
    )
    for h in aux_heads:
        nn.init.normal_(h.weight, mean=0.0, std=0.02)

    n_params = sum(p.numel() for p in model.parameters())
    aux_params = sum(p.numel() for p in aux_heads.parameters()) + \
                 sum(p.numel() for p in aux_lns.parameters())
    print(f"DGL: {n_params/1e6:.2f}M model params, "
          f"{aux_params/1e6:.2f}M aux-head params (training-only), "
          f"{n_layers} layers, d={d_model}, h={n_heads}")

    t0 = time.monotonic()

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp
        else nullcontext()
    )

    for layer_idx in range(n_layers):
        cur_block = model.blocks[layer_idx]
        cur_ln = aux_lns[layer_idx]
        cur_head = aux_heads[layer_idx]

        # Trainable params for this DGL stage
        params = list(cur_block.parameters()) \
                + list(cur_ln.parameters()) \
                + list(cur_head.parameters())
        if layer_idx == 0:
            params += list(model.tok_emb.parameters())

        decay, no_decay = [], []
        for p in params:
            (decay if p.dim() >= 2 else no_decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=peak_lr, betas=(0.9, 0.95), eps=1e-9,
        )

        for step in range(n_steps_per_layer):
            lr = _lr_at(step, peak_lr=peak_lr, warmup=warmup_steps,
                        total=n_steps_per_layer)
            for g in opt.param_groups:
                g["lr"] = lr

            idx = torch.randint(
                0, train_ids.numel() - max_len - 1,
                (batch_size,), device=device,
            )
            x = torch.stack([train_ids[i:i + max_len] for i in idx])
            y = torch.stack([train_ids[i + 1:i + 1 + max_len] for i in idx])

            opt.zero_grad(set_to_none=True)

            with autocast_ctx:
                # ---- frozen forward through 0..L-1 (no_grad: NO activation cache) ----
                if layer_idx == 0:
                    cos, sin = _rope_cos_sin(model.d_head, 0, max_len, device=device)
                    h = model.tok_emb(x)
                else:
                    with torch.no_grad():
                        cos, sin = _rope_cos_sin(model.d_head, 0, max_len, device=device)
                        h = model.tok_emb(x)
                        for prev_idx in range(layer_idx):
                            h, _, _ = model.blocks[prev_idx](h, cos, sin)
                    # detach: gradient does NOT flow back into prev blocks
                    h = h.detach()

                # ---- single-block forward+backward (only THIS block's activations live) ----
                h, _, _ = cur_block(h, cos, sin)
                logits = _softcap(cur_head(cur_ln(h)))  # match eval transform
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

            if step % 200 == 0 or step == n_steps_per_layer - 1:
                elapsed = time.monotonic() - t0
                print(f"  layer {layer_idx} step {step:4d}  loss {loss.item():.4f}"
                      f"  lr {lr:.2e}  elapsed {elapsed:.0f}s", flush=True)

        # Free this layer's optimizer state immediately
        del opt

    # Hook up the last layer's aux LN + head as the final ln_f + head
    with torch.no_grad():
        model.ln_f.load_state_dict(aux_lns[-1].state_dict())
        model.head.weight.copy_(aux_heads[-1].weight)

    elapsed = time.monotonic() - t0
    print(f"DGL training done in {elapsed:.0f}s", flush=True)
    return DGLCharModel(model, device)


# ---------------------------------------------------------------------------
# Streaming inference (KV-cached, RoPE absolute-position counter, sliding trim)
# ---------------------------------------------------------------------------

class DGLCharModel(CharModel):
    def __init__(self, model: DGLNet, device: str):
        self.model = model
        self.device = device
        self.model.eval()
        self._kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._next_logits: torch.Tensor | None = None
        self._pos: int = 0

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        self._pos = 0
        x = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        logits, self._kv = self.model(x, None, position=self._pos)
        self._next_logits = logits[0, -1]
        self._pos = 1

    @torch.no_grad()
    def predict(self) -> dict[str, float]:
        if self._next_logits is None:
            raise RuntimeError("predict() called before reset()")
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
            logits, self._kv = self.model(x, self._kv, position=self._pos)
            self._next_logits = logits[0, -1]
            self._pos += 1

    def _maybe_trim_cache(self) -> None:
        if self._kv is None:
            return
        cur = self._kv[0][0].shape[2]
        if cur < self.model.max_len:
            return
        keep = self.model.max_len - 1
        self._kv = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in self._kv]
