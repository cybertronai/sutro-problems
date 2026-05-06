"""DGL v2 — warm-started aux heads + inference ensemble.

v1 found that with random aux-head init, deeper layers' local NLL
plateaued at ≈ layer-0 levels (each new layer + random head had to
relearn the char distribution from scratch — the dominant fraction
of each layer's training budget went into rediscovering what layer
0 already knew). v2 changes two things:

1. Each new layer's aux head + aux LN are **warm-started** from the
   previous layer's. Layer L thus *starts* at layer L-1's prediction
   quality and can only improve. Combined with the residual init
   (random Block(x) ≈ x at step 0), the loss curve at each new layer
   begins exactly where the previous left off rather than at log V.

2. **Inference ensemble**: at evaluation, the final logits are the
   mean of every layer's aux head's predictions (after softcap). Each
   aux head was independently trained to predict next-char from its
   own layer's features; uniform-weighted logit averaging is a
   training-free ensemble that consistently beats any single member
   when the members make different mistakes (which they do here —
   the layers see different receptive fields / abstraction levels).

These two changes preserve the no-backprop guarantee — the autograd
graph during training is still a single block + one head, and
inference doesn't update parameters.

Other adjustments vs v1:
  * 6 layers (was 5): the energy headroom from v1 (54 kJ used, 46 kJ
    free) is enough for one more block.
  * 4000 steps / layer (was 3000): with warm-starts the first ~500
    steps per layer are no longer "burn-in to vocab distribution",
    so more of each layer's training budget goes into actual
    refinement. Total fits in ~85 kJ projected.
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
# RoPE / Block — identical to v1
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
    """Pre-norm transformer block with ReLU² MLP."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        # Zero-init the residual outs so a freshly-initialised block is
        # near-identity. This is the v2 detail that lets warm-started
        # aux heads from the previous layer keep working at step 0.
        nn.init.zeros_(self.attn.out.weight)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x, cos, sin, k_cache=None, v_cache=None):
        h, k, v = self.attn(self.ln1(x), cos, sin, k_cache, v_cache)
        x = x + h
        u = F.relu(self.fc1(self.ln2(x)))
        x = x + self.fc2(u * u)
        return x, k, v


SOFTCAP = 30.0


def _softcap(z):
    return SOFTCAP * torch.tanh(z / SOFTCAP)


class DGLNet(nn.Module):
    """Inference network with per-layer aux heads kept around for the
    inference-time ensemble."""

    def __init__(self, vocab_size=256, d_model=384, n_layers=6,
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
        # One LN + linear head per layer (kept for the inference ensemble).
        self.aux_lns = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )
        self.aux_heads = nn.ModuleList(
            [nn.Linear(d_model, vocab_size, bias=False) for _ in range(n_layers)]
        )

    def forward(self, x, kv_caches=None, *, position=0):
        """At inference: return softcapped *mean-of-layer-logits* and the
        new KV caches.

        Each aux head sees its own layer's features through its own LN.
        Logits are softcapped per-layer (so each member of the ensemble
        is on the same scale as it was trained on), then averaged.
        """
        T = x.shape[1]
        h = self.tok_emb(x)
        cos, sin = _rope_cos_sin(self.d_head, position, T, device=x.device)
        new_caches = []
        logit_sum = None
        for i, block in enumerate(self.blocks):
            kc, vc = (None, None) if kv_caches is None else kv_caches[i]
            h, k, v = block(h, cos, sin, kc, vc)
            new_caches.append((k, v))
            layer_logits = _softcap(self.aux_heads[i](self.aux_lns[i](h)))
            logit_sum = layer_logits if logit_sum is None else logit_sum + layer_logits
        return logit_sum / self.n_layers, new_caches


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
# DGL training (with warm-started aux heads)
# ---------------------------------------------------------------------------

def train(train_text, valid_text=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    vocab_size = 256
    d_model = 384
    n_layers = 6
    n_heads = 6
    max_len = 512

    batch_size = 64
    # 6 layers × 3500 steps × (L+3) summed L=0..5 = 33 → 115,500 layer-fwd-eqs.
    # v1 measured ~0.72 J / layer-fwd-eq → projected energy ≈ 83 kJ.
    n_steps_per_layer = 3500
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
    # Apply default init (Linear/Embedding standard normal). Block.__init__
    # then re-zeros the residual outs, so a fresh block is exactly identity.
    model.apply(_init_weights)
    # The Block constructor already zeroed the residual-out weights; redo
    # after the global apply() to be safe.
    for blk in model.blocks:
        nn.init.zeros_(blk.attn.out.weight)
        nn.init.zeros_(blk.fc2.weight)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DGL v2: {n_params/1e6:.2f}M model params, "
          f"{n_layers} layers, d={d_model}, h={n_heads}, "
          f"steps/layer={n_steps_per_layer}, peak_lr={peak_lr:.0e}")

    t0 = time.monotonic()

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp
        else nullcontext()
    )

    for layer_idx in range(n_layers):
        cur_block = model.blocks[layer_idx]
        cur_ln = model.aux_lns[layer_idx]
        cur_head = model.aux_heads[layer_idx]

        # ---- v2 warm-start: copy previous layer's aux head + LN ----
        if layer_idx == 0:
            # First layer: random init for aux head, default for LN
            nn.init.normal_(cur_head.weight, mean=0.0, std=0.02)
        else:
            with torch.no_grad():
                prev_ln = model.aux_lns[layer_idx - 1]
                prev_head = model.aux_heads[layer_idx - 1]
                cur_ln.load_state_dict(prev_ln.state_dict())
                cur_head.weight.copy_(prev_head.weight)
        # The block was zero-initialised on the residual out, so its
        # forward at step 0 is exactly identity → previous layer's
        # aux head is now applied to identical features → loss at
        # step 0 ≈ previous layer's final loss, not log(V).

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
                if layer_idx == 0:
                    cos, sin = _rope_cos_sin(model.d_head, 0, max_len, device=device)
                    h = model.tok_emb(x)
                else:
                    with torch.no_grad():
                        cos, sin = _rope_cos_sin(model.d_head, 0, max_len, device=device)
                        h = model.tok_emb(x)
                        for prev_idx in range(layer_idx):
                            h, _, _ = model.blocks[prev_idx](h, cos, sin)
                    h = h.detach()

                h, _, _ = cur_block(h, cos, sin)
                logits = _softcap(cur_head(cur_ln(h)))
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size), y.reshape(-1),
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

            if step % 500 == 0 or step == n_steps_per_layer - 1:
                elapsed = time.monotonic() - t0
                print(f"  layer {layer_idx} step {step:4d}  loss {loss.item():.4f}"
                      f"  lr {lr:.2e}  elapsed {elapsed:.0f}s", flush=True)

        del opt

    elapsed = time.monotonic() - t0
    print(f"DGL v2 training done in {elapsed:.0f}s — using inference-ensemble", flush=True)
    return DGLCharModel(model, device)


# ---------------------------------------------------------------------------
# Streaming inference (KV-cached, ensemble across all layers' aux heads)
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
