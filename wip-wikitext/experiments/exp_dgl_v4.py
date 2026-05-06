"""DGL v4 — DGL features + CE-trained linear readout (with per-layer LN).

RESULT: 0.6555 char-acc, 77 kJ. Essentially tied with v2's 0.6553 — the
CE-trained readout did NOT improve over uniform-mean ensembling of
per-layer aux heads. This is strong evidence that the layers' features
are highly redundant under DGL training: the optimal *linear*
classifier from concatenated features carries no more argmax
information than averaging individually-trained linear heads. To
actually break the ~0.65 ceiling, future iterations should switch
paradigms (DFA, target-prop) or add inter-layer pressure during DGL
training, not just rework the readout.


v3 found that ridge regression (L2 on one-hot) is the wrong objective
for an argmax-classification readout: it ended up *worse* than v2's
uniform-mean ensemble (0.59 vs 0.66 char-acc). Two fixes for v4:

1. **Replace ridge with cross-entropy training** on the readout. A
   single 2304→256 linear layer trained with CE on the frozen DGL
   features. This matches the eval objective (greedy argmax over
   softmax of logits) directly. CE training is gradient-based, but
   the gradient only flows through *one* linear layer — the DGL
   blocks above are forwarded under `torch.no_grad()`. Strictly
   one-layer's-activations live during the readout training step,
   so the no-backprop constraint is preserved.

2. **Per-layer LayerNorm before concatenation.** v3 concatenated raw
   block outputs whose per-layer scales drift through the residual
   stream (deeper layers have larger magnitudes). Apply a fixed-init
   LN to each layer's output, *then* concat. Now each layer's
   contribution to the readout has comparable magnitude.

Pipeline:
  1. DGL train (warm-start aux heads, as v2/v3) — 2500 steps/layer.
  2. Discard aux heads. Compute frozen features → per-layer LN →
     concat (2304-d).
  3. CE-train a single 2304→256 linear readout on those features
     for ~4000 steps with AdamW. Energy: ≈25 kJ.
  4. Inference: forward through frozen blocks, per-layer LN, concat,
     readout, softcap, softmax.

Energy projection:
  DGL: 6 × 2500 × 33 × 0.72 J ≈ 59 kJ
  Readout: 4000 × (6 layer-fwds + 1 readout-fwd + 1 readout-bwd)
         = 4000 × 7 × 0.72 J ≈ 20 kJ
  Total ≈ 79 kJ, ~21 kJ margin.
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
# Architecture (identical to v2/v3)
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
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
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


class DGLNetV4(nn.Module):
    """Backbone: DGL-trained blocks, per-layer LN, single CE-trained
    linear readout from concatenated features."""

    def __init__(self, vocab_size=256, d_model=384, n_layers=6,
                 n_heads=6, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads) for _ in range(n_layers)]
        )
        # Per-layer LN applied to each block's output before concat.
        self.feat_lns = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )
        # CE-trained linear readout from concatenated features.
        self.readout = nn.Linear(n_layers * d_model, vocab_size, bias=False)

    def features(self, x, kv_caches=None, *, position=0):
        T = x.shape[1]
        h = self.tok_emb(x)
        cos, sin = _rope_cos_sin(self.d_head, position, T, device=x.device)
        per_layer = []
        new_caches = []
        for i, block in enumerate(self.blocks):
            kc, vc = (None, None) if kv_caches is None else kv_caches[i]
            h, k, v = block(h, cos, sin, kc, vc)
            new_caches.append((k, v))
            per_layer.append(self.feat_lns[i](h))
        feat = torch.cat(per_layer, dim=-1)
        return feat, new_caches

    def forward(self, x, kv_caches=None, *, position=0):
        feat, new_caches = self.features(x, kv_caches, position=position)
        logits = _softcap(self.readout(feat))
        return logits, new_caches


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
# Training: DGL + CE-trained linear readout
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
    n_steps_per_layer = 2500
    peak_lr = 5e-4
    warmup_steps = 200
    weight_decay = 0.1
    grad_clip = 1.0

    n_steps_readout = 4000
    readout_peak_lr = 1e-3
    readout_warmup = 200

    train_ids = torch.tensor(
        list(train_text.encode("utf-8")), dtype=torch.long, device=device,
    )

    model = DGLNetV4(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=max_len,
    ).to(device)
    model.apply(_init_weights)
    for blk in model.blocks:
        nn.init.zeros_(blk.attn.out.weight)
        nn.init.zeros_(blk.fc2.weight)

    aux_lns = nn.ModuleList(
        [nn.LayerNorm(d_model).to(device) for _ in range(n_layers)]
    )
    aux_heads = nn.ModuleList(
        [nn.Linear(d_model, vocab_size, bias=False).to(device)
         for _ in range(n_layers)]
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DGL v4: {n_params/1e6:.2f}M params, {n_layers} layers, d={d_model}; "
          f"DGL_steps/layer={n_steps_per_layer}, readout_steps={n_steps_readout}")

    t0 = time.monotonic()

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp
        else nullcontext()
    )

    # ----- Phase 1: DGL training (warm-start aux heads) -----
    for layer_idx in range(n_layers):
        cur_block = model.blocks[layer_idx]
        cur_ln = aux_lns[layer_idx]
        cur_head = aux_heads[layer_idx]

        if layer_idx == 0:
            nn.init.normal_(cur_head.weight, mean=0.0, std=0.02)
        else:
            with torch.no_grad():
                cur_ln.load_state_dict(aux_lns[layer_idx - 1].state_dict())
                cur_head.weight.copy_(aux_heads[layer_idx - 1].weight)

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

    del aux_heads, aux_lns

    elapsed = time.monotonic() - t0
    print(f"DGL training done in {elapsed:.0f}s — training CE readout", flush=True)

    # ----- Phase 2: CE-trained linear readout from concatenated features -----
    # Only model.feat_lns and model.readout are trainable. DGL blocks are frozen.
    readout_params = list(model.feat_lns.parameters()) + list(model.readout.parameters())
    decay_r, no_decay_r = [], []
    for p in readout_params:
        (decay_r if p.dim() >= 2 else no_decay_r).append(p)
    opt_r = torch.optim.AdamW(
        [{"params": decay_r, "weight_decay": weight_decay},
         {"params": no_decay_r, "weight_decay": 0.0}],
        lr=readout_peak_lr, betas=(0.9, 0.95), eps=1e-9,
    )

    model.eval()  # blocks are forward-only — keep eval mode for them
    # But feat_lns and readout still train. Manually toggle:
    for ln in model.feat_lns:
        ln.train()
    model.readout.train()

    for step in range(n_steps_readout):
        lr = _lr_at(step, peak_lr=readout_peak_lr, warmup=readout_warmup,
                    total=n_steps_readout)
        for g in opt_r.param_groups:
            g["lr"] = lr

        idx = torch.randint(
            0, train_ids.numel() - max_len - 1,
            (batch_size,), device=device,
        )
        x = torch.stack([train_ids[i:i + max_len] for i in idx])
        y = torch.stack([train_ids[i + 1:i + 1 + max_len] for i in idx])

        opt_r.zero_grad(set_to_none=True)

        with autocast_ctx:
            # Frozen forward through blocks under no_grad — no autograd
            # cache for the multi-layer stack. Only the readout + feat_lns
            # have grad-tracked activations (single linear layer + LN per
            # layer on the OUTPUT, not chained; each LN is independent).
            with torch.no_grad():
                cos, sin = _rope_cos_sin(model.d_head, 0, max_len, device=device)
                h = model.tok_emb(x)
                per_layer_raw = []
                for block in model.blocks:
                    h, _, _ = block(h, cos, sin)
                    per_layer_raw.append(h.detach())  # detach: no autograd to blocks

            # Now apply trainable feat_lns + readout (with grad)
            per_layer_normed = [
                self_ln(h_raw)
                for self_ln, h_raw in zip(model.feat_lns, per_layer_raw)
            ]
            feat = torch.cat(per_layer_normed, dim=-1)
            logits = _softcap(model.readout(feat))
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1),
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(readout_params, grad_clip)
        opt_r.step()

        if step % 500 == 0 or step == n_steps_readout - 1:
            elapsed = time.monotonic() - t0
            print(f"  readout step {step:4d}  loss {loss.item():.4f}"
                  f"  lr {lr:.2e}  elapsed {elapsed:.0f}s", flush=True)

    elapsed = time.monotonic() - t0
    print(f"DGL v4 done in {elapsed:.0f}s — readout ready", flush=True)
    return DGLV4CharModel(model, device)


# ---------------------------------------------------------------------------
# Streaming inference
# ---------------------------------------------------------------------------

class DGLV4CharModel(CharModel):
    def __init__(self, model: DGLNetV4, device: str):
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
