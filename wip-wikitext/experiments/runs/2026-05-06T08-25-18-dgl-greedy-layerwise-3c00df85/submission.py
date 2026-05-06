"""DGL v3 — closed-form ridge readout from concatenated multi-layer features.

v1 / v2 found a hard ~1.20 nats/char ceiling on every individual aux
head's local NLL: each layer plateaus at the same per-position
predictive-power level because no inter-layer gradient can shape its
features for downstream use. Logit averaging across heads (v2) helped a
little but didn't break through.

v3's bet: the *features* at each layer carry complementary information,
even when the per-layer aux head can't extract it on its own. Replace
the per-layer linear-head ensemble with a single **closed-form ridge
regression** from the concatenation of all layers' features to the
next-character one-hot. Solving `W = (XᵀX + λI)⁻¹ XᵀY` is gradient-free
(no autograd graph at all — pure linear algebra) and the optimal
linear combination of those features is exactly what each independent
local head couldn't construct.

Pipeline:

  1. DGL train as in v2 (warm-start aux heads, zero-init residual outs,
     6 layers × 3500 steps, AdamW). The aux heads are now used only as
     a *training signal* for shaping each layer's features — they are
     discarded at inference.
  2. Forward a sample of training positions through the 6 frozen blocks
     (`torch.no_grad()` — no activation cache anywhere in the stack)
     and concatenate `(layer_0_out, …, layer_5_out)` per position into
     a 6·d-dim feature vector.
  3. Solve a Tikhonov-regularised normal-equations ridge regression:
     `W = (XᵀX + λI)⁻¹ Xᵀ Y` where `Y` is one-hot next-char.
  4. At inference, run the streaming forward through the 6 blocks,
     concatenate outputs, multiply by `W`, optionally softcap, and
     softmax.

The constraint on no-backprop is preserved everywhere:
  * DGL training (steps 1) only ever holds one block's activations live
    — same as v1/v2.
  * Step 2 is `torch.no_grad()` — no activations saved.
  * Step 3 is closed-form linear algebra — *no autograd at all*. The
    XᵀX accumulation is streamed in chunks so memory stays bounded.
  * Inference is `torch.no_grad()` only.
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
# Architecture (identical to v2)
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


class DGLNetV3(nn.Module):
    """Backbone for inference. Holds the trained blocks plus a single
    ridge-regression weight matrix that maps concatenated layer features
    to next-char logits. The aux heads used during DGL training are
    discarded."""

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
        # Ridge readout from concatenated 6-layer features.
        # Filled in after DGL + ridge solve.
        self.register_buffer(
            "ridge_W",
            torch.zeros(n_layers * d_model, vocab_size),
        )

    def features(self, x, kv_caches=None, *, position=0):
        """Run all blocks and return concatenated per-layer features +
        new KV caches. Shape (B, T, n_layers * d_model)."""
        T = x.shape[1]
        h = self.tok_emb(x)
        cos, sin = _rope_cos_sin(self.d_head, position, T, device=x.device)
        per_layer = []
        new_caches = []
        for i, block in enumerate(self.blocks):
            kc, vc = (None, None) if kv_caches is None else kv_caches[i]
            h, k, v = block(h, cos, sin, kc, vc)
            new_caches.append((k, v))
            per_layer.append(h)
        feat = torch.cat(per_layer, dim=-1)  # (B, T, n_layers * d)
        return feat, new_caches

    def forward(self, x, kv_caches=None, *, position=0):
        feat, new_caches = self.features(x, kv_caches, position=position)
        logits = _softcap(feat @ self.ridge_W)
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
# Training: DGL + ridge readout
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
    n_steps_per_layer = 3000
    peak_lr = 5e-4
    warmup_steps = 200
    weight_decay = 0.1
    grad_clip = 1.0
    ridge_lambda = 1.0  # tikhonov regularisation on (n_layers * d_model)-dim X^T X
    ridge_n_windows = 80  # 80 * 64 * 512 ≈ 2.6 M positions of features for the solve

    train_ids = torch.tensor(
        list(train_text.encode("utf-8")), dtype=torch.long, device=device,
    )

    model = DGLNetV3(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=max_len,
    ).to(device)
    model.apply(_init_weights)
    for blk in model.blocks:
        nn.init.zeros_(blk.attn.out.weight)
        nn.init.zeros_(blk.fc2.weight)

    # Aux heads (training-only, discarded at inference)
    aux_lns = nn.ModuleList(
        [nn.LayerNorm(d_model).to(device) for _ in range(n_layers)]
    )
    aux_heads = nn.ModuleList(
        [nn.Linear(d_model, vocab_size, bias=False).to(device)
         for _ in range(n_layers)]
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DGL v3: {n_params/1e6:.2f}M backbone params, "
          f"{n_layers} layers, d={d_model}; ridge_λ={ridge_lambda}, "
          f"ridge_n_windows={ridge_n_windows}")

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

    # Aux heads can be freed now — they were just for shaping features.
    del aux_heads, aux_lns

    elapsed = time.monotonic() - t0
    print(f"DGL training done in {elapsed:.0f}s — solving ridge readout", flush=True)

    # ----- Phase 2: streaming X^T X / X^T Y accumulation -----
    feat_dim = n_layers * d_model
    XtX = torch.zeros(feat_dim, feat_dim, dtype=torch.float32, device=device)
    XtY = torch.zeros(feat_dim, vocab_size, dtype=torch.float32, device=device)
    n_seen = 0

    model.eval()
    with torch.no_grad():
        cos, sin = _rope_cos_sin(model.d_head, 0, max_len, device=device)
        for w in range(ridge_n_windows):
            idx = torch.randint(
                0, train_ids.numel() - max_len - 1,
                (batch_size,), device=device,
            )
            x = torch.stack([train_ids[i:i + max_len] for i in idx])
            y = torch.stack([train_ids[i + 1:i + 1 + max_len] for i in idx])

            with autocast_ctx:
                h = model.tok_emb(x)
                feats = []
                for block in model.blocks:
                    h, _, _ = block(h, cos, sin)
                    feats.append(h)
                feat = torch.cat(feats, dim=-1)  # (B, T, feat_dim)

            # Cast to fp32 for the linear-algebra accumulation.
            feat32 = feat.reshape(-1, feat_dim).to(torch.float32)
            y_oh = F.one_hot(y.reshape(-1), vocab_size).to(torch.float32)
            XtX += feat32.T @ feat32
            XtY += feat32.T @ y_oh
            n_seen += feat32.shape[0]

            if w % 20 == 0 or w == ridge_n_windows - 1:
                elapsed = time.monotonic() - t0
                print(f"  ridge window {w+1}/{ridge_n_windows}  "
                      f"n_seen={n_seen:,}  elapsed {elapsed:.0f}s", flush=True)

    # Tikhonov-regularised solve.
    XtX += ridge_lambda * torch.eye(feat_dim, device=device, dtype=torch.float32)
    print(f"  solving ridge: feat_dim={feat_dim}, λ={ridge_lambda}, "
          f"n_seen={n_seen:,}", flush=True)
    W = torch.linalg.solve(XtX, XtY)  # (feat_dim, vocab_size)
    model.ridge_W = W.contiguous()

    elapsed = time.monotonic() - t0
    print(f"DGL v3 done in {elapsed:.0f}s — ridge W ready", flush=True)
    return DGLV3CharModel(model, device)


# ---------------------------------------------------------------------------
# Streaming inference (KV cached, ridge readout from concatenated features)
# ---------------------------------------------------------------------------

class DGLV3CharModel(CharModel):
    def __init__(self, model: DGLNetV3, device: str):
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
