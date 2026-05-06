"""DGL v9 — v7 (DGL+Muon, 4×512) backbone + CE-trained ridge-style readout.

The story so far:
  v1 (plain DGL, 5×384 AdamW)       0.6440
  v2 (warm-start)                   0.6553  (+1.1)
  v4 (CE-trained readout on v2)     0.6555  (no gain — features redundant)
  v6 (warm-start, 4×512)            0.6638  (+0.85)
  v7 (warm-start + 4×512 + Muon)    0.6789  (+1.5)
  v8 (warm-start + 5×512 + Muon)    0.6785  (no gain from depth at fixed budget)

v4 didn't help over v2 because plain-AdamW DGL produced highly
redundant per-layer features. Muon's full-rank orthogonalised
updates in v7 ought to make per-layer features *less* redundant —
each layer's matmul receives a maximally-informative direction in
weight space, so successive blocks can specialise more.

v9 puts a CE-trained linear readout on top of *v7-style features*
to test that hypothesis. Configuration:
  - DGL training: 4 × d=512 × 3500 steps + Muon (lr=0.02) +
    AdamW (lr=2e-3) + WSD schedule, warm-start aux heads.
  - After DGL: discard aux heads. Per-layer LayerNorm each block's
    output, concatenate to 4·512 = 2048-dim features, train a single
    2048→256 linear readout with CE for 3000 steps (AdamW, lr=1e-3,
    cosine decay).

Both phases satisfy the no-backprop constraint:
  - DGL: only one block's autograd graph live at a time (same as v7).
  - Readout: blocks forward under torch.no_grad() (no activation
    cache), then per-layer LNs and the readout sit in a *single-
    layer* autograd graph (LNs are independent per-layer — they
    don't chain — so depth ≤ 2 even there).

Energy: v7 used 74 kJ for 4×4000 steps. v9 uses 4×3500 (~65 kJ for
DGL) + ~20 kJ for 3000 readout steps ≈ 85 kJ total. ~15 kJ headroom.
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
# Muon (ported from submission_modded.py / v7)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    assert G.ndim >= 2
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = torch.zeros_like(g)
                    state["momentum_buffer"] = buf
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf
                ortho = _zeropower_via_newtonschulz5(update, steps=ns_steps)
                fan_out, fan_in = ortho.shape[-2], ortho.shape[-1]
                scale = max(1.0, fan_out / fan_in) ** 0.5
                if wd:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(ortho.type_as(p.data), alpha=-lr * scale)
        return loss


# ---------------------------------------------------------------------------
# Architecture
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
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

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
        return self.proj(out), k, v


class Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)
        nn.init.zeros_(self.attn.proj.weight)
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


class DGLNetV9(nn.Module):
    """DGL-trained backbone + per-layer LN + concat-feature CE readout."""

    def __init__(self, vocab_size=256, d_model=512, n_layers=4,
                 n_heads=8, max_len=512):
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
        self.feat_lns = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )
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
        return _softcap(self.readout(feat)), new_caches


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


def _wsd_lr(step, *, total, warmup, decay_frac=0.2):
    decay_start = total - int(decay_frac * total)
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if step >= decay_start:
        return max(0.0, 1.0 - (step - decay_start) / max(1, total - decay_start))
    return 1.0


def _cosine_lr(step, *, peak_lr, warmup, total, min_lr_frac=0.1):
    if step < warmup:
        return peak_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1.0 - min_lr_frac) * cosine)


# ---------------------------------------------------------------------------
# Training: DGL+Muon (v7 recipe) → CE-trained linear readout
# ---------------------------------------------------------------------------

def train(train_text, valid_text=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")

    vocab_size = 256
    d_model = 512
    n_layers = 4
    n_heads = 8
    max_len = 512

    batch_size = 64
    n_steps_per_layer = 3500
    muon_lr = 0.02
    adam_lr = 2e-3
    warmup_steps = 250
    weight_decay = 0.0
    grad_clip = 1.0

    n_steps_readout = 3000
    readout_peak_lr = 1.5e-3
    readout_warmup = 200

    train_ids = torch.tensor(
        list(train_text.encode("utf-8")), dtype=torch.long, device=device,
    )

    model = DGLNetV9(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=max_len,
    ).to(device)
    model.apply(_init_weights)
    for blk in model.blocks:
        nn.init.zeros_(blk.attn.proj.weight)
        nn.init.zeros_(blk.fc2.weight)

    aux_lns = nn.ModuleList(
        [nn.LayerNorm(d_model).to(device) for _ in range(n_layers)]
    )
    aux_heads = nn.ModuleList(
        [nn.Linear(d_model, vocab_size, bias=False).to(device)
         for _ in range(n_layers)]
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DGL v9: {n_params/1e6:.2f}M params, {n_layers} × d={d_model}, "
          f"DGL_steps/layer={n_steps_per_layer}, readout_steps={n_steps_readout}",
          flush=True)

    t0 = time.monotonic()

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp
        else nullcontext()
    )

    # ----- Phase 1: DGL+Muon training (v7 recipe) -----
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

        # Split: hidden 2-D matmuls → Muon; LNs/aux head/embed → AdamW.
        muon_params = []
        adam_params = []
        for name, p in cur_block.named_parameters():
            if (p.ndim == 2 and
                ("attn.qkv" in name or "attn.proj" in name
                 or "fc1" in name or "fc2" in name)):
                muon_params.append(p)
            else:
                adam_params.append(p)
        adam_params += list(cur_ln.parameters())
        adam_params += list(cur_head.parameters())
        if layer_idx == 0:
            adam_params += list(model.tok_emb.parameters())

        opt_muon = Muon(muon_params, lr=muon_lr, momentum=0.95,
                        nesterov=True, ns_steps=5, weight_decay=weight_decay)
        opt_adam = torch.optim.AdamW(
            adam_params, lr=adam_lr, betas=(0.9, 0.95),
            weight_decay=weight_decay, eps=1e-10,
        )

        for step in range(n_steps_per_layer):
            scale = _wsd_lr(step, total=n_steps_per_layer, warmup=warmup_steps)
            for g in opt_muon.param_groups:
                g["lr"] = muon_lr * scale
            for g in opt_adam.param_groups:
                g["lr"] = adam_lr * scale

            idx = torch.randint(
                0, train_ids.numel() - max_len - 1,
                (batch_size,), device=device,
            )
            x = torch.stack([train_ids[i:i + max_len] for i in idx])
            y = torch.stack([train_ids[i + 1:i + 1 + max_len] for i in idx])

            opt_muon.zero_grad(set_to_none=True)
            opt_adam.zero_grad(set_to_none=True)

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
            torch.nn.utils.clip_grad_norm_(muon_params + adam_params, grad_clip)
            opt_muon.step()
            opt_adam.step()

            if step % 500 == 0 or step == n_steps_per_layer - 1:
                elapsed = time.monotonic() - t0
                print(f"  layer {layer_idx} step {step:4d}  loss {loss.item():.4f}"
                      f"  scale {scale:.3f}  elapsed {elapsed:.0f}s", flush=True)

        del opt_muon, opt_adam

    del aux_heads, aux_lns

    elapsed = time.monotonic() - t0
    print(f"DGL phase done in {elapsed:.0f}s — training CE readout", flush=True)

    # ----- Phase 2: CE-trained linear readout from concat features -----
    readout_params = list(model.feat_lns.parameters()) + list(model.readout.parameters())
    decay_r, no_decay_r = [], []
    for p in readout_params:
        (decay_r if p.dim() >= 2 else no_decay_r).append(p)
    opt_r = torch.optim.AdamW(
        [{"params": decay_r, "weight_decay": 0.1},
         {"params": no_decay_r, "weight_decay": 0.0}],
        lr=readout_peak_lr, betas=(0.9, 0.95), eps=1e-10,
    )

    model.eval()
    for ln in model.feat_lns:
        ln.train()
    model.readout.train()

    for step in range(n_steps_readout):
        lr = _cosine_lr(step, peak_lr=readout_peak_lr,
                        warmup=readout_warmup, total=n_steps_readout)
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
            with torch.no_grad():
                cos, sin = _rope_cos_sin(model.d_head, 0, max_len, device=device)
                h = model.tok_emb(x)
                per_layer_raw = []
                for block in model.blocks:
                    h, _, _ = block(h, cos, sin)
                    per_layer_raw.append(h.detach())

            per_layer_normed = [
                ln(h_raw)
                for ln, h_raw in zip(model.feat_lns, per_layer_raw)
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
    print(f"DGL v9 done in {elapsed:.0f}s", flush=True)
    return DGLV9CharModel(model, device)


class DGLV9CharModel(CharModel):
    def __init__(self, model: DGLNetV9, device: str):
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
