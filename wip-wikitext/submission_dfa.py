"""Char-level wikitext submission — Direct Feedback Alignment (DFA).

Trains all 6 transformer blocks *in parallel* every step using fixed
random feedback projections of the output error, never holding more
than one block's autograd graph live at once. This is the only path
in this branch that gives every block a global error signal without
chaining gradients across the stack.

Recipe (per FINDINGS.md §"Future directions / 2. DFA Parallelized"):

  - Block-DFA: forward pass is L block forwards with `.detach()` between
    blocks, so each block has an independent autograd graph rooted at
    its detached input. Per-block backwards use
    `torch.autograd.grad(block_output, block.params, grad_outputs=g_L)`
    with the synthetic output gradient
        g_L = (softmax(logits) - one_hot(y)) / (B*T) @ B_L.
    Only one block's autograd graph is live at any backward call —
    same memory profile as DGL.
  - **Sign-projection feedback** B_L = sign(N(0,1)) * (1/sqrt(V)) per
    block; sampled once at init, frozen forever. Sign vectors give
    more concentrated update directions than dense Gaussians at the
    same Frobenius norm.
  - **Muon** for hidden 2-D matmul weights (qkv, proj, fc1, fc2) of every
    block. The synthetic per-block gradient is what NS5 sees as `update`
    — orthogonalising it to full effective rank is exactly the fix DGL
    v11 demonstrated for low-rank gradient updates (+1.51 pp). DFA's
    synthetic grads are even more rank-deficient than DGL's local-NLL
    grads, so this is the highest-leverage transfer.
  - **AdamW** for embeddings / LayerNorm / bias / lm_head. The lm_head
    receives a *real* gradient via `loss.backward()` (it sits below the
    detach cut closest to the output and trains end-to-end with the
    final logits).
  - **Embedding gradient via DFA** through B_0: tok_emb sits above the
    first detach cut, so it cannot be reached by `loss.backward()`. We
    project the same output error through B_0 and apply it to the
    embedding's autograd graph (rooted at h_embed, which still has
    autograd to tok_emb).
  - WSD (warmup-stable-decay) schedule, ReLU² MLP, RoPE, logit softcap
    — same architecture knobs as the modded backprop baseline.

Energy: a 4k-step run measured 12.9 J/step (51.5 kJ wall) on the
pinned A100 SXM4. 7000 steps projects to ~90 kJ — fits under the
100 kJ watchdog with ~10 % margin. The 4k loss curve plateaued from
~step 1500 through ~step 3000 and only dropped further during the
WSD decay phase; bumping steps lengthens both the flat phase and
the decay tail, so the additional improvement is concentrated in
the cooldown.
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
# Muon optimizer (ported from submission_modded.py / exp_dgl_v11.py)
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
    """MomentUm Orthogonalized by Newton-schulz."""

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
# Architecture (RoPE + transformer block + softcap, same as DGL v11)
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
        # bias=False so the weight is a pure 2-D matrix for Muon.
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

    def forward(self, x, cos, sin, k_cache=None, v_cache=None):
        h, k, v = self.attn(self.ln1(x), cos, sin, k_cache, v_cache)
        x = x + h
        u = F.relu(self.fc1(self.ln2(x)))
        x = x + self.fc2(u * u)
        return x, k, v


SOFTCAP = 30.0


def _softcap(z):
    return SOFTCAP * torch.tanh(z / SOFTCAP)


class DFANet(nn.Module):
    def __init__(self, vocab_size=256, d_model=384, n_layers=6,
                 n_heads=6, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model
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


def _wsd_lr(step, *, total, warmup, decay_frac=0.2):
    decay_start = total - int(decay_frac * total)
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if step >= decay_start:
        return max(0.0, 1.0 - (step - decay_start) / max(1, total - decay_start))
    return 1.0


# ---------------------------------------------------------------------------
# DFA training
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
    n_steps = 7000
    muon_lr = 0.02
    adam_lr = 3e-3
    warmup_steps = 200
    grad_clip = 1.0

    # Sign-projection feedback: B_L = sign(N(0,1)) * 1/sqrt(V).
    # Per-entry std = 1/sqrt(V), same scale as the Gaussian baseline so
    # the synthetic-grad magnitude is comparable, but with sign-only
    # entries giving more concentrated update directions.
    fb_std = 1.0 / math.sqrt(vocab_size)

    train_ids = torch.tensor(
        list(train_text.encode("utf-8")), dtype=torch.long, device=device,
    )

    model = DFANet(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=max_len,
    ).to(device)
    model.apply(_init_weights)
    # Zero-init residual outputs so each block starts as ≈ identity.
    for blk in model.blocks:
        nn.init.zeros_(blk.attn.proj.weight)
        nn.init.zeros_(blk.fc2.weight)

    B_mats = []
    for _ in range(n_layers):
        B = torch.sign(torch.randn(vocab_size, d_model, device=device))
        # sign() can return 0 when randn produces an exact 0 (rare); map
        # to +1 so every entry is ±1.
        B = torch.where(B == 0, torch.ones_like(B), B)
        B = B * fb_std
        B.requires_grad_(False)
        B_mats.append(B)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DFA submission: {n_params/1e6:.2f}M params, {n_layers} layers, "
          f"d={d_model}, n_steps={n_steps}, muon_lr={muon_lr}, "
          f"adam_lr={adam_lr}, fb_std={fb_std:.4f} (sign)", flush=True)

    # Split parameters: hidden 2-D matmul weights → Muon, rest → AdamW.
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        is_hidden_2d = (
            p.ndim == 2
            and ("attn.qkv.weight" in name or "attn.proj.weight" in name
                 or "fc1.weight" in name or "fc2.weight" in name)
        )
        if is_hidden_2d:
            muon_params.append(p)
        else:
            adam_params.append(p)
    print(f"  muon params: {sum(p.numel() for p in muon_params)/1e6:.2f}M  "
          f"adam params: {sum(p.numel() for p in adam_params)/1e6:.2f}M",
          flush=True)

    opt_muon = Muon(muon_params, lr=muon_lr, momentum=0.95,
                    nesterov=True, ns_steps=5, weight_decay=0.0)
    opt_adam = torch.optim.AdamW(
        adam_params, lr=adam_lr, betas=(0.9, 0.95),
        weight_decay=0.0, eps=1e-10,
    )

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp
        else nullcontext()
    )

    t0 = time.monotonic()
    for step in range(n_steps):
        scale = _wsd_lr(step, total=n_steps, warmup=warmup_steps)
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
            cos, sin = _rope_cos_sin(model.d_head, 0, max_len, device=device)
            # Embedding forward keeps autograd live so tok_emb can receive
            # a DFA gradient via h_embed.
            h_embed = model.tok_emb(x)
            block_outputs = []
            h = h_embed.detach()
            h.requires_grad_(False)
            for L, block in enumerate(model.blocks):
                h_out, _, _ = block(h, cos, sin)
                block_outputs.append(h_out)
                h = h_out.detach()
                h.requires_grad_(False)

            # Final head + ln_f: single-layer autograd graph over the
            # detached final block output. loss.backward() will give head
            # + ln_f real gradients; everything upstream is already
            # detached so it gets nothing here.
            logits = _softcap(model.head(model.ln_f(h)))
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1),
            )

        # Real gradient on head + ln_f.
        loss.backward()

        # Synthetic per-block gradients: g_L = (softmax(logits) - y) @ B_L.
        with torch.no_grad():
            probs = F.softmax(logits.detach().float(), dim=-1)
            y_oh = F.one_hot(y, vocab_size).to(probs.dtype)
            e = (probs - y_oh) / (batch_size * max_len)  # (B, T, V)
            e_flat = e.reshape(-1, vocab_size)

        for L, block in enumerate(model.blocks):
            g_L_flat = e_flat @ B_mats[L]  # (B*T, d_model), fp32
            g_L = g_L_flat.reshape(batch_size, max_len, d_model)
            g_L = g_L.to(block_outputs[L].dtype)
            grads = torch.autograd.grad(
                outputs=block_outputs[L],
                inputs=list(block.parameters()),
                grad_outputs=g_L,
                retain_graph=False,
            )
            for p, g in zip(block.parameters(), grads):
                p.grad = g

        # tok_emb gets the same DFA signal projected through B_0. Its
        # autograd graph is rooted at h_embed, which still has autograd
        # to tok_emb (we only `.detach()`-ed the input to block 0).
        g_emb = (e_flat @ B_mats[0]).reshape(
            batch_size, max_len, d_model
        ).to(h_embed.dtype)
        emb_grads = torch.autograd.grad(
            outputs=h_embed, inputs=list(model.tok_emb.parameters()),
            grad_outputs=g_emb, retain_graph=False,
        )
        for p, g in zip(model.tok_emb.parameters(), emb_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(
            muon_params + adam_params, grad_clip,
        )
        opt_muon.step()
        opt_adam.step()

        if step % 200 == 0 or step == n_steps - 1:
            elapsed = time.monotonic() - t0
            print(f"  step {step:5d}  loss {loss.item():.4f}  "
                  f"scale {scale:.3f}  elapsed {elapsed:.0f}s", flush=True)

    elapsed = time.monotonic() - t0
    print(f"DFA training done in {elapsed:.0f}s", flush=True)
    return DFACharModel(model, device)


# ---------------------------------------------------------------------------
# Streaming inference (KV-cached)
# ---------------------------------------------------------------------------

class DFACharModel(CharModel):
    def __init__(self, model: DFANet, device: str):
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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(self.device == "cuda")):
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
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(self.device == "cuda")):
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
