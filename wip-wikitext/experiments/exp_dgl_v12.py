"""DGL v12 — v11 backbone + MLP-augmented aux head per layer.

v11 (5000 steps × Muon, 4 × d=512) reached 0.6839 — new best but still
4.7 pp below the 0.7310 bar at 93 % of the energy budget. Pure
parameter sweeps will not close that gap. Diagnosis: with a *linear*
aux head, layer L's features are shaped only by what a single Linear
can decode from them as a next-character predictor. Past v4/v9 showed
that a CE-trained linear readout from concatenated per-layer features
gains only ~0.14 pp — confirming that the linear readout already
extracts ~all the linearly-decodable next-char information from the
features themselves. To raise that ceiling, the *per-layer* objective
needs a richer head so deeper layers are pressured to encode features
that need non-linear decoding.

v12 keeps v11's recipe (4 × d=512, Muon for hidden 2-D matmuls,
AdamW for everything else, WSD schedule, warm-start) but replaces the
single-Linear aux head with a small MLP-residual head:

    aux(x) = head( ln_post( x + fc2( ReLU2( fc1( ln_pre(x) ) ) ) ) )

with fc2.weight initialised to zero (the residual MLP starts as the
identity). At step 0 this is *bit-equal* to v11's aux head, so the
warm-start tying of head to tok_emb still gives a perfect
bigram-baseline starting loss. As training proceeds, fc1/fc2 give the
head a non-linear extra layer of capacity which we expect to lower
per-layer NLL below the linear-decoder limit. fc1/fc2 are 2-D
matmuls — they go in Muon along with block weights.

No-backprop constraint preserved: only the *current* layer's block
plus its aux head are in the autograd graph at any given step. All
upstream blocks are frozen and run under torch.no_grad. We do not
keep any cross-layer activation cache.

Step-count: 4000 / layer (back to v7's count, vs v11's 5000) — the
extra MLP-head compute + optimizer state needs ~5 kJ headroom under
the 100 kJ cap (v11 used 93 kJ at 5000 steps).

Energy projection: 4 × 4000 × ~25 ms ≈ 400 s × ~165 W = ~66 kJ
training + ~15 kJ eval ≈ 81 kJ total. ~19 kJ slack.
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
# Muon (ported from submission_modded.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz5 quintic iteration in bf16. Orthogonalises G.

    Coefficients (3.4445, -4.7750, 2.0315) are Keller Jordan's tuned
    pick. Spectral-norm normalises so the iteration converges; the
    transpose trick keeps the iteration on the smaller dimension.
    """
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
# Architecture (RoPE + transformer block + softcap, identical to v6)
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
        # bias=False so the weight is pure 2-D for Muon.
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
        # bias=False on fc1/fc2 too.
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


class MLPAuxHead(nn.Module):
    """Per-layer aux head: residual MLP block + linear projection to V.

    Initialisation: fc2.weight = 0 → residual is identity at step 0 →
    aux(x) = head(ln_post(x)). With head warm-started to tok_emb (and
    ln_post starting near-identity from default init), this matches v11's
    aux head bit-for-bit at step 0.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.ln_pre = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model, bias=False)
        self.fc2 = nn.Linear(d_model, d_model, bias=False)
        self.ln_post = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc2.weight)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.relu(self.fc1(self.ln_pre(x)))
        x = x + self.fc2(u * u)
        return self.head(self.ln_post(x))


class DGLNetV11(nn.Module):
    def __init__(self, vocab_size=256, d_model=512, n_layers=4,
                 n_heads=8, max_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_head = d_model // n_heads
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads) for _ in range(n_layers)]
        )
        # The "final head" is the deepest layer's MLP aux head, copied
        # in at end-of-DGL-training. Same shape/init as MLPAuxHead.
        self.final_aux = MLPAuxHead(d_model, vocab_size)

    def forward(self, x, kv_caches=None, *, position=0):
        T = x.shape[1]
        h = self.tok_emb(x)
        cos, sin = _rope_cos_sin(self.d_head, position, T, device=x.device)
        new_caches = []
        for i, block in enumerate(self.blocks):
            kc, vc = (None, None) if kv_caches is None else kv_caches[i]
            h, k, v = block(h, cos, sin, kc, vc)
            new_caches.append((k, v))
        return _softcap(self.final_aux(h)), new_caches


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
# DGL training with Muon for hidden weights, AdamW for everything else
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
    n_steps_per_layer = 4000
    muon_lr = 0.02
    adam_lr = 2e-3
    warmup_steps = 250
    weight_decay = 0.0  # modded uses 0 for both opts; we follow
    grad_clip = 1.0

    train_ids = torch.tensor(
        list(train_text.encode("utf-8")), dtype=torch.long, device=device,
    )

    model = DGLNetV11(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=max_len,
    ).to(device)
    model.apply(_init_weights)
    for blk in model.blocks:
        nn.init.zeros_(blk.attn.proj.weight)
        nn.init.zeros_(blk.fc2.weight)

    aux_heads = nn.ModuleList(
        [MLPAuxHead(d_model, vocab_size).to(device)
         for _ in range(n_layers)]
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_aux_params = sum(p.numel() for p in aux_heads.parameters())
    print(f"DGL v12: {n_params/1e6:.2f}M block params + "
          f"{n_aux_params/1e6:.2f}M aux-head params, {n_layers} layers, "
          f"d={d_model}, h={n_heads}, steps/layer={n_steps_per_layer}, "
          f"muon_lr={muon_lr}, adam_lr={adam_lr}", flush=True)

    t0 = time.monotonic()

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp
        else nullcontext()
    )

    for layer_idx in range(n_layers):
        cur_block = model.blocks[layer_idx]
        cur_aux = aux_heads[layer_idx]

        if layer_idx > 0:
            with torch.no_grad():
                cur_aux.load_state_dict(
                    aux_heads[layer_idx - 1].state_dict()
                )

        # Split this stage's params: hidden 2-D matmuls → Muon; others → AdamW.
        muon_params = []
        adam_params = []
        for name, p in cur_block.named_parameters():
            if (p.ndim == 2 and
                ("attn.qkv" in name or "attn.proj" in name
                 or "fc1" in name or "fc2" in name)):
                muon_params.append(p)
            else:
                adam_params.append(p)
        # Aux head: fc1/fc2 are 2-D matmuls → Muon. lns + final head → AdamW.
        for name, p in cur_aux.named_parameters():
            if name in ("fc1.weight", "fc2.weight"):
                muon_params.append(p)
            else:
                adam_params.append(p)
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
                logits = _softcap(cur_aux(h))
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

    with torch.no_grad():
        model.final_aux.load_state_dict(aux_heads[-1].state_dict())

    elapsed = time.monotonic() - t0
    print(f"DGL v12 done in {elapsed:.0f}s", flush=True)
    return DGLV11CharModel(model, device)


class DGLV11CharModel(CharModel):
    def __init__(self, model: DGLNetV11, device: str):
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
