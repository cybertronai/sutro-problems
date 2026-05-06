"""Char-level wikitext submission, modded-nanogpt-flavored.

modded-nanogpt itself is tuned for 8xH100 BPE training on FineWeb (FP8
matmul, FlashAttention-3 varlen sliding-window kernels, sparse comms,
parameter-bank sharding, MTP, value embeddings). Most of that doesn't
apply to single-A100 char-level training under a 5-min energy budget.

What *does* transfer cleanly, and is implemented here:

  - Muon optimizer for hidden 2D matmul weights (qkv, proj, fc1, fc2)
    with AdamW for embeddings / lm_head / LayerNorm. The headline
    speedrun gain.
  - Newton-Schulz5 orthogonalization in bf16 (Keller Jordan reference
    impl; Polar Express was added later as an in-place upgrade — we
    use the classic version because it's simpler and the gap is small
    at this scale).
  - QK-norm (RMSNorm on q,k after RoPE) for training stability.
  - ReLU² MLP activation (modded-nanogpt's choice; small but free).
  - Logit softcapping `30 * tanh(z / 30)` (Gemma-2 style), applied
    consistently in train and eval so streaming inference matches
    training distribution.
  - Per-group learning rates: high LR for embedding/head (which scale
    differently from hidden matmuls), Muon LR for hidden weights.
  - WSD schedule: linear warmup → flat → linear cooldown to 0
    (better than cosine for short trapezoidal runs).
  - Tied input/output embeddings (GPT-2 convention; halves embedding
    energy footprint, helpful at this scale).
  - bf16 autocast on CUDA, tf32 matmul precision.

Streaming inference uses a per-instance KV-cache with sliding-window
trim at `max_len`, mirroring `baseline_transformer.TransformerModel`'s
RoPE-aware position counter so a >max_len test stream stays inside
the trained relative-offset range.
"""
import math
import time
from dataclasses import dataclass
from typing import cast, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from wikitext import CharModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6  # head_dim = 64
    max_len: int = 512
    softcap: float = 30.0


# ---------------------------------------------------------------------------
# Muon optimizer (single-GPU, non-sharded)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz5 quintic iteration on bf16 — orthogonalizes G.

    Coefficients (3.4445, -4.7750, 2.0315) chosen by Keller Jordan to
    maximize slope at zero subject to staying in the convergence region
    on [0, sqrt(2)].

    G can be 2-D or batched 3-D; iterations are applied along the last
    two dims independently.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    assert G.ndim >= 2
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-2, -1)
    # Spectral-norm normalize so the iteration converges.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-2, -1)
    return X


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz.

    Reference: https://kellerjordan.github.io/posts/muon/

    Applied only to hidden-layer 2-D matmul weights (qkv, proj, mlp).
    Embeddings, biases, LayerNorms, and the lm_head go through AdamW.
    """

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
                # LR scaling by aspect ratio: matches modded-nanogpt's
                # `max(1, fan_out/fan_in)**0.5` heuristic.
                fan_out, fan_in = ortho.shape[-2], ortho.shape[-1]
                scale = max(1.0, fan_out / fan_in) ** 0.5
                if wd:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(ortho.type_as(p.data), alpha=-lr * scale)
        return loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _rope_cos_sin(d_head: int, offset: int, T: int, *, device,
                  base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (
        torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head
    ))
    pos = torch.arange(offset, offset + T, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_e, x_o = x[..., 0::2], x[..., 1::2]
    c = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)
    s = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    rot_e = x_e * c - x_o * s
    rot_o = x_e * s + x_o * c
    return torch.stack((rot_e, rot_o), dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    """RMSNorm without learnable scale — modded-nanogpt's `norm()`."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability.
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.qk_norm = RMSNorm()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # QK-norm @Grad62304977.
        q = self.qk_norm(q)
        k = self.qk_norm(k)
        if k_cache is not None and v_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        is_causal = (k_cache is None) and T > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out), k, v


class ReLUSqMLP(nn.Module):
    """ReLU² MLP — modded-nanogpt's choice for hidden activations."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff, bias=False)
        self.proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        h = F.relu(h).square()
        return self.proj(h)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = RMSNorm()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.norm2 = RMSNorm()
        self.mlp = ReLUSqMLP(d_model, 4 * d_model)

    def forward(self, x, cos, sin, k_cache=None, v_cache=None):
        h, k, v = self.attn(self.norm1(x), cos, sin, k_cache, v_cache)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, k, v


class CharGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.d_head = cfg.d_model // cfg.n_heads
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers)]
        )
        self.norm_f = RMSNorm()
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie input + output embeddings (GPT-2 convention).
        self.head.weight = self.tok_emb.weight
        self.apply(_init_weights)

    def forward(
        self,
        x: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        position: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        T = x.shape[1]
        h = self.tok_emb(x)
        cos, sin = _rope_cos_sin(self.d_head, position, T, device=x.device)
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            kc, vc = (None, None) if kv_caches is None else kv_caches[i]
            h, k, v = cast("Block", block)(h, cos, sin, kc, vc)
            new_caches.append((k, v))
        h = self.norm_f(h)
        logits = self.head(h)
        # Logit softcapping (Gemma-2 style; consistent train+eval so the
        # streaming distribution matches the training distribution).
        cap = self.cfg.softcap
        logits = cap * torch.tanh(logits / cap)
        return logits, new_caches


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

def _wsd_lr(step: int, *, total: int, warmup: int, decay_frac: float = 0.2) -> float:
    """Warmup-Stable-Decay schedule.

    Linear warmup over `warmup` steps, flat, then linear cooldown to 0
    over the final `decay_frac × total` steps. Better than cosine for
    short trapezoidal runs (Hägele et al. 2024).
    """
    decay_start = total - int(decay_frac * total)
    if step < warmup:
        return (step + 1) / max(1, warmup)
    if step >= decay_start:
        return max(0.0, 1.0 - (step - decay_start) / max(1, total - decay_start))
    return 1.0


def _split_params_for_optim(model: CharGPT):
    """Split parameters into (muon_2d_hidden, adam_other).

    Muon: 2-D weights of qkv / proj / fc / proj inside CausalSelfAttention
    and ReLUSqMLP. Adam: token embedding (tied with head), biases, scalars.
    """
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_hidden_2d = (
            p.ndim == 2
            and ("attn.qkv.weight" in name or "attn.proj.weight" in name
                 or "mlp.fc.weight" in name or "mlp.proj.weight" in name)
        )
        if is_hidden_2d:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


def _train(
    train_text: str,
    *,
    cfg: Config,
    n_steps: int,
    batch_size: int,
    muon_lr: float,
    adam_lr: float,
    warmup: int,
    log_every: int,
) -> CharGPT:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"[modded] device={device}  use_amp={use_amp}")

    train_ids = torch.tensor(
        list(train_text.encode("utf-8")), dtype=torch.long, device=device,
    )
    if train_ids.numel() < cfg.max_len + 1:
        raise ValueError(
            f"need at least {cfg.max_len + 1} bytes; got {train_ids.numel()}"
        )

    model = CharGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Subtract tied head weight (counted twice via .parameters()).
    if model.head.weight is model.tok_emb.weight:
        n_params -= model.head.weight.numel()
    print(f"[modded] params: {n_params/1e6:.2f}M  cfg={cfg}")

    muon_params, adam_params = _split_params_for_optim(model)
    print(f"[modded] muon params: {sum(p.numel() for p in muon_params)/1e6:.2f}M  "
          f"adam params: {sum(p.numel() for p in adam_params)/1e6:.2f}M")

    opt_muon = Muon(muon_params, lr=muon_lr, momentum=0.95, nesterov=True,
                    ns_steps=5, weight_decay=0.0)
    opt_adam = torch.optim.AdamW(
        adam_params, lr=adam_lr, betas=(0.9, 0.95),
        weight_decay=0.0, eps=1e-10,
    )

    model.train()
    t0 = time.monotonic()
    for step in range(n_steps):
        scale = _wsd_lr(step, total=n_steps, warmup=warmup)
        for g in opt_muon.param_groups:
            g["lr"] = muon_lr * scale
        for g in opt_adam.param_groups:
            g["lr"] = adam_lr * scale

        idx = torch.randint(
            0, train_ids.numel() - cfg.max_len - 1,
            (batch_size,), device=device,
        )
        x = torch.stack([train_ids[i : i + cfg.max_len] for i in idx])
        y = torch.stack([train_ids[i + 1 : i + 1 + cfg.max_len] for i in idx])

        opt_muon.zero_grad(set_to_none=True)
        opt_adam.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), y.reshape(-1),
                )
        else:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), y.reshape(-1),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], 1.0,
        )
        opt_muon.step()
        opt_adam.step()

        if log_every and (step % log_every == 0 or step == n_steps - 1):
            elapsed = time.monotonic() - t0
            print(f"[modded] step {step:6d}  loss {loss.item():.4f}  "
                  f"lr {muon_lr * scale:.2e}/{adam_lr * scale:.2e}  "
                  f"elapsed {elapsed:.0f}s", flush=True)
    return model


# ---------------------------------------------------------------------------
# Streaming CharModel adapter
# ---------------------------------------------------------------------------

class ModdedCharModel(CharModel):
    """KV-cached streaming wrapper around `CharGPT`.

    Same RoPE-aware sliding-window cache trim as
    baseline_transformer.TransformerModel: a true absolute-position
    counter is fed to every forward so RoPE rotates the new query at
    its real position; the cache is trimmed to `max_len - 1` once the
    stream exceeds the trained window.
    """

    def __init__(self, model: CharGPT):
        self.model = model
        self.device = next(model.parameters()).device.type
        self.model.eval()
        self._kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._next_logits: torch.Tensor | None = None
        self._pos: int = 0

    @torch.no_grad()
    def reset(self) -> None:
        self._kv = None
        self._pos = 0
        # Seed with byte 0 (stream-start sentinel; same convention as
        # baseline_transformer).
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
        if cur < self.model.cfg.max_len:
            return
        keep = self.model.cfg.max_len - 1
        self._kv = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in self._kv]


# ---------------------------------------------------------------------------
# Submission entry point
# ---------------------------------------------------------------------------

def train(train_text: str, valid_text: str | None = None) -> CharModel:
    """Wikitext submission entry point.

    Trains a ~10M-param char-level GPT with Muon+AdamW under the
    submitter's energy budget. Returns a streaming-ready CharModel.
    """
    cfg = Config(
        vocab_size=256,
        d_model=384,
        n_layers=6,
        n_heads=6,
        max_len=512,
        softcap=30.0,
    )
    model = _train(
        train_text,
        cfg=cfg,
        # Sized to fit comfortably inside the 100 kJ / ~5 min budget on
        # a Lambda A100 SXM4 40GB. 3000 steps × ~70-90 ms/step ≈ 4 min.
        # The remainder is amp/compile warmup, optimizer init, and a
        # safety margin. Tune in follow-up iterations.
        n_steps=3000,
        batch_size=64,
        muon_lr=0.02,
        adam_lr=3e-3,
        warmup=200,
        log_every=100,
    )
    return ModdedCharModel(model)
