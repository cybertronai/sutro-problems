"""Tiny smoke variant of exp_dfa_v1.py — runs end-to-end on CPU in seconds."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math, time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_dfa_v1 import (
    DFANet, DFACharModel, _init_weights, _lr_at, _rope_cos_sin, _softcap,
)


def train(train_text, valid_text=None):
    device = "cpu"
    vocab_size = 256
    d_model = 64
    n_layers = 4
    n_heads = 4
    max_len = 64
    batch_size = 16
    n_steps = 80
    peak_lr = 1e-3
    warmup_steps = 10

    train_ids = torch.tensor(list(train_text.encode("utf-8")), dtype=torch.long, device=device)
    model = DFANet(vocab_size, d_model, n_layers, n_heads, max_len).to(device)
    model.apply(_init_weights)

    fb_std = 1.0 / math.sqrt(vocab_size)
    B_mats = [torch.randn(vocab_size, d_model) * fb_std for _ in range(n_layers)]
    for B in B_mats:
        B.requires_grad_(False)

    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr, betas=(0.9, 0.95), eps=1e-9,
    )

    t0 = time.monotonic()
    for step in range(n_steps):
        idx = torch.randint(0, train_ids.numel() - max_len - 1, (batch_size,), device=device)
        x = torch.stack([train_ids[i:i+max_len] for i in idx])
        y = torch.stack([train_ids[i+1:i+1+max_len] for i in idx])

        opt.zero_grad(set_to_none=True)
        cos, sin = _rope_cos_sin(model.d_head, 0, max_len, device=device)
        h_embed = model.tok_emb(x)
        block_outputs = []
        h = h_embed.detach()
        h.requires_grad_(False)
        for L, block in enumerate(model.blocks):
            h_out, _, _ = block(h, cos, sin)
            block_outputs.append(h_out)
            h = h_out.detach()
            h.requires_grad_(False)

        logits_pre = model.head(model.ln_f(h))
        logits = _softcap(logits_pre)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        loss.backward()

        with torch.no_grad():
            probs = F.softmax(logits.detach().float(), dim=-1)
            y_oh = F.one_hot(y, vocab_size).to(probs.dtype)
            e = (probs - y_oh) / (batch_size * max_len)
            e_flat = e.reshape(-1, vocab_size)

        for L, block in enumerate(model.blocks):
            g_L = (e_flat @ B_mats[L]).reshape(batch_size, max_len, d_model)
            g_L = g_L.to(block_outputs[L].dtype)
            grads = torch.autograd.grad(
                outputs=block_outputs[L], inputs=list(block.parameters()),
                grad_outputs=g_L, retain_graph=False,
            )
            for p, g in zip(block.parameters(), grads):
                p.grad = g

        g_emb = (e_flat @ B_mats[0]).reshape(batch_size, max_len, d_model).to(h_embed.dtype)
        emb_grads = torch.autograd.grad(
            outputs=h_embed, inputs=list(model.tok_emb.parameters()),
            grad_outputs=g_emb, retain_graph=False,
        )
        for p, g in zip(model.tok_emb.parameters(), emb_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 20 == 0 or step == n_steps - 1:
            print(f"  step {step:3d} loss {loss.item():.3f} elapsed {time.monotonic()-t0:.1f}s", flush=True)

    print(f"smoke DFA done in {time.monotonic()-t0:.1f}s", flush=True)
    return DFACharModel(model, device)
