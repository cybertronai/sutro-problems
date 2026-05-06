"""Tiny smoke variant of exp_dgl_v2.py — runs end-to-end on CPU in seconds."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_dgl_v2 as base
from exp_dgl_v2 import DGLNet, DGLCharModel, _init_weights, _lr_at, _rope_cos_sin, _softcap
import math, time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


def train(train_text, valid_text=None):
    device = "cpu"
    vocab_size = 256
    d_model = 64
    n_layers = 4
    n_heads = 4
    max_len = 64
    batch_size = 16
    n_steps_per_layer = 60
    peak_lr = 1e-3
    warmup_steps = 10

    train_ids = torch.tensor(list(train_text.encode("utf-8")), dtype=torch.long, device=device)
    model = DGLNet(vocab_size, d_model, n_layers, n_heads, max_len).to(device)
    model.apply(_init_weights)
    for blk in model.blocks:
        nn.init.zeros_(blk.attn.out.weight)
        nn.init.zeros_(blk.fc2.weight)

    t0 = time.monotonic()
    for layer_idx in range(n_layers):
        cur_block = model.blocks[layer_idx]
        cur_ln = model.aux_lns[layer_idx]
        cur_head = model.aux_heads[layer_idx]
        if layer_idx == 0:
            nn.init.normal_(cur_head.weight, mean=0.0, std=0.02)
        else:
            with torch.no_grad():
                cur_ln.load_state_dict(model.aux_lns[layer_idx - 1].state_dict())
                cur_head.weight.copy_(model.aux_heads[layer_idx - 1].weight)
        params = list(cur_block.parameters()) + list(cur_ln.parameters()) + list(cur_head.parameters())
        if layer_idx == 0:
            params += list(model.tok_emb.parameters())
        decay, no_decay = [], []
        for p in params:
            (decay if p.dim() >= 2 else no_decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
            lr=peak_lr, betas=(0.9, 0.95),
        )
        for step in range(n_steps_per_layer):
            lr = _lr_at(step, peak_lr=peak_lr, warmup=warmup_steps, total=n_steps_per_layer)
            for g in opt.param_groups:
                g["lr"] = lr
            idx = torch.randint(0, train_ids.numel() - max_len - 1, (batch_size,), device=device)
            x = torch.stack([train_ids[i:i+max_len] for i in idx])
            y = torch.stack([train_ids[i+1:i+1+max_len] for i in idx])
            opt.zero_grad(set_to_none=True)
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
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 30 == 0 or step == n_steps_per_layer - 1:
                elapsed = time.monotonic() - t0
                print(f"  L{layer_idx} step {step:3d} loss {loss.item():.3f} elapsed {elapsed:.1f}s", flush=True)
    print(f"smoke v2 training done in {time.monotonic()-t0:.1f}s", flush=True)
    return DGLCharModel(model, device)
