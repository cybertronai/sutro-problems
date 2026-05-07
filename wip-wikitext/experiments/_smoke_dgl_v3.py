"""Tiny smoke variant of exp_dgl_v3.py — runs end-to-end on CPU in seconds."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_dgl_v3 as base
from exp_dgl_v3 import (
    DGLNetV3, DGLV3CharModel, _init_weights, _lr_at, _rope_cos_sin, _softcap,
)
import math, time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


def train(train_text, valid_text=None):
    device = "cpu"
    vocab_size = 256
    d_model = 32
    n_layers = 3
    n_heads = 4
    max_len = 64
    batch_size = 8
    n_steps_per_layer = 30
    ridge_lambda = 0.1
    ridge_n_windows = 4

    train_ids = torch.tensor(list(train_text.encode("utf-8")), dtype=torch.long, device=device)
    model = DGLNetV3(vocab_size, d_model, n_layers, n_heads, max_len).to(device)
    model.apply(_init_weights)
    for blk in model.blocks:
        nn.init.zeros_(blk.attn.out.weight)
        nn.init.zeros_(blk.fc2.weight)
    aux_lns = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
    aux_heads = nn.ModuleList([nn.Linear(d_model, vocab_size, bias=False) for _ in range(n_layers)])

    t0 = time.monotonic()
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
        params = list(cur_block.parameters()) + list(cur_ln.parameters()) + list(cur_head.parameters())
        if layer_idx == 0:
            params += list(model.tok_emb.parameters())
        decay, no_decay = [], []
        for p in params:
            (decay if p.dim() >= 2 else no_decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
            lr=1e-3, betas=(0.9, 0.95),
        )
        for step in range(n_steps_per_layer):
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
            opt.step()
        print(f"  L{layer_idx} done loss {loss.item():.3f}", flush=True)

    # Phase 2: ridge solve
    feat_dim = n_layers * d_model
    XtX = torch.zeros(feat_dim, feat_dim, dtype=torch.float32, device=device)
    XtY = torch.zeros(feat_dim, vocab_size, dtype=torch.float32, device=device)
    n_seen = 0
    model.eval()
    with torch.no_grad():
        cos, sin = _rope_cos_sin(model.d_head, 0, max_len, device=device)
        for w in range(ridge_n_windows):
            idx = torch.randint(0, train_ids.numel() - max_len - 1, (batch_size,), device=device)
            x = torch.stack([train_ids[i:i+max_len] for i in idx])
            y = torch.stack([train_ids[i+1:i+1+max_len] for i in idx])
            h = model.tok_emb(x)
            feats = []
            for blk in model.blocks:
                h, _, _ = blk(h, cos, sin)
                feats.append(h)
            feat = torch.cat(feats, dim=-1)
            feat32 = feat.reshape(-1, feat_dim).to(torch.float32)
            y_oh = F.one_hot(y.reshape(-1), vocab_size).to(torch.float32)
            XtX += feat32.T @ feat32
            XtY += feat32.T @ y_oh
            n_seen += feat32.shape[0]
    XtX += ridge_lambda * torch.eye(feat_dim, device=device, dtype=torch.float32)
    W = torch.linalg.solve(XtX, XtY)
    model.ridge_W = W.contiguous()
    print(f"  ridge solve done; n_seen={n_seen}, feat_dim={feat_dim}", flush=True)
    return DGLV3CharModel(model, device)
