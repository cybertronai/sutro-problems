#!POPCORN leaderboard mnist-a100-1
#!POPCORN gpu A100
#!POPCORN function custom_kernel
# Submission by @yaroslavvb, based on example.py by @yaroslavvb.
# Faster variant: 200 steps, TF32 matmuls, and fused AdamW.
"""A fresh 60-1024-1024-10 MLP for each draw.

Compared with example.py, halve training from 400 to 200 steps, enable TF32,
and fuse AdamW updates. Keep architecture, regularization, batch size and LR.
All parameters and optimizer state are initialized anew on every call.
"""

import torch
import torch.nn.functional as F


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

WIDTH, STEPS, BATCH, LR = 1024, 200, 512, 4e-3


def mlp(train_x, train_y, test_x):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(60, WIDTH), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(WIDTH, WIDTH), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(WIDTH, 10),
    ).to(train_x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01,
                            fused=train_x.device.type == "cuda")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    for step in range(STEPS):
        idx = torch.randint(0, train_x.shape[0], (BATCH,), device=train_x.device)
        x = train_x[idx] + 0.3 * torch.randn(BATCH, 60, device=train_x.device)  # input noise
        loss = F.cross_entropy(model(x), train_y[idx], label_smoothing=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return model(test_x).argmax(1)


def custom_kernel(train_x, train_y, test_x):
    return mlp(train_x, train_y, test_x)
