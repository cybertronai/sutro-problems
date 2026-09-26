"""A two-hidden-layer MLP, trained from scratch on every call. Passes difficulty 1: 96.7% on MNIST,
about 700 ms per call on an A100-80GB (median of three sandboxed runs).

    python example.py            # score at difficulty 1 (needs a CUDA GPU; an A100 for official times)
"""

import torch
import torch.nn.functional as F

import mnist

WIDTH, STEPS, BATCH, LR = 1024, 400, 512, 4e-3


def mlp(train_x, train_y, test_x):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(60, WIDTH), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(WIDTH, WIDTH), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(WIDTH, 10),
    ).to(train_x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
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


if __name__ == "__main__":
    try:
        print(mnist.score(mlp, difficulty=1), "ms per call")
    except mnist.Disqualified:
        raise SystemExit(1)
