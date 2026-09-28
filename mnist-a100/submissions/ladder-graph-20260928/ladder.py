"""Ladder network for mnist-a100: the recipe that sets the difficulty bands, one training step in a CUDA graph.

The recipe is `ladder-xlong-s11` of mnist/experiments/release-cutoffs-20260925 (below-2pct/ladder_model.py
and neural.py's _fit_ladder_once): the fully supervised AMLP[2,2] Ladder of Pezeshki et al. (ICML 2016),
60-1000-500-250-250-250-10, input noise and per-layer noise 0.3, inputs scaled by 0.6, input
reconstruction weight 2000, Adam at 0.002 with the published 100/150 linear decay, 250 labelled and 250
unlabelled rows per step, the unlabelled rows drawn from train_x and test_x together (transductive), every
minibatch full, and BatchNorm statistics calibrated on the training rows at the end.

What differs from that code, none of it in the model or the objective:
* The whole step (noisy encoder, decoder, loss, backward, fused Adam) is captured once in a CUDA graph on
  the harness's untimed warm-up call and replayed. Each call re-initialises the weights, Adam's moments,
  the step counter and the random streams in place, so only the graph carries over between calls.
* Every epoch's shuffles are drawn up front on the GPU (one argsort per stream) and the graph reads its
  minibatch by step number, instead of a numpy permutation per epoch.
* The clean pass that updated BatchNorm running statistics every step is dropped: calibrate_bn replaces
  those statistics before prediction, so the pass never reached an output.
* Adam is PyTorch's fused implementation.
"""

import math

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

STEPS, BATCH = 5000, 250
DIMS = (60, 1000, 500, 250, 250, 250, 10)
NOISE, INPUT_SCALE, RECONSTRUCTION = 0.3, 0.6, 2000.0
LR, EPOCHS_CONFIG, DECAY_START_CONFIG = 0.002, 150, 100
EPS, COMBINATOR_STD = 1e-10, 0.025
SEED = 11
CACHE = {}


def normalize(x):
    """Batch normalisation without affine terms, over the rows of each stream (dim -2)."""
    variance, mean = torch.var_mean(x, dim=-2, unbiased=False, keepdim=True)
    return (x - mean) * torch.rsqrt(variance + EPS)


def combinator(params, lateral, vertical):
    """An independent 3 -> 2 -> 2 -> 1 leaky-ReLU MLP at every unit, on [vertical, lateral, product]."""
    h = torch.stack((vertical, lateral, vertical * lateral), dim=-1)
    for index in range(3):
        h = (h.unsqueeze(-1) * params[2 * index]).sum(-2) + params[2 * index + 1]
        if index < 2:
            h = F.leaky_relu(h, 0.1)
    return h.squeeze(-1)


def encode(params, h, layer, z):
    """The encoder's activation after normalisation: ReLU(z + beta), or (z + beta) * gamma at the top."""
    beta = params["beta"][layer]
    if layer == len(DIMS) - 2:
        return (z + beta) * params["gamma"]
    return F.relu(z + beta)


def loss(params, x_labelled, y_labelled, x_unlabelled):
    h = torch.stack((x_labelled, x_unlabelled))  # the two streams, normalised separately
    h = h + NOISE * torch.randn_like(h)
    lateral = [h[1]]
    for layer, weight in enumerate(params["encoder"]):
        z = normalize(h @ weight)
        z = z + NOISE * torch.randn_like(z)
        lateral.append(z[1])
        h = encode(params, h, layer, z)
    vertical = normalize(F.softmax(h[1], dim=-1))
    reconstruction = combinator(params["combinators"][-1], lateral[-1], vertical)
    for layer in range(len(DIMS) - 2, -1, -1):
        vertical = normalize(reconstruction @ params["decoder"][layer])
        reconstruction = combinator(params["combinators"][layer], lateral[layer], vertical)
    return F.cross_entropy(h[0], y_labelled) + RECONSTRUCTION * F.mse_loss(reconstruction, x_unlabelled)


class Trainer:
    def __init__(self, n, pool_n, device):
        self.device = device
        self.steps_per_epoch = math.ceil(n / BATCH)
        self.epochs = max(1, math.ceil(STEPS / self.steps_per_epoch))
        self.total = self.epochs * self.steps_per_epoch
        decay_start = min(self.epochs, max(1, round(self.epochs * DECAY_START_CONFIG / EPOCHS_CONFIG)))
        factor = [1.0 if self.epochs <= decay_start else
                  max(0.0, min(1.0, (self.epochs - e) / (self.epochs - decay_start))) for e in range(self.epochs)]
        self.schedule = (LR * torch.tensor(factor, device=device)).repeat_interleave(self.steps_per_epoch)
        self.n, self.pool_n = n, pool_n
        self.x = torch.zeros(n, DIMS[0], device=device)
        self.y = torch.zeros(n, dtype=torch.long, device=device)
        self.pool = torch.zeros(pool_n, DIMS[0], device=device)
        self.labelled = torch.zeros(self.total, BATCH, dtype=torch.long, device=device)
        self.unlabelled = torch.zeros(self.total, BATCH, dtype=torch.long, device=device)
        self.counter = torch.zeros(1, dtype=torch.long, device=device)
        zeros = lambda *shape: torch.zeros(*shape, device=device, requires_grad=True)
        self.params = {
            "encoder": [zeros(a, b) for a, b in zip(DIMS[:-1], DIMS[1:])],
            "beta": [zeros(b) for b in DIMS[1:]],
            "gamma": zeros(DIMS[-1]),
            "decoder": [zeros(b, a) for a, b in zip(DIMS[:-1], DIMS[1:])],
            "combinators": [[zeros(s, 3, 2), zeros(s, 2), zeros(s, 2, 2), zeros(s, 2), zeros(s, 2, 1), zeros(s, 1)]
                            for s in DIMS],
        }
        self.flat = (self.params["encoder"] + self.params["beta"] + [self.params["gamma"]] + self.params["decoder"]
                     + [p for c in self.params["combinators"] for p in c])
        cuda = device.type == "cuda"
        self.lr = torch.tensor(LR, device=device) if cuda else LR
        self.optimizer = torch.optim.Adam(self.flat, lr=self.lr, betas=(0.9, 0.999), eps=1e-8,
                                          fused=cuda, capturable=cuda)
        self.graph = None
        self.initialise()
        if cuda:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.graph):
                self.step()

    def step(self):
        labelled = self.labelled.index_select(0, self.counter).squeeze(0)
        unlabelled = self.unlabelled.index_select(0, self.counter).squeeze(0)
        if self.device.type == "cuda":
            self.lr.copy_(self.schedule.index_select(0, self.counter).squeeze(0))
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = self.schedule[int(self.counter)].item()
            self.optimizer.zero_grad(set_to_none=True)
        loss(self.params, self.x[labelled], self.y[labelled], self.pool[unlabelled]).backward()
        self.optimizer.step()
        with torch.no_grad():
            self.counter.add_(1)

    @torch.no_grad()
    def initialise(self):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(SEED)
        normal = lambda p, std: p.copy_(torch.randn(p.shape, generator=generator, device=self.device) * std)
        for weight in self.params["encoder"] + self.params["decoder"]:
            normal(weight, weight.shape[0] ** -0.5)
        for combinator_params in self.params["combinators"]:
            for index, p in enumerate(combinator_params):
                normal(p, COMBINATOR_STD) if index % 2 == 0 else p.zero_()
        for p in self.params["beta"]:
            p.zero_()
        self.params["gamma"].fill_(1.0)
        for p in self.flat:
            state = self.optimizer.state.get(p)
            if state:
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
                state["step"].zero_() if torch.is_tensor(state["step"]) else state.update(step=0)
        self.counter.zero_()
        # Each epoch's shuffles, drawn at once: labelled rows padded to full minibatches with a second
        # shuffle, unlabelled rows the first steps_per_epoch * BATCH of a shuffle of train and test rows.
        epochs, spe = self.epochs, self.steps_per_epoch
        order = torch.rand(epochs, self.n, generator=generator, device=self.device).argsort(1)
        if spe * BATCH > self.n:
            extra = torch.rand(epochs, self.n, generator=generator, device=self.device).argsort(1)
            order = torch.cat((order, extra[:, :spe * BATCH - self.n]), 1)
        self.labelled.copy_(order.reshape(self.total, BATCH))
        recon = torch.rand(epochs, self.pool_n, generator=generator, device=self.device).argsort(1)
        self.unlabelled.copy_(recon[:, :spe * BATCH].reshape(self.total, BATCH))
        if self.device.type == "cuda":
            torch.cuda.manual_seed(SEED + 1)
        else:
            torch.manual_seed(SEED + 1)

    @torch.no_grad()
    def predict(self, q):
        """Calibrate BatchNorm on the training rows (minibatch means and unbiased variances, averaged over
        one pass, deeper layers fed minibatch-normalised activations), then run the clean encoder on q."""
        params = self.params
        batches = self.n // BATCH
        generator = torch.Generator(device=self.device)
        generator.manual_seed(1)
        rows = torch.randperm(self.n, generator=generator, device=self.device)[:batches * BATCH]
        h = self.x[rows].reshape(batches, BATCH, DIMS[0])
        stats = []
        for layer, weight in enumerate(params["encoder"]):
            raw = h @ weight
            variance, mean = torch.var_mean(raw, dim=1, unbiased=False, keepdim=True)
            stats.append((mean.mean(0), variance.mean(0) * BATCH / (BATCH - 1)))
            h = encode(params, h, layer, (raw - mean) * torch.rsqrt(variance + EPS))
        h = q
        for layer, weight in enumerate(params["encoder"]):
            mean, variance = stats[layer]
            h = encode(params, h, layer, (h @ weight - mean) * torch.rsqrt(variance + EPS))
        return h


def ladder(train_x, train_y, test_x):
    device = train_x.device
    x = train_x.reshape(train_x.shape[0], -1).float() * INPUT_SCALE
    q = test_x.reshape(test_x.shape[0], -1).float() * INPUT_SCALE
    key = (tuple(x.shape), tuple(q.shape), str(device), STEPS)
    if key not in CACHE:
        CACHE.clear()
        CACHE[key] = Trainer(x.shape[0], x.shape[0] + q.shape[0], device)
    trainer = CACHE[key]
    with torch.no_grad():
        trainer.x.copy_(x)
        trainer.y.copy_(train_y.long())
        trainer.pool.copy_(torch.cat((x, q)))
    trainer.initialise()
    if trainer.graph is not None:
        for _ in range(trainer.total):
            trainer.graph.replay()
    else:
        for _ in range(trainer.total):
            trainer.step()
    return trainer.predict(q).argmax(1)
