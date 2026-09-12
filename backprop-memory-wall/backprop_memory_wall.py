#!/usr/bin/env python3
"""
backprop_memory_wall.py
Compute-to-commute model of one transformer training step (forward + backward + update).

Every number below is a parameter. Defaults and their provenance:
  * Wire / add / SRAM energies: Dally, AHA retreat keynote, 31 Aug 2023
      - on-chip communication ~100 fJ/bit-mm; add ~1 fJ/bit; small RAM ~50 fJ/bit
      - HBM access ~5 pJ/bit at the interface; GPU-die round trip 100 fJ/b-mm x path length
      => "an add is worth 10 um of movement"; 1 byte across 16 mm = 1600 8-bit adds
  * A100 / H100 peak, bandwidth, TDP: NVIDIA data sheets (dense BF16 tensor-core peak).
  * e_flop (all-in on-chip energy per FLOP at peak) is *derived*: (TDP - static - HBM) / peak.
  * Activation-storage constant 34*d bytes/token/layer: Korthikanti et al. 2022 (Megatron).
  * Traffic constants b_w, a, b_opt: counted from the kernel sequence of a fused, FlashAttention
    transformer layer (see report, Section 2). Change them if your stack differs.
Usage:  python3 backprop_memory_wall.py   -> prints tables, writes fig_backprop_memory_wall.png
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------
# 1. Physical constants (Dally 2023) and the "add-equivalent" unit
# ----------------------------------------------------------------------------------------------
E_WIRE_BIT_MM = 100e-15     # J per bit per mm of on-chip wire
E_ADD_BIT     = 1e-15       # J per bit for an add
E_SRAM_BIT    = 50e-15      # J per bit, small local RAM
AE            = 8 * E_ADD_BIT   # one int8 add = 8 fJ = one byte moved 10 um  (the unit used in the text)

def ae(joules):            # convert an energy to add-equivalents
    return joules / AE

# ----------------------------------------------------------------------------------------------
# 2. Machines
# ----------------------------------------------------------------------------------------------
GPUS = {
    # peak: dense BF16 FLOP/s; bw: HBM B/s; l2: bytes; e_hbm: J/byte all-in (interface + on-die transport)
    # e_flop: all-in on-chip J/FLOP at peak (derived from power budget); e_dp: datapath-only estimate
    # p_static: idle/leakage W; fabric: per-direction B/s for NVLink and one IB NIC
    "A100-80GB": dict(peak=312e12, bw=2.039e12, l2=40e6, e_hbm=60e-12, e_flop=0.90e-12, e_dp=0.10e-12,
                      p_static=60.0, tdp=400.0, nvlink=300e9, ib=25e9),
    "A100-40GB": dict(peak=312e12, bw=1.555e12, l2=40e6, e_hbm=60e-12, e_flop=0.90e-12, e_dp=0.10e-12,
                      p_static=60.0, tdp=400.0, nvlink=300e9, ib=25e9),
    "H100-SXM":  dict(peak=989e12, bw=3.35e12, l2=50e6, e_hbm=50e-12, e_flop=0.55e-12, e_dp=0.07e-12,
                      p_static=90.0, tdp=700.0, nvlink=450e9, ib=50e9),
    "B200 (approx.)": dict(peak=2.25e15, bw=8.0e12, l2=126e6, e_hbm=45e-12, e_flop=0.33e-12, e_dp=0.05e-12,
                      p_static=150.0, tdp=1000.0, nvlink=900e9, ib=50e9),
}

def balance(g):            # machine balance beta_t (FLOP/byte at HBM) and energy balance beta_E
    return g["peak"] / g["bw"], g["e_hbm"] / g["e_flop"]

# ----------------------------------------------------------------------------------------------
# 3. Traffic model of one transformer layer, one micro-batch of T tokens on one weight-holding device
# ----------------------------------------------------------------------------------------------
B_W_DEFAULT   = 12    # bytes/param per micro-batch at HBM: read W (bf16) fwd 2 + bwd 2 + fp32 grad RMW 8
A_DEFAULT     = 200   # inter-kernel activation traffic, bytes per token per layer, per unit d (fused, FlashAttn)
A_STORE       = 34    # saved-activation *capacity*, bytes/token/layer per unit d (Korthikanti et al. 2022)
B_OPT_DEFAULT = 30    # bytes/param per optimizer step (Adam, fp32 master + m + v, bf16 copy)
B_AR_DEFAULT  = 4     # bytes/param per step over the DP fabric (ring all-reduce of bf16 grads, both directions)

def layer_step(d, T, g, b_w=B_W_DEFAULT, a=A_DEFAULT, s=None, l2_frac=0.5):
    """FLOPs, HBM bytes (weights, activations), time and energy for one layer x one micro-batch."""
    P = 12 * d * d                                  # params/layer (4d^2 attention + 8d^2 FFN)
    F = 72 * d * d * T                              # fwd 24d^2 + bwd 48d^2 per token
    if s:                                           # optional attention term (fwd 4sd, bwd ~8sd w/ recompute)
        F += 12 * s * d * T
    Z = l2_frac * g["l2"] / 2                       # usable L2 in bf16 elements
    # weight-side bytes: W streams once per micro-batch (b_w bytes/param) plus, once the T x d input no
    # longer fits in L2, one extra stream per L2-sized input chunk (Hong-Kung: GEMM traffic >= MNK/sqrt(Z)
    # elements).  The sum is an upper envelope of the two regimes; it is smooth and conservative.
    B_W = P * (b_w + 6 * T / math.sqrt(Z))
    B_act = a * d * T
    t_mm = max(F / g["peak"], B_W / g["bw"])        # GEMM phase: weights stream while tensor cores run
    t_ew = B_act / g["bw"]                          # element-wise/inter-kernel phase: memory-bound, cores idle
    t = t_mm + t_ew
    E_arith = F * g["e_flop"]
    E_hbm_w = B_W * g["e_hbm"]
    E_hbm_a = B_act * g["e_hbm"]
    E_static = g["p_static"] * t
    return dict(F=F, B_W=B_W, B_act=B_act, I=F / (B_W + B_act), t=t, t_mm=t_mm, t_ew=t_ew,
                E_arith=E_arith, E_hbm_w=E_hbm_w, E_hbm_a=E_hbm_a, E_static=E_static,
                E=E_arith + E_hbm_w + E_hbm_a + E_static)

def T_min(d, g, b_w=B_W_DEFAULT, a=A_DEFAULT, l2_frac=0.5):
    """Smallest micro-batch (tokens per weight visit) with HBM intensity >= machine balance.
    Closed form with the L2 re-blocking term folded into an effective a:  a_eff = a + 72 d / sqrt(Z)."""
    beta = g["peak"] / g["bw"]
    Z = l2_frac * g["l2"] / 2
    a_eff = a + 72 * d / math.sqrt(Z)
    den = 72 * d - a_eff * beta
    return math.inf if den <= 0 else 12 * b_w * d * beta / den

def d_min(g, a=A_DEFAULT, l2_frac=0.5):
    """Below this hidden size no batch size reaches the HBM balance (activation traffic alone exceeds it)."""
    beta = g["peak"] / g["bw"]
    Z = l2_frac * g["l2"] / 2
    return a * beta / (72 * (1 - beta / math.sqrt(Z)))

# ----------------------------------------------------------------------------------------------
# 4. Tables
# ----------------------------------------------------------------------------------------------
def table_boundaries():
    print("\n== Table 1: cost of a byte at each boundary (A100-class, 7 nm; estimates, see report) ==")
    rows = [
        ("int8 add (datapath)",          8e-15,      "1 fJ/bit (Dally 2023)"),
        ("BF16 FLOP, datapath only",     0.10e-12,   "HMMA-class MAC, 45nm->7nm scaled"),
        ("BF16 FLOP, all-in on A100",    0.90e-12,   "(400 W - 60 W static - 40 W HBM)/312 TFLOP/s"),
        ("1 byte moved 1 mm",            8*100e-15,  "100 fJ/bit-mm"),
        ("1 byte across a 16 mm die",    8*1.6e-12,  "= 1600 int8 adds (Dally's slide)"),
        ("1 byte from register file",    0.3e-12,    "~40 fJ/bit incl. local wire"),
        ("1 byte from SMEM/L1 (192 KB)", 1.0e-12,    "50 fJ/bit array + ~0.5 mm round trip"),
        ("1 byte from L2 (40 MB)",       12e-12,     "~0.5 pJ/bit array + ~10-20 mm round trip"),
        ("1 byte from HBM2e (interface)",40e-12,     "5 pJ/bit access (Dally 2023)"),
        ("1 byte from HBM2e (all-in)",   60e-12,     "+ ~25 mm on-die round trip"),
        ("1 byte from HBM2e (far corner)",78e-12,    "+ 48 mm round trip (Dally's 28 mm die)"),
    ]
    print(f"{'item':34s} {'pJ':>8s} {'int8-adds':>10s}   note")
    for name, e, note in rows:
        print(f"{name:34s} {e*1e12:8.3f} {ae(e):10.0f}   {note}")

def table_balances():
    print("\n== Table 2: machine balances at HBM ==")
    print(f"{'GPU':16s} {'peak TF/s':>9s} {'HBM TB/s':>8s} {'beta_t FLOP/B':>13s} {'beta_E FLOP/B':>13s} {'d_min':>6s} {'T_min(inf d)=2beta':>18s}")
    for name, g in GPUS.items():
        bt, bE = balance(g)
        print(f"{name:16s} {g['peak']/1e12:9.0f} {g['bw']/1e12:8.2f} {bt:13.0f} {bE:13.0f} {d_min(g):6.0f} {B_W_DEFAULT*bt/6:18.0f}")

def table_tmin():
    print("\n== Table 3: micro-batch tokens per weight visit needed for HBM intensity >= balance ==")
    ds = [512, 768, 1024, 2048, 4096, 8192, 12288]
    names = list(GPUS.keys())
    print(f"{'d':>6s} " + " ".join(f"{n:>16s}" for n in names))
    for d in ds:
        vals = []
        for n in names:
            v = T_min(d, GPUS[n])
            vals.append(f"{v:16.0f}" if math.isfinite(v) else f"{'never':>16s}")
        print(f"{d:6d} " + " ".join(vals))

def table_elementwise_tax():
    print("\n== Table 4: element-wise (memory-bound) share of step time, a*beta/(72 d), and HBM energy floor a*e_B/(72 d e_F) ==")
    for name in ["A100-80GB", "H100-SXM"]:
        g = GPUS[name]; bt, _ = balance(g)
        print(name)
        for d in [768, 1024, 2048, 4096, 12288]:
            tfrac = A_DEFAULT * bt / (72 * d)
            efloor = A_DEFAULT * g["e_hbm"] / (72 * d * g["e_flop"])
            print(f"   d={d:6d}: element-wise time / GEMM time = {tfrac:5.2f}   HBM-activation energy / arithmetic energy = {efloor*100:5.2f}%")

def worked_example():
    print("\n== Table 5: 7B-class model (d=4096, L=32, s=2048) on A100-80GB: per-token cost by regime ==")
    g = GPUS["A100-80GB"]; d, L, s = 4096, 32, 2048
    P = 12 * d * d * L + 2 * 50000 * d              # + embeddings (rough)
    print(f"   params ~ {P/1e9:.2f} B; saved activations per 2048-token sequence (34 d s L) = {A_STORE*d*s*L/1e9:.1f} GB;"
          f" Adam mixed-precision state (16 B/param) = {16*P/1e9:.0f} GB")
    scen = [("online, 1 token/step, 1 GPU",           1,     1,   1, "hbm"),
            ("1 sequence (2048 tok)/step, 1 GPU",     2048,  1,   1, "hbm"),
            ("1 seq/GPU, 8 GPUs NVLink (DP)",         2048,  1,   8, "nvlink"),
            ("1 seq/GPU, 64 GPUs over IB (DP)",       2048,  1,  64, "ib"),
            ("8 seq/GPU (grad-accum), 64 GPUs IB",    2048,  8,  64, "ib"),
            ("64 seq/GPU (grad-accum), 64 GPUs IB",   2048, 64,  64, "ib")]
    print(f"{'scenario':40s} {'tok/step/GPU':>12s} {'ms/token':>9s} {'mJ/token':>9s} {'arith%':>7s} {'HBM%':>6s} {'static%':>8s} {'I FLOP/B':>9s} {'slowdown':>9s}")
    for name, Tm, nm, N, fab in scen:
        r = layer_step(d, Tm, g, s=s)
        # per micro-batch, all layers
        t_layers = r["t"] * L * nm; E_arith = r["E_arith"] * L * nm; E_hbm = (r["E_hbm_w"] + r["E_hbm_a"]) * L * nm
        # per-step terms: optimizer (sharded over N), all-reduce over fabric (not overlapped)
        B_opt = B_OPT_DEFAULT * P / N
        t_opt = B_opt / g["bw"]; E_hbm += B_opt * g["e_hbm"]
        t_ar = 0.0
        if N > 1:
            t_ar = B_AR_DEFAULT * P * (N - 1) / N / g[fab]
        t = t_layers + t_opt + t_ar
        E_static = g["p_static"] * t
        E = E_arith + E_hbm + E_static
        T_step = Tm * nm
        F_step = r["F"] * L * nm
        t_ideal = F_step / g["peak"]
        I_step = F_step / ((r["B_W"] + r["B_act"]) * L * nm + B_opt)
        print(f"{name:40s} {T_step:12d} {1e3*t/T_step:9.3f} {1e3*E/T_step:9.3f} {100*E_arith/E:7.1f} {100*E_hbm/E:6.1f} {100*E_static/E:8.1f} {I_step:9.0f} {t/t_ideal:9.2f}")

def table_squeeze():
    print("\n== Table 6: the batch-size squeeze: time-optimal global batch B_opt = sqrt(B_crit * N * T_fixed) ==")
    g = GPUS["A100-80GB"]; d, L = 4096, 32; P = 12*d*d*L
    Bcrit = 1e6
    for fab, label in [("nvlink", "NVLink 300 GB/s"), ("ib", "IB HDR 25 GB/s")]:
        for N in [8, 64, 256, 1024]:
            t_fixed = B_AR_DEFAULT * P * (N-1)/N / g[fab] + B_OPT_DEFAULT * P / N / g["bw"]
            T_fixed = t_fixed * g["peak"] / (6 * P)          # fixed per-step cost in token-equivalents
            Bopt = math.sqrt(Bcrit * N * T_fixed)
            print(f"   {label:16s} N={N:5d}: T_fixed={T_fixed:7.0f} tok-eq/GPU  B_opt={Bopt/1e6:5.2f} M tokens  "
                  f"T_dev,opt={Bopt/N:7.0f} tok/GPU  (comm parity at {T_fixed:.0f})")

# ----------------------------------------------------------------------------------------------
# 5. Figure
# ----------------------------------------------------------------------------------------------
def figure(path="fig_backprop_memory_wall.png", stacked=False):
    """stacked=False: one wide row of three panels (screen); stacked=True: three rows (print)."""
    if stacked:
        fig, axes = plt.subplots(3, 1, figsize=(7.4, 9.8))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    g = GPUS["A100-80GB"]
    T = np.logspace(0, 5, 300)

    # (a) HBM operational intensity of a training step vs tokens per weight visit
    ax = axes[0]
    for d, c in [(768, "#b03a2e"), (2048, "#d68910"), (4096, "#1f618d"), (12288, "#117a65")]:
        I = np.array([layer_step(d, t, g)["I"] for t in T])
        ax.plot(T, I, color=c, lw=2, label=f"d = {d}")
    for name, ls in [("A100-80GB", "--"), ("H100-SXM", ":")]:
        bt, _ = balance(GPUS[name]); ax.axhline(bt, color="k", ls=ls, lw=1.2)
        ax.text(1.2, bt*1.12, f"{name}: β = {bt:.0f} FLOP/B", fontsize=8.5)
    _, bE = balance(g); ax.axhline(bE, color="gray", ls="-.", lw=1)
    ax.text(1.2, bE*0.62, f"β_E(A100) = {bE:.0f}", fontsize=8.5, color="gray")
    ax.plot(T, T/2, color="0.6", lw=0.8); ax.text(4.0, 0.9, "I ≈ T/2", fontsize=8, color="0.4", rotation=39)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 1e5); ax.set_ylim(0.3, 4000)
    ax.set_xlabel("tokens per weight visit  T  (micro-batch × sequence length, per device)")
    ax.set_ylabel("HBM intensity, fwd+bwd  [FLOP / byte]" if stacked else "HBM operational intensity of fwd+bwd  [FLOP / byte]")
    ax.set_title("(a) Training-step intensity vs. batch (A100 traffic model)", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.3, which="both")

    # (b) required tokens per weight visit vs d
    ax = axes[1]
    ds = np.logspace(math.log10(300), math.log10(30000), 400)
    for name, c in [("A100-40GB", "#5dade2"), ("A100-80GB", "#1f618d"), ("H100-SXM", "#b03a2e"), ("B200 (approx.)", "#7d3c98")]:
        gg = GPUS[name]; Tm = np.array([T_min(d, gg) for d in ds])
        ax.plot(ds, Tm, color=c, lw=2, label=f"{name} (d_min ≈ {d_min(gg):.0f}, 2β ≈ {2*gg['peak']/gg['bw']:.0f})")
    for x, lab in [(768, "GPT-2 small"), (1600, "GPT-2 xl"), (4096, "7B / 13B"), (12288, "GPT-3 175B")]:
        ax.axvline(x, color="0.75", lw=0.8); ax.text(x*1.04, 108, lab, fontsize=7.5, color="0.35", rotation=90, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(300, 30000); ax.set_ylim(100, 3e4)
    ax.set_xlabel("hidden size d"); ax.set_ylabel("T_min  [tokens per weight visit]" if stacked else "minimum tokens per weight visit  T_min")
    ax.set_title("(b) Batch needed to leave the HBM wall:  T_min = 12·b_w·d·β / (72d − a·β)", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3, which="both")

    # (c) energy breakdown vs T for d = 4096 on A100
    ax = axes[2]
    d = 4096
    parts = {"on-chip dynamic (datapath ≈10%, operand delivery ≈90%)": [], "HBM: weights + grads": [],
             "HBM: activations": [], "static / leakage (∝ wall time)": []}
    keys = list(parts.keys())
    for t in T:
        r = layer_step(d, t, g)
        parts[keys[0]].append(r["E_arith"] / r["E"])
        parts[keys[1]].append(r["E_hbm_w"] / r["E"])
        parts[keys[2]].append(r["E_hbm_a"] / r["E"])
        parts[keys[3]].append(r["E_static"] / r["E"])
    ax.stackplot(T, *[np.array(v) for v in parts.values()], labels=list(parts.keys()),
                 colors=["#1f618d", "#c0392b", "#e59866", "#95a5a6"], alpha=0.9)
    ax.set_xscale("log"); ax.set_xlim(1, 1e5); ax.set_ylim(0, 1)
    ax.set_xlabel("tokens per weight visit  T"); ax.set_ylabel("energy share per token-layer" if stacked else "share of energy per token-layer")
    ax.set_title("(c) Where the joules go, d = 4096 on A100-80GB", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92); ax.grid(alpha=0.3)
    for tt in [T_min(d, g)]:
        ax.axvline(tt, color="k", ls="--", lw=1); ax.text(tt*1.1, 0.45, f"T_min ≈ {tt:.0f}", fontsize=8.5)

    fig.tight_layout(); fig.savefig(path, dpi=170); print(f"\nwrote {path}")

if __name__ == "__main__":
    table_boundaries(); table_balances(); table_tmin(); table_elementwise_tax(); worked_example(); table_squeeze()
    figure()
    figure("fig_backprop_memory_wall_tall.png", stacked=True)
