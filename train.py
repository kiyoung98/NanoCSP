"""
NanoCSP DiffCSP baseline trainer.

Single-file CSP baseline for the MP20 polymorph split, evaluated by cRMSE.
This branch implements the original DiffCSP formulation
(arXiv:2309.04475, jiaor17/DiffCSP @ pl_modules/diffusion.py):

  - CSPNet (DiffCSP-style equivariant message passing) with two heads:
    `pred_l` (per-graph 3x3 lattice noise) and `pred_x` (per-atom 3 score
    on the torus). No species head — CSP keeps the composition fixed.
  - Lattice diffusion: Gaussian DDPM with cosine β-schedule. Network
    predicts ε ~ N(0, I); MSE(pred_l, rand_l).
  - Fractional coords diffusion: Wrapped Normal noise on the 3-torus
    with log-linear σ schedule (σ_begin=0.005, σ_end=0.5, T=1000).
    Network predicts the score; target = d_log_p_wrapped_normal / sqrt(σ_norm).
  - Sampling: predictor-corrector (PC) for fractional coords, ancestral
    DDPM for lattice. Frac coords wrapped to [0, 1) at every step.

The script runs for `--epochs` epochs of training (with per-epoch
val_loss driving ReduceLROnPlateau), then writes one CIF per test
entry to `runs/<run>/test_samples/{idx:05d}.cif`. `evaluate.py` reads
that directory and scores it.

Leaderboard time is the external wall-clock measured by the maintainer
between script invocation and exit. Pick `epochs` so the run fits the
12 h honor-system budget on a single RTX 3090.

Submit by editing only this file (and optionally `requirements.txt`).
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, scatter
from tqdm import tqdm

# ============================================================================
# Hyperparameters
# ============================================================================


@dataclasses.dataclass
class HParams:
    # Data
    data_dir: str = "./data"
    train_file: str = "mp20_ps_train.pt"
    val_file: str = "mp20_ps_val.pt"
    test_file: str = "mp20_ps_test.pt"

    # Model (CSPNet baseline; mirrors diffcsp/conf/model/decoder/cspnet.yaml)
    hidden_dim: int = 512
    num_layers: int = 6
    num_freqs: int = 128            # SinusoidsEmbedding for frac diffs
    time_dim: int = 256             # SinusoidalTimeEmbeddings dim
    max_atom_num: int = 100
    layer_norm: bool = True
    use_inner_product: bool = True  # ip=True in DiffCSP

    # Diffusion (DiffCSP defaults: conf/model/diffusion.yaml + sigma_scheduler/wrapped.yaml)
    num_diffusion_steps: int = 1000
    pos_sigma_min: float = 0.005     # sigma_begin (log-linear σ)
    pos_sigma_max: float = 0.5       # sigma_end
    cell_beta_start: float = 1e-4    # only used if cell_schedule = "linear"
    cell_beta_end: float = 0.02
    cell_schedule: str = "cosine"    # DiffCSP default
    pos_loss_weight: float = 1.0     # cost_coord
    cell_loss_weight: float = 1.0    # cost_lattice
    pc_step_lr: float = 1e-5         # corrector step size base for PC sampler

    # `evaluate.py` prints `hp.integration_steps`; mirror it to the
    # diffusion step count so the same field name keeps working.
    integration_steps: int = 1000

    # Lattice base distribution (kept for evaluate.py contract; unused in
    # DiffCSP because the sampler initialises lattice from N(0, I)).
    mp20_length_log_means: tuple = (1.575442910194397, 1.7017393112182617, 1.9781638383865356)
    mp20_length_log_stds: tuple = (0.24437622725963593, 0.26526379585266113, 0.3535512685775757)

    # Optimizer / training. Method-specific hyperparameters (lr, weight decay,
    # batch size, gradient clip, LR scheduler) match DiffCSP MP-20 official:
    # conf/optim/default.yaml + conf/data/mp_20.yaml. The 12 h honor-system
    # budget covers training (with per-epoch val_loss for ReduceLROnPlateau)
    # plus final 1000-step PC test sampling (heavy: ~75 min). Tune `epochs`
    # so the combined wall-clock fits.
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 3400
    grad_clip: float = 0.5           # gradient_clip_val in default.yaml
    weight_decay: float = 0.0
    seed: int = 42                   # random_seed in train/default.yaml
    # ReduceLROnPlateau (replaces cosine warmup).
    lr_scheduler_factor: float = 0.6
    lr_scheduler_patience: int = 30
    lr_scheduler_min_lr: float = 1e-4

    # Final test sampling. Writes one CIF per test entry to
    # runs/<run>/test_samples/{idx:05d}.cif.
    test_sample_seed: int = 0
    test_sample_batch_size: int = 64

    # Logging / IO
    log_every: int = 50
    out_dir: str = "./runs"
    run_name: str = "diffcsp"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Sanity / dev knobs
    compile: bool = False


# ============================================================================
# Data loading & batching
# ============================================================================


class CrystalDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.records = torch.load(str(path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def collate_pyg(batch):
    num_atoms = torch.tensor([r["atomic_numbers"].numel() for r in batch], dtype=torch.long)
    atom_types = torch.cat([r["atomic_numbers"] for r in batch], dim=0).long()
    frac_coords = torch.cat([r["frac_coords"] for r in batch], dim=0).float()
    lattices = torch.stack([r["lattice"] for r in batch], dim=0).float()
    node2graph = torch.repeat_interleave(
        torch.arange(len(batch), dtype=torch.long), num_atoms
    )
    return {
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "lattices": lattices,
        "num_atoms": num_atoms,
        "node2graph": node2graph,
    }


# ============================================================================
# Model: CSPNet (shared with all NanoCSP baselines)
# ============================================================================


class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=time.device) * -scale)
        emb = time[:, None] * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies: int = 128, n_space: int = 3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.register_buffer(
            "frequencies",
            2 * math.pi * torch.arange(n_frequencies).float(),
            persistent=False,
        )
        self.dim = n_frequencies * 2 * n_space

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :]
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class CSPLayer(nn.Module):
    def __init__(self, hidden_dim: int, dis_emb: SinusoidsEmbedding, ln: bool, ip: bool):
        super().__init__()
        self.dis_emb = dis_emb
        self.ip = ip
        edge_in = hidden_dim * 2 + 9 + dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.ln = ln
        if ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, frac_coords, lattices, edge_index, edge2graph, frac_diff):
        h_in = h
        if self.ln:
            h = self.layer_norm(h_in)
        hi, hj = h[edge_index[0]], h[edge_index[1]]
        if self.ip:
            lat_ip = lattices @ lattices.transpose(-1, -2)
        else:
            lat_ip = lattices
        lat_ip_edges = lat_ip.view(-1, 9)[edge2graph]
        frac_diff_emb = self.dis_emb(frac_diff)
        e = self.edge_mlp(torch.cat([hi, hj, lat_ip_edges, frac_diff_emb], dim=1))
        agg = scatter(e, edge_index[0], dim=0, reduce="mean", dim_size=h.shape[0])
        out = self.node_mlp(torch.cat([h, agg], dim=1))
        return h_in + out


class CSPNetFull(nn.Module):
    """Two-head CSPNet. For the DiffCSP baseline, the two outputs are
    interpreted as (score_pos, eps_cell): the per-atom score on the torus
    and the per-graph Gaussian noise prediction for the lattice.
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self.dis_emb = SinusoidsEmbedding(n_frequencies=hp.num_freqs)
        self.time_emb = SinusoidalTimeEmbeddings(hp.time_dim)
        self.node_embedding = nn.Embedding(hp.max_atom_num, hp.hidden_dim)
        self.atom_latent_emb = nn.Linear(hp.hidden_dim + hp.time_dim, hp.hidden_dim)
        self.layers = nn.ModuleList(
            [CSPLayer(hp.hidden_dim, self.dis_emb, hp.layer_norm, hp.use_inner_product)
             for _ in range(hp.num_layers)]
        )
        if hp.layer_norm:
            self.final_layer_norm = nn.LayerNorm(hp.hidden_dim)
        self.coord_out = nn.Linear(hp.hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hp.hidden_dim, 9, bias=False)

    def gen_edges(self, num_atoms, frac_coords):
        blocks = [torch.ones(int(n), int(n), device=num_atoms.device) for n in num_atoms]
        adj = torch.block_diag(*blocks)
        edge_index, _ = dense_to_sparse(adj)
        frac_diff = (frac_coords[edge_index[1]] - frac_coords[edge_index[0]]) % 1.0
        return edge_index, frac_diff

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):
        edge_index, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edge_index[0]]
        h = self.node_embedding((atom_types - 1).clamp_min(0))
        t_emb = self.time_emb(t)
        t_per_atom = t_emb[node2graph]
        h = self.atom_latent_emb(torch.cat([h, t_per_atom], dim=1))
        for layer in self.layers:
            h = layer(h, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        if self.hp.layer_norm:
            h = self.final_layer_norm(h)
        pos_v = self.coord_out(h)
        graph_h = scatter(h, node2graph, dim=0, reduce="mean")
        cell_v = self.lattice_out(graph_h).view(-1, 3, 3)
        if self.hp.use_inner_product:
            cell_v = torch.einsum("bij,bjk->bik", cell_v, lattices)
        return pos_v, cell_v


# ============================================================================
# Diffusion schedules (port of DiffCSP/pl_modules/diff_utils.py).
# ============================================================================


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def p_wrapped_normal(x: torch.Tensor, sigma: torch.Tensor, N: int = 10, T: float = 1.0) -> torch.Tensor:
    p_ = 0
    for i in range(-N, N + 1):
        p_ = p_ + torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
    return p_


def d_log_p_wrapped_normal(x: torch.Tensor, sigma: torch.Tensor, N: int = 10, T: float = 1.0) -> torch.Tensor:
    p_ = 0
    num = 0
    for i in range(-N, N + 1):
        e = torch.exp(-(x + T * i) ** 2 / 2 / sigma ** 2)
        num = num + (x + T * i) / sigma ** 2 * e
        p_ = p_ + e
    return num / p_


def _sigma_norm(sigma: torch.Tensor, T: float = 1.0, sn: int = 10000) -> torch.Tensor:
    """Per-σ Monte-Carlo normalization: E[(d_log_p_wrapped_normal)²]."""
    sigmas = sigma[None, :].expand(sn, -1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
    return (normal_ ** 2).mean(dim=0)


class DiffusionSchedules(nn.Module):
    """Combines DiffCSP's BetaScheduler (lattice DDPM) and SigmaScheduler
    (frac coords Wrapped-Normal score). Stored as buffers so they move
    with `.to(device)`.

    Convention follows DiffCSP: t indexes 1..T (0 reserved as the
    boundary). All buffers are length T+1 with t=0 entries
    pre-populated (β=0, α_cumprod=1, σ=0, σ_norm=1).
    """

    def __init__(self, timesteps: int, sigma_begin: float, sigma_end: float,
                 cell_schedule: str = "cosine", beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.timesteps = timesteps

        if cell_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif cell_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"unknown cell_schedule={cell_schedule}")

        betas = torch.cat([torch.zeros([1]), betas], dim=0)        # (T+1,)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas_post = torch.zeros_like(betas)
        sigmas_post[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        sigmas_post = torch.sqrt(sigmas_post)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("post_sigmas", sigmas_post)

        # Wrapped-normal log-linear σ schedule.
        sigmas_x = torch.exp(torch.linspace(math.log(sigma_begin), math.log(sigma_end), timesteps))
        sigmas_norm = _sigma_norm(sigmas_x)
        self.register_buffer("sigmas_x", torch.cat([torch.zeros([1]), sigmas_x], dim=0))
        self.register_buffer("sigmas_norm", torch.cat([torch.ones([1]), sigmas_norm], dim=0))

    def uniform_sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.timesteps + 1, (batch_size,), device=device)


# ============================================================================
# Lattice base (kept for evaluate.py contract; sample_structures ignores it).
# ============================================================================


def _cellpar_to_matrix(lengths: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    a, b, c = lengths
    al, be, ga = np.deg2rad(angles_deg)
    cos_al, cos_be, cos_ga = math.cos(al), math.cos(be), math.cos(ga)
    sin_ga = math.sin(ga)
    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * cos_ga, b * sin_ga, 0.0
    cx = c * cos_be
    cy = c * (cos_al - cos_be * cos_ga) / sin_ga
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([[ax, ay, az], [bx, by, bz], [cx, cy, cz]], dtype=np.float64)


class InformedLatticeDistribution:
    """Kept identical to the OMatG baseline for API compatibility with
    evaluate.py. DiffCSP itself starts the lattice from a unit Gaussian
    during sampling (see `sample_structures` below); this class is only
    instantiated and passed through.
    """

    def __init__(self, log_means: tuple, log_stds: tuple):
        self.log_means = torch.tensor(log_means, dtype=torch.float32)
        self.log_stds = torch.tensor(log_stds, dtype=torch.float32)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        normal = torch.randn(batch_size, 3) * self.log_stds + self.log_means
        lengths = torch.exp(normal).sort(dim=-1).values.numpy()
        angles = (np.random.rand(batch_size, 3) * 60.0 + 60.0)
        cells = np.stack([_cellpar_to_matrix(l, a) for l, a in zip(lengths, angles)], axis=0)
        return torch.from_numpy(cells).float().to(device)


# ============================================================================
# Loss & sampling (DiffCSP).
# ============================================================================


def wrap_unit(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, 1.0)


def diffusion_loss(
    model: CSPNetFull,
    batch: dict,
    schedules: DiffusionSchedules,
    hp: HParams,
):
    """Single training step loss (port of CSPDiffusion.forward)."""
    device = batch["atom_types"].device
    num_atoms = batch["num_atoms"]
    node2graph = batch["node2graph"]
    B = num_atoms.shape[0]

    times = schedules.uniform_sample_t(B, device)
    alphas_cumprod = schedules.alphas_cumprod[times]
    c0 = torch.sqrt(alphas_cumprod)
    c1 = torch.sqrt(1.0 - alphas_cumprod)

    sigmas_x = schedules.sigmas_x[times]
    sigmas_norm = schedules.sigmas_norm[times]
    sigmas_per_atom = sigmas_x.repeat_interleave(num_atoms)[:, None]
    sigmas_norm_per_atom = sigmas_norm.repeat_interleave(num_atoms)[:, None]

    lattices = batch["lattices"]
    frac_coords = batch["frac_coords"]

    rand_l = torch.randn_like(lattices)
    rand_x = torch.randn_like(frac_coords)

    input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
    input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.0

    # Network input time is the integer timestep cast to float (matches
    # DiffCSP, which feeds `times` directly into the sinusoidal embedding).
    pred_x, pred_l = model(
        times.float(),
        batch["atom_types"],
        input_frac_coords,
        input_lattice,
        num_atoms,
        node2graph,
    )

    tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

    loss_lattice = F.mse_loss(pred_l, rand_l)
    loss_coord = F.mse_loss(pred_x, tar_x)
    loss = hp.pos_loss_weight * loss_coord + hp.cell_loss_weight * loss_lattice

    return loss, {
        "loss": loss.detach().item(),
        "pos_loss": loss_coord.detach().item(),
        "cell_loss": loss_lattice.detach().item(),
    }


@torch.no_grad()
def sample_structures(
    model: CSPNetFull,
    atom_types_batch: torch.Tensor,
    num_atoms_batch: torch.Tensor,
    base_lattice: InformedLatticeDistribution,  # unused; for API parity
    hp: HParams,
    device: torch.device,
):
    """Predictor-corrector sampler (port of CSPDiffusion.sample)."""
    model.eval()
    schedules = _maybe_get_schedules(model, hp, device)
    B = num_atoms_batch.shape[0]
    n_total = int(num_atoms_batch.sum())
    node2graph = torch.repeat_interleave(
        torch.arange(B, device=device), num_atoms_batch
    )

    l_T = torch.randn(B, 3, 3, device=device)
    x_T = torch.rand(n_total, 3, device=device)

    x_t = x_T % 1.0
    l_t = l_T

    step_lr = hp.pc_step_lr
    sigma_begin = hp.pos_sigma_min

    for t in range(schedules.timesteps, 0, -1):
        times = torch.full((B,), float(t), device=device)
        alphas = schedules.alphas[t]
        alphas_cumprod = schedules.alphas_cumprod[t]
        post_sigma = schedules.post_sigmas[t]

        sigma_x = schedules.sigmas_x[t]
        sigma_norm = schedules.sigmas_norm[t]

        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1.0 - alphas) / torch.sqrt(1.0 - alphas_cumprod)

        # === Corrector (Langevin) on positions ===
        rand_x = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
        step_size = step_lr * (sigma_x / sigma_begin) ** 2
        std_x = torch.sqrt(2.0 * step_size)

        pred_x, _ = model(times, atom_types_batch, x_t, l_t, num_atoms_batch, node2graph)
        pred_x = pred_x * torch.sqrt(sigma_norm)

        x_t_half = x_t - step_size * pred_x + std_x * rand_x
        l_t_half = l_t  # corrector doesn't touch the lattice

        # === Predictor: ancestral DDPM step on lattice + Euler-Maruyama on positions ===
        rand_l = torch.randn_like(l_t) if t > 1 else torch.zeros_like(l_t)
        rand_x = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)

        adjacent_sigma_x = schedules.sigmas_x[t - 1]
        step_size_p = sigma_x ** 2 - adjacent_sigma_x ** 2
        std_xp = torch.sqrt(
            (adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2 + 1e-12)
        )

        pred_x, pred_l = model(times, atom_types_batch, x_t_half, l_t_half, num_atoms_batch, node2graph)
        pred_x = pred_x * torch.sqrt(sigma_norm)

        x_t = x_t_half - step_size_p * pred_x + std_xp * rand_x
        l_t = c0 * (l_t_half - c1 * pred_l) + post_sigma * rand_l
        x_t = x_t % 1.0

    return x_t, l_t


# Cache the schedules on the model so we don't rebuild them per sample call.
# Use object.__setattr__ to bypass nn.Module.__setattr__'s automatic submodule
# registration — otherwise DiffusionSchedules (an nn.Module) gets pulled into
# model.state_dict() and breaks evaluate.py's load_state_dict.
def _maybe_get_schedules(model: CSPNetFull, hp: HParams, device: torch.device) -> DiffusionSchedules:
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    sch = getattr(inner, "_diffcsp_schedules", None)
    if sch is None or sch.timesteps != hp.num_diffusion_steps:
        sch = DiffusionSchedules(
            hp.num_diffusion_steps, hp.pos_sigma_min, hp.pos_sigma_max,
            cell_schedule=hp.cell_schedule,
            beta_start=hp.cell_beta_start, beta_end=hp.cell_beta_end,
        ).to(device)
        object.__setattr__(inner, "_diffcsp_schedules", sch)
    return sch


# ============================================================================
# Main training loop.
# ============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_test_samples(
    model: CSPNetFull,
    hp: HParams,
    base_lattice: InformedLatticeDistribution,
    device: torch.device,
    out_dir: Path,
    print0=print,
) -> Path:
    """Sample one structure per test composition and write CIFs.

    Reads `mp20_ps_test.pt` and uses ONLY `atomic_numbers` from each test
    record — ground-truth frac_coords / lattice are never touched here
    (leakage prevention; those are read by evaluate.py for matching).

    Writes `{idx:05d}.cif` files (indexed by the order of the test split)
    into `out_dir / "test_samples"`. Returns that path.
    """
    from pymatgen.core import Lattice, Structure  # noqa: PLC0415

    test_records = torch.load(str(Path(hp.data_dir) / hp.test_file))
    n_test = len(test_records)
    samples_dir = out_dir / "test_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    print0(f"[test] sampling {n_test} structures → {samples_dir}")

    # Deterministic RNG so re-running the same checkpoint yields the same CIFs.
    torch.manual_seed(hp.test_sample_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(hp.test_sample_seed)

    species_list = [torch.as_tensor(r["atomic_numbers"], dtype=torch.long)
                    for r in test_records]
    num_atoms_all = torch.tensor([t.numel() for t in species_list], dtype=torch.long)
    species_flat = torch.cat(species_list, dim=0)

    model.eval()
    bs = hp.test_sample_batch_size
    cursor_atom = 0
    cif_idx = 0
    for i in range(0, n_test, bs):
        nb = num_atoms_all[i:i + bs].to(device)
        n_atoms_chunk = int(nb.sum())
        ab = species_flat[cursor_atom:cursor_atom + n_atoms_chunk].to(device)
        with torch.no_grad():
            pos, cell = sample_structures(model, ab, nb, base_lattice, hp, device)
        pos_cpu = pos.cpu().numpy()
        cell_cpu = cell.cpu().numpy()
        atom_offset = 0
        for j, n_j in enumerate(nb.tolist()):
            sp = species_list[i + j].numpy()
            f = pos_cpu[atom_offset:atom_offset + n_j]
            l = cell_cpu[j]
            struct = Structure(Lattice(l), sp, f, coords_are_cartesian=False)
            struct.to(filename=str(samples_dir / f"{cif_idx:05d}.cif"), fmt="cif")
            atom_offset += n_j
            cif_idx += 1
        cursor_atom += n_atoms_chunk
    print0(f"[test] wrote {cif_idx} CIFs")
    return samples_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    hp = HParams()
    for f in dataclasses.fields(hp):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(hp, f.name, v)

    set_seed(hp.seed)
    device = torch.device(hp.device)
    print0 = print

    print0(f"[setup] device={device}  hp={hp}")
    out_dir = Path(hp.out_dir) / hp.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CrystalDataset(Path(hp.data_dir) / hp.train_file)
    val_ds = CrystalDataset(Path(hp.data_dir) / hp.val_file)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=collate_pyg,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print0(f"[data] train={len(train_ds)}  val={len(val_ds)}  steps/epoch≈{len(train_loader)}")

    model = CSPNetFull(hp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print0(f"[model] CSPNet params={n_params/1e6:.2f}M")

    schedules = DiffusionSchedules(
        hp.num_diffusion_steps, hp.pos_sigma_min, hp.pos_sigma_max,
        cell_schedule=hp.cell_schedule,
        beta_start=hp.cell_beta_start, beta_end=hp.cell_beta_end,
    ).to(device)
    # Cache on model so sample_structures finds it. Use object.__setattr__
    # to bypass nn.Module's auto-submodule registration so the schedule
    # buffers don't leak into model.state_dict().
    object.__setattr__(model, "_diffcsp_schedules", schedules)

    if hp.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, dynamic=True)
            print0("[model] torch.compile enabled")
        except Exception as e:
            print0(f"[model] torch.compile failed, continuing eager: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    # Official DiffCSP LR schedule: ReduceLROnPlateau on val_loss.
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=hp.lr_scheduler_factor,
        patience=hp.lr_scheduler_patience,
        min_lr=hp.lr_scheduler_min_lr,
    )

    base_lattice = InformedLatticeDistribution(
        hp.mp20_length_log_means, hp.mp20_length_log_stds
    )

    total_steps = hp.epochs * len(train_loader)
    print0(f"[sched] total_steps={total_steps}  T={hp.num_diffusion_steps}  "
           f"lr_plateau(factor={hp.lr_scheduler_factor}, patience={hp.lr_scheduler_patience}, min_lr={hp.lr_scheduler_min_lr})")

    step = 0
    smooth_loss = 0.0
    ema_beta = 0.9

    # val_loader drives the per-epoch val_loss pass that feeds
    # ReduceLROnPlateau (DiffCSP's LR scheduler).
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=hp.batch_size, shuffle=False,
        collate_fn=collate_pyg, drop_last=False, num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print0(f"[val] val_loss every epoch on {len(val_ds)} records (drives ReduceLROnPlateau)")
    print0(f"[train] starting; out_dir={out_dir}")

    for epoch in range(1, hp.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = diffusion_loss(model, batch, schedules, hp)
            loss.backward()
            # Official DiffCSP uses gradient_clip_algorithm='value'.
            torch.nn.utils.clip_grad_value_(model.parameters(), hp.grad_clip)
            optimizer.step()

            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * metrics["loss"]
            debiased = smooth_loss / (1 - ema_beta ** (step + 1))

            if step % hp.log_every == 0:
                print0(
                    f"epoch {epoch:5d}  step {step:7d}  loss {debiased:.5f}  "
                    f"pos {metrics['pos_loss']:.5f}  cell {metrics['cell_loss']:.5f}"
                )
            step += 1

        # Per-epoch val_loss pass — drives ReduceLROnPlateau. Part of
        # DiffCSP's training loop (the official code steps the scheduler
        # every epoch via Lightning's val_check_interval=1).
        model.eval()
        with torch.no_grad():
            losses = []
            for v_batch in val_loader:
                v_batch = {k: v.to(device, non_blocking=True) for k, v in v_batch.items()}
                v_loss, _ = diffusion_loss(model, v_batch, schedules, hp)
                losses.append(v_loss.item())
            val_loss = float(np.mean(losses))
        lr_sched.step(val_loss)
        if epoch % hp.log_every == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print0(f"[val] epoch {epoch}  val_loss={val_loss:.5f}  lr={current_lr:.2e}")

    # Final test sampling: write one CIF per test entry.
    samples_dir = _write_test_samples(
        model=model,
        hp=hp,
        base_lattice=base_lattice,
        device=device,
        out_dir=out_dir,
        print0=print0,
    )
    print0(f"[done] test samples → {samples_dir}")
    print0(f"Run: python evaluate.py --samples_dir {samples_dir}")


if __name__ == "__main__":
    main()
