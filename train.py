"""
NanoCSP baseline trainer.

Single-file CSP baseline for the MP20 polymorph split, evaluated by cRMSE.
Architecture is a port of OMatG's `csp_linear_ode_mp_20.yaml`:

  - CSPNetFull (DiffCSP-style equivariant message passing) with a velocity
    head (lattice + frac coords). No species head — CSP keeps the
    composition fixed.
  - Linear ODE stochastic interpolant (a.k.a. flow matching with no noise):
      x_t = (1 - t) * x_0 + t * x_1,  v_target = x_1 - x_0
      loss = MSE(v_pred, v_target)
    Position uses the periodic linear interpolant (fractional coords on the
    torus); lattice uses the plain linear interpolant.
  - Sampling: Euler ODE integration with velocity annealing, fractional
    coords wrapped to [0, 1) at every step.

The script runs for `--epochs` epochs of training, then writes one CIF
per test entry to `runs/<run>/test_samples/{idx:05d}.cif` (indexed by
the order of `mp20_ps_test.pt`). `evaluate.py` reads that directory and
scores it.

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
    test_file: str = "mp20_ps_test.pt"

    # Model (CSPNetFull baseline; mirrors csp_linear_ode_mp_20.yaml)
    hidden_dim: int = 512
    num_layers: int = 6
    num_freqs: int = 128            # SinusoidsEmbedding for frac diffs
    time_dim: int = 256             # SinusoidalTimeEmbeddings dim
    max_atom_num: int = 100
    layer_norm: bool = True
    use_inner_product: bool = True  # ip=True in OMatG yaml

    # Flow matching (linear ODE)
    pos_velocity_annealing: float = 10.182659004291072
    cell_velocity_annealing: float = 1.824475401606087
    pos_loss_weight: float = 0.9994149341846618
    cell_loss_weight: float = 0.0005850658153382233
    correct_com_motion: bool = True  # for fractional coords; FlowMM-style
    integration_steps: int = 210
    small_time: float = 1e-3
    big_time: float = 1.0 - 1e-3

    # MP20-PS lattice base distribution (LogNormal lengths + Uniform[60,120] angles)
    # Numbers from OMatG omg/sampler/cell_distributions.py for "mp_20".
    mp20_length_log_means: tuple = (1.575442910194397, 1.7017393112182617, 1.9781638383865356)
    mp20_length_log_stds: tuple = (0.24437622725963593, 0.26526379585266113, 0.3535512685775757)

    # Optimizer / training. Calibration on a single RTX 3090 (24 GB) with
    # hidden_dim=512, batch_size=256 runs at ~118 ms/step (106 steps/epoch
    # ≈ 12.5 s/epoch). epochs=3400 → ~11.8 h training + ~6 min final test
    # sampling fits the 12 h honor-system budget.
    batch_size: int = 256
    lr: float = 6.689636445843722e-4
    epochs: int = 3400
    grad_clip: float = 0.5
    warmup_steps: int = 200
    weight_decay: float = 0.0
    seed: int = 0

    # Final test sampling. Writes one CIF per test entry to
    # runs/<run>/test_samples/{idx:05d}.cif.
    test_sample_seed: int = 0
    test_sample_batch_size: int = 64

    # Logging / IO
    log_every: int = 50              # steps
    out_dir: str = "./runs"
    run_name: str = "baseline"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Sanity / dev knobs
    compile: bool = False            # torch.compile (skip on torch<2.1)


# ============================================================================
# Data loading & batching
# ============================================================================


class CrystalDataset(torch.utils.data.Dataset):
    """Backed by the .pt list produced by prepare_data.py."""

    def __init__(self, path: str | Path):
        self.records = torch.load(str(path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def collate_pyg(batch: list[dict]) -> dict:
    """Concatenate variable-size structures into a single PyG-style graph batch."""
    num_atoms = torch.tensor([r["atomic_numbers"].numel() for r in batch], dtype=torch.long)
    atom_types = torch.cat([r["atomic_numbers"] for r in batch], dim=0).long()
    frac_coords = torch.cat([r["frac_coords"] for r in batch], dim=0).float()
    lattices = torch.stack([r["lattice"] for r in batch], dim=0).float()  # (B, 3, 3)
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
# Model: CSPNetFull (ported from OMatG diffcsp_copies.py + cspnet_full.py)
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
    """Frequency embedding for frac-diff vectors (DiffCSP convention)."""

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
    """Single message-passing block. Equivariant under translations along
    the lattice basis (frac_diff is taken modulo 1).
    """

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

    def forward(
        self,
        h: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        edge_index: torch.Tensor,
        edge2graph: torch.Tensor,
        frac_diff: torch.Tensor,
    ) -> torch.Tensor:
        h_in = h
        if self.ln:
            h = self.layer_norm(h_in)
        # Edge features: sender, receiver, lattice IP, frac diff embedding.
        hi, hj = h[edge_index[0]], h[edge_index[1]]
        if self.ip:
            lat_ip = lattices @ lattices.transpose(-1, -2)
        else:
            lat_ip = lattices
        lat_ip_edges = lat_ip.view(-1, 9)[edge2graph]
        frac_diff_emb = self.dis_emb(frac_diff)
        e = self.edge_mlp(torch.cat([hi, hj, lat_ip_edges, frac_diff_emb], dim=1))
        # Aggregate edges back to nodes.
        agg = scatter(e, edge_index[0], dim=0, reduce="mean", dim_size=h.shape[0])
        out = self.node_mlp(torch.cat([h, agg], dim=1))
        return h_in + out


class CSPNetFull(nn.Module):
    """Velocity-field GNN. Predicts `pos_v` (per-atom, 3) and `cell_v`
    (per-graph, 3x3). Output is in the same fractional / lattice space as
    the inputs, so it can be fed directly into the linear interpolant
    targets.
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

    def gen_edges(
        self,
        num_atoms: torch.Tensor,
        frac_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fully-connected graph within each structure (edge_style='fc')."""
        blocks = [torch.ones(int(n), int(n), device=num_atoms.device) for n in num_atoms]
        adj = torch.block_diag(*blocks)
        edge_index, _ = dense_to_sparse(adj)
        # Frac diff wrapped to [0, 1) (DiffCSP convention).
        frac_diff = (frac_coords[edge_index[1]] - frac_coords[edge_index[0]]) % 1.0
        return edge_index, frac_diff

    def forward(
        self,
        t: torch.Tensor,        # (B,) in [0, 1]
        atom_types: torch.Tensor,
        frac_coords: torch.Tensor,
        lattices: torch.Tensor,
        num_atoms: torch.Tensor,
        node2graph: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pos_v, cell_v): shapes (sum_atoms, 3) and (B, 3, 3)."""
        edge_index, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edge_index[0]]

        # OMatG node embedding indexes from atomic_number - 1 (species_shift=1).
        h = self.node_embedding((atom_types - 1).clamp_min(0))
        t_emb = self.time_emb(t)                              # (B, time_dim)
        t_per_atom = t_emb[node2graph]                        # (sum_atoms, time_dim)
        h = self.atom_latent_emb(torch.cat([h, t_per_atom], dim=1))

        for layer in self.layers:
            h = layer(h, frac_coords, lattices, edge_index, edge2graph, frac_diff)

        if self.hp.layer_norm:
            h = self.final_layer_norm(h)

        pos_v = self.coord_out(h)                             # (sum_atoms, 3)
        graph_h = scatter(h, node2graph, dim=0, reduce="mean")
        cell_v = self.lattice_out(graph_h).view(-1, 3, 3)     # (B, 3, 3)
        if self.hp.use_inner_product:
            # OMatG multiplies the lattice output by the current lattice
            # (matches DiffCSP); keeps the velocity in cell-relative space.
            cell_v = torch.einsum("bij,bjk->bik", cell_v, lattices)
        return pos_v, cell_v


# ============================================================================
# Linear ODE flow matching (no score, no noise).
# ============================================================================


def wrap_unit(x: torch.Tensor) -> torch.Tensor:
    """Wrap fractional coords back to [0, 1)."""
    return torch.remainder(x, 1.0)


def shortest_frac_separation(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """Return x_1 - x_0 mod 1 in [-0.5, 0.5) (closest periodic image)."""
    return torch.remainder((x_1 - x_0) + 0.5, 1.0) - 0.5


def interpolate_frac(t_atom: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Periodic linear interpolant on the torus.

    Returns (x_t in [0,1), v_target = x_1' - x_0') where x_1' is the closest
    image of x_1 to x_0 (so v_target lies in [-0.5, 0.5)^3).
    """
    x_0 = wrap_unit(x_0)
    sep = shortest_frac_separation(x_0, x_1)        # in [-0.5, 0.5)
    x_1_unwrapped = x_0 + sep
    x_t = (1.0 - t_atom) * x_0 + t_atom * x_1_unwrapped
    return wrap_unit(x_t), sep                        # v_target = sep


def interpolate_cell(t_graph: torch.Tensor, c_0: torch.Tensor, c_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain linear interpolant on lattice matrices."""
    c_t = (1.0 - t_graph) * c_0 + t_graph * c_1
    return c_t, c_1 - c_0


# ============================================================================
# Base distributions for x_0 (uniform on torus + informed lattice).
# ============================================================================


def sample_pos_base(num_atoms: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.rand(int(num_atoms.sum()), 3, device=device)


def _cellpar_to_matrix(lengths: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Convert (a, b, c, α, β, γ) → 3×3 lattice matrix (rows = basis vectors).

    Standard crystallographic convention; matches ASE's cellpar_to_cell up to
    floating-point roundoff for our inputs.
    """
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
    """LogNormal lengths + uniform angles in [60°, 120°] (FlowMM/OMatG)."""

    def __init__(self, log_means: tuple, log_stds: tuple):
        self.log_means = torch.tensor(log_means, dtype=torch.float32)
        self.log_stds = torch.tensor(log_stds, dtype=torch.float32)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # LogNormal: exp(N(μ, σ)). We sort lengths (a ≤ b ≤ c) per OMatG's
        # parameter convention.
        normal = torch.randn(batch_size, 3) * self.log_stds + self.log_means
        lengths = torch.exp(normal).sort(dim=-1).values.numpy()
        angles = (np.random.rand(batch_size, 3) * 60.0 + 60.0)
        cells = np.stack([_cellpar_to_matrix(l, a) for l, a in zip(lengths, angles)], axis=0)
        return torch.from_numpy(cells).float().to(device)


# ============================================================================
# Loss & sampling.
# ============================================================================


def flow_matching_loss(
    model: CSPNetFull,
    batch: dict,
    base_lattice: InformedLatticeDistribution,
    hp: HParams,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single training step loss for linear-ODE flow matching."""
    device = batch["atom_types"].device
    num_atoms = batch["num_atoms"]
    node2graph = batch["node2graph"]
    B = num_atoms.shape[0]

    # Random times (clamped to [small_time, big_time] like OMatG).
    t = torch.rand(B, device=device) * (hp.big_time - hp.small_time) + hp.small_time
    t_graph = t.view(B, 1, 1)
    t_atom = t[node2graph].unsqueeze(-1)  # (sum_atoms, 1)

    # Base samples.
    x_0_pos = sample_pos_base(num_atoms, device)
    x_0_cell = base_lattice.sample(B, device)

    # Targets.
    x_t_pos, v_target_pos = interpolate_frac(t_atom, x_0_pos, batch["frac_coords"])
    x_t_cell, v_target_cell = interpolate_cell(t_graph, x_0_cell, batch["lattices"])

    # CoM motion correction on positions (FlowMM-style): subtract per-graph
    # mean of v_target_pos before computing the loss, since the
    # translationally invariant model can't predict it anyway.
    if hp.correct_com_motion:
        mean_v = scatter(v_target_pos, node2graph, dim=0, reduce="mean")[node2graph]
        v_target_pos = v_target_pos - mean_v

    pos_v_pred, cell_v_pred = model(
        t,
        batch["atom_types"],
        x_t_pos,
        x_t_cell,
        num_atoms,
        node2graph,
    )

    pos_loss = F.mse_loss(pos_v_pred, v_target_pos)
    cell_loss = F.mse_loss(cell_v_pred, v_target_cell)
    loss = hp.pos_loss_weight * pos_loss + hp.cell_loss_weight * cell_loss
    metrics = {
        "loss": loss.detach().item(),
        "pos_loss": pos_loss.detach().item(),
        "cell_loss": cell_loss.detach().item(),
    }
    return loss, metrics


@torch.no_grad()
def sample_structures(
    model: CSPNetFull,
    atom_types_batch: torch.Tensor,
    num_atoms_batch: torch.Tensor,
    base_lattice: InformedLatticeDistribution,
    hp: HParams,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler ODE sampler. Returns (frac_coords, lattices) for the batch."""
    model.eval()
    B = num_atoms_batch.shape[0]
    node2graph = torch.repeat_interleave(
        torch.arange(B, device=device), num_atoms_batch
    )
    x_pos = sample_pos_base(num_atoms_batch, device)
    x_cell = base_lattice.sample(B, device)

    n_steps = hp.integration_steps
    times = torch.linspace(hp.small_time, hp.big_time, n_steps + 1, device=device)
    for k in range(n_steps):
        t = times[k]
        dt = times[k + 1] - times[k]
        t_batch = t.expand(B)
        v_pos, v_cell = model(
            t_batch, atom_types_batch, wrap_unit(x_pos), x_cell, num_atoms_batch, node2graph
        )
        # Velocity annealing factor as in OMatG.
        x_pos = x_pos + dt * (1.0 + hp.pos_velocity_annealing * t) * v_pos
        x_cell = x_cell + dt * (1.0 + hp.cell_velocity_annealing * t) * v_cell

    return wrap_unit(x_pos), x_cell


# ============================================================================
# Main training loop.
# ============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    """Linear warmup + cosine decay to 0 (slowrun-compatible LR schedule)."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


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
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override HParams.epochs.")
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
    print0 = print  # single-process; placeholder for slowrun parity

    print0(f"[setup] device={device}  hp={hp}")
    out_dir = Path(hp.out_dir) / hp.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_ds = CrystalDataset(Path(hp.data_dir) / hp.train_file)
    # num_workers=0: the .pt cache is already in RAM as torch tensors and
    # collate is trivial (torch.cat/stack), so worker-process pickle/IPC
    # overhead exceeds the prefetch benefit. Measured ~4% faster than
    # num_workers=2 on a single RTX 3090.
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=collate_pyg,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print0(f"[data] train={len(train_ds)}  steps/epoch≈{len(train_loader)}")

    # Model
    model = CSPNetFull(hp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print0(f"[model] CSPNetFull params={n_params/1e6:.2f}M")

    if hp.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, dynamic=True)  # graphs are dynamic by structure size
            print0("[model] torch.compile enabled")
        except Exception as e:
            print0(f"[model] torch.compile failed, continuing eager: {e}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    base_lattice = InformedLatticeDistribution(
        hp.mp20_length_log_means, hp.mp20_length_log_stds
    )

    total_steps = hp.epochs * len(train_loader)
    print0(f"[sched] total_steps={total_steps}  warmup={hp.warmup_steps}")

    step = 0
    smooth_loss = 0.0
    ema_beta = 0.9

    print0(f"[train] starting; out_dir={out_dir}")

    for epoch in range(1, hp.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            for g in optimizer.param_groups:
                g["lr"] = cosine_lr(step, total_steps, hp.warmup_steps, hp.lr)

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = flow_matching_loss(model, batch, base_lattice, hp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            optimizer.step()

            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * metrics["loss"]
            debiased = smooth_loss / (1 - ema_beta ** (step + 1))

            if step % hp.log_every == 0:
                print0(
                    f"epoch {epoch:5d}  step {step:7d}  loss {debiased:.5f}  "
                    f"pos {metrics['pos_loss']:.5f}  cell {metrics['cell_loss']:.5f}"
                )
            step += 1

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
