"""
NanoCSP FlowMM baseline trainer.

Single-file CSP baseline for the MP20 polymorph split, evaluated by cRMSE.
This branch implements FlowMM (arXiv:2406.04713,
facebookresearch/flowmm @ scripts_model/conf/model/null_params.yaml):

  - CSPNet with a 6-dim lattice head, where the lattice manifold is the
    `lattice_params` parametrization: lengths (positive) + angles
    (degrees in (59.9, 120.1) bijected to R via sigmoid). Lengths use a
    LogNormal base; angles use Uniform-via-sigmoid base.
  - Riemannian flow matching on the product manifold
    (FlatTorus³_per_atom × R⁶_per_graph). The torus uses the periodic
    linear interpolant (closest-image geodesic); the lattice_params
    space is Euclidean-flat under the manifold's chosen inner product.
  - Sampling: Euler integration on the manifold; fractional coords
    wrapped to [0, 1) at every step; the 6-D lattice state decoded back
    to a 3x3 cell via the standard cellpar formula at every step (so
    the network's IP feature stays meaningful).

The script runs for `--epochs` epochs of training, then writes one CIF
per test entry to `runs/<run>/test_samples/{idx:05d}.cif`. Following
FlowMM convention, the EMA shadow weights are swapped in before final
sampling. `evaluate.py` reads that directory and scores it.

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

    # Model (FlowMM CSPNet; mirrors scripts_model/conf/vectorfield/rfm_cspnet.yaml).
    hidden_dim: int = 512
    num_layers: int = 6
    num_freqs: int = 128
    time_dim: int = 256
    num_atomic_types: int = 100         # NUM_ATOMIC_TYPES in flowmm
    max_atom_num: int = 100             # one_hot_dim for represent_num_atoms
    layer_norm: bool = True
    concat_sum_pool: bool = True
    represent_num_atoms: bool = True
    represent_angle_edge_to_lattice: bool = True
    use_log_map: bool = True            # closest-image FlatTorus01.logmap
    self_edges: bool = False            # remove diagonal from FC graph

    # FlowMM RFM (lattice_params manifold). Defaults from
    # scripts_model/conf/model/null_params.yaml + default.yaml/integrate.
    pos_loss_weight: float = 400.0    # cost_coord
    cell_loss_weight: float = 1.0     # cost_lattice
    correct_com_motion: bool = True   # f_manifold.proju subtracts the mean
    integration_steps: int = 1000
    small_time: float = 0.0
    big_time: float = 1.0
    inference_anneal_slope: float = 0.0   # disabled in FlowMM null_params
    inference_anneal_offset: float = 0.0
    angle_low_deg: float = 59.9            # FlowMM LatticeParams bounds
    angle_high_deg: float = 120.1

    # MP20-PS lattice base distribution (LogNormal lengths + Uniform-via-sigmoid angles).
    # Length stats imported from OMatG mp_20 (matches what FlowMM would compute).
    mp20_length_log_means: tuple = (1.575442910194397, 1.7017393112182617, 1.9781638383865356)
    mp20_length_log_stds: tuple = (0.24437622725963593, 0.26526379585266113, 0.3535512685775757)

    # Optimizer / training. Method-specific hyperparameters (AdamW lr=3e-4,
    # weight_decay=0, EMA decay=0.999, gradient_clip 0.5 by value,
    # batch_size=256, CosineAnnealingLR with eta_min=1e-5) match FlowMM
    # scripts_model/conf/default.yaml. CosineAnnealingLR.T_max is linked
    # to `epochs`. The 12 h honor-system budget covers training + (optional)
    # in-line validation + final test sampling (heavy: 1000-step
    # integration) — tune `epochs` so the run fits.
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.0
    epochs: int = 2800
    grad_clip: float = 0.5
    seed: int = 42
    ema_decay: float = 0.999
    lr_eta_min: float = 1e-5

    # Final test sampling. FlowMM convention: EMA shadow weights are
    # swapped in before sampling. Writes one CIF per test entry to
    # runs/<run>/test_samples/{idx:05d}.cif.
    test_sample_seed: int = 0
    test_sample_batch_size: int = 64

    # Logging / IO
    log_every: int = 50              # steps
    out_dir: str = "./runs"
    run_name: str = "flowmm"
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


class SinusoidsEmbedding(nn.Module):
    """Frequency embedding for frac-diff vectors. Same in DiffCSP and FlowMM."""

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


def flat_torus_logmap(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Closest-image periodic distance on T^d (FlowMM `FlatTorus01.logmap`).

    Equivalent to `((y - x + 0.5) % 1.0) - 0.5`, returns values in
    [-0.5, 0.5)^d. The atan2 form mirrors the FlowMM source verbatim.
    """
    z = 2 * math.pi * (y - x)
    return torch.atan2(torch.sin(z), torch.cos(z)) / (2 * math.pi)


class CSPLayer(nn.Module):
    """FlowMM CSPLayer. Edge features:
        [unit_dots(3), hi, hj, num_atoms_emb(hidden_dim), lattice_flat(6),
         dis_emb(frac_diff)]
    The 6-D lattice_flat goes in directly (no L L^T); the 3-D unit_dots
    encodes the angle between cartesian edge vector and lattice rows.
    """

    def __init__(
        self, hidden_dim: int, dis_emb: SinusoidsEmbedding, ln: bool,
        dim_l: int, represent_num_atoms: bool, max_atom_num: int,
        represent_angle_edge_to_lattice: bool,
    ):
        super().__init__()
        self.dis_emb = dis_emb
        self.represent_num_atoms = represent_num_atoms
        # FlowMM declares num_atom_embedding per-layer (not shared).
        if represent_num_atoms:
            self.num_atom_embedding = nn.Linear(max_atom_num, hidden_dim, bias=False)
        else:
            self.num_atom_embedding = None
        self.represent_angle_edge_to_lattice = represent_angle_edge_to_lattice

        angle_edge_dims = 3 if represent_angle_edge_to_lattice else 0
        num_hidden_vecs = 3 if represent_num_atoms else 2  # hi, hj (+num_atoms emb)
        edge_in = angle_edge_dims + hidden_dim * num_hidden_vecs + dim_l + dis_emb.dim

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

    @staticmethod
    def _unit_dots_ltlf(non_zscored_lattice: torch.Tensor, edge2graph: torch.Tensor,
                       frac_diff: torch.Tensor) -> torch.Tensor:
        ltl = non_zscored_lattice @ non_zscored_lattice.transpose(-1, -2)  # (B, 3, 3)
        dots = torch.einsum("...ij,...j->...i", ltl[edge2graph], frac_diff)  # (E, 3)
        unit_dots = dots / (dots.norm(dim=-1, keepdim=True) + 1e-12)
        return unit_dots

    def forward(
        self,
        h: torch.Tensor,
        lattice_flat: torch.Tensor,            # (B, dim_l) — 6-D for lattice_params
        non_zscored_lattice: torch.Tensor | None,  # (B, 3, 3) decoded cell
        edge_index: torch.Tensor,
        edge2graph: torch.Tensor,
        frac_diff: torch.Tensor,                # (E, 3) closest-image
        num_atoms_one_hot: torch.Tensor | None,  # (B, max_atom_num)
    ) -> torch.Tensor:
        h_in = h
        if self.ln:
            h = self.layer_norm(h_in)

        edge_features = []
        if self.represent_angle_edge_to_lattice:
            unit_dots = self._unit_dots_ltlf(non_zscored_lattice, edge2graph, frac_diff)
            edge_features.append(unit_dots)

        frac_diff_emb = self.dis_emb(frac_diff)
        hi, hj = h[edge_index[0]], h[edge_index[1]]
        lattice_flat_edges = lattice_flat[edge2graph]
        edge_features.extend([hi, hj, lattice_flat_edges, frac_diff_emb])

        if self.represent_num_atoms:
            num_atoms_emb = self.num_atom_embedding(num_atoms_one_hot)[edge2graph]
            edge_features.append(num_atoms_emb)

        e = self.edge_mlp(torch.cat(edge_features, dim=1))
        agg = scatter(e, edge_index[0], dim=0, reduce="mean", dim_size=h.shape[0])
        out = self.node_mlp(torch.cat([h, agg], dim=1))
        return h_in + out


class CSPNetFull(nn.Module):
    """FlowMM CSPNet (lattice_params variant). Predicts `pos_v` (per-atom, 3)
    on the torus and `cell_v` (per-graph, 6) in the lattice_params tangent
    space (3 length-velocities + 3 angle-uncon-velocities).
    """

    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self.dis_emb = SinusoidsEmbedding(n_frequencies=hp.num_freqs)
        # FlowMM: time emb = Linear(1, time_dim, bias=False), NOT sinusoidal.
        self.time_emb = nn.Linear(1, hp.time_dim, bias=False)
        # FlowMM: atom emb = Linear on one-hot, NOT nn.Embedding lookup.
        self.node_embedding = nn.Linear(hp.num_atomic_types, hp.hidden_dim, bias=False)
        self.atom_latent_emb = nn.Linear(hp.hidden_dim + hp.time_dim, hp.hidden_dim, bias=False)
        dim_l = 6  # lattice_params: lengths(3) + angles_uncon(3)
        self.layers = nn.ModuleList([
            CSPLayer(
                hp.hidden_dim, self.dis_emb, hp.layer_norm,
                dim_l=dim_l,
                represent_num_atoms=hp.represent_num_atoms,
                max_atom_num=hp.max_atom_num,
                represent_angle_edge_to_lattice=hp.represent_angle_edge_to_lattice,
            )
            for _ in range(hp.num_layers)
        ])
        if hp.layer_norm:
            self.final_layer_norm = nn.LayerNorm(hp.hidden_dim)
        self.coord_out = nn.Linear(hp.hidden_dim, 3, bias=False)
        # FlowMM concat_sum_pool: lattice head input dim doubles when on.
        self.lattice_out = nn.Linear(
            (2 if hp.concat_sum_pool else 1) * hp.hidden_dim, dim_l,
        )
        # FlowMM also declares a type-prediction head (unused for CSP, but
        # we keep it so the param count and state_dict shape match).
        self.type_out = nn.Linear(hp.hidden_dim, hp.num_atomic_types)

    def gen_edges(
        self,
        num_atoms: torch.Tensor,
        frac_coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FlowMM 'fc' edge style with self_edges configurable + log-map frac_diff."""
        if self.hp.self_edges:
            blocks = [torch.ones(int(n), int(n), device=num_atoms.device) for n in num_atoms]
        else:
            blocks = [
                torch.ones(int(n), int(n), device=num_atoms.device)
                - torch.eye(int(n), device=num_atoms.device)
                for n in num_atoms
            ]
        adj = torch.block_diag(*blocks)
        edge_index, _ = dense_to_sparse(adj)
        if self.hp.use_log_map:
            frac_diff = flat_torus_logmap(frac_coords[edge_index[0]], frac_coords[edge_index[1]])
        else:
            frac_diff = (frac_coords[edge_index[1]] - frac_coords[edge_index[0]]) % 1.0
        return edge_index, frac_diff

    def forward(
        self,
        t: torch.Tensor,                  # (B,) in [0, 1]
        atom_types: torch.Tensor,         # (sum_atoms,) integer atomic numbers
        frac_coords: torch.Tensor,        # (sum_atoms, 3) in [0, 1)
        lattice_params: torch.Tensor,     # (B, 6) lengths + angles_uncon
        num_atoms: torch.Tensor,          # (B,)
        node2graph: torch.Tensor,         # (sum_atoms,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pos_v, cell_v): shapes (sum_atoms, 3) and (B, 6)."""
        edge_index, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = node2graph[edge_index[0]]

        # One-hot atom encoding (FlowMM convention; species_shift=1).
        atom_one_hot = F.one_hot(
            (atom_types - 1).clamp_min(0), num_classes=self.hp.num_atomic_types,
        ).to(dtype=lattice_params.dtype)
        h = self.node_embedding(atom_one_hot)

        # Linear time projection (FlowMM-specific). t_emb shape: (B, time_dim).
        t_emb = self.time_emb(t.view(-1, 1))
        t_per_atom = t_emb[node2graph]
        h = self.atom_latent_emb(torch.cat([h, t_per_atom], dim=1))

        # Decode 6-D lattice_params → 3x3 cell for unit_dots edge feature.
        non_zscored_lattice = (
            params_uncon_to_lattice(lattice_params, self.hp.angle_low_deg, self.hp.angle_high_deg)
            if self.hp.represent_angle_edge_to_lattice else None
        )
        # One-hot num_atoms (FlowMM clamps to one_hot_dim=100).
        if self.hp.represent_num_atoms:
            num_atoms_one_hot = F.one_hot(
                num_atoms.clamp_max(self.hp.max_atom_num - 1),
                num_classes=self.hp.max_atom_num,
            ).to(dtype=lattice_params.dtype)
        else:
            num_atoms_one_hot = None

        for layer in self.layers:
            h = layer(
                h, lattice_params, non_zscored_lattice, edge_index, edge2graph,
                frac_diff, num_atoms_one_hot,
            )

        if self.hp.layer_norm:
            h = self.final_layer_norm(h)

        pos_v = self.coord_out(h)                              # (sum_atoms, 3)
        if self.hp.concat_sum_pool:
            graph_h = torch.cat([
                scatter(h, node2graph, dim=0, reduce="mean"),
                scatter(h, node2graph, dim=0, reduce="sum"),
            ], dim=-1)
        else:
            graph_h = scatter(h, node2graph, dim=0, reduce="mean")
        cell_v = self.lattice_out(graph_h)                     # (B, 6)
        return pos_v, cell_v


# ============================================================================
# FlowMM RFM on the (FlatTorus³ × R⁶_lattice_params) product manifold.
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


# ----------------------------------------------------------------------------
# Lattice parametrization (lengths + angles_uncon).
#
# Port of `flowmm.rfm.manifolds.lattice_params.LatticeParams`. Angles in
# (low, high) degrees are bijected to R via the inverse-CDF of the
# Uniform[low, high] distribution (sigmoid):
#   uncon = logit((angle_deg - low) / (high - low))
#   angle_deg = low + (high - low) * sigmoid(uncon)
# Lengths flow in raw positive R³ space (LogNormal base).
# ----------------------------------------------------------------------------


def angles_deg_to_uncon(angles_deg: torch.Tensor, low: float, high: float) -> torch.Tensor:
    u = (angles_deg - low) / (high - low)
    u = u.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(u) - torch.log1p(-u)


def angles_uncon_to_deg(angles_uncon: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(angles_uncon)


def lattice_matrix_to_lengths_angles(L: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """L: (B, 3, 3) with rows = basis vectors. Returns (lengths, angles_deg)."""
    a, b, c = L[:, 0], L[:, 1], L[:, 2]
    la = torch.linalg.norm(a, dim=-1)
    lb = torch.linalg.norm(b, dim=-1)
    lc = torch.linalg.norm(c, dim=-1)
    cos_alpha = (b * c).sum(-1) / (lb * lc + 1e-12)
    cos_beta = (a * c).sum(-1) / (la * lc + 1e-12)
    cos_gamma = (a * b).sum(-1) / (la * lb + 1e-12)
    angles_rad = torch.stack([
        torch.acos(cos_alpha.clamp(-1.0, 1.0)),
        torch.acos(cos_beta.clamp(-1.0, 1.0)),
        torch.acos(cos_gamma.clamp(-1.0, 1.0)),
    ], dim=-1)
    return torch.stack([la, lb, lc], dim=-1), torch.rad2deg(angles_rad)


def lattice_params_to_matrix_torch(lengths: torch.Tensor, angles_deg: torch.Tensor) -> torch.Tensor:
    """Port of DiffCSP/flowmm `lattice_params_to_matrix_torch` (used by
    FlowMM at decode time). Note: this convention places vector_a in the
    (x, 0, z) plane via `sin(β)/cos(β)`, and vector_c along z. It is NOT
    the same orientation as ASE/standard cellpar_to_cell, but matches
    what FlowMM uses end-to-end.
    """
    angles_r = torch.deg2rad(angles_deg)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)
    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = torch.clamp(val, -1.0, 1.0)
    gamma_star = torch.arccos(val)
    zero = torch.zeros_like(lengths[:, 0])
    vector_a = torch.stack([lengths[:, 0] * sins[:, 1], zero, lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0],
    ], dim=1)
    vector_c = torch.stack([zero, zero, lengths[:, 2]], dim=1)
    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def lattice_to_params_uncon(L: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """3x3 cell → 6-vector [lengths, angles_uncon]."""
    lengths, angles_deg = lattice_matrix_to_lengths_angles(L)
    angles_uncon = angles_deg_to_uncon(angles_deg, low, high)
    return torch.cat([lengths, angles_uncon], dim=-1)


def params_uncon_to_lattice(params: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """6-vector [lengths, angles_uncon] → 3x3 cell."""
    lengths = params[:, :3]
    angles_uncon = params[:, 3:]
    angles_deg = angles_uncon_to_deg(angles_uncon, low, high)
    return lattice_params_to_matrix_torch(lengths, angles_deg)


def interpolate_cell_params(
    t_graph: torch.Tensor, p_0: torch.Tensor, p_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euclidean linear interpolant in 6-D lattice_params space.

    The FlowMM `LatticeParams` manifold inherits from `Euclidean`, so the
    geodesic interpolant collapses to plain linear interpolation and the
    velocity is constant.
    """
    p_t = (1.0 - t_graph) * p_0 + t_graph * p_1
    return p_t, p_1 - p_0


# ============================================================================
# Base distributions for x_0 (uniform on torus + informed lattice).
# ============================================================================


def sample_pos_base(num_atoms: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.rand(int(num_atoms.sum()), 3, device=device)


class InformedLatticeDistribution:
    """FlowMM `LatticeParams` base distribution: LogNormal lengths +
    Uniform[low, high] angles bijected to R via inverse-sigmoid.

    `sample` returns a 3x3 cell matrix (used by evaluate.py and by main
    for the OMatG-style API). `sample_params` returns the raw 6-D
    representation used during training/sampling on the manifold.
    """

    def __init__(self, log_means: tuple, log_stds: tuple,
                 angle_low_deg: float = 59.9, angle_high_deg: float = 120.1):
        self.log_means = torch.tensor(log_means, dtype=torch.float32)
        self.log_stds = torch.tensor(log_stds, dtype=torch.float32)
        self.angle_low_deg = angle_low_deg
        self.angle_high_deg = angle_high_deg

    def sample_params(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # LogNormal lengths (no sort: FlowMM does not sort).
        normal = torch.randn(batch_size, 3, device=device) * self.log_stds.to(device) + self.log_means.to(device)
        lengths = torch.exp(normal)
        # Angles: sample uniform in (low, high) deg, then bijection-invert
        # to R. This is identical to drawing the unconstrained angle from
        # the pushforward of Uniform under the logit transform, which is
        # what FlowMM's UnconstrainedCompact does internally.
        u = torch.rand(batch_size, 3, device=device)
        angles_deg = self.angle_low_deg + (self.angle_high_deg - self.angle_low_deg) * u
        angles_uncon = angles_deg_to_uncon(angles_deg, self.angle_low_deg, self.angle_high_deg)
        return torch.cat([lengths, angles_uncon], dim=-1)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        params = self.sample_params(batch_size, device)
        return params_uncon_to_lattice(params, self.angle_low_deg, self.angle_high_deg)


# ============================================================================
# Loss & sampling.
# ============================================================================


def flow_matching_loss(
    model: CSPNetFull,
    batch: dict,
    base_lattice: InformedLatticeDistribution,
    hp: HParams,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single training step loss for FlowMM RFM on
    (FlatTorus³_per_atom × R⁶_lattice_params).
    """
    device = batch["atom_types"].device
    num_atoms = batch["num_atoms"]
    node2graph = batch["node2graph"]
    B = num_atoms.shape[0]

    t = torch.rand(B, device=device) * (hp.big_time - hp.small_time) + hp.small_time
    t_graph = t.view(B, 1)
    t_atom = t[node2graph].unsqueeze(-1)

    # Base samples (x_0): uniform on torus + LatticeParams base.
    x_0_pos = sample_pos_base(num_atoms, device)
    x_0_params = base_lattice.sample_params(B, device)

    # Data target (x_1): convert the data 3x3 cell into 6-D params.
    x_1_params = lattice_to_params_uncon(
        batch["lattices"], hp.angle_low_deg, hp.angle_high_deg,
    )

    x_t_pos, v_target_pos = interpolate_frac(t_atom, x_0_pos, batch["frac_coords"])
    x_t_params, v_target_params = interpolate_cell_params(t_graph, x_0_params, x_1_params)

    # CoM motion correction (FlatTorus.proju subtracts the per-graph mean).
    if hp.correct_com_motion:
        mean_v = scatter(v_target_pos, node2graph, dim=0, reduce="mean")[node2graph]
        v_target_pos = v_target_pos - mean_v

    # FlowMM's CSPNet takes the 6-D lattice_params directly (it decodes to
    # a 3x3 cell internally for the unit_dots edge feature).
    pos_v_pred, cell_v_pred = model(
        t,
        batch["atom_types"],
        x_t_pos,
        x_t_params,
        num_atoms,
        node2graph,
    )

    # FlowMM loss (model_pl.py:rfm_loss_fn:664-679) uses
    #   manifold.inner(x_t, diff, diff).mean() / dim_per_atom
    # which on Euclidean / FlatTorus reduces to a per-graph squared-error
    # sum, averaged over graphs, divided by the per-atom dimensionality.
    # This is NOT the same as F.mse_loss (which averages over all flat
    # elements): atoms in larger graphs get implicitly more weight in MSE
    # and less weight in FlowMM. Reproducing the FlowMM convention here.
    diff_pos = pos_v_pred - v_target_pos                       # (sum_atoms, 3)
    pos_per_atom_sq = (diff_pos ** 2).sum(dim=-1)              # (sum_atoms,)
    pos_per_graph_sq = scatter(pos_per_atom_sq, node2graph,
                               dim=0, reduce="sum", dim_size=B)  # (B,)
    pos_loss = pos_per_graph_sq.mean() / 3.0                    # /dim_f_per_atom

    diff_cell = cell_v_pred - v_target_params                   # (B, 6)
    cell_loss = (diff_cell ** 2).sum(dim=-1).mean() / 6.0       # /dims.l

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
    """Euler integration on (FlatTorus³ × R⁶_lattice_params). Returns
    (frac_coords, 3x3 lattices)."""
    model.eval()
    B = num_atoms_batch.shape[0]
    node2graph = torch.repeat_interleave(
        torch.arange(B, device=device), num_atoms_batch
    )
    x_pos = sample_pos_base(num_atoms_batch, device)
    x_params = base_lattice.sample_params(B, device)

    n_steps = hp.integration_steps
    times = torch.linspace(hp.small_time, hp.big_time, n_steps + 1, device=device)
    for k in range(n_steps):
        t = times[k]
        dt = times[k + 1] - times[k]
        t_batch = t.expand(B)
        v_pos, v_cell = model(
            t_batch, atom_types_batch, wrap_unit(x_pos), x_params,
            num_atoms_batch, node2graph,
        )
        # Optional inference annealing factor (1 + c * t + b); FlowMM
        # null_params disables it (slope=offset=0 → factor=1).
        anneal = 1.0 + hp.inference_anneal_slope * t + hp.inference_anneal_offset
        x_pos = x_pos + dt * anneal * v_pos
        x_params = x_params + dt * anneal * v_cell

    x_cell_final = params_uncon_to_lattice(x_params, hp.angle_low_deg, hp.angle_high_deg)
    return wrap_unit(x_pos), x_cell_final


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

    Caller is responsible for swapping in EMA weights before calling
    this (FlowMM convention: final evaluation uses EMA shadow).
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


class EMAState:
    """Minimal exponential-moving-average shadow weights, FlowMM convention.

    Mirrors `manifm.ema.EMA` (the EMA module used by ~/refs/flowmm via
    model_pl.py:21 `from manifm.ema import EMA`):
        decay_t = min(decay, (1 + t) / (10 + t))
    so the shadow weights track the live model closely for ~30 early
    updates, then settle to the configured decay (0.999). Using a
    constant decay from step 0 leaves the shadow weights stuck near
    their initialization for the first hundred-or-so steps and would
    produce meaningless EMA-based validation early in training.

    `apply_to(model)` swaps the shadow weights into the model and
    returns the original parameters; `restore(model, backup)` undoes
    the swap.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            name: p.detach().clone() for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        # FlowMM/manifm.ema decay warmup: ramps from ~1/11 at t=1 toward
        # `self.decay` as t grows (crosses 0.99 around t ≈ 990 for
        # decay=0.999, but already > 0.9 by t=90).
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, p in model.named_parameters():
            if name in backup:
                p.copy_(backup[name])


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

    # Optimizer (FlowMM: AdamW lr=3e-4, weight_decay=0).
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    # FlowMM: CosineAnnealingLR(T_max=epochs, eta_min=1e-5), stepped per epoch.
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp.epochs, eta_min=hp.lr_eta_min,
    )

    # FlowMM: maintain EMA shadow weights (decay=0.999). These are
    # swapped in for the final test sampling (FlowMM convention).
    ema = EMAState(model, hp.ema_decay)

    base_lattice = InformedLatticeDistribution(
        hp.mp20_length_log_means, hp.mp20_length_log_stds,
        hp.angle_low_deg, hp.angle_high_deg,
    )

    total_steps = hp.epochs * len(train_loader)
    print0(f"[sched] total_steps={total_steps}  cosine(eta_min={hp.lr_eta_min})  ema_decay={hp.ema_decay}")

    step = 0
    smooth_loss = 0.0
    ema_beta = 0.9

    print0(f"[train] starting; out_dir={out_dir}")

    for epoch in range(1, hp.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = flow_matching_loss(model, batch, base_lattice, hp)
            loss.backward()
            # FlowMM gradient_clip_algorithm='value'.
            torch.nn.utils.clip_grad_value_(model.parameters(), hp.grad_clip)
            optimizer.step()
            ema.update(model)

            smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * metrics["loss"]
            debiased = smooth_loss / (1 - ema_beta ** (step + 1))

            if step % hp.log_every == 0:
                print0(
                    f"epoch {epoch:5d}  step {step:7d}  loss {debiased:.5f}  "
                    f"pos {metrics['pos_loss']:.5f}  cell {metrics['cell_loss']:.5f}"
                )
            step += 1

        # End-of-epoch: step LR scheduler (interval='epoch').
        lr_sched.step()

    # Final test sampling: swap in EMA weights (FlowMM convention) and
    # write one CIF per test entry.
    ema.apply_to(model)
    samples_dir = _write_test_samples(
        model=model,
        hp=hp,
        base_lattice=base_lattice,
        device=device,
        out_dir=out_dir,
        print0=print0,
    )
    print0(f"[done] test samples → {samples_dir} (EMA weights)")
    print0(f"Run: python evaluate.py --samples_dir {samples_dir}")


if __name__ == "__main__":
    main()
