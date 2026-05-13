"""
METRe match rate (and diagnostic RMSD/cRMSE) for crystal structures.

Slim port of OMatG `omg/analysis/analysis.py::metre_rmsds` and its helpers
(commit fcb9ba2c2cfd70505b0f142a5b3c44944d78e7f0, MIT License — see
third_party/OMatG-LICENSE). Strip-down notes:

  - NanoCSP ranks submissions by the **METRe match rate** (single metric,
    higher is better, in [0, 1]) — the fraction of reference structures
    with at least one matching generation under pymatgen StructureMatcher.
    cRMSE and mean RMSD are reported as diagnostics. OMatG's
    `metre_rmsds` returns an 8-tuple; we extract the scalars we need.
  - We DROP `ValidAtoms`'s SMACT / CrystalNN / Magpie fingerprint validation
    (the heavy `smact` / `matminer` deps) — NanoCSP does not condition
    the METRe match rate on validity, so we never need those checks.
  - StructureMatcher tolerances default to the NanoCSP track values
    (stol=0.5, ltol=0.3, angle_tol=10.0), which match the METRe paper
    (arXiv 2509.12178 §5). Do **not** change them for leaderboard numbers.

The protocol: for each test composition C with multiplicity K_C in the
reference set, the model generates K_C structures conditioned on C. Total
generated count == reference count. For every reference structure we find
the smallest pymatgen RMS distance over generated structures (within the
same composition); references with no match contribute `stol` to the mean.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Optional, Sequence

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from tqdm import tqdm


class CrystalRecord:
    """Pair of (pymatgen Structure, atomic-number array) used by METRe.

    The `valid` flag is kept in the API for parity with OMatG's `ValidAtoms`
    but is always True here; we don't compute SMACT/CrystalNN validity.
    """

    __slots__ = ("structure", "numbers", "valid")

    def __init__(self, structure: Structure, atomic_numbers: Sequence[int]):
        self.structure = structure
        self.numbers = np.asarray(atomic_numbers, dtype=np.int64)
        self.valid = True


def _element_check(numbers_a: np.ndarray, numbers_b: np.ndarray, check_reduced: bool = True) -> bool:
    """Return True if two structures have the same composition (or are simple multiples)."""
    if check_reduced:
        max_z = int(max(numbers_a.max(), numbers_b.max())) + 1
        ca = np.bincount(numbers_a, minlength=max_z).astype(np.float64)
        cb = np.bincount(numbers_b, minlength=max_z).astype(np.float64)
        ca_min = ca[ca > 0].min()
        cb_min = cb[cb > 0].min()
        return np.allclose(ca / ca_min, cb / cb_min)
    return np.array_equal(np.sort(numbers_a), np.sort(numbers_b))


def _rms_dist(s_a: Structure, s_b: Structure, ltol: float, stol: float, angle_tol: float) -> Optional[float]:
    """Return normalized RMS distance from pymatgen StructureMatcher, or None on no match."""
    sm = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    res = sm.get_rms_dist(s_a, s_b)
    return None if res is None else float(res[0])


def _match_one_to_many(
    gen_record: CrystalRecord,
    ref_records: Sequence[CrystalRecord],
    ltol: float,
    stol: float,
    angle_tol: float,
    check_reduced: bool,
) -> list[tuple[float, int]]:
    """For one generated record, return list of (rmsd, ref_index) for matches."""
    out: list[tuple[float, int]] = []
    for idx, ref in enumerate(ref_records):
        if not _element_check(gen_record.numbers, ref.numbers, check_reduced):
            continue
        rms = _rms_dist(gen_record.structure, ref.structure, ltol, stol, angle_tol)
        if rms is not None:
            out.append((rms, idx))
    return out


def _best_rmsd_per_ref(
    gen_list: Sequence[CrystalRecord],
    ref_list: Sequence[CrystalRecord],
    *,
    ltol: float,
    stol: float,
    angle_tol: float,
    num_workers: Optional[int],
    check_reduced: bool,
    enable_progress_bar: bool,
    desc: str,
) -> list[Optional[float]]:
    """Return, for each reference index, the smallest matching RMSD across
    `gen_list` (or None if no generated structure matched).

    Shared inner loop for `metre_metrics`.
    """
    if len(gen_list) > len(ref_list):
        raise ValueError(
            f"len(gen_list)={len(gen_list)} cannot exceed len(ref_list)={len(ref_list)}."
        )

    func = partial(
        _match_one_to_many,
        ref_records=list(ref_list),
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        check_reduced=check_reduced,
    )

    best_rmsd: list[Optional[float]] = [None] * len(ref_list)

    def _absorb(matches: list[tuple[float, int]]) -> None:
        for rmsd, ref_idx in matches:
            cur = best_rmsd[ref_idx]
            if cur is None or rmsd < cur:
                best_rmsd[ref_idx] = rmsd

    if num_workers is None or num_workers <= 1:
        iterator = tqdm(
            gen_list, total=len(gen_list), desc=desc,
            disable=not enable_progress_bar,
        )
        for gen in iterator:
            _absorb(func(gen))
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(func, gen): i for i, gen in enumerate(gen_list)}
            iterator = tqdm(
                as_completed(futures), total=len(futures), desc=desc,
                disable=not enable_progress_bar,
            )
            for fut in iterator:
                _absorb(fut.result())

    return best_rmsd


def metre_metrics(
    gen_list: Sequence[CrystalRecord],
    ref_list: Sequence[CrystalRecord],
    *,
    ltol: float = 0.3,
    stol: float = 0.5,
    angle_tol: float = 10.0,
    num_workers: Optional[int] = None,
    check_reduced: bool = True,
    enable_progress_bar: bool = True,
    desc: str = "metre",
) -> dict:
    """Run a single StructureMatcher pass and return all aggregate scalars
    derivable from it. `match_rate` (the METRe match rate) is the
    leaderboard metric; the others are diagnostic — they help
    disentangle "matched but imprecise" from "did not match".

    Returns dict with keys (each a Python float / int):
        match_rate   n_matched / n_total
                     — METRe match rate, primary metric, range [0, 1]
        mean_rmsd    mean over matched refs of best RMSD
                     — NaN if 0 matches
        cRMSE        mean over refs of (best RMSD or stol if no match)
                     — diagnostic, range [0, stol]
        n_matched    int, refs with at least one matching gen
        n_total      int, len(ref_list)
    """
    best_rmsd = _best_rmsd_per_ref(
        gen_list, ref_list,
        ltol=ltol, stol=stol, angle_tol=angle_tol,
        num_workers=num_workers, check_reduced=check_reduced,
        enable_progress_bar=enable_progress_bar, desc=desc,
    )
    n_total = len(best_rmsd)
    matched = [r for r in best_rmsd if r is not None]
    n_matched = len(matched)
    crmse = float(np.mean([r if r is not None else stol for r in best_rmsd]))
    mean_rmsd = float(np.mean(matched)) if n_matched > 0 else float("nan")
    return {
        "cRMSE": crmse,
        "match_rate": n_matched / n_total,
        "mean_rmsd": mean_rmsd,
        "n_matched": n_matched,
        "n_total": n_total,
    }
