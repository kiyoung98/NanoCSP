"""
Score a NanoCSP submission from a directory of CIF samples.

`train.py` writes one CIF per test entry to
`runs/<run>/test_samples/{idx:05d}.cif`, indexed by the order of
`data/mp20_ps_test.pt`. This script loads those CIFs, builds the
reference test set from the ground-truth `.pt` tensors, and computes the
full METRe metric set at the fixed NanoCSP tolerances
(`stol=0.5, ltol=0.3, angle_tol=10.0`).

METRe match rate (higher is better, range [0, 1]) is the leaderboard
metric. cRMSE (lower is better) and mean_rmsd (precision when matched)
are reported as diagnostics.

Usage:
    python evaluate.py --samples_dir runs/<run>/test_samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifParser

from _vendored.metre import CrystalRecord, metre_metrics


def _ref_from_test(rec) -> CrystalRecord:
    """Build a reference CrystalRecord from a test-split record."""
    frac = rec["frac_coords"].numpy()
    lat = rec["lattice"].numpy()
    species = rec["atomic_numbers"].numpy()
    return CrystalRecord(
        Structure(Lattice(lat), species, frac, coords_are_cartesian=False),
        species,
    )


def _gen_from_cif(path: Path) -> CrystalRecord:
    """Build a generated CrystalRecord from a CIF file.

    Uses pymatgen's default ``occupancy_tolerance``. If a CIF can't be
    parsed (e.g. atoms collapsed to identical fractional coords →
    occupancy > tolerance, or a lattice axis collapsed below the
    minimum-thickness threshold), this raises; the caller catches and
    skips that gen from the matching pool. METRe's per-composition
    matching makes the absence a graceful "no candidate" rather than a
    hard error.
    """
    struct = CifParser(str(path)).parse_structures(primitive=False)[0]
    return CrystalRecord(struct, struct.atomic_numbers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples_dir", type=str, required=True,
                        help="Directory containing {idx:05d}.cif files written by train.py.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--test_file", type=str, default="mp20_ps_test.pt")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, evaluate on first N test entries only (debug).")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Process pool size for StructureMatcher (0 = sequential, "
                             "default). Sequential is fastest in practice: our parallel "
                             "path pickles the full reference list per submitted task, "
                             "and IPC overhead exceeds StructureMatcher CPU cost. "
                             "Measured on a 512-record subset: 17.6 s sequential vs "
                             "22-41 s with 2-16 workers.")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.is_dir():
        raise SystemExit(f"samples_dir not found: {samples_dir}")

    # Load test split (ground truth used only as reference for matching).
    test_records = torch.load(str(Path(args.data_dir) / args.test_file))
    if args.limit > 0:
        test_records = test_records[:args.limit]
    n = len(test_records)
    print(f"[data] test split: {n} structures")

    print("[ref] building reference pymatgen structures ...")
    ref_list = [_ref_from_test(r) for r in tqdm(test_records, desc="ref")]

    # Load CIFs at fixed index order. METRe matching is composition-based,
    # not index-based — a missing or unparseable gen at index i simply
    # reduces the pool of candidates for refs of that composition. Other
    # refs of the same composition can still match against the remaining
    # gens. So we don't abort here; we collect skips and report.
    print(f"[gen] loading CIFs from {samples_dir} ...")
    gen_list: list[CrystalRecord] = []
    missing: list[int] = []           # file doesn't exist
    unparseable: list[tuple[int, str]] = []  # (idx, short reason)
    for i in range(n):
        path = samples_dir / f"{i:05d}.cif"
        if not path.exists():
            missing.append(i)
            continue
        try:
            gen_list.append(_gen_from_cif(path))
        except Exception as e:
            unparseable.append((i, type(e).__name__ + ": " + str(e)[:120]))
    n_dropped = len(missing) + len(unparseable)
    if missing:
        print(f"[gen] missing {len(missing)} CIFs "
              f"(first few: {missing[:5]})")
    if unparseable:
        print(f"[gen] unparseable {len(unparseable)} CIFs "
              f"(first few: {[idx for idx, _ in unparseable[:5]]})")
        for idx, reason in unparseable[:5]:
            print(f"[gen]   {idx:05d}.cif: {reason}")
    print(f"[gen] loaded gen_list: {len(gen_list)}/{n} "
          f"({n_dropped} dropped, {100*n_dropped/n:.2f}%)")

    if len(gen_list) == 0:
        sys.stderr.write(
            f"[eval] all {n} CIFs missing or unparseable — nothing to score.\n"
        )
        raise SystemExit(2)

    # Compute METRe at the fixed leaderboard tolerances.
    print(f"[metre] computing METRe (stol=0.5, ltol=0.3, angle_tol=10.0, "
          f"workers={args.num_workers or 'seq'}) ...")
    m = metre_metrics(
        gen_list,
        ref_list,
        ltol=0.3,
        stol=0.5,
        angle_tol=10.0,
        num_workers=args.num_workers if args.num_workers > 0 else None,
        check_reduced=True,
        enable_progress_bar=True,
        desc="metre",
    )
    print("=" * 60)
    print(f"NanoCSP  METRe match rate (test, n={n}, higher is better) = "
          f"{m['match_rate']*100:.2f}%   ({m['n_matched']}/{m['n_total']})   ← leaderboard")
    print(f"         mean RMSD over matched only                       = {m['mean_rmsd']:.4f}")
    print(f"         cRMSE (lower is better, in [0, stol=0.5])         = {m['cRMSE']:.4f}")
    print("=" * 60)
    return m


if __name__ == "__main__":
    main()
