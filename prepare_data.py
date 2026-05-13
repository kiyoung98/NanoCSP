"""Set up the MP20 polymorph split dataset for NanoCSP.

Streams OMatG's mp_20_ps {train,val,test}.lmdb at a pinned commit, converts
each structure into a slim torch dict, and writes only the result to
`./data/mp20_ps_*.pt`:
  - lattice: (3, 3) float32 tensor (rows = basis vectors)
  - frac_coords: (N, 3) float32 tensor in [0, 1)
  - atomic_numbers: (N,) int64 tensor
  - identifier: str

OMatG stores positions as Cartesian; we convert to fractional once here so
train.py never has to. Convention: cartesian = frac @ lattice, hence
frac = cartesian @ inv(lattice). We then wrap to [0, 1).

Idempotent: any split whose .pt is already present is skipped (no
re-download, no re-conversion). Intermediate LMDB files are written to a
temp directory and removed after conversion — no `omatg_data/` left behind.

Usage (no manual file placement; downloads ~110 MB on first run):
    python prepare_data.py
"""
from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import torch
from tqdm import tqdm

from _vendored.lmdb_loader import count_lmdb, iter_lmdb

SPLITS = ("train", "val", "test")

# Pinned to the same commit as the vendored cRMSE code (see
# _vendored/metre.py). Bump deliberately — leaderboard cRMSE numbers are
# only comparable across submissions that use the same dataset bytes.
OMATG_COMMIT = "fcb9ba2c2cfd70505b0f142a5b3c44944d78e7f0"
OMATG_RAW_BASE = (
    f"https://raw.githubusercontent.com/FERMat-ML/OMatG/{OMATG_COMMIT}/omg/data/mp_20_ps"
)


def cartesian_to_fractional(pos_cart: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """Convert Cartesian positions to fractional, wrapped to [0, 1).

    Convention: lattice rows are basis vectors, so a Cartesian position p
    corresponds to fractional f = p @ inv(lattice).
    """
    inv_lat = torch.linalg.inv(lattice)
    frac = pos_cart @ inv_lat
    return frac - torch.floor(frac)


def convert_split(lmdb_path: Path) -> list[dict]:
    n = count_lmdb(lmdb_path)
    out: list[dict] = []
    for rec in tqdm(iter_lmdb(lmdb_path), total=n, desc=f"converting {lmdb_path.name}"):
        lattice = rec["cell"].to(torch.float32)
        pos_cart = rec["pos"].to(torch.float32)
        frac = cartesian_to_fractional(pos_cart, lattice)
        out.append({
            "lattice": lattice,
            "frac_coords": frac,
            "atomic_numbers": rec["atomic_numbers"].to(torch.int64),
            "identifier": rec["identifier"],
        })
    return out


def _download_with_progress(url: str, out_path: Path) -> None:
    """Stream `url` to `out_path` with a tqdm bar."""
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        with open(out_path, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"download {out_path.name}"
        ) as bar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./data"),
        help="Output directory for .pt files. Defaults to ./data.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Use a temp dir for intermediate LMDBs so nothing persists outside
    # `out_dir/mp20_ps_*.pt` after we exit.
    with tempfile.TemporaryDirectory(prefix="nanocsp_lmdb_") as tmp:
        tmp_dir = Path(tmp)
        for split in SPLITS:
            pt_path = args.out_dir / f"mp20_ps_{split}.pt"
            if pt_path.is_file() and pt_path.stat().st_size > 0:
                print(f"  found {pt_path} — skipping")
                continue
            lmdb_path = tmp_dir / f"{split}.lmdb"
            url = f"{OMATG_RAW_BASE}/{split}.lmdb"
            _download_with_progress(url, lmdb_path)
            records = convert_split(lmdb_path)
            torch.save(records, pt_path)
            print(f"  wrote {pt_path}  ({len(records)} structures)")
            lmdb_path.unlink()  # free disk before next split


if __name__ == "__main__":
    main()
