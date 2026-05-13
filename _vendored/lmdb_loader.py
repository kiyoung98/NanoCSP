"""
Minimal LMDB record reader for OMatG-format crystal structure data.

Derived from OMatG `omg/datamodule/structure_dataset.py::StructureDataset._from_lmdb`,
commit fcb9ba2c2cfd70505b0f142a5b3c44944d78e7f0. MIT License (see
third_party/OMatG-LICENSE).

Each LMDB record is a pickled dict with at least:
  - "cell": 3x3 torch.Tensor (lattice vectors; row i is the i-th basis vector
            in Cartesian coordinates)
  - "atomic_numbers": (N,) int torch.Tensor
  - "pos": (N, 3) float torch.Tensor of CARTESIAN atomic positions
  - "identifier" (or legacy "ids"): str, optional
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator, Optional

import lmdb
import torch


def iter_lmdb(path: str | Path) -> Iterator[dict]:
    """Yield each record from an OMatG-format LMDB as a dict.

    The dict has keys: 'cell' (3,3 float tensor), 'atomic_numbers' (N int
    tensor), 'pos' (N,3 float tensor of Cartesian coords), and optionally
    'identifier' (str).
    """
    path = str(path)
    with lmdb.Environment(
        path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False
    ) as env, env.begin() as txn:
        for enc_key, data in txn.cursor():
            rec = pickle.loads(data)
            _validate(rec, enc_key)
            ident = _extract_identifier(rec, enc_key)
            yield {
                "cell": rec["cell"],
                "atomic_numbers": rec["atomic_numbers"],
                "pos": rec["pos"],
                "identifier": ident,
            }


def count_lmdb(path: str | Path) -> int:
    """Return number of entries in an LMDB."""
    with lmdb.Environment(
        str(path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False
    ) as env, env.begin() as txn:
        return txn.stat()["entries"]


def _extract_identifier(rec: dict, enc_key: bytes) -> str:
    if "identifier" in rec and "ids" in rec:
        raise KeyError(f"Record {enc_key!r} has both 'identifier' and 'ids'.")
    if "identifier" in rec:
        return str(rec["identifier"])
    if "ids" in rec:
        return str(rec["ids"])
    return enc_key.decode()


def _validate(rec: dict, enc_key: bytes) -> None:
    key = enc_key.decode(errors="replace")
    for required in ("cell", "atomic_numbers", "pos"):
        if required not in rec:
            raise KeyError(f"LMDB record {key!r} missing required field '{required}'")
    if not isinstance(rec["cell"], torch.Tensor) or not torch.is_floating_point(rec["cell"]):
        raise TypeError(f"LMDB record {key!r}: 'cell' must be a float torch.Tensor")
    if rec["cell"].shape != (3, 3):
        raise TypeError(f"LMDB record {key!r}: 'cell' must be shape (3,3), got {tuple(rec['cell'].shape)}")
    if not isinstance(rec["atomic_numbers"], torch.Tensor) or rec["atomic_numbers"].dtype not in (torch.int64, torch.int32):
        raise TypeError(f"LMDB record {key!r}: 'atomic_numbers' must be int torch.Tensor")
    if not isinstance(rec["pos"], torch.Tensor) or not torch.is_floating_point(rec["pos"]):
        raise TypeError(f"LMDB record {key!r}: 'pos' must be a float torch.Tensor")
    n = rec["atomic_numbers"].shape[0]
    if rec["pos"].shape != (n, 3):
        raise TypeError(
            f"LMDB record {key!r}: 'pos' must be shape ({n},3), got {tuple(rec['pos'].shape)}"
        )
