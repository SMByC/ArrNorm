#!/usr/bin/env python3
"""
Generate expected regression outputs for the ArrNorm test suite.

Run ONCE from the project root:
    python tests/generate_expected.py

Outputs are saved to tests/data/expected/ and committed as regression
baselines. Re-running overwrites any existing files.
"""
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from arrnorm import Normalization  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
EXPECTED_DIR = DATA_DIR / "expected"


def _setup(tmpdir):
    for name in ("ref.tif", "ref_adjusted2target.tif", "target.tif"):
        shutil.copy(str(DATA_DIR / name), str(tmpdir / name))


def _run(tmpdir, ref_name, **kw):
    norm = Normalization(
        count=0,
        img_ref=str(tmpdir / ref_name),
        img_target=str(tmpdir / "target.tif"),
        max_iters=kw.get("max_iters", 30),
        threshold=kw.get("threshold", 0.95),
        reg=False,
        onlyreg=False,
        noneg=False,
        mask=kw.get("mask", False),
        warpband=2,
        chunksize=None,
        graphics=False,
    )
    norm.run()
    return norm


# Each entry: (description, ref_name, run_kwargs, [(src_attr, dest_filename), ...])
SCENARIOS = [
    (
        "Full ref — requires CRS reprojection",
        "ref.tif", {},
        [("img_norm", "target_norm_full_ref.tif")],
    ),
    (
        "Pre-aligned ref, 30 iters, t=0.95",
        "ref_adjusted2target.tif", {},
        [("img_norm", "target_norm_prealigned.tif")],
    ),
    (
        "Pre-aligned ref, 5 iters",
        "ref_adjusted2target.tif", {"max_iters": 5},
        [("img_norm", "target_norm_iter5.tif")],
    ),
    (
        "Pre-aligned ref, 15 iters",
        "ref_adjusted2target.tif", {"max_iters": 15},
        [("img_norm", "target_norm_iter15.tif")],
    ),
    (
        "Pre-aligned ref, t=0.90",
        "ref_adjusted2target.tif", {"threshold": 0.90},
        [("img_norm", "target_norm_t090.tif")],
    ),
    (
        "Pre-aligned ref, with mask (auto nodata → 0)",
        "ref_adjusted2target.tif", {"mask": None},
        [
            ("img_norm",   "target_norm_masked.tif"),
            ("mask_file",  "target_mask.tif"),
        ],
    ),
]


def main():
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating expected outputs → {EXPECTED_DIR}\n{'=' * 60}")

    for description, ref_name, run_kwargs, outputs in SCENARIOS:
        print(f"\n--- {description}")
        with tempfile.TemporaryDirectory() as td:
            tmpdir = Path(td)
            _setup(tmpdir)
            norm = _run(tmpdir, ref_name, **run_kwargs)
            for attr, dest_name in outputs:
                src = Path(getattr(norm, attr))
                dst = EXPECTED_DIR / dest_name
                shutil.copy(str(src), str(dst))
                print(f"  saved: {dst.name}")

    print(f"\nDone — {len(SCENARIOS)} scenarios written to {EXPECTED_DIR}\n")


if __name__ == "__main__":
    main()
