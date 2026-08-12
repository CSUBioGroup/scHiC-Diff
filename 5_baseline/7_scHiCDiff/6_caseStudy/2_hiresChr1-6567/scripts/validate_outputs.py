#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz


OUTPUT_NAMES = (
    "denoise_recon_inv",
    "denoise_recon",
    "denoise_target",
    "raw_x",
)


def validate_outputs(result_dir: Path, expected_shape: tuple[int, int]) -> dict:
    files = {}
    for name in OUTPUT_NAMES:
        path = result_dir / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing output: {path}")
        matrix = load_npz(path)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{name} shape {matrix.shape} does not match {expected_shape}"
            )
        values = matrix.data
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite stored values")
        if (values < 0).any():
            raise ValueError(f"{name} contains negative stored values")
        files[name] = {
            "path": str(path),
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
            "min_stored": float(values.min()) if values.size else None,
            "max_stored": float(values.max()) if values.size else None,
        }
    return {
        "status": "passed",
        "expected_shape": list(expected_shape),
        "files": files,
    }


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="ascii")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--expected-cols", required=True, type=int)
    args = parser.parse_args()

    payload = validate_outputs(
        args.result_dir,
        (args.expected_rows, args.expected_cols),
    )
    atomic_write(
        args.result_dir / "validation.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    atomic_write(args.result_dir / "done.flag", "validation passed\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
