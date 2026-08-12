#!/usr/bin/env python3
"""Evaluate one counts-like NPZ under the corrected scHiC-Diff benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


BENCHMARK_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunIdentity:
    input_npz: Path
    label: str
    slug: str
    output_root: Path


@dataclass(frozen=True)
class GeneratedConfig:
    benchmark_path: Path
    methods_path: Path
    provenance_path: Path


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot derive a slug from {value!r}")
    return slug


def derive_identity(
    input_npz: str | Path,
    label: str | None = None,
    output_root: str | Path | None = None,
) -> RunIdentity:
    input_npz = Path(input_npz)
    label = label or input_npz.parent.name
    slug = slugify(label)
    root = (
        Path(output_root)
        if output_root is not None
        else BENCHMARK_DIR / "results_auto" / slug
    )
    return RunIdentity(input_npz=input_npz, label=label, slug=slug, output_root=root)


def ensure_safe_output_root(
    output_root: str | Path,
    input_npz: str | Path,
    *,
    force: bool,
) -> Path:
    """Return a safe, resolved output directory without modifying it."""
    output_root = Path(output_root).resolve()
    input_parent = Path(input_npz).resolve().parent
    try:
        output_root.relative_to(input_parent)
    except ValueError:
        pass
    else:
        raise ValueError("output root must not be nested beneath the input NPZ directory")
    if output_root.exists() and not force:
        raise FileExistsError(f"output root already exists: {output_root}")
    return output_root


def write_generated_config(
    input_npz: str | Path,
    label: str,
    slug: str,
    output_root: str | Path,
) -> GeneratedConfig:
    """Write a one-method config derived from the default corrected benchmark."""
    input_npz = Path(input_npz).resolve()
    output_root = Path(output_root).resolve()
    template_path = BENCHMARK_DIR / "configs" / "benchmark.json"
    methods_template_path = BENCHMARK_DIR / "configs" / "methods.json"
    benchmark = json.loads(template_path.read_text())
    methods_template = json.loads(methods_template_path.read_text())

    config_dir = output_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    methods_path = config_dir / "methods.json"
    benchmark_path = config_dir / "benchmark.json"
    provenance_path = config_dir / "provenance.json"

    shared_inputs = {
        key: str(_resolve_template_path(methods_template_path.parent, methods_template[key]))
        for key in ("canonical_h5ad", "canonical_named_npz", "early_neurons_npz")
    }
    methods = {
        **{key: value for key, value in methods_template.items() if key != "methods"},
        **shared_inputs,
        "methods": [
            {
                "name": label,
                "slug": slug,
                "role": "imputation",
                "input_npz": str(input_npz),
                "include_diagonal": True,
            }
        ],
    }
    benchmark.update(
        {
            "case_name": f"corrected_{slug}",
            "methods_config": str(methods_path),
            "loci_config": str((BENCHMARK_DIR / "configs" / "loci.json").resolve()),
            "legacy_loop_script": str(
                (BENCHMARK_DIR.parent / "1_experiment_loop_compare" / "scripts" / "call_loops_from_npz.py").resolve()
            ),
            "output_root": str(output_root),
        }
    )
    apa = dict(benchmark["apa"])
    for key in ("java_bin", "juicer_jar", "reference_hic"):
        value = Path(apa[key])
        apa[key] = str(value if value.is_absolute() else (template_path.parent / value).resolve())
    local_juicer_jar = Path.home() / "Downloads" / "juicer_tools.2.20.00.jar"
    if local_juicer_jar.is_file():
        apa["juicer_jar"] = str(local_juicer_jar.resolve())
    benchmark["apa"] = apa

    methods_path.write_text(json.dumps(methods, indent=2) + "\n")
    benchmark_path.write_text(json.dumps(benchmark, indent=2) + "\n")
    provenance_path.write_text(
        json.dumps(
            {
                "input_npz": str(input_npz),
                "label": label,
                "slug": slug,
                "output_root": str(output_root),
                "template_benchmark": str(template_path.resolve()),
                "template_methods": str(methods_template_path.resolve()),
            },
            indent=2,
        )
        + "\n"
    )
    return GeneratedConfig(benchmark_path, methods_path, provenance_path)


def _resolve_template_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def build_stage_commands(
    benchmark_path: str | Path,
    *,
    python_bin: str | Path,
    force: bool,
) -> list[list[str]]:
    """Return the benchmark, APA, and summary commands in required order."""
    benchmark_path = str(Path(benchmark_path))
    python_bin = str(python_bin)
    scripts = BENCHMARK_DIR / "scripts"
    benchmark_command = [
        python_bin,
        str(scripts / "run_benchmark.py"),
        "--config",
        benchmark_path,
        "--stage",
        "all",
    ]
    if force:
        benchmark_command.append("--force")
    return [
        benchmark_command,
        [python_bin, str(scripts / "run_apa.py"), "--config", benchmark_path],
        [python_bin, str(scripts / "summarize_results.py"), "--config", benchmark_path],
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one counts-like scHiC-Diff NPZ with the corrected benchmark."
    )
    parser.add_argument("--input-npz", required=True, help="Path to denoise_recon_inv.npz")
    parser.add_argument("--label", help="Display name; defaults to the NPZ parent directory name")
    parser.add_argument(
        "--output-root",
        help="Independent result directory; defaults to results_auto/<derived label>",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter with anndata, scipy, pandas, and matplotlib installed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow resuming an existing output root and rerun loop calls",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the generated configuration and provenance without running metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identity = derive_identity(args.input_npz, label=args.label, output_root=args.output_root)
    if not identity.input_npz.is_file():
        raise FileNotFoundError(f"input NPZ does not exist: {identity.input_npz}")
    output_root = ensure_safe_output_root(
        identity.output_root, identity.input_npz, force=args.force
    )
    generated = write_generated_config(
        identity.input_npz, identity.label, identity.slug, output_root
    )
    print(f"Generated config: {generated.benchmark_path}")
    print(f"Output root: {output_root}")
    if args.prepare_only:
        return
    for command in build_stage_commands(
        generated.benchmark_path, python_bin=args.python_bin, force=args.force
    ):
        print("+ " + " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
