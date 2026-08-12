#!/usr/bin/env python3
"""Convert loop caller BEDPE output into Juicer-compatible BEDPE."""

import argparse
from pathlib import Path


def convert_single(in_file, out_file, chrom="chr1"):
    in_file = Path(in_file)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        raise FileNotFoundError(in_file)

    with open(in_file) as src, open(out_file, "w") as dst:
        for line in src:
            parts = line.strip().split("\t")
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(f"Expected at least 5 columns in {in_file}, got {len(parts)}")
            dst.write(f"{chrom}\t{parts[0]}\t{parts[1]}\t{chrom}\t{parts[2]}\t{parts[3]}\t.\t{parts[4]}\n")


def convert_directory(loop_dir, output_dir, chrom="chr1"):
    loop_dir = Path(loop_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_dir in sorted(path for path in loop_dir.iterdir() if path.is_dir()):
        in_file = sample_dir / "loops.loop.bedpe"
        if not in_file.exists():
            continue
        out_file = output_dir / f"{sample_dir.name}.bedpe"
        convert_single(in_file=in_file, out_file=out_file, chrom=chrom)
        print(f"Converted {in_file} -> {out_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert loops.loop.bedpe to Juicer BEDPE")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-bedpe")
    group.add_argument("--loop-dir", help="Directory containing */loops.loop.bedpe")
    parser.add_argument("--output-bedpe")
    parser.add_argument("--output-dir")
    parser.add_argument("--chrom", default="chr1")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_bedpe:
        if not args.output_bedpe:
            raise SystemExit("--output-bedpe is required with --input-bedpe")
        convert_single(args.input_bedpe, args.output_bedpe, chrom=args.chrom)
    else:
        if not args.output_dir:
            raise SystemExit("--output-dir is required with --loop-dir")
        convert_directory(args.loop_dir, args.output_dir, chrom=args.chrom)


if __name__ == "__main__":
    main()
