#!/usr/bin/env python3
"""Draw the corrected T-FLAMINGO portion of panel A with audited TADs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pandas as pd

import plot_fig_2test as style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="FLAMINGO_fixed_contact")
    parser.add_argument("--display-name", default="T-FLAMINGO")
    parser.add_argument("--pcc-col", default="pcc_8x8_full")
    parser.add_argument("--norm", choices=["q99", "shared"], default="q99")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "results_fixed"),
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    style.set_gr_style()
    pcc_table = style.load_pcc_table(args.method)
    if pcc_table is None:
        raise SystemExit(f"PCC table is missing for {args.method}")

    width_mm = 85.0
    map_mm = 20.5
    gap_mm = 1.5
    top_mm = 8.0
    bottom_mm = 13.0
    grid_mm = len(style.cfg.CELL_TYPES) * map_mm + (
        len(style.cfg.CELL_TYPES) - 1
    ) * gap_mm
    height_mm = top_mm + grid_mm + bottom_mm
    x0_mm = (width_mm - map_mm) / 2.0
    fig = style.plt.figure(figsize=(width_mm * style.MM, height_mm * style.MM))

    start, end = style.cfg.PDGFRA_SUB_BINS
    source_rows = []
    audit_rows = []

    for row_index, cell_type in enumerate(style.cfg.CELL_TYPES):
        trial_id, median_pcc = style.find_median_trial(
            pcc_table,
            cell_type,
            args.pcc_col,
        )
        selected = pcc_table[
            (pcc_table.cell_type == cell_type)
            & (pcc_table.trial_id == trial_id)
        ].iloc[0]

        target_full = style.load_target(cell_type)
        method_full = style.load_trial_matrix(args.method, cell_type, trial_id)
        target = target_full[start:end, start:end]
        method = method_full[start:end, start:end]
        target_norm = style.normalize(target, args.norm, ref=target)
        method_norm = style.normalize(method, args.norm, ref=target)

        target_report = {}
        target_path = style.find_tad_tsv("target", cell_type, 0)
        target_tads = style.clip_tads(
            style.parse_tad_tsv(target_path, target_report),
            start,
            end,
            target_report,
        )
        audit_rows.append(
            {
                "role": "target",
                "method": "target",
                "cell_type": cell_type,
                "trial_id": "",
                "path": target_path,
                **target_report,
            }
        )

        method_report = {}
        method_path = style.find_tad_tsv(args.method, cell_type, trial_id)
        method_tads = style.clip_tads(
            style.parse_tad_tsv(method_path, method_report),
            start,
            end,
            method_report,
        )
        audit_rows.append(
            {
                "role": "method",
                "method": args.method,
                "cell_type": cell_type,
                "trial_id": trial_id,
                "path": method_path,
                **method_report,
            }
        )

        y_mm = top_mm + row_index * (map_mm + gap_mm)
        ax = fig.add_axes(
            style._rect(
                x0_mm,
                y_mm,
                map_mm,
                map_mm,
                width_mm,
                height_mm,
            )
        )
        displayed_pcc = style.pcc(method, target)
        style.draw_map(
            ax,
            method_norm,
            target_norm,
            cell_type,
            displayed_pcc,
            method_tads,
            target_tads,
            tad_mode="supertad",
            highlight=True,
        )
        if row_index == 0:
            ax.set_title(
                args.display_name,
                fontsize=style.FS_HEAD,
                pad=2.5,
                fontweight="bold",
                color=style.ACCENT,
            )
        ax.set_ylabel(
            cell_type,
            fontsize=style.FS_HEAD,
            labelpad=3,
            color=style.CT_COLORS.get(cell_type, style.INK),
        )

        source_rows.append(
            {
                "cell_type": cell_type,
                "method": args.method,
                "display_name": args.display_name,
                "trial_id": trial_id,
                "selection_metric": args.pcc_col,
                "selection_value": median_pcc,
                "displayed_pcc_8x8": displayed_pcc,
                "pcc_8x8_full": float(selected["pcc_8x8_full"]),
                "pcc_49x49_full": float(selected["pcc_49x49_full"]),
                "normalization": args.norm,
                "method_tads_drawn": method_report.get("drawn", 0),
                "target_tads_drawn": target_report.get("drawn", 0),
            }
        )

    fig.text(
        2.5 / width_mm,
        1 - 1.0 / height_mm,
        "A",
        fontsize=style.FS_TAG,
        fontweight="bold",
        va="top",
        ha="left",
    )

    legend_y_mm = top_mm + grid_mm + 2.0
    legend_ax = fig.add_axes(
        style._rect(10.0, legend_y_mm, 65.0, 5.0, width_mm, height_mm)
    )
    legend_ax.set_axis_off()
    legend_handles = [
        style.Line2D(
            [],
            [],
            color=style.TAD_CORE,
            lw=0.9,
            ls=style.TAD_DASH,
            path_effects=style._halo(style.TAD_CORE, style.TAD_HALO, 0.9),
            label="SuperTAD domain",
        ),
        style.Line2D(
            [],
            [],
            color=style.DIAG_CORE,
            lw=0.7,
            ls=(0, (3, 2)),
            path_effects=style._halo(style.DIAG_CORE, style.DIAG_HALO, 0.7),
            label="Diagonal",
        ),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=style.FS_SMALL,
        handlelength=2.0,
        columnspacing=1.2,
        borderaxespad=0,
        handletextpad=0.5,
    )
    fig.text(
        0.5,
        1 - (legend_y_mm + 7.0) / height_mm,
        "lower: T-FLAMINGO   |   upper: target",
        fontsize=style.FS_SMALL,
        ha="center",
        va="top",
        color="#4D4D4D",
    )

    stem = os.path.join(args.outdir, "PanelA_TFLAMINGO_fixed_contact")
    output_files = style.save_all(fig, stem, dpi=args.dpi)
    style.plt.close(fig)

    source_path = os.path.join(
        args.outdir,
        "PanelA_TFLAMINGO_fixed_contact_source_data.csv",
    )
    audit_path = os.path.join(
        args.outdir,
        "PanelA_TFLAMINGO_fixed_contact_tad_audit.csv",
    )
    pd.DataFrame(source_rows).to_csv(source_path, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    provenance_path = os.path.join(
        args.outdir,
        "PanelA_TFLAMINGO_fixed_contact_provenance.json",
    )
    with open(provenance_path, "w") as handle:
        json.dump(
            {
                "method": args.method,
                "display_name": args.display_name,
                "selection_metric": args.pcc_col,
                "normalization": args.norm,
                "subregion_bins_zero_based_half_open": [start, end],
                "supertad_input_coordinates": "one-based inclusive",
                "plot_coordinates": "zero-based inclusive after TSV conversion",
                "outputs": output_files,
                "source_data": source_path,
                "tad_audit": audit_path,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            handle,
            indent=2,
        )

    for row in source_rows:
        print(
            f"{row['cell_type']}: trial={row['trial_id']:03d}, "
            f"PCC8={row['displayed_pcc_8x8']:.4f}, "
            f"method_tads={row['method_tads_drawn']}, "
            f"target_tads={row['target_tads_drawn']}"
        )
    for path in output_files + [source_path, audit_path, provenance_path]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
