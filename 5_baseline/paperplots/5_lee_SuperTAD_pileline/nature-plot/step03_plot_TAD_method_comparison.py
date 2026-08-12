#!/usr/bin/env python3
"""
scHi-C imputation benchmark figure  (Genome Research submission format)
======================================================================
Figure 1
  A  cell types (rows) x methods (columns); lower triangle = method,
     upper triangle = target, SuperTAD domains overlaid, Pearson's r inset.
  B  estimation plot per cell type:
       top    violins of Pearson's r over N imputation trials
              (white dot = median, thick bar = IQR, thin line = range)
       bottom bootstrap mean difference in r versus the Raw baseline,
              with a 95% confidence interval.

Column order is exactly the order given to --methods (Raw first, ours last).
No null-hypothesis significance stars: with N trials under the author's own
control, any P value can be driven arbitrarily small (pseudoreplication).
Effect size + interval is reported instead, per Ho et al. 2019 Nat Methods.

Genome Research digital-art compliance
--------------------------------------
  * Arial/Helvetica; panel tags 12 pt bold; compact map annotations 6.2 pt
  * every rule >= 0.4 pt (GR floor is 0.25 pt)
  * RGB; no pale fills; CVD-safe; no red/green contrast
  * one accent colour (navy) marks our method; everything else is greyscale
  * any colour key sits in the figure body; titles and glyph
    definitions belong in the legend, not in the artwork
  * export: vector PDF (fonts embedded) + 600 dpi LZW TIFF (combination art)

Usage
-----
python plot_panel_grid.py \
    --methods Raw,scHiCluster,Higashi,scVI-3D,DeepLoop,SnapHiC,Query,scHiC-Diff \
    --highlight scHiC-Diff --baseline Raw \
    --norm q99 --tad supertad --paired
"""
from __future__ import annotations

import os
import io
import glob
import json
import sys
import argparse
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator
from scipy.sparse import load_npz
from scipy.stats import pearsonr, gaussian_kde

import TAD_method_comparison_config as cfg

# --------------------------------------------------------------------------
# 0. Constants
# --------------------------------------------------------------------------
MM = 1 / 25.4

FS_SMALL, FS_BASE, FS_HEAD = 8, 9, 10     # spread == 2 pt (GR rule)
FS_TAG = 12                                # panel tags, bold
MAP_CORNER_LABEL_SIZE = 6.2                # shared by PCC and Target
METHOD_LABEL_SIZE = 6.2                    # one label below each Panel A column
PANEL_AB_GAP_MM = 5.0                      # complete-figure A-to-B spacing
SHARED_COORDINATE_LABEL = "chr4:55.09–55.17 Mb"
SHARED_COORDINATE_LABEL_SIZE = 6.2
SHARED_COORDINATE_TEXT_Y_MM = 4.6
SHARED_COORDINATE_RULE_WIDTH_COLUMNS = 2
SHARED_COORDINATE_RULE_LABEL_GAP_MM = 1.5
SHARED_COORDINATE_RULE_LENGTH_FRACTION = 0.60
SHARED_COORDINATE_RULE_MAX_LENGTH_FRACTION = 2 / 3
SHARED_COORDINATE_RULE_COLOR = "#4A4A4A"
SHARED_COORDINATE_RULE_WIDTH_PT = 0.55

# Sequential, single-hue, CVD-safe. Warm half of RdBu; no pale yellow.
HIC_CMAP = LinearSegmentedColormap.from_list(
    "hic_red", ["#FFFFFF", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B", "#67001F"]
)

ACCENT = "#08306B"        # the single accent: our method (labels, spines, violins)
BASE_GREY = "#6E7377"     # the Raw baseline
OTHER_GREY = "#A9B0B5"    # competing methods (dark enough to survive print)
INK = "#1A1A1A"

# The heat map runs white -> #67001F, so NO single line colour is legible at
# both ends: navy scores 12.8:1 on white but 1.03:1 on the darkest red; cyan is
# the mirror image.  Every overlay is therefore drawn as a bright core inside a
# dark (or dark core inside a light) stroke -- the halo carries whichever end
# the core loses.  Dash pattern and lightness both differ between the two TAD
# sets, so they survive greyscale printing and colour-vision deficiency.
TAD_CORE, TAD_HALO = "#22D3EE", "#08306B"     # SuperTAD  : dashed, cyan on navy
REF_CORE, REF_HALO = "#FFFFFF", "#333333"     # reference : dotted, white on grey
DIAG_COLOR = "#666666"                          # diagonal  : one dashed rule

TARGET_TAD_CORE, TARGET_TAD_HALO = REF_CORE, REF_HALO

TAD_DASH = (0, (2.2, 1.4))
REF_DOT = (0, (0.8, 1.3))

# Cell-type colours shared with the companion Lee-data figures. Used only for
# cell-type labels, never for data marks, so the method accent stays distinct.
CT_COLORS = {
    "Astro": "#3F6FAE",
    "Endo": "#D6A33B",
    "ODC": "#8A6BB1",
    "OPC": "#3C927D",
}

METHOD_DISPLAY_NAMES = {
    "Raw": "Raw",
    "scHiCluster": "scHiCluster",
    "Higashi-nbr0": "Higashi-nbr0",
    "Higashi-nbr5": "Higashi-nbr5",
    "scVI-3D": "scVI-3D",
    "HiCImpute": "HiCImpute",
    "T-FLAMINGO": "T-FLAMINGO",
    "scHiC-Diff": "scHiC-Diff",
}

# Published (Lee et al.) reference TAD skeleton, PDGFRA sub-window bin coords
LEE_TADS = {
    "Astro": [[(0, 2), (1, 2)], [(1, 2), (1, 4)], [(1, 4), (3, 4)], [(3, 4), (3, 8)]],
    "Endo":  [[(0, 2), (1, 2)], [(1, 2), (1, 3)], [(1, 3), (2, 3)], [(2, 3), (2, 4)],
              [(2, 4), (3, 4)], [(3, 4), (3, 6)], [(3, 6), (5, 6)], [(5, 6), (5, 7)],
              [(5, 7), (6, 7)], [(6, 7), (6, 8)]],
    "ODC":   [[(0, 2), (1, 2)], [(1, 2), (1, 3)], [(1, 3), (2, 3)], [(2, 3), (2, 4)],
              [(2, 4), (3, 4)], [(3, 4), (3, 7)], [(3, 7), (6, 7)], [(6, 7), (6, 8)]],
    "OPC":   [[(0, 4), (3, 4)], [(3, 4), (3, 5)], [(3, 5), (4, 5)],
              [(4, 5), (4, 7)], [(4, 7), (6, 7)], [(6, 7), (6, 8)]],
}


# --------------------------------------------------------------------------
# 1. Style
# --------------------------------------------------------------------------
def set_gr_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": FS_BASE,
        "axes.labelsize": FS_BASE,
        "axes.titlesize": FS_HEAD,
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        "legend.fontsize": FS_SMALL,
        "legend.frameon": False,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,          # embed real fonts, editable in Illustrator
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": None,        # absolute mm layout; never crop
        "figure.dpi": 200,
    })


def role_color(method: str, highlight: str, baseline: str) -> str:
    if method == highlight:
        return ACCENT
    if method == baseline:
        return BASE_GREY
    return OTHER_GREY


# --------------------------------------------------------------------------
# 2. Canonical compact IO helpers
# --------------------------------------------------------------------------
def load_target(cell_type: str, target_root=None) -> np.ndarray:
    root = Path(target_root or cfg.TARGET_ROOT)
    return load_npz(str(root / f"{cell_type}_target.npz")).toarray()


def representative_matrix_path(method, cell_type, trial_id,
                               representative_root=None):
    root = Path(representative_root or cfg.REPRESENTATIVE_MATRIX_ROOT)
    return root / method / f"{cell_type}_trial{trial_id:03d}.npz"


def load_trial_matrix(method: str, cell_type: str, trial_id: int,
                      representative_root=None) -> np.ndarray:
    path = representative_matrix_path(
        method, cell_type, trial_id, representative_root
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return load_npz(str(path)).toarray()


def load_pcc_table(method: str, pcc_root=None):
    root = Path(pcc_root or cfg.PCC_RESULTS_ROOT)
    path = root / method / f"{method}_PCC_trials.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    table["method"] = method
    return table


def load_trial_metadata(method, pcc_root=None):
    root = Path(pcc_root or cfg.PCC_RESULTS_ROOT)
    path = root / method / f"{method}_PCC_calculation_information.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        metadata = json.load(handle)
    metadata["metadata_path"] = path.as_posix()
    return metadata


def find_median_trial(df, cell_type, pcc_col="pcc_8x8_full"):
    sub = df[df.cell_type == cell_type].sort_values(pcc_col)
    if len(sub) == 0:
        return 0, np.nan
    row = sub.iloc[len(sub) // 2]
    return int(row.trial_id), float(row[pcc_col])


def parse_tad_tsv(tsv_path, report=None):
    """Read SuperTAD domains as (start_bin, end_bin) from columns 1 and 5.

    Rows with fewer than 8 tab fields are skipped -- that silent skip is the
    single most common reason a method ends up with no boundaries, so when
    `report` (a dict) is supplied we record exactly what was seen.
    """
    tads, n_lines, n_bad = [], 0, 0
    if not tsv_path or not os.path.exists(tsv_path):
        if report is not None:
            report.update(found=False, lines=0, parsed=0, skipped=0)
        return tads
    with open(tsv_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            n_lines += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                n_bad += 1
                continue
            try:
                start_bin = int(parts[1]) - 1
                end_bin = int(parts[5]) - 1
                if end_bin < start_bin:
                    n_bad += 1
                    continue
                tads.append((start_bin, end_bin))
            except ValueError:
                n_bad += 1
    if report is not None:
        report.update(found=True, lines=n_lines, parsed=len(tads), skipped=n_bad)
    return tads


TAD_AUDIT = []            # filled while panel A is drawn, printed afterwards


def select_deepest_non_singletons(tads):
    """Return leaf domains in the non-singleton containment hierarchy."""
    domains = sorted({(int(a), int(b)) for a, b in tads if int(b) > int(a)})
    return [
        domain
        for domain in domains
        if not any(
            domain != child
            and domain[0] <= child[0]
            and child[1] <= domain[1]
            for child in domains
        )
    ]


def map_tads_to_window(tads, s, e, report=None):
    """Translate intersecting domains into local coordinates without clamping.

    Keeping the original endpoints lets the axes clip the true line segments.
    It avoids creating a synthetic TAD edge at either side of the window.
    """
    visible = []
    outside = 0
    for a, b in tads:
        a, b = int(a), int(b)
        if b < s or a >= e:
            outside += 1
            continue
        visible.append((a - s, b - s))
    if report is not None:
        report.update(visible=len(visible), outside_window=outside)
    return visible


def prepare_tads_for_window(tads, s, e, report=None):
    selected = select_deepest_non_singletons(tads)
    if report is not None:
        report.update(non_singleton=len({d for d in tads if d[1] > d[0]}),
                      selected_depth=len(selected))
    return map_tads_to_window(selected, s, e, report)


def clip_tads(tads, s, e, report=None):
    """Compatibility wrapper; endpoints are mapped, never clipped."""
    return prepare_tads_for_window(tads, s, e, report)


def build_TAD_boundary_plot_check(audit, representatives, target_root=None):
    """Attach the matching upper/lower heatmap source to each TAD audit row."""
    target_root = Path(target_root or cfg.TARGET_ROOT)
    representative_lookup = {
        (row["method"], row["cell_type"]): row for row in representatives
    }
    rows = []
    for record in audit:
        row = dict(record)
        if str(row["method"]).lower() == "target":
            row["heatmap_half"] = "upper"
            row["domain_role"] = "Target SuperTAD"
            row["matrix_path"] = (
                target_root / f"{row['cell_type']}_target.npz"
            ).as_posix()
        else:
            selected = representative_lookup[(row["method"], row["cell_type"])]
            row["heatmap_half"] = "lower"
            row["domain_role"] = "Method SuperTAD"
            row["matrix_path"] = selected["matrix_path"]
        rows.append(row)
    return pd.DataFrame(rows)


def find_tad_tsv(method: str, cell_type: str, trial_id: int,
                 supertad_results_root=None):
    """Resolve one canonical Target or representative SuperTAD TSV."""
    root = Path(supertad_results_root or cfg.SUPERTAD_DOMAIN_ROOT)
    if method.lower() == "target":
        path = root / "target" / f"{cell_type}_target.tsv"
    else:
        path = (
            root / "representatives" / method
            / f"{cell_type}_trial{trial_id:03d}.tsv"
        )
    return path if path.is_file() else None


def resolve_panel_a_sources(method, cell_type, pcc_table,
                            pcc_col="pcc_8x8_full",
                            representative_root=None,
                            supertad_results_root=None):
    """Pair the lower heatmap and lower TAD with one representative trial."""
    trial_id, r_value = find_median_trial(pcc_table, cell_type, pcc_col)
    matrix_path = representative_matrix_path(
        method, cell_type, trial_id, representative_root
    )
    target_tad = find_tad_tsv(
        "target", cell_type, 0, supertad_results_root
    )
    method_tad = find_tad_tsv(
        method, cell_type, trial_id, supertad_results_root
    )
    missing = [
        path for path in (matrix_path, target_tad, method_tad)
        if path is None or not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing canonical Panel A source(s): "
            + ", ".join(str(path) for path in missing)
        )
    return {
        "upper": {
            "source": "target",
            "trial_id": None,
            "tad_path": target_tad,
        },
        "lower": {
            "source": method,
            "trial_id": trial_id,
            "r_value": r_value,
            "matrix_path": matrix_path,
            "tad_path": method_tad,
        },
    }


# --------------------------------------------------------------------------
# 3. Numerics
# --------------------------------------------------------------------------
def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.copy(), b.copy()
    np.fill_diagonal(a, 0)
    np.fill_diagonal(b, 0)
    x, y = a.flatten(), b.flatten()
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(pearsonr(x, y)[0])


def normalize(mat: np.ndarray, mode: str, ref=None) -> np.ndarray:
    """Put every method on a comparable 0-1 scale before colour mapping.

    q99    -- divide by the matrix's own 99th off-diagonal percentile, so a
              sparse Raw map and a dense imputed map are judged on STRUCTURE.
    shared -- scale everything by the TARGET's 99th percentile, preserving the
              absolute-magnitude gap (use to emphasise Raw under-sampling).
    Whichever is chosen must be stated in the legend.
    """
    m = mat.astype(float).copy()
    np.fill_diagonal(m, 0)
    src = m if (mode == "q99" or ref is None) else np.asarray(ref, float)
    off = src[~np.eye(src.shape[0], dtype=bool)]
    hi = np.percentile(off, 99) if off.size and np.any(off > 0) else 1.0
    return np.clip(m / (hi if hi > 0 else 1.0), 0, 1)


def bootstrap_mean_diff(a, b, n_boot=10000, paired=False, rng=None):
    """Δ = mean(a) - mean(b), with a percentile bootstrap 95% CI. b = baseline.

    paired=True requires a and b to be aligned trial-by-trial (same seed /
    same down-sampled cells). It is the correct choice when run_trials.py
    reuses one seed set across methods, and gives a much tighter interval.
    """
    rng = rng or np.random.default_rng(0)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    if paired:
        if len(a) != len(b):
            raise ValueError("paired bootstrap needs equal-length, aligned arrays")
        d = a - b
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot = d[idx].mean(axis=1)
    else:
        ia = rng.integers(0, len(a), size=(n_boot, len(a)))
        ib = rng.integers(0, len(b), size=(n_boot, len(b)))
        boot = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs, lo, hi, boot


def celltype_values(long_df, cell_type, methods, pcc_col, paired):
    """dict {method: 1-D array}. If paired, align every method on trial_id."""
    sub = long_df[long_df.cell_type == cell_type]
    if not paired:
        return {m: sub[sub.method == m][pcc_col].values for m in methods}

    wide = sub.pivot_table(index="trial_id", columns="method", values=pcc_col)
    missing = [m for m in methods if m not in wide.columns]
    if missing:
        raise ValueError(f"paired mode: {missing} absent for {cell_type}")
    wide = wide[methods].dropna(axis=0, how="any")
    if wide.empty:
        raise ValueError(f"paired mode: no shared trial_id across methods "
                         f"for {cell_type}; rerun with --no-paired")
    return {m: wide[m].values for m in methods}


# --------------------------------------------------------------------------
# 4. Panel A primitives
# --------------------------------------------------------------------------
def _rect(x_mm, ytop_mm, w_mm, h_mm, W_mm, H_mm):
    """mm-from-top-left -> matplotlib figure-fraction rect."""
    return [x_mm / W_mm, 1 - (ytop_mm + h_mm) / H_mm, w_mm / W_mm, h_mm / H_mm]


def _halo(core, halo, lw):
    """Bright core inside a dark stroke (or vice versa) so the line stays
    legible over both the white and the near-black end of the colour map."""
    return [pe.Stroke(linewidth=lw + 0.7, foreground=halo), pe.Normal()]


def draw_tad_triangle(ax, s, e, side, lw=0.7, zorder=4,
                      core=TAD_CORE, halo=TAD_HALO, linestyle=TAD_DASH):
    """Two legs of the SuperTAD domain triangle [s, e] (imshow, origin='upper')."""
    lo, hi = s - 0.5, e + 0.5
    if side == "upper":
        pts = [((lo, hi), (lo, lo)), ((hi, hi), (lo, hi))]
    else:
        pts = [((lo, lo), (lo, hi)), ((lo, hi), (hi, hi))]
    for (x1, x2), (y1, y2) in pts:
        ax.add_line(Line2D([x1, x2], [y1, y2], color=core, lw=lw,
                           ls=linestyle, dash_capstyle="butt", zorder=zorder,
                           path_effects=_halo(core, halo, lw)))


def draw_lee_reference(ax, cell_type, lw=0.7):
    for (x1, y1), (x2, y2) in LEE_TADS.get(cell_type, []):
        for xs, ys in (([x1 - .5, x2 - .5], [y1 - .5, y2 - .5]),
                       ([y1 - .5, y2 - .5], [x1 - .5, x2 - .5])):
            ax.add_line(Line2D(xs, ys, color=REF_CORE, ls=REF_DOT, lw=lw,
                               zorder=3,
                               path_effects=_halo(REF_CORE, REF_HALO, lw)))


def draw_map(ax, method_mat, target_mat, cell_type, r_value,
             tads_lower=None, tads_upper=None, tad_mode="supertad",
             highlight=False):
    n = method_mat.shape[0]
    tri_lo = np.tril(np.ones((n, n), bool), k=0)
    tri_up = np.triu(np.ones((n, n), bool), k=1)

    canvas = np.full((n, n), np.nan)
    canvas[tri_lo] = method_mat[tri_lo]
    canvas[tri_up] = target_mat[tri_up]
    ax.imshow(canvas, cmap=HIC_CMAP, vmin=0, vmax=1,
              origin="upper", interpolation="nearest", rasterized=True)

    # A domain whose boundary sits at the first or last bin has legs at exactly
    # -0.5 / n-0.5, i.e. underneath the axes frame. Pad the view so it shows.
    pad = 0.22
    ax.set_xlim(-0.5 - pad, n - 0.5 + pad)
    ax.set_ylim(n - 0.5 + pad, -0.5 - pad)

    ax.add_line(Line2D([-.5, n - .5], [-.5, n - .5], color=DIAG_COLOR,
                       ls=(0, (3, 2)), lw=0.55, zorder=5))

    if tad_mode in ("supertad", "both"):
        for s, e in (tads_lower or []):
            draw_tad_triangle(ax, s, e, "lower")
        for s, e in (tads_upper or []):
            draw_tad_triangle(ax, s, e, "upper",
                              core=TARGET_TAD_CORE, halo=TARGET_TAD_HALO,
                              linestyle=REF_DOT)
    if tad_mode in ("lee", "both"):
        draw_lee_reference(ax, cell_type)

    if np.isfinite(r_value):
        pcc_label = ax.text(
            0.04,
            0.04,
            f"{r_value:.3f}",
            transform=ax.transAxes,
            fontsize=MAP_CORNER_LABEL_SIZE,
            fontweight="normal",
            ha="left",
            va="bottom",
            color=INK,
            bbox=dict(
                boxstyle="square,pad=0.12",
                facecolor="white",
                edgecolor="none",
                alpha=0.78,
            ),
            zorder=6,
        )
        pcc_label.set_gid("panel_A_pcc_label")

    target_label = ax.text(
        0.96,
        0.96,
        "Target",
        transform=ax.transAxes,
        fontsize=MAP_CORNER_LABEL_SIZE,
        fontweight="normal",
        ha="right",
        va="top",
        color="#3F3F3F",
        bbox=dict(
            boxstyle="square,pad=0.10",
            facecolor="white",
            edgecolor="none",
            alpha=0.72,
        ),
        zorder=7,
    )
    target_label.set_gid("panel_A_target_label")

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0 if highlight else 0.5)
        sp.set_edgecolor(ACCENT if highlight else "#333333")
    ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------------------
# 5. Panel A
# --------------------------------------------------------------------------
def panel_grid(fig, geo, methods, pcc_tables, highlight, norm_mode, tad_mode,
               pcc_col="pcc_8x8_full", colorbar="none", audit=None,
               representative_trials=None):
    """Rows = cell types, columns = methods. Absolute mm placement, so the
    square heat maps tile exactly with no aspect-ratio gutters."""
    W, H = geo["W"], geo["H"]
    x0, y0 = geo["x0"], geo["A_top"] + geo["HEAD"]
    cw, gap = geo["cell"], geo["gap"]

    s, e = cfg.PDGFRA_SUB_BINS
    cts = list(cfg.CELL_TYPES)
    targets = {ct: load_target(ct) for ct in cts}
    im_handle = None
    audit = TAD_AUDIT if audit is None else audit
    representative_trials = ([] if representative_trials is None
                             else representative_trials)

    for i, ct in enumerate(cts):
        tgt_sub = targets[ct][s:e, s:e]
        tgt_norm = normalize(tgt_sub, norm_mode, ref=tgt_sub)
        rep = {}
        target_tad_path = find_tad_tsv("target", ct, 0)
        tads_up = prepare_tads_for_window(
            parse_tad_tsv(target_tad_path, rep), s, e, rep
        )
        audit.append(dict(method="target", cell_type=ct,
                          tad_path=target_tad_path, **rep))

        for j, m in enumerate(methods):
            ax = fig.add_axes(_rect(x0 + j * (cw + gap), y0 + i * (cw + gap),
                                    cw, cw, W, H))
            df = pcc_tables.get(m)
            if df is None:
                ax.set_axis_off()
                continue
            sources = resolve_panel_a_sources(m, ct, df, pcc_col)
            tid = sources["lower"]["trial_id"]
            mat_sub = load_npz(sources["lower"]["matrix_path"]).toarray()[s:e, s:e]
            mat_norm = normalize(mat_sub, norm_mode, ref=tgt_sub)
            rep = {}
            tads_lo = prepare_tads_for_window(
                parse_tad_tsv(sources["lower"]["tad_path"], rep), s, e, rep
            )
            audit.append(dict(method=m, cell_type=ct, trial=tid,
                              tad_path=sources["lower"]["tad_path"], **rep))
            representative_trials.append(dict(
                method=m, cell_type=ct, trial_id=tid,
                pcc_8x8_full=sources["lower"]["r_value"],
                matrix_path=sources["lower"]["matrix_path"],
                method_tad_path=sources["lower"]["tad_path"],
                target_tad_path=sources["upper"]["tad_path"],
            ))

            draw_map(ax, mat_norm, tgt_norm, ct, pcc(mat_sub, tgt_sub),
                     tads_lo, tads_up, tad_mode, highlight=(m == highlight))
            im_handle = ax.images[0]

            if j == 0:
                ax.set_ylabel(ct, fontsize=FS_HEAD, labelpad=3,
                              color=CT_COLORS.get(ct, INK))

    grid_h = len(cts) * cw + (len(cts) - 1) * gap
    grid_w = len(methods) * cw + (len(methods) - 1) * gap
    strip_y = y0 + grid_h + 1.0

    # A single compact method label below each column pairs the lower triangle
    # with its method while the upper triangle is identified inside every map.
    for column, method in enumerate(methods):
        label = fig.text(
            (x0 + column * (cw + gap) + cw / 2) / W,
            1 - strip_y / H,
            METHOD_DISPLAY_NAMES.get(method, method),
            fontsize=METHOD_LABEL_SIZE,
            fontweight="normal",
            ha="center",
            va="top",
            color=ACCENT if method == highlight else INK,
        )
        label.set_gid("panel_A_method_label")

    # ---- optional colour key.  GR only requires that a key, IF PRESENT, sits
    # in the figure body rather than the legend; with per-matrix q99 scaling the
    # bar carries no absolute units, so 'none' is defensible.  'inline' puts a
    # slim horizontal Low->High gradient in the strip below the grid.
    if colorbar == "right":
        cb_h = min(24.0, grid_h * 0.42)
        cax = fig.add_axes(_rect(x0 + grid_w + 2.5, y0 + (grid_h - cb_h) / 2,
                                 2.4, cb_h, W, H))
        cb = fig.colorbar(im_handle, cax=cax, ticks=[0, 1])
        cb.ax.set_yticklabels(["Low", "High"], fontsize=FS_SMALL)
        cb.ax.tick_params(width=0.4, length=1.6, pad=1.5)
        cb.outline.set_linewidth(0.4)
        cb.set_label("Contact frequency", fontsize=FS_SMALL, labelpad=2)
    elif colorbar == "inline":
        cax = fig.add_axes(_rect(x0 + grid_w - 26.0, strip_y + 3.4,
                                 16.0, 1.6, W, H))
        cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=HIC_CMAP)
        cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4); sp.set_edgecolor("#5A5A5A")
        cax.text(-0.06, 0.5, "Low", transform=cax.transAxes, fontsize=FS_SMALL,
                 ha="right", va="center", color="#4D4D4D")
        cax.text(1.06, 0.5, "High", transform=cax.transAxes, fontsize=FS_SMALL,
                 ha="left", va="center", color="#4D4D4D")

    return strip_y + 5.6         # bottom of panel A, in mm


# --------------------------------------------------------------------------
# 6. Panel B  (estimation plot: violins + bootstrap mean difference)
# --------------------------------------------------------------------------
def _draw_violins(ax, methods, data, highlight, baseline, ylim):
    pos = np.arange(len(methods))
    parts = ax.violinplot([data[m] for m in methods], positions=pos,
                          widths=0.78, showmeans=False, showmedians=False,
                          showextrema=False)
    for m, body in zip(methods, parts["bodies"]):
        hero = (m == highlight)
        body.set_facecolor(role_color(m, highlight, baseline))
        body.set_edgecolor("#2B2B2B" if hero else "#5A5A5A")
        body.set_alpha(0.95 if hero else 0.75)
        body.set_linewidth(0.9 if hero else 0.45)
        body.set_zorder(2)

    if baseline in data:                       # reference line at Raw median
        ax.axhline(np.median(data[baseline]), color=BASE_GREY, lw=0.6,
                   ls=(0, (4, 3)), zorder=1)

    for i, m in enumerate(methods):            # median dot / IQR / range
        v = data[m]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.vlines(i, v.min(), v.max(), color=INK, lw=0.5, zorder=4)
        ax.vlines(i, q1, q3, color=INK, lw=2.2, zorder=4)
        ax.scatter(i, med, s=7, color="white", edgecolor=INK,
                   linewidth=0.5, zorder=5)

    ax.set_ylim(*ylim)
    ax.set_xlim(-0.7, len(methods) - 0.3)
    ax.set_xticks(pos)
    ax.tick_params(axis="x", length=0, labelbottom=False)


def _draw_delta(ax, methods, res, highlight, baseline, ylim, annotate):
    ax.axhline(0, color=BASE_GREY, lw=0.6, ls=(0, (4, 3)), zorder=1)
    for i, m in enumerate(methods):
        obs, lo, hi, boot = res[m]
        c = role_color(m, highlight, baseline)
        hero = (m == highlight)
        if boot is None:                        # the baseline sits at Δ = 0
            ax.scatter(i, 0.0, s=9, color=BASE_GREY, zorder=4)
            continue
        kde = gaussian_kde(boot)
        ys = np.linspace(boot.min(), boot.max(), 120)
        xs = kde(ys); xs = xs / xs.max() * 0.40
        ax.fill_betweenx(ys, i, i + xs, color=c,
                         alpha=0.90 if hero else 0.55, lw=0, zorder=2)
        ax.vlines(i, lo, hi, color=INK, lw=1.0, zorder=4)
        ax.scatter(i, obs, s=12 if hero else 8, color=c,
                   edgecolor=INK, linewidth=0.5, zorder=5)
        if annotate and hero:
            ax.text(i, hi + (ylim[1] - ylim[0]) * 0.05, f"{obs:+.2f}",
                    ha="center", va="bottom", fontsize=FS_SMALL,
                    color=ACCENT, fontweight="bold", zorder=6)

    ax.set_ylim(*ylim)
    ax.set_xlim(-0.7, len(methods) - 0.3)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=90, ha="center", fontsize=FS_SMALL)
    for tl, m in zip(ax.get_xticklabels(), methods):
        if m == highlight:
            tl.set_color(ACCENT); tl.set_fontweight("bold")


def panel_estimation(fig, geo, methods, long_df, highlight, baseline,
                     pcc_col, paired, n_boot, y_top):
    """Per cell type: violins of r (top) over the bootstrap Δr vs Raw (bottom)."""
    W, H = geo["W"], geo["H"]
    cts = list(cfg.CELL_TYPES)
    n_ct = len(cts)

    L2, R2, gap_f = 17.0, 3.0, 6.0
    fw = (W - L2 - R2 - (n_ct - 1) * gap_f) / n_ct
    h_v, h_gap, h_d = geo["h_v"], 1.6, geo["h_d"]

    # ---- gather data & effect sizes once, so the axes can share limits ----
    data = {ct: celltype_values(long_df, ct, methods, pcc_col, paired)
            for ct in cts}
    rng = np.random.default_rng(1)
    res = {}
    for ct in cts:
        d = data[ct]
        res[ct] = {m: (0.0, 0.0, 0.0, None) if m == baseline else
                   bootstrap_mean_diff(d[m], d[baseline], n_boot=n_boot,
                                       paired=paired, rng=rng)
                   for m in methods}

    allv = np.concatenate([data[ct][m] for ct in cts for m in methods])
    v_lo = max(0.0, np.floor((allv.min() - 0.02) * 10) / 10)
    ylim_v = (v_lo, 1.0)

    los = [lo for ct in cts for _, lo, _, b in res[ct].values() if b is not None]
    his = [hi for ct in cts for _, _, hi, b in res[ct].values() if b is not None]
    d_lo, d_hi = min(0.0, min(los)), max(his)
    padd = (d_hi - d_lo) * 0.18
    ylim_d = (d_lo - padd * 0.4, d_hi + padd)

    for j, ct in enumerate(cts):
        x = L2 + j * (fw + gap_f)
        ax_v = fig.add_axes(_rect(x, y_top + 4.5, fw, h_v, W, H))
        ax_d = fig.add_axes(_rect(x, y_top + 4.5 + h_v + h_gap, fw, h_d, W, H))

        _draw_violins(ax_v, methods, data[ct], highlight, baseline, ylim_v)
        _draw_delta(ax_d, methods, res[ct], highlight, baseline, ylim_d,
                    annotate=True)

        ax_v.set_title(ct, fontsize=FS_HEAD, pad=3,
                       color=CT_COLORS.get(ct, INK))
        ax_v.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5]))
        ax_d.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5]))
        if j == 0:
            ax_v.set_ylabel("Pearson's $r$", fontsize=FS_BASE, labelpad=2)
            ax_d.set_ylabel("$\\Delta r$ vs " + baseline, fontsize=FS_BASE,
                            labelpad=2)
        else:
            ax_v.tick_params(labelleft=False)
            ax_d.tick_params(labelleft=False)
        for a in (ax_v, ax_d):
            a.tick_params(width=0.6, length=2.2)


# --------------------------------------------------------------------------
# 7. Export (GR: vector PDF + 600 dpi LZW TIFF for combination art)
# --------------------------------------------------------------------------
def save_all(fig, stem: str, dpi: int = 600):
    out = []
    for ext in ("pdf", "png"):
        p = f"{stem}.{ext}"
        tmp = f"{p}.tmp"
        fig.savefig(tmp, format=ext, dpi=dpi, facecolor="white")
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            raise RuntimeError(f"empty figure export: {tmp}")
        os.replace(tmp, p)
        out.append(p)
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
    buf.seek(0)
    p = f"{stem}.tif"
    tmp = f"{p}.tmp"
    Image.open(buf).convert("RGB").save(
        tmp, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        raise RuntimeError(f"empty figure export: {tmp}")
    os.replace(tmp, p)
    out.append(p)
    return out


def build_export_manifest(outdir):
    stems = (
        "TAD_method_comparison",
        "panel_A_TAD_heatmaps",
        "panel_B_PCC_violin_plots",
    )
    return [os.path.join(str(outdir), f"{stem}.{ext}")
            for stem in stems for ext in ("pdf", "png", "tif")]


def compute_layout(methods, width_mm=170.0, colorbar="none"):
    W = float(width_mm)
    ncol, nrow = len(methods), len(cfg.CELL_TYPES)
    L, HEAD, gap = 13.0, 6.0, 1.1
    R = 17.0 if colorbar == "right" else 4.0
    avail = W - L - R
    cell = min((avail - (ncol - 1) * gap) / ncol, 24.0)
    grid_w = ncol * cell + (ncol - 1) * gap
    x0 = L + max((avail - grid_w) / 2.0, 0.0)
    grid_h = nrow * cell + (nrow - 1) * gap
    h_A = 1.0 + HEAD + grid_h + 6.6
    h_v, h_d = 32.0, 15.0
    lab_h = 2.0 + 0.55 * max(len(m) for m in methods) * (FS_SMALL / 8.0) * 2.2
    h_B = 4.5 + h_v + 1.6 + h_d + lab_h
    return dict(W=W, L=L, R=R, HEAD=HEAD, gap=gap, cell=cell,
                grid_w=grid_w, grid_h=grid_h, x0=x0, h_A=h_A,
                h_v=h_v, h_d=h_d, h_B=h_B)


def add_shared_coordinate_annotation(figure, methods):
    """Centre the shared genomic range between balanced Panel A side rules."""
    layout = compute_layout(methods)
    width_mm = layout["W"]
    height_mm = figure.get_figheight() / MM
    if len(methods) < SHARED_COORDINATE_RULE_WIDTH_COLUMNS:
        raise ValueError("shared-coordinate rule requires at least two methods")

    first_column = (len(methods) - SHARED_COORDINATE_RULE_WIDTH_COLUMNS) // 2
    last_column = first_column + SHARED_COORDINATE_RULE_WIDTH_COLUMNS - 1
    anchor_left_mm = layout["x0"] + first_column * (
        layout["cell"] + layout["gap"]
    )
    anchor_right_mm = (
        layout["x0"]
        + last_column * (layout["cell"] + layout["gap"])
        + layout["cell"]
    )
    center_x = (anchor_left_mm + anchor_right_mm) / (2 * width_mm)
    label_y = 1 - SHARED_COORDINATE_TEXT_Y_MM / height_mm

    label = figure.text(
        center_x,
        label_y,
        SHARED_COORDINATE_LABEL,
        fontsize=SHARED_COORDINATE_LABEL_SIZE,
        fontweight="normal",
        color=INK,
        ha="center",
        va="center",
    )
    label.set_gid("panel_A_shared_coordinate_label")

    figure.canvas.draw()
    label_bbox = label.get_window_extent(renderer=figure.canvas.get_renderer())
    label_left, _ = figure.transFigure.inverted().transform(
        (label_bbox.x0, label_bbox.y0)
    )
    label_right, _ = figure.transFigure.inverted().transform(
        (label_bbox.x1, label_bbox.y1)
    )
    label_gap = SHARED_COORDINATE_RULE_LABEL_GAP_MM / width_mm
    left_inner = label_left - label_gap
    right_inner = label_right + label_gap
    if (
        SHARED_COORDINATE_RULE_LENGTH_FRACTION
        > SHARED_COORDINATE_RULE_MAX_LENGTH_FRACTION
    ):
        raise ValueError("shared-coordinate side rules exceed the approved maximum")
    rule_length = (
        SHARED_COORDINATE_RULE_LENGTH_FRACTION
        * (label_right - label_left)
    )
    left_outer = left_inner - rule_length
    right_outer = right_inner + rule_length
    if left_outer <= 0 or right_outer >= 1:
        raise ValueError("shared-coordinate side rules extend beyond the figure")

    for gid, xdata in (
        ("panel_A_shared_coordinate_left_rule", [left_outer, left_inner]),
        ("panel_A_shared_coordinate_right_rule", [right_inner, right_outer]),
    ):
        rule = Line2D(
            xdata,
            [label_y, label_y],
            transform=figure.transFigure,
            color=SHARED_COORDINATE_RULE_COLOR,
            linewidth=SHARED_COORDINATE_RULE_WIDTH_PT,
            solid_capstyle="butt",
            clip_on=False,
            zorder=label.get_zorder() - 1,
        )
        rule.set_gid(gid)
        figure.add_artist(rule)


def add_panel_tag(fig, letter, x_mm, ytop_mm, W, H):
    fig.text(max(x_mm - 9.0, 0.5) / W, 1 - ytop_mm / H, letter,
             fontsize=FS_TAG, fontweight="bold", va="top", ha="left")


def build_full_figure(methods, pcc_tables, long_df, args):
    layout = compute_layout(methods, args.width_mm, args.colorbar)
    W = layout["W"]
    H = layout["h_A"] + PANEL_AB_GAP_MM + layout["h_B"] + 3.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    geo = dict(W=W, H=H, L=layout["L"], R=layout["R"],
               HEAD=layout["HEAD"], cell=layout["cell"], gap=layout["gap"],
               A_top=1.0, x0=layout["x0"], h_v=layout["h_v"],
               h_d=layout["h_d"])
    audit, representatives = [], []
    a_bottom = panel_grid(
        fig, geo, methods, pcc_tables, args.highlight, args.norm, args.tad,
        args.pcc_col, args.colorbar, audit=audit,
        representative_trials=representatives,
    )
    add_shared_coordinate_annotation(fig, methods)
    add_panel_tag(fig, "A", layout["L"], 0.5, W, H)
    b_top = a_bottom + PANEL_AB_GAP_MM
    panel_estimation(fig, geo, methods, long_df, args.highlight, args.baseline,
                     args.pcc_col, args.paired, args.n_boot, b_top)
    add_panel_tag(fig, "B", layout["L"], b_top, W, H)
    return fig, audit, representatives


def build_panel_a_figure(methods, pcc_tables, args, include_panel_tag=False):
    layout = compute_layout(methods, args.width_mm, args.colorbar)
    W, H = layout["W"], layout["h_A"] + 1.5
    fig = plt.figure(figsize=(W * MM, H * MM))
    geo = dict(W=W, H=H, L=layout["L"], R=layout["R"],
               HEAD=layout["HEAD"], cell=layout["cell"], gap=layout["gap"],
               A_top=1.0, x0=layout["x0"], h_v=layout["h_v"],
               h_d=layout["h_d"])
    panel_grid(fig, geo, methods, pcc_tables, args.highlight, args.norm,
               args.tad, args.pcc_col, args.colorbar, audit=[],
               representative_trials=[])
    add_shared_coordinate_annotation(fig, methods)
    if include_panel_tag:
        add_panel_tag(fig, "A", layout["L"], 0.5, W, H)
    return fig


def build_panel_b_figure(methods, long_df, args, include_panel_tag=False):
    layout = compute_layout(methods, args.width_mm, args.colorbar)
    W, H = layout["W"], 1.0 + layout["h_B"] + 3.0
    fig = plt.figure(figsize=(W * MM, H * MM))
    geo = dict(W=W, H=H, h_v=layout["h_v"], h_d=layout["h_d"])
    panel_estimation(fig, geo, methods, long_df, args.highlight, args.baseline,
                     args.pcc_col, args.paired, args.n_boot, 1.0)
    if include_panel_tag:
        add_panel_tag(fig, "B", layout["L"], 0.5, W, H)
    return fig


def build_arg_parser():
    ap = argparse.ArgumentParser(description="scHi-C benchmark figure (GR format)")
    ap.add_argument(
        "--methods",
        default=",".join(cfg.FIGURE_METHOD_ORDER),
        help="comma-separated; this is the panel and violin order",
    )
    ap.add_argument("--highlight", default="scHiC-Diff")
    ap.add_argument("--baseline", default="Raw")
    ap.add_argument("--norm", choices=["q99", "shared"], default="q99")
    ap.add_argument("--tad", choices=["none", "lee", "supertad", "both"],
                    default="supertad")
    ap.add_argument("--pcc-col", default="pcc_8x8_full")
    ap.add_argument("--colorbar", choices=["none", "inline", "right"],
                    default="none",
                    help="'none' omits the colour key; 'inline' places it under "
                         "the grid; 'right' uses a vertical bar")
    ap.add_argument("--paired", dest="paired", action="store_true", default=False,
                    help="use only with verified cell-level trial pairing")
    ap.add_argument("--no-paired", dest="paired", action="store_false",
                    help="unpaired bootstrap (corrected default)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--width-mm", type=float, default=170.0,
                    help="170 = GR double column, 85 = single column")
    ap.add_argument("--dpi", type=int, default=600,
                    help="raster dpi; GR wants 600-900 for combination art")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-standalone-panels", dest="standalone_panels",
                    action="store_false", default=True)
    return ap


def _portable(path):
    return Path(os.path.relpath(Path(path).resolve(), Path.cwd().resolve())).as_posix()


def build_provenance(methods, trial_metadata, args):
    """Build relative-path-only provenance for the canonical compact project."""
    return {
        "command": ["python", "step03_plot_TAD_method_comparison.py"],
        "methods": list(methods),
        "trial_input_methods": {
            method: trial_metadata[method]["input_key"] for method in methods
        },
        "paths": {
            "target_root": str(cfg.TARGET_ROOT),
            "pcc_root": str(cfg.PCC_RESULTS_ROOT),
            "representative_root": str(cfg.REPRESENTATIVE_MATRIX_ROOT),
            "supertad_domain_root": str(cfg.SUPERTAD_DOMAIN_ROOT),
            "figure_root": _portable(args.outdir or cfg.FIGURE_ROOT),
            "result_root": str(cfg.RESULTS_ROOT),
        },
        "tad_coordinate_system": "SuperTAD 1-based inclusive -> 0-based inclusive",
        "tad_hierarchy": "deepest non-singleton",
        "tad_window_mapping": "translate without endpoint clamping",
        "bootstrap_mode": "paired" if args.paired else "unpaired",
        "normalization": args.norm,
        "method_order_scope": "figure_only",
        "standalone_panel_tags": False,
        "panel_ab_gap_mm": PANEL_AB_GAP_MM,
        "panel_a_genomic_range": SHARED_COORDINATE_LABEL,
        "panel_a_coordinate_annotation": {
            "text_y_mm": SHARED_COORDINATE_TEXT_Y_MM,
            "rule_length_fraction": SHARED_COORDINATE_RULE_LENGTH_FRACTION,
            "rule_max_length_fraction": SHARED_COORDINATE_RULE_MAX_LENGTH_FRACTION,
            "rule_label_gap_mm": SHARED_COORDINATE_RULE_LABEL_GAP_MM,
        },
        "panel_a_method_display_names": {
            method: METHOD_DISPLAY_NAMES.get(method, method) for method in methods
        },
        "dpi": args.dpi,
    }


# --------------------------------------------------------------------------
# 8. Main
# --------------------------------------------------------------------------
def render_TAD_method_comparison(args):
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [method for method in methods if method not in cfg.MAIN_METHOD_SOURCES]
    if unknown:
        raise ValueError(f"methods outside the approved main mapping: {unknown}")
    final_outdir = Path(args.outdir or cfg.FIGURE_ROOT)
    result_files = (
        cfg.PCC_SUMMARY_FILE,
        cfg.TAD_PLOT_CHECK_FILE,
        cfg.RUN_INFORMATION_FILE,
    )
    if (final_outdir.exists() or any(path.exists() for path in result_files)) and not args.force:
        raise FileExistsError(
            "refusing to replace existing TAD method comparison without --force"
        )
    stage_outdir = final_outdir.parent / f".staging_{final_outdir.name}"
    if stage_outdir.exists():
        if not args.force:
            raise FileExistsError(f"stale Figure 1 staging directory: {stage_outdir}")
        shutil.rmtree(stage_outdir)
    stage_result_dir = cfg.RESULTS_ROOT / ".staging_TAD_method_comparison_results"
    if stage_result_dir.exists():
        if not args.force:
            raise FileExistsError(f"stale result staging directory: {stage_result_dir}")
        shutil.rmtree(stage_result_dir)
    stage_outdir.mkdir(parents=True)
    stage_result_dir.mkdir(parents=True)
    set_gr_style()

    try:
        print("=" * 64)
        print(f"TAD comparison | {len(methods)} methods x {len(cfg.CELL_TYPES)} cell types")
        print(f"          paired={args.paired}  n_boot={args.n_boot}  norm={args.norm}")
        print("=" * 64)

        pcc_tables = {method: load_pcc_table(method) for method in methods}
        if args.baseline not in methods:
            raise ValueError(f"baseline '{args.baseline}' missing; cannot compute delta r")
        long_df = pd.concat([pcc_tables[method] for method in methods], ignore_index=True)
        trial_metadata = {method: load_trial_metadata(method) for method in methods}
        if (
            "T-FLAMINGO" in methods
            and trial_metadata["T-FLAMINGO"].get("input_key")
            != "FLAMINGO_fixed_contact"
        ):
            raise RuntimeError("T-FLAMINGO must use FLAMINGO_fixed_contact")

        TAD_AUDIT.clear()
        fig, tad_audit, representatives = build_full_figure(
            methods, pcc_tables, long_df, args
        )
        save_all(fig, str(stage_outdir / "TAD_method_comparison"), dpi=args.dpi)
        plt.close(fig)

        if args.standalone_panels:
            fig_a = build_panel_a_figure(methods, pcc_tables, args)
            save_all(fig_a, str(stage_outdir / "panel_A_TAD_heatmaps"), dpi=args.dpi)
            plt.close(fig_a)

            fig_b = build_panel_b_figure(methods, long_df, args)
            save_all(fig_b, str(stage_outdir / "panel_B_PCC_violin_plots"), dpi=args.dpi)
            plt.close(fig_b)

        build_TAD_boundary_plot_check(tad_audit, representatives).to_csv(
            stage_result_dir / cfg.TAD_PLOT_CHECK_FILE.name, index=False
        )

        rows = []
        rng = np.random.default_rng(1)
        for cell_type in cfg.CELL_TYPES:
            data = celltype_values(
                long_df, cell_type, methods, args.pcc_col, args.paired
            )
            for method in methods:
                values = data[method]
                if method == args.baseline:
                    observed = low = high = 0.0
                else:
                    observed, low, high, _ = bootstrap_mean_diff(
                        values,
                        data[args.baseline],
                        n_boot=args.n_boot,
                        paired=args.paired,
                        rng=rng,
                    )
                rows.append(
                    {
                        "cell_type": cell_type,
                        "method": method,
                        "n_trials": len(values),
                        "r_mean": values.mean(),
                        "r_sd": values.std(ddof=1),
                        "r_median": np.median(values),
                        "delta_r": observed,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_mode": "paired" if args.paired else "unpaired",
                    }
                )
        pd.DataFrame(rows).to_csv(
            stage_result_dir / cfg.PCC_SUMMARY_FILE.name, index=False
        )

        provenance = build_provenance(methods, trial_metadata, args)
        with (stage_result_dir / cfg.RUN_INFORMATION_FILE.name).open("w") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
            handle.write("\n")

        expected_exports = build_export_manifest(stage_outdir)
        if not args.standalone_panels:
            expected_exports = expected_exports[:3]
        missing = [
            path for path in expected_exports
            if not Path(path).is_file() or Path(path).stat().st_size == 0
        ]
        if missing:
            raise RuntimeError(f"missing figure exports: {missing}")

        if args.force and final_outdir.exists():
            shutil.rmtree(final_outdir)
        os.replace(stage_outdir, final_outdir)
        for final_path in result_files:
            staged_path = stage_result_dir / final_path.name
            if args.force and final_path.exists():
                final_path.unlink()
            os.replace(staged_path, final_path)
        stage_result_dir.rmdir()
        return {
            "outdir": final_outdir,
            "export_count": len(expected_exports),
            "representative_count": len(representatives),
        }
    except Exception:
        plt.close("all")
        if stage_outdir.exists():
            shutil.rmtree(stage_outdir)
        if stage_result_dir.exists():
            shutil.rmtree(stage_result_dir)
        raise


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg.validate_project_cwd()
    result = render_TAD_method_comparison(args)
    print(
        f"Published {result['export_count']} TAD method comparison exports "
        f"in {result['outdir']}."
    )


if __name__ == "__main__":
    main()
