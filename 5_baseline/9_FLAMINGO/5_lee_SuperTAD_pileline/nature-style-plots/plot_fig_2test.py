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
  * Arial/Helvetica; all text 8-10 pt (spread <= 2 pt); panel tags 12 pt bold
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
import argparse
import warnings

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

import config as cfg

# --------------------------------------------------------------------------
# 0. Constants
# --------------------------------------------------------------------------
MM = 1 / 25.4

FS_SMALL, FS_BASE, FS_HEAD = 8, 9, 10     # spread == 2 pt (GR rule)
FS_TAG = 12                                # panel tags, bold

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
DIAG_CORE, DIAG_HALO = "#FFFFFF", "#5A5A5A"   # diagonal  : dashed, hairline

TAD_DASH = (0, (2.2, 1.4))
REF_DOT = (0, (0.8, 1.3))

# Okabe-Ito subset. Used ONLY for cell-type labels, never for data marks,
# so the accent stays unambiguous.
CT_COLORS = {"Astro": "#0072B2", "Endo": "#E69F00",
             "ODC": "#009E73", "OPC": "#CC79A7"}

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
# 2. IO helpers
# --------------------------------------------------------------------------
def load_target(cell_type: str) -> np.ndarray:
    return load_npz(os.path.join(cfg.TARGET_DIR, f"{cell_type}_target.npz")).toarray()


def load_trial_matrix(method: str, cell_type: str, trial_id: int) -> np.ndarray:
    path = os.path.join(cfg.TRIALS_DIR, method, "matrices",
                        f"{cell_type}_trial{trial_id:03d}.npz")
    return load_npz(path).toarray()


def load_pcc_table(method: str):
    path = os.path.join(cfg.TRIALS_DIR, method, "pcc_results.csv")
    if not os.path.exists(path):
        warnings.warn(f"missing {path}; method '{method}' will be skipped")
        return None
    df = pd.read_csv(path)
    df["method"] = method
    return df


def find_median_trial(df, cell_type, pcc_col="pcc_8x8_full"):
    sub = df[df.cell_type == cell_type].sort_values(pcc_col)
    if len(sub) == 0:
        return 0, np.nan
    row = sub.iloc[len(sub) // 2]
    return int(row.trial_id), float(row[pcc_col])


def parse_tad_tsv(tsv_path, report=None):
    """Read SuperTAD domains as (start_bin, end_bin) from columns 1 and 5.

    SuperTAD writes one-based inclusive bin indices; convert them to the
    zero-based matrix coordinates used by NumPy and the plotting window.

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
                tads.append((int(parts[1]) - 1, int(parts[5]) - 1))
            except ValueError:
                n_bad += 1
    if report is not None:
        report.update(found=True, lines=n_lines, parsed=len(tads), skipped=n_bad)
    return tads


TAD_AUDIT = []            # filled while panel A is drawn, printed afterwards


def clip_tads(tads, s, e, report=None):
    """Map SuperTAD domains onto the display window [s, e).

    SuperTAD runs on the FULL matrix, so domains carry whole-matrix bin indices
    and most of them straddle or miss an 8-bin sub-window. Requiring full
    containment silently drops every one of them; clip instead.

    A domain that strictly ENCLOSES the window (a < s and b > e-1, e.g. the root
    node of the hierarchy) has no boundary anywhere inside the view. Clipping it
    would place two legs exactly on the panel frame, which reads as a TAD call
    that was never made. Those are dropped and counted, not drawn.
    """
    out, enclosing, degenerate = [], 0, 0
    for a, b in tads:
        a, b = int(a), int(b)
        if a < s and b > e - 1:
            enclosing += 1
            continue
        a2, b2 = max(a, s), min(b, e - 1)
        if b2 - a2 < 1:
            degenerate += 1
            continue
        out.append((a2 - s, b2 - s))
    if report is not None:
        report.update(drawn=len(out), enclosing=enclosing, outside=degenerate)
    return out


def find_tad_tsv(method: str, cell_type: str, trial_id: int):
    """Tolerant lookup: pipelines name SuperTAD output slightly differently."""
    if method.lower() == "target":
        cands = [os.path.join(cfg.SUPERTAD_DIR, "target", f"{cell_type}_target.tsv")]
    else:
        base = os.path.join(cfg.SUPERTAD_DIR, method)
        cands = [
            os.path.join(base, f"{cell_type}_trial{trial_id:03d}.tsv"),
            os.path.join(base, "tads", f"{cell_type}_trial{trial_id:03d}.tsv"),
            *sorted(glob.glob(os.path.join(
                base, "**", f"{cell_type}_trial{trial_id:03d}*.tsv"), recursive=True)),
        ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


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


def draw_tad_triangle(ax, s, e, side, lw=0.7, zorder=4):
    """Two legs of the SuperTAD domain triangle [s, e] (imshow, origin='upper')."""
    lo, hi = s - 0.5, e + 0.5
    if side == "upper":
        pts = [((lo, hi), (lo, lo)), ((hi, hi), (lo, hi))]
    else:
        pts = [((lo, lo), (lo, hi)), ((lo, hi), (hi, hi))]
    for (x1, x2), (y1, y2) in pts:
        ax.add_line(Line2D([x1, x2], [y1, y2], color=TAD_CORE, lw=lw,
                           ls=TAD_DASH, dash_capstyle="butt", zorder=zorder,
                           path_effects=_halo(TAD_CORE, TAD_HALO, lw)))


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

    ax.add_line(Line2D([-.5, n - .5], [-.5, n - .5], color=DIAG_CORE,
                       ls=(0, (3, 2)), lw=0.45, zorder=5,
                       path_effects=_halo(DIAG_CORE, DIAG_HALO, 0.45)))

    if tad_mode in ("supertad", "both"):
        for s, e in (tads_lower or []):
            draw_tad_triangle(ax, s, e, "lower")
        for s, e in (tads_upper or []):
            draw_tad_triangle(ax, s, e, "upper")
    if tad_mode in ("lee", "both"):
        draw_lee_reference(ax, cell_type)

    if np.isfinite(r_value):
        ax.text(0.04, 0.04, f"{r_value:.2f}", transform=ax.transAxes,
                fontsize=FS_SMALL, ha="left", va="bottom", color=INK,
                bbox=dict(boxstyle="square,pad=0.12", fc="white",
                          ec="none", alpha=0.78), zorder=6)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0 if highlight else 0.5)
        sp.set_edgecolor(ACCENT if highlight else "#333333")
    ax.set_xticks([]); ax.set_yticks([])


# --------------------------------------------------------------------------
# 5. Panel A
# --------------------------------------------------------------------------
def panel_grid(fig, geo, methods, pcc_tables, highlight, norm_mode, tad_mode,
               pcc_col="pcc_8x8_full", colorbar="none"):
    """Rows = cell types, columns = methods. Absolute mm placement, so the
    square heat maps tile exactly with no aspect-ratio gutters."""
    W, H = geo["W"], geo["H"]
    x0, y0 = geo["x0"], geo["A_top"] + geo["HEAD"]
    cw, gap = geo["cell"], geo["gap"]

    s, e = cfg.PDGFRA_SUB_BINS
    cts = list(cfg.CELL_TYPES)
    targets = {ct: load_target(ct) for ct in cts}
    im_handle = None

    for i, ct in enumerate(cts):
        tgt_sub = targets[ct][s:e, s:e]
        tgt_norm = normalize(tgt_sub, norm_mode, ref=tgt_sub)
        rep = {}
        tads_up = clip_tads(parse_tad_tsv(find_tad_tsv("target", ct, 0), rep),
                            s, e, rep)
        TAD_AUDIT.append(dict(method="target", cell_type=ct, **rep))

        for j, m in enumerate(methods):
            ax = fig.add_axes(_rect(x0 + j * (cw + gap), y0 + i * (cw + gap),
                                    cw, cw, W, H))
            df = pcc_tables.get(m)
            if df is None:
                ax.set_axis_off()
                continue
            tid, _ = find_median_trial(df, ct, pcc_col)
            mat_sub = load_trial_matrix(m, ct, tid)[s:e, s:e]
            mat_norm = normalize(mat_sub, norm_mode, ref=tgt_sub)
            rep = {}
            tads_lo = clip_tads(parse_tad_tsv(find_tad_tsv(m, ct, tid), rep),
                                s, e, rep)
            TAD_AUDIT.append(dict(method=m, cell_type=ct, trial=tid, **rep))

            draw_map(ax, mat_norm, tgt_norm, ct, pcc(mat_sub, tgt_sub),
                     tads_lo, tads_up, tad_mode, highlight=(m == highlight))
            im_handle = ax.images[0]

            if i == 0:
                ax.set_title(m, fontsize=FS_HEAD, pad=2.5,
                             fontweight="bold" if m == highlight else "normal",
                             color=ACCENT if m == highlight else INK)
            if j == 0:
                ax.set_ylabel(ct, fontsize=FS_HEAD, labelpad=3,
                              color=CT_COLORS.get(ct, INK))

    grid_h = len(cts) * cw + (len(cts) - 1) * gap
    grid_w = len(methods) * cw + (len(methods) - 1) * gap
    strip_y = y0 + grid_h + 1.0

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
        cax = fig.add_axes(_rect(x0 + grid_w - 26.0, strip_y + 0.9,
                                 16.0, 1.6, W, H))
        cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=HIC_CMAP)
        cax.set_xticks([]); cax.set_yticks([])
        for sp in cax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.4); sp.set_edgecolor("#5A5A5A")
        cax.text(-0.06, 0.5, "Low", transform=cax.transAxes, fontsize=FS_SMALL,
                 ha="right", va="center", color="#4D4D4D")
        cax.text(1.06, 0.5, "High", transform=cax.transAxes, fontsize=FS_SMALL,
                 ha="left", va="center", color="#4D4D4D")

    # ---- line key: dash pattern is the identifier, so it reads in greyscale --
    handles = [
        Line2D([], [], color=TAD_CORE, lw=0.9, ls=TAD_DASH,
               path_effects=_halo(TAD_CORE, TAD_HALO, 0.9), label="SuperTAD domain"),
        Line2D([], [], color=DIAG_CORE, lw=0.7, ls=(0, (3, 2)),
               path_effects=_halo(DIAG_CORE, DIAG_HALO, 0.7), label="Diagonal"),
    ]
    if tad_mode in ("lee", "both"):
        handles.insert(1, Line2D([], [], color=REF_CORE, lw=0.9, ls=REF_DOT,
                                 path_effects=_halo(REF_CORE, REF_HALO, 0.9),
                                 label="Reference TAD"))
    lax = fig.add_axes(_rect(x0, strip_y, min(cw * len(methods), 92.0), 4.4, W, H))
    lax.set_axis_off()
    lax.legend(handles=handles, loc="upper left", ncol=len(handles),
               fontsize=FS_SMALL, handlelength=2.0, columnspacing=1.2,
               borderaxespad=0, handletextpad=0.5)

    if colorbar != "inline":     # keep the strip's right end free for this note
        fig.text((x0 + grid_w) / W, 1 - (strip_y + 1.0) / H,
                 "lower: method   |   upper: target",
                 fontsize=FS_SMALL, ha="right", va="top", color="#4D4D4D")

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
        fig.savefig(p, dpi=dpi, facecolor="white")
        out.append(p)
    try:
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
        buf.seek(0)
        p = f"{stem}.tif"
        Image.open(buf).convert("RGB").save(
            p, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
        out.append(p)
    except Exception as exc:                              # pragma: no cover
        warnings.warn(f"TIFF export skipped: {exc}")
    return out


# --------------------------------------------------------------------------
# 8. Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="scHi-C benchmark figure (GR format)")
    ap.add_argument("--methods", required=True,
                    help="comma-separated; this IS the column order, ours last")
    ap.add_argument("--highlight", default="scHiC-Diff")
    ap.add_argument("--baseline", default="Raw")
    ap.add_argument("--norm", choices=["q99", "shared"], default="q99")
    ap.add_argument("--tad", choices=["none", "lee", "supertad", "both"],
                    default="supertad")
    ap.add_argument("--pcc-col", default="pcc_8x8_full")
    ap.add_argument("--colorbar", choices=["none", "inline", "right"],
                    default="none",
                    help="'none' omits the colour key (state the scaling in the "
                         "legend); 'inline' tucks a slim Low->High bar under the "
                         "grid; 'right' restores the old vertical bar")
    ap.add_argument("--paired", dest="paired", action="store_true", default=True,
                    help="align trials by trial_id across methods (default)")
    ap.add_argument("--no-paired", dest="paired", action="store_false",
                    help="use when methods were run with independent seeds")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--width-mm", type=float, default=170.0,
                    help="170 = GR double column, 85 = single column")
    ap.add_argument("--dpi", type=int, default=600,
                    help="raster dpi; GR wants 600-900 for combination art")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    outdir = args.outdir or os.path.join(cfg.FIGURES_DIR, "combined")
    os.makedirs(outdir, exist_ok=True)
    set_gr_style()

    print("=" * 64)
    print(f"Figure 1 | {len(methods)} methods x {len(cfg.CELL_TYPES)} cell types")
    print(f"          paired={args.paired}  n_boot={args.n_boot}  norm={args.norm}")
    print("=" * 64)

    pcc_tables = {m: load_pcc_table(m) for m in methods}
    methods = [m for m in methods if pcc_tables[m] is not None]   # order preserved
    if args.baseline not in methods:
        raise SystemExit(f"baseline '{args.baseline}' missing; cannot compute Δr")
    long_df = pd.concat([pcc_tables[m] for m in methods], ignore_index=True)

    # ---------------- geometry (millimetres) ------------------------------
    W_mm = float(args.width_mm)
    ncol, nrow = len(methods), len(cfg.CELL_TYPES)
    L, HEAD, gap = 13.0, 6.0, 1.1
    R = 17.0 if args.colorbar == "right" else 4.0
    avail = W_mm - L - R
    cell = min((avail - (ncol - 1) * gap) / ncol, 24.0)
    grid_w = ncol * cell + (ncol - 1) * gap
    x0 = L + max((avail - grid_w) / 2.0, 0.0)
    grid_h = nrow * cell + (nrow - 1) * gap

    h_A = 1.0 + HEAD + grid_h + 6.6
    h_v, h_d = 32.0, 15.0
    lab_h = 2.0 + 0.55 * max(len(m) for m in methods) * (FS_SMALL / 8.0) * 2.2
    h_B = 4.5 + h_v + 1.6 + h_d + lab_h
    H_mm = h_A + 11.0 + h_B + 3.0

    fig = plt.figure(figsize=(W_mm * MM, H_mm * MM))
    geo = dict(W=W_mm, H=H_mm, L=L, R=R, HEAD=HEAD, cell=cell, gap=gap,
               A_top=1.0, x0=x0, h_v=h_v, h_d=h_d)

    def tag(letter, x_mm, ytop_mm):
        fig.text(max(x_mm - 9.0, 0.5) / W_mm, 1 - ytop_mm / H_mm, letter,
                 fontsize=FS_TAG, fontweight="bold", va="top", ha="left")

    a_bottom = panel_grid(fig, geo, methods, pcc_tables, args.highlight,
                          args.norm, args.tad, args.pcc_col, args.colorbar)
    tag("A", L, 0.5)

    b_top = a_bottom + 11.0
    panel_estimation(fig, geo, methods, long_df, args.highlight, args.baseline,
                     args.pcc_col, args.paired, args.n_boot, b_top)
    tag("B", L, b_top)

    if args.tad in ("supertad", "both") and TAD_AUDIT:
        aud = pd.DataFrame(TAD_AUDIT)
        agg = (aud.groupby("method", sort=False)
                  .agg(files_found=("found", "sum"),
                       rows=("lines", "sum"), skipped_rows=("skipped", "sum"),
                       domains_parsed=("parsed", "sum"),
                       domains_drawn=("drawn", "sum"),
                       enclosing_window=("enclosing", "sum"),
                       outside_window=("outside", "sum")))
        print("\nSuperTAD audit (summed over cell types):")
        print(agg.to_string())
        dead = agg.index[agg.domains_drawn == 0].tolist()
        if dead:
            print(f"\n  !! no TAD boundary drawn for: {', '.join(dead)}")
            print("     run  python diagnose_tads.py --methods "
                  + ",".join(dead) + "  to find out why\n")
        aud.to_csv(os.path.join(outdir, "Fig1_tad_audit.csv"), index=False)

    stem = os.path.join(outdir, "Fig1_method_comparison")
    for f in save_all(fig, stem, dpi=args.dpi):
        print(f"  Saved: {f}")
    plt.close(fig)

    # ---- machine-readable source data (GR encourages this) ---------------
    rows = []
    rng = np.random.default_rng(1)
    for ct in cfg.CELL_TYPES:
        d = celltype_values(long_df, ct, methods, args.pcc_col, args.paired)
        for m in methods:
            v = d[m]
            if m == args.baseline:
                obs = lo = hi = 0.0
            else:
                obs, lo, hi, _ = bootstrap_mean_diff(
                    v, d[args.baseline], n_boot=args.n_boot,
                    paired=args.paired, rng=rng)
            rows.append(dict(cell_type=ct, method=m, n_trials=len(v),
                             r_mean=v.mean(), r_sd=v.std(ddof=1),
                             r_median=np.median(v),
                             delta_r=obs, ci_low=lo, ci_high=hi))
    src = os.path.join(outdir, "Fig1_source_data.csv")
    pd.DataFrame(rows).to_csv(src, index=False)
    print(f"  Saved: {src}")
    print("Done.")


if __name__ == "__main__":
    main()
