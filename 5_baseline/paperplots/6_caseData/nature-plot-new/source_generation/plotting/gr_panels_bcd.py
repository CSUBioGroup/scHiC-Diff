#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gr_panels_bcd.py
================
正文主图的 B / C / D 面板。与 gr_stagefig.py 共用同一套 GR 样式源。

设计原则（也是这次重构的核心）：**每个 draw_* 只画，不加载、不计算、不 savefig。**
它们接收 ax（或 fig + 毫米几何）+ 已算好的数据。这是能把 A/B/C/D 拼进一张 174 mm
整图的前提 —— 你现有的 plot_comparison_grid / plot_results / plot_apa_from_directory
各自在一个函数里做完 load→compute→plot→savefig，所以永远拼不到一起。

对外
----
draw_loop_grid(fig, ...)     面板 B：行=方法、列=cell count 的接触图 + loop 叠加
draw_map2_curve(ax, ...)     面板 C：Map2 已知 loop 的富集 vs 聚合细胞数
draw_apa_row(fig, ...)       面板 D：一行 APA 热图 + Juicer 真 P2LL
hic_cmap()                   非负接触计数用的序列色图（白→深红，无浅黄，CVD 安全）
coverage_normalize(m)        按总覆盖度归一，比较结构而非深度
offdiag_vmax(m, q, k)        排除对角线带之后取分位数做 vmax
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import matplotlib.patheffects as pe

from gr_stagefig import (LW_HAIR, OKABE_ITO, PT_BASE, PT_SMALL, TEXT_MAIN,
                         TEXT_MUTED, resolve_method_styles)


# ----------------------------------------------------------------------------
# 色图与归一化
# ----------------------------------------------------------------------------
# 注意：本模块的 imshow 一律**不加** rasterized=True。
# imshow 在 PDF 后端本来就以嵌入图像输出；加 rasterized 会让 MixedModeRenderer
# 为每个 artist 单独分配一次整幅 savefig-dpi 缓冲（174x202 mm @600 dpi ≈ 78 MB），
# 20 个面板就是 1.5 GB，直接 OOM。只有散点（大量矢量点）才需要 rasterized。


def hic_cmap():
    """
    非负接触计数的序列色图：白 → 深红。

    不用 'Reds'：它高端只到 #67000D 但中段偏粉，缩到 20 mm 面板后层次糊掉。
    更不用 icefire/coolwarm 这类发散色图 —— 接触计数是非负的，发散色图会
    凭空造出「冷/暖两极」这个不存在的语义。
    """
    return LinearSegmentedColormap.from_list(
        "hic_fall", ["#FFFFFF", "#FDD9C7", "#F7A98A", "#E8623C", "#B32316", "#5C0A06"]
    )


def coverage_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    按总覆盖度归一（除以上三角总和）。

    为什么必须做：Raw 的 10 细胞聚合和 100 细胞聚合差一个数量级，共享色标下
    前者会几乎全白。归一之后比较的是**结构**而不是**深度** —— 这是 Hi-C 领域
    的标准做法，也是共享色标能成立的前提。
    """
    m = np.asarray(matrix, dtype=float)
    total = np.triu(m, k=1).sum()
    return m / total if total > 0 else m


def offdiag_vmax(matrix: np.ndarray, q: float = 99.0, k: int = 2) -> float:
    """
    排除主对角线及其相邻 k 条带之后取分位数。

    你现有代码用 np.percentile(matrix, 98) 取全矩阵 —— 但 Hi-C 的对角线永远最亮，
    色标会被对角线定死，loop 那点信号全被压进最低那一档。
    """
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    i, j = np.indices((n, n))
    off = m[(j - i) > k]
    if off.size == 0:
        return float(m.max()) or 1.0
    v = float(np.percentile(off, q))
    return v if v > 0 else (float(off.max()) or 1.0)


def shared_scale(matrices: Sequence[np.ndarray], q: float = 99.0,
                 k: int = 2) -> tuple[float, float]:
    """整格网共用一个 (vmin, vmax)。这是这次重构里最要紧的一处改动，见 docstring。"""
    vmaxes = [offdiag_vmax(m, q=q, k=k) for m in matrices if m is not None]
    return 0.0, float(np.median(vmaxes)) if vmaxes else 1.0


# ----------------------------------------------------------------------------
# 面板 B：loop 对比格网
# ----------------------------------------------------------------------------
def draw_loop_grid(
    fig,
    matrices: dict,
    loops: dict,
    methods: Sequence[str],
    cell_counts: Sequence[int],
    highlight: str,
    fig_w_mm: float,
    fig_h_mm: float,
    x_mm: float,
    top_mm: float,
    width_mm: float,
    left_gutter_mm: float = 13.0,
    gap_mm: float = 1.2,
    header_mm: float = 5.0,
    cbar_mm: float = 4.0,
    scale: str = "sum",
    vq: float = 99.0,
    resolution: int = 20_000,
) -> float:
    """
    行=方法、列=cell count 的接触图格网，called loop 以空心圈叠加。

    matrices: {(method, count): (n_bins, n_bins) ndarray}
    loops   : {(method, count): (n_loops, 2) 的 bin 坐标, 或 None}

    与你现有 plot_comparison_grid 的三处关键差异
    ------------------------------------------
    1. **共享色标**。原代码每个面板 `vmax = np.percentile(matrix, 98)` 各自归一，
       于是一个把接触数整体放大 8 倍的方法，和没放大的那个，长得一模一样 ——
       「方法间强度差异」这个信息被逐面板归一化整个抹掉了。
    2. **先按覆盖度归一再共享色标**，否则 10 细胞那列会全白。
    3. **vmax 取非对角分位数**，不让对角线定死色标。
    返回格网底边距图顶的毫米数。
    """
    n_col, n_row = len(cell_counts), len(methods)
    avail = width_mm - left_gutter_mm - gap_mm * (n_col - 1)
    cell = avail / n_col
    grid_top = top_mm + header_mm

    prep = {}
    for m in methods:
        for c in cell_counts:
            mat = matrices.get((m, c))
            if mat is None:
                prep[(m, c)] = None
                continue
            prep[(m, c)] = coverage_normalize(mat) if scale == "sum" else np.asarray(mat, float)

    vmin, vmax = shared_scale([v for v in prep.values() if v is not None], q=vq)
    cmap, image = hic_cmap(), None

    for r, meth in enumerate(methods):
        for c, cnt in enumerate(cell_counts):
            x0 = x_mm + left_gutter_mm + c * (cell + gap_mm)
            y0 = grid_top + r * (cell + gap_mm)
            ax = fig.add_axes([x0 / fig_w_mm, 1 - (y0 + cell) / fig_h_mm,
                               cell / fig_w_mm, cell / fig_h_mm])
            mat = prep[(meth, cnt)]
            if mat is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, fontsize=PT_SMALL, color="#5F5E5A")
                ax.set_facecolor("#F1EFE8")
            else:
                image = ax.imshow(mat, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax,
                                  interpolation="nearest")
                lp = loops.get((meth, cnt))
                if lp is not None and len(lp):
                    # R1：loop 圈是**标注**不是数据 -> 不占颜色预算。
                    # 黑圈 + 白描边（双描边）在白底和暗红底上都看得见；
                    # v1 用的 #0072B2 会和面板 A 的「Blue 组」同色异义（deltaE = 0）。
                    lp = np.asarray(lp)
                    ax.scatter(lp[:, 1], lp[:, 0], s=6, facecolors="none",
                               edgecolors=TEXT_MAIN, linewidths=0.5,
                               path_effects=[pe.withStroke(linewidth=1.4,
                                                           foreground="white")])

            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_linewidth(LW_HAIR); s.set_edgecolor("#444441")

            if r == 0:
                ax.text(0.5, 1.05, f"{cnt} cells", transform=ax.transAxes,
                        fontsize=PT_BASE, ha="center", va="bottom", color=TEXT_MAIN)
            if c == 0:
                ax.text(-0.07, 0.5, meth, transform=ax.transAxes, fontsize=PT_BASE,
                        ha="right", va="center", rotation=90,
                        fontweight="bold" if meth == highlight else "normal")

        if meth == highlight:
            y0 = grid_top + r * (cell + gap_mm)
            bar = fig.add_axes([(x_mm + left_gutter_mm - 1.8) / fig_w_mm,
                                1 - (y0 + cell) / fig_h_mm, 0.6 / fig_w_mm,
                                cell / fig_h_mm])
            bar.set_axis_off()
            bar.imshow(np.ones((2, 1)), cmap=LinearSegmentedColormap.from_list(
                "acc", [TEXT_MAIN, TEXT_MAIN]), aspect="auto")

    grid_bot = grid_top + n_row * cell + (n_row - 1) * gap_mm

    # 一根共享 colorbar，不是 16 根 inset colorbar
    if image is not None:
        cx = x_mm + left_gutter_mm + avail * 0.25
        cax = fig.add_axes([cx / fig_w_mm, 1 - (grid_bot + 3.0 + cbar_mm * 0.35) / fig_h_mm,
                            (avail * 0.5) / fig_w_mm, (cbar_mm * 0.35) / fig_h_mm])
        cb = fig.colorbar(image, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(LW_HAIR)
        cb.set_ticks([vmin, vmax])
        lab = "Normalized contacts" if scale == "sum" else "Contacts"
        cb.set_ticklabels(["0", "max"])
        cax.tick_params(labelsize=PT_SMALL, width=LW_HAIR, length=1.5, pad=1,
                        colors=TEXT_MUTED)
        cax.set_xlabel(lab, fontsize=PT_SMALL, labelpad=1, color=TEXT_MUTED)
    return grid_bot + cbar_mm + 4.0


# ----------------------------------------------------------------------------
# 面板 C：Map2 已知 loop 的富集
# ----------------------------------------------------------------------------
def draw_map2_curve(
    ax,
    df,
    methods: Sequence[str],
    highlight: str,
    baseline: str = "Raw",
    metric: str = "log2_enrichment",
    ylabel: str = "Map2 loop enrichment (log$_2$)",
    xlabel: str = "Aggregated cells",
    null_line: float = 0.0,
) -> None:
    """
    Map2 已知 enhancer–promoter loop 的富集 vs 聚合细胞数。

    df 需含列：method, cell_number, <metric>。**不要**传经过 maybe_log1p_transform
    的数 —— 见 make_main_figure.py 顶部那段说明，条件 log1p 会让不同深度的方法
    拿到不同的变换，富集比之间就不可比了。

    null_line：无富集的参考位置。log2 富集 → 0；比值 → 1。
    """
    styles = resolve_method_styles(methods, highlight, baseline)
    ax.axhline(null_line, color="#5F5E5A", lw=0.4, ls=":", zorder=1)

    for m in methods:
        sub = df[df["method"] == m].sort_values("cell_number")
        if sub.empty:
            continue
        st = styles[m]
        ax.plot(sub["cell_number"].to_numpy(), sub[metric].to_numpy(),
                **st, clip_on=False)

    ax.set_xscale("log")
    counts = sorted(df["cell_number"].unique())
    ax.set_xticks(counts)
    ax.set_xticklabels([str(int(c)) for c in counts])
    ax.minorticks_off()
    ax.set_xlabel(xlabel, fontsize=PT_BASE, color=TEXT_MUTED)
    ax.set_ylabel(ylabel, fontsize=PT_BASE, color=TEXT_MUTED)
    ax.tick_params(axis="both", labelsize=PT_SMALL, width=LW_HAIR, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(LW_HAIR)

    handles = [Line2D([], [], **{k: v for k, v in styles[m].items() if k != "zorder"},
                      label=m) for m in methods]
    ax.legend(handles=handles, loc="upper left", fontsize=PT_SMALL, frameon=False,
              handlelength=2.0, borderaxespad=0.2, labelspacing=0.25)


# ----------------------------------------------------------------------------
# 面板 D：APA
# ----------------------------------------------------------------------------
def draw_apa_row(
    fig,
    apa: dict,
    p2ll: dict,
    methods: Sequence[str],
    highlight: str,
    fig_w_mm: float,
    fig_h_mm: float,
    x_mm: float,
    top_mm: float,
    width_mm: float,
    gap_mm: float = 1.5,
    header_mm: float = 4.5,
    cbar_mm: float = 4.0,
    resolution: int = 20_000,
) -> float:
    """
    一行 APA 热图 + Juicer 的真 P2LL。

    apa  : {method: (w, w) ndarray}，Juicer APA 输出（已 O/E 归一）
    p2ll : {method: float}，从 Juicer measures.txt 读的 **P2LL**
           —— 即中心像素 / 左下角区域均值（Rao et al. 2014 的定义）。
           不要把 calculate_p2ll.py 那个「中心 5×5 / 周围环形」的数塞进来，
           它是另一个量（见 make_main_figure.py 顶部）。

    APA 已经是 O/E，所以直接共享色标即可，不需要覆盖度归一。
    """
    n = len(methods)
    cell = (width_mm - gap_mm * (n - 1)) / n
    grid_top = top_mm + header_mm

    mats = [apa[m] for m in methods if apa.get(m) is not None]
    vmin, vmax = 0.0, float(np.percentile(np.concatenate([m.ravel() for m in mats]), 99.5))
    cmap, image = hic_cmap(), None

    for c, meth in enumerate(methods):
        x0 = x_mm + c * (cell + gap_mm)
        ax = fig.add_axes([x0 / fig_w_mm, 1 - (grid_top + cell) / fig_h_mm,
                           cell / fig_w_mm, cell / fig_h_mm])
        m = apa.get(meth)
        if m is None:
            ax.set_axis_off(); continue
        w = m.shape[0]
        half = (w * resolution) / 2 / 1000
        image = ax.imshow(m, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax,
                          extent=(-half, half, -half, half), aspect="equal",
                          interpolation="nearest")
        ax.set_xticks([-half, 0, half]); ax.set_yticks([-half, 0, half])
        ax.set_xticklabels([f"{-half:.0f}", "0", f"{half:.0f}"], fontsize=PT_SMALL)
        ax.set_yticklabels([] if c else [f"{-half:.0f}", "0", f"{half:.0f}"],
                           fontsize=PT_SMALL)
        ax.tick_params(width=LW_HAIR, length=1.5, pad=1, colors=TEXT_MUTED)
        for s in ax.spines.values():
            s.set_linewidth(LW_HAIR); s.set_edgecolor("#444441")

        ax.text(0.5, 1.04, meth, transform=ax.transAxes, fontsize=PT_BASE,
                ha="center", va="bottom",
                fontweight="bold" if meth == highlight else "normal")
        v = p2ll.get(meth)
        if v is not None:  # R1：数字是数据，一律黑色常规
            ax.text(0.96, 0.04, f"{v:.2f}", transform=ax.transAxes, fontsize=PT_SMALL,
                    ha="right", va="bottom", color=TEXT_MAIN)
        if c == 0:
            ax.set_ylabel("kb", fontsize=PT_SMALL, labelpad=1, color=TEXT_MUTED)

    bot = grid_top + cell
    if image is not None:
        cx = x_mm + width_mm * 0.25
        cax = fig.add_axes([cx / fig_w_mm, 1 - (bot + 5.5 + cbar_mm * 0.35) / fig_h_mm,
                            (width_mm * 0.5) / fig_w_mm, (cbar_mm * 0.35) / fig_h_mm])
        cb = fig.colorbar(image, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(LW_HAIR)
        cb.set_ticks([vmin, vmax]); cb.set_ticklabels(["0", f"{vmax:.1f}"])
        cax.tick_params(labelsize=PT_SMALL, width=LW_HAIR, length=1.5, pad=1,
                        colors=TEXT_MUTED)
        cax.set_xlabel("APA (obs / exp)", fontsize=PT_SMALL, labelpad=1,
                       color=TEXT_MUTED)
    return bot + cbar_mm + 6.0
