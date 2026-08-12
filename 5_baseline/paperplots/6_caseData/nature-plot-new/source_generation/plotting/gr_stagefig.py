#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gr_stagefig.py
==============
分阶段（developmental stage）× 多方法的 UMAP + silhouette 出版级绘图模块。
Nature 正文审美 + Genome Research (CSHL) 数字图稿规范。

对外 API
--------
set_gr_style()                  一次性写入 rcParams（每个入口调一次）
StageData                       单个 (method, stage) 的数据容器
silhouette_with_ci(...)         silhouette + 子采样重抽样 95% CI
plot_silhouette_by_stage(ax,..) 面板 a：silhouette–stage 折线，方法为线
plot_stage_grid(fig, ...)       面板 b：行=方法、列=stage 的 UMAP 网格
compose_figure(...)             a + b 合成整图（174 mm 双栏）
save_figure(fig, stem, outdir)  同时导出矢量 PDF（首选）与 600 dpi PNG
mm(x) / COL1 / COL15 / COL2     毫米→英寸；单栏 / 1.5 栏 / 双栏宽度

已编码的 GR 硬规范（references/figure-guidelines.md）
---------------------------------------------------
  * Arial/Helvetica；缺失时退到度量兼容的 Liberation Sans 并 warn
  * 图内文字 8–9 pt（同图浮动 1 pt，GR 上限 2 pt）；面板字母 12 pt 粗体大写
  * 所有线/框线 ≥0.25 pt（本模块用 0.4 pt，留余量）
  * 避免浅色；多色在色谱上离得远；对色觉障碍(CVD)友好 → Okabe–Ito + 形状冗余
  * 提交 RGB；线条图首选真矢量 PDF（不受 dpi 约束），字体嵌入（fonttype 42）
  * 色标放图内；描述性面板标题留给图注，不写进图稿
  * 不使用 bbox_inches='tight' —— 它会改变实际栏宽，8 pt 在成品上就不再是 8 pt

设计取舍（与你现有图的差异，见回复正文）
--------------------------------------
  * 主色只编码「被度量的那个分组」（Red/Blue 二分），不编码 25 个 celltype。
    25 类彩虹色带既非 CVD 安全、又暗示了不存在的顺序、且无人能对回图例。
    需要 celltype 全貌 → mode='celltype' 出 Supplemental 版。
  * silhouette 默认在**高维嵌入**上算，不在 UMAP 2D 上算（见 docstring）。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------------------------
# 尺寸：GR 双栏 ≤174 mm，单栏 ≤85 mm
# ----------------------------------------------------------------------------
MM_PER_IN = 25.4


def mm(x: float) -> float:
    """毫米 → 英寸。"""
    return x / MM_PER_IN


COL1 = mm(85.0)    # 单栏
COL15 = mm(114.0)  # 1.5 栏
COL2 = mm(174.0)   # 双栏满宽

# ----------------------------------------------------------------------------
# 字号 / 线宽
# ----------------------------------------------------------------------------
PT_SMALL = 8   # 刻度、图内数字、图例
PT_BASE = 9    # 轴标题、行列标签
PT_TAG = 12    # 面板字母（GR 指定 12 pt 粗体大写）
LW_HAIR = 0.4  # GR 下限 0.25 pt，留余量


def _pick_font() -> str:
    import matplotlib.font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "Arimo"):
        if name in have:
            if name not in ("Arial", "Helvetica"):
                warnings.warn(
                    f"未找到 Arial/Helvetica，退到度量兼容的 {name}。"
                    "版式与字宽一致，但投稿终稿建议在装有 Arial 的机器上重跑。",
                    stacklevel=2,
                )
            return name
    warnings.warn("未找到任何 Arial 兼容字体，退到 DejaVu Sans —— 不符合 GR 字体规范。", stacklevel=2)
    return "DejaVu Sans"


def set_gr_style() -> None:
    """写入 GR 合规的 rcParams。每个绘图入口调用一次。"""
    font = _pick_font()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font, "DejaVu Sans"],
            "font.size": PT_BASE,
            "axes.labelsize": PT_BASE,
            "axes.titlesize": PT_BASE,
            "xtick.labelsize": PT_SMALL,
            "ytick.labelsize": PT_SMALL,
            "legend.fontsize": PT_SMALL,
            "axes.linewidth": LW_HAIR,
            "xtick.major.width": LW_HAIR,
            "ytick.major.width": LW_HAIR,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 0.8,
            "patch.linewidth": LW_HAIR,
            "grid.linewidth": LW_HAIR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 1.0,
            "legend.labelspacing": 0.3,
            # mathtext 默认回退到 DejaVu —— 一个 $\log_2$ 就会把 DejaVuSans
            # 塞进 PDF 字体表，GR 制版会挑出来。强制走同一套无衬线。
            "mathtext.fontset": "custom",
            "mathtext.rm": font,
            "mathtext.it": f"{font}:italic",
            "mathtext.bf": f"{font}:bold",
            # fontset='custom' 会连带解析 cal/sf/tt，其中 cal 默认指向 'cursive'；
            # 机器上没有就回退 DejaVu 并刷警告。全部绑到同一套。
            "mathtext.cal": f"{font}:italic",
            "mathtext.sf": font,
            "mathtext.tt": "monospace",
            "mathtext.default": "regular",
            # 真矢量输出 + 字体嵌入（GR: embed all fonts）
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "pdf.compression": 6,
            "figure.dpi": 200,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


# ----------------------------------------------------------------------------
# 调色板：Okabe–Ito（CVD 安全），无浅色
# ----------------------------------------------------------------------------
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

#: 次要文字（n=、轴标题、色标签）。用灰度做层级，不用颜色做层级。
TEXT_MUTED = "#5F5E5A"
TEXT_MAIN = "#000000"
#: 只在「方法本身就是数据维度」的面板里用（如折线图的 series）。
#: **不要**拿它做「哪一行是我的方法」这种强调 —— 见 DESIGN_RULES。
ACCENT = OKABE_ITO["vermillion"]

DESIGN_RULES = """
样式规则（v2，由 verify_palette() 的实测结果推出）
--------------------------------------------------
R1  一个面板内，一个颜色只能有一个意思。
    v1 实测：面板 A 里 #D55E00 同时是「Red 组」(数据) 和「本文方法」(强调)，
    两者 deltaE = 0；#0072B2 同时是「Blue 组」和「loop 标记」，同样 0。
    这不是配色问题 —— 任何调色板都救不了同色异义。只能改分工：
      * 「哪一行是我的方法」是关于图的**元信息**，不是数据 -> 归排版（字重 + 黑色细规线）
      * 「这里有个 called loop」是**标注**，不是数据 -> 归形状（黑圈 + 白描边，achromatic）
R2  颜色编码「该面板的数据维度」，所以同一条规则会给出不同结论：
      * 面板 A：方法是版面维度(行)，Red/Blue 才是数据 -> 颜色给 Red/Blue
      * 面板 C：方法就是数据维度(series)      -> 颜色给方法
    跨面板复用颜色可以容忍（每个面板自带 key），面板内同色异义不行。
R3  有意义的坐标范围才配框线。UMAP 的 extent 无意义 -> 无框无刻度；
    接触图/APA 的 extent 是基因组坐标 -> 有框有刻度。
    A 和 B/D 长得不一样是**有理由的**，不是忘了统一。
R4  层级用字号 + 灰度，不用颜色。主要文字黑 9 pt；次要文字 #5F5E5A 8 pt。
R5  序列色图必须感知明度单调。hic_fall 实测 J' 100->22 单调、步长 std 0.82，通过。
"""

#: 被度量的二分组：色 + 形状双编码（GR 要求 CVD 友好的冗余编码）
GROUP_STYLE = {
    "Red": dict(color=OKABE_ITO["vermillion"], marker="o"),
    "Blue": dict(color=OKABE_ITO["blue"], marker="^"),
}

#: Supplemental 用：celltype 分组色。>8 类时按 lineage 折叠，不要上彩虹色带。
LINEAGE_COLORS = {
    "Ectoderm / neural": OKABE_ITO["vermillion"],
    "Mesoderm": OKABE_ITO["blue"],
    "Endoderm": OKABE_ITO["green"],
    "Extra-embryonic": OKABE_ITO["orange"],
    "Blood": OKABE_ITO["black"],
    "Epithelial": OKABE_ITO["purple"],
    "Other": OKABE_ITO["sky"],
}
LINEAGE_MARKERS = ["o", "^", "s", "D", "v", "P", "X"]


def resolve_method_styles(methods: Sequence[str], highlight: str,
                          baseline: str = "Raw") -> dict:
    """
    方法 → 线型。列表顺序即入图顺序，不按指标动态排序（审稿人会认为你在挑顺序）。

    本文方法：唯一暖色 + 最粗线 + 唯一实心 marker（三重冗余，去色也能认）。
    Raw：黑色虚线 —— 它是地板，值得被一眼认出。
    其余基线：Okabe–Ito 冷色 + 空心 marker，按给定顺序确定性分配。
    """
    cool = [OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["purple"], OKABE_ITO["sky"]]
    marks = ["s", "^", "v", "P"]
    out, k = {}, 0
    for m in methods:
        if m == highlight:
            out[m] = dict(color=ACCENT, ls="-", lw=1.4, marker="D", ms=3.2,
                          mfc=ACCENT, mec=ACCENT, mew=LW_HAIR, zorder=5)
        elif m == baseline:
            out[m] = dict(color=OKABE_ITO["black"], ls="--", lw=0.8, marker="o", ms=2.8,
                          mfc="white", mec=OKABE_ITO["black"], mew=0.6, zorder=3)
        else:
            out[m] = dict(color=cool[k % len(cool)], ls="-", lw=0.8,
                          marker=marks[k % len(marks)], ms=2.8, mfc="white",
                          mec=cool[k % len(cool)], mew=0.6, zorder=4)
            k += 1
    return out


# ----------------------------------------------------------------------------
# 数据容器
# ----------------------------------------------------------------------------
@dataclass
class StageData:
    """单个 (method, stage) 的数据。emb 用于算指标，xy 仅用于画图。"""

    xy: np.ndarray                       # (n, 2) UMAP 坐标
    group: np.ndarray                    # (n,) 二元分组，取值需落在 GROUP_STYLE 的键里
    emb: Optional[np.ndarray] = None     # (n, d) 高维嵌入；silhouette 应在这上面算
    celltype: Optional[np.ndarray] = None  # (n,) 可选，细粒度标签（Supplemental 用）

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=float)
        self.group = np.asarray(self.group)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError(f"xy 应为 (n, 2)，得到 {self.xy.shape}")
        if len(self.group) != len(self.xy):
            raise ValueError(f"group 长度 {len(self.group)} != xy 行数 {len(self.xy)}")
        if self.emb is not None:
            self.emb = np.asarray(self.emb, dtype=float)
            if len(self.emb) != len(self.xy):
                raise ValueError(f"emb 行数 {len(self.emb)} != xy 行数 {len(self.xy)}")

    @property
    def n(self) -> int:
        return len(self.xy)

    def metric_space(self) -> np.ndarray:
        """算 silhouette 用的空间。有 emb 就用 emb，否则退到 xy 并 warn。"""
        if self.emb is not None:
            return self.emb
        warnings.warn(
            "未提供 emb，silhouette 将在 UMAP 2D 上计算。UMAP 不保距，"
            "该数值会随 UMAP 的 seed / n_neighbors / min_dist 漂移，审稿人会问。"
            "请传入高维嵌入（如 total_decomp.npz 的 20 维）。",
            stacklevel=2,
        )
        return self.xy


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def silhouette_with_ci(
    X: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 200,
    sub_n: int = 400,
    seed: int = 0,
    metric: str = "euclidean",
) -> tuple[float, float, float]:
    """
    silhouette + 子采样重抽样百分位 CI。

    返回 (observed, lo95, hi95)。

    诚实说明 —— 这个 CI 度量的是什么、不度量什么
    -------------------------------------------
    做法：从 n 个细胞里**无放回**抽 m = min(n, sub_n) 个，算一次 silhouette，
    重复 n_boot 次，取 2.5/97.5 百分位。用无放回而非有放回，是因为有放回会产生
    重复点，令 a(i) 被 0 距离污染、silhouette 被系统性抬高。

    它反映：**细胞抽样**在样本量 m 下的波动。
    它不反映：UMAP 的 seed、插补的 seed、SVD 的 seed 带来的波动。
    后者要靠多 seed 重跑整条管线才能拿到 —— 如果你要在正文里声称"方法 A 显著优于
    方法 B"，需要的是那个，不是这个。这个 CI 只够回答"这个数和 0 能不能分开"。
    """
    X = np.asarray(X, dtype=float)
    lab = np.asarray(labels)
    if len(np.unique(lab)) < 2:
        return np.nan, np.nan, np.nan

    obs = float(silhouette_score(X, lab, metric=metric))

    rng = np.random.default_rng(seed)
    n = len(lab)
    m = min(n, sub_n)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=m, replace=False)
        if len(np.unique(lab[idx])) < 2:
            continue
        boots.append(silhouette_score(X[idx], lab[idx], metric=metric))
    if len(boots) < 20:
        return obs, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, float(lo), float(hi)


def compute_all_silhouettes(data: dict, methods: Sequence[str], stages: Sequence[str],
                            **kw) -> dict:
    """对 data[method][stage] 全表算 silhouette，返回 {(method, stage): (obs, lo, hi)}。"""
    out = {}
    for meth in methods:
        for st in stages:
            sd = data[meth][st]
            out[(meth, st)] = silhouette_with_ci(sd.metric_space(), sd.group, **kw)
    return out


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
def panel_letter(fig, x_mm: float, y_mm: float, letter: str, fig_w_mm: float,
                 fig_h_mm: float) -> None:
    """GR: 面板字母 = 大写、12 pt 粗体、Helvetica/Arial。坐标按毫米给，左上角对齐。"""
    fig.text(x_mm / fig_w_mm, 1 - y_mm / fig_h_mm, letter.upper(),
             fontsize=PT_TAG, fontweight="bold", ha="left", va="top")


def _axis_indicator(fig, x_mm: float, top_mm: float, size_mm: float,
                    fig_w_mm: float, fig_h_mm: float, xlabel: str = "UMAP1",
                    ylabel: str = "UMAP2") -> None:
    """
    L 形轴指示器，替代 7×4 = 28 组重复的坐标框与轴标题。
    UMAP 各面板各自独立跑，坐标系之间本就不可比 —— 画刻度反而暗示可比。
    x_mm / top_mm 都是自图左、图顶起算的毫米，与模块其余部分一致。
    """
    ax = fig.add_axes([x_mm / fig_w_mm, 1 - (top_mm + size_mm) / fig_h_mm,
                       size_mm / fig_w_mm, size_mm / fig_h_mm])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate("", xy=(0.52, 0.0), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color=TEXT_MUTED,
                                mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(0.0, 0.52), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color=TEXT_MUTED,
                                mutation_scale=5, shrinkA=0, shrinkB=0))
    ax.text(0.26, -0.16, xlabel, fontsize=PT_SMALL, ha="center", va="top",
            color=TEXT_MUTED)
    ax.text(-0.16, 0.26, ylabel, fontsize=PT_SMALL, ha="right", va="center",
            rotation=90, color=TEXT_MUTED)


def _fmt_sil(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    return f"{v:+.3f}".replace("+0.", "0.").replace("-0.", "\u22120.")


# ----------------------------------------------------------------------------
# 面板 a：silhouette–stage 折线
# ----------------------------------------------------------------------------
def plot_silhouette_by_stage(
    ax,
    sil: dict,
    methods: Sequence[str],
    stages: Sequence[str],
    highlight: str,
    baseline: str = "Raw",
    show_ci: bool = True,
    ylabel: str = "Silhouette (Red vs Blue)",
) -> None:
    """
    方法为线、stage 为 x 的折线图。这是整张图真正的量化落点 ——
    它把 28 个 UMAP 面板压成一条可读的曲线。

    sil: {(method, stage): (obs, lo, hi)}，来自 compute_all_silhouettes。
    """
    styles = resolve_method_styles(methods, highlight, baseline)
    x = np.arange(len(stages))

    # 0 参考线：silhouette 在 0 附近时，这条线是读者唯一的锚
    ax.axhline(0.0, color="#5F5E5A", lw=0.4, ls=":", zorder=1)

    for meth in methods:
        st = styles[meth]
        obs = np.array([sil[(meth, s)][0] for s in stages], dtype=float)
        if show_ci:
            lo = np.array([sil[(meth, s)][1] for s in stages], dtype=float)
            hi = np.array([sil[(meth, s)][2] for s in stages], dtype=float)
            ok = np.isfinite(lo) & np.isfinite(hi)
            if ok.any():
                ax.fill_between(x[ok], lo[ok], hi[ok], color=st["color"],
                                alpha=0.14, lw=0, zorder=st["zorder"] - 1)
        ax.plot(x, obs, **{k: v for k, v in st.items()}, clip_on=False)

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_xlim(-0.35, len(stages) - 0.65)
    ax.set_ylabel(ylabel, fontsize=PT_BASE)
    ax.tick_params(axis="both", labelsize=PT_SMALL, width=LW_HAIR, length=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(LW_HAIR)

    handles = [Line2D([], [], **{k: v for k, v in styles[m].items() if k != "zorder"},
                      label=m) for m in methods]
    ax.legend(handles=handles, loc="upper left", fontsize=PT_SMALL, frameon=False,
              handlelength=2.2, borderaxespad=0.2, labelspacing=0.25)


# ----------------------------------------------------------------------------
# 面板 b：方法 × stage 的 UMAP 网格
# ----------------------------------------------------------------------------
def plot_stage_grid(
    fig,
    data: dict,
    sil: dict,
    methods: Sequence[str],
    stages: Sequence[str],
    highlight: str,
    fig_w_mm: float,
    fig_h_mm: float,
    top_mm: float,
    left_mm: float = 17.0,
    right_mm: float = 1.0,
    gap_mm: float = 1.2,
    header_mm: float = 6.5,
    key_mm: float = 6.0,
    stage_n: Optional[dict] = None,
    point_size: float = 1.4,
    alpha: float = 0.75,
    mode: str = "group",
    lineage_of=None,
) -> float:
    """
    行=方法、列=stage 的 UMAP 网格。所有几何按毫米算 —— 面板是正方形（aspect equal），
    UMAP 坐标不能被拉伸。

    mode='group'    ：只画被度量的二分组（正文用）
    mode='lineage'  ：按 lineage 折叠上色，需给 lineage_of: celltype -> lineage（Supplemental 用）

    返回网格底边距图顶的毫米数，便于调用方继续排下面的内容。
    """
    n_col, n_row = len(stages), len(methods)
    avail = fig_w_mm - left_mm - right_mm - gap_mm * (n_col - 1)
    cell = avail / n_col  # 正方形边长

    grid_top = top_mm + header_mm

    for r, meth in enumerate(methods):
        for c, st in enumerate(stages):
            x0 = left_mm + c * (cell + gap_mm)
            y0 = grid_top + r * (cell + gap_mm)
            ax = fig.add_axes([x0 / fig_w_mm, 1 - (y0 + cell) / fig_h_mm,
                               cell / fig_w_mm, cell / fig_h_mm])
            sd = data[meth][st]

            if mode == "group":
                keys, styles = list(GROUP_STYLE), GROUP_STYLE
                lab = sd.group
            else:
                if lineage_of is None or sd.celltype is None:
                    raise ValueError("mode='lineage' 需要 lineage_of 与 StageData.celltype")
                lab = np.array([lineage_of(t) for t in sd.celltype])
                keys = [k for k in LINEAGE_COLORS if k in set(lab)]
                styles = {k: dict(color=LINEAGE_COLORS[k],
                                  marker=LINEAGE_MARKERS[i % len(LINEAGE_MARKERS)])
                          for i, k in enumerate(LINEAGE_COLORS)}

            for k in keys:
                sel = lab == k
                if not sel.any():
                    continue
                ax.scatter(sd.xy[sel, 0], sd.xy[sel, 1], s=point_size,
                           c=styles[k]["color"], marker=styles[k]["marker"],
                           linewidths=0, alpha=alpha, rasterized=True)

            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            # 图内 silhouette：右下角，8 pt，本文方法加粗
            v = sil[(meth, st)][0]
            # R1：数字是数据，一律黑色常规。哪一行是本文方法，靠行标签的字重认。
            ax.text(0.97, 0.03, _fmt_sil(v), transform=ax.transAxes,
                    fontsize=PT_SMALL, ha="right", va="bottom", color=TEXT_MAIN)

            # 列标题：只在第一行
            if r == 0:
                ax.text(0.5, 1.20, st, transform=ax.transAxes, fontsize=PT_BASE,
                        ha="center", va="bottom", color=TEXT_MAIN)
                if stage_n:  # R4：n 是次要信息，降一级字号 + 灰度
                    ax.text(0.5, 1.04, f"n = {stage_n[st]}", transform=ax.transAxes,
                            fontsize=PT_SMALL, ha="center", va="bottom", color=TEXT_MUTED)

            # 行标签：只在第一列
            if c == 0:
                ax.text(-0.10, 0.5, meth, transform=ax.transAxes, fontsize=PT_BASE,
                        ha="right", va="center", rotation=90,
                        fontweight="bold" if meth == highlight else "normal")

        # R1：本文方法的强调是**元信息**，不占颜色预算 -> 黑色细规线 + 粗体行标签
        if meth == highlight:
            y0 = grid_top + r * (cell + gap_mm)
            bar = fig.add_axes([(left_mm - 1.8) / fig_w_mm, 1 - (y0 + cell) / fig_h_mm,
                                0.6 / fig_w_mm, cell / fig_h_mm])
            bar.set_axis_off()
            bar.add_patch(plt.Rectangle((0, 0), 1, 1, transform=bar.transAxes,
                                        color=TEXT_MAIN, lw=0))

    grid_bot = grid_top + n_row * cell + (n_row - 1) * gap_mm

    # 色标：GR 明文要求放图内，不放图注。横排一次。
    if mode == "group":
        entries = [(k, GROUP_STYLE[k]["color"], GROUP_STYLE[k]["marker"])
                   for k in GROUP_STYLE]
    else:
        seen = []
        for meth in methods:
            for st in stages:
                sd = data[meth][st]
                if sd.celltype is not None:
                    seen += [lineage_of(t) for t in sd.celltype]
        # marker 必须按 k 在 LINEAGE_COLORS 里的固定位置取，
        # 不能按 uniq 的位置 —— 否则色标 marker 会和面板里的对不上。
        order = list(LINEAGE_COLORS)
        entries = [(k, LINEAGE_COLORS[k],
                    LINEAGE_MARKERS[order.index(k) % len(LINEAGE_MARKERS)])
                   for k in order if k in set(seen)]

    kax = fig.add_axes([left_mm / fig_w_mm, 1 - (grid_bot + key_mm) / fig_h_mm,
                        (fig_w_mm - left_mm - right_mm) / fig_w_mm, key_mm / fig_h_mm])
    kax.set_axis_off()
    handles = [Line2D([], [], ls="none", marker=mk, ms=3.0, mfc=col, mec=col, label=k)
               for k, col, mk in entries]
    kax.legend(handles=handles, loc="center", ncol=min(len(entries), 7),
               fontsize=PT_SMALL, frameon=False, handletextpad=0.3,
               columnspacing=1.4, borderaxespad=0)

    _axis_indicator(fig, x_mm=4.0, top_mm=grid_bot - 9.0, size_mm=6.0,
                    fig_w_mm=fig_w_mm, fig_h_mm=fig_h_mm)
    return grid_bot + key_mm


# ----------------------------------------------------------------------------
# 合成整图
# ----------------------------------------------------------------------------
def compose_figure(
    data: dict,
    sil: dict,
    methods: Sequence[str],
    stages: Sequence[str],
    highlight: str,
    baseline: str = "Raw",
    stage_n: Optional[dict] = None,
    mode: str = "group",
    lineage_of=None,
    fig_w_mm: float = 174.0,
    a_h_mm: float = 40.0,
) -> plt.Figure:
    """174 mm 双栏整图：a = silhouette–stage 折线，b = 方法 × stage 的 UMAP 网格。"""
    n_col, n_row = len(stages), len(methods)
    left_mm, right_mm, gap_mm = 17.0, 1.0, 1.2
    cell = (fig_w_mm - left_mm - right_mm - gap_mm * (n_col - 1)) / n_col

    a_top, a_left = 5.0, 17.0
    b_top = a_top + a_h_mm + 8.0
    header_mm, key_mm = 6.5, 6.0
    fig_h_mm = b_top + header_mm + n_row * cell + (n_row - 1) * gap_mm + key_mm + 3.0

    fig = plt.figure(figsize=(mm(fig_w_mm), mm(fig_h_mm)))

    ax_a = fig.add_axes([a_left / fig_w_mm, 1 - (a_top + a_h_mm) / fig_h_mm,
                         (fig_w_mm - a_left - 2.0) / fig_w_mm, a_h_mm / fig_h_mm])
    plot_silhouette_by_stage(ax_a, sil, methods, stages, highlight, baseline)

    plot_stage_grid(fig, data, sil, methods, stages, highlight,
                    fig_w_mm=fig_w_mm, fig_h_mm=fig_h_mm, top_mm=b_top,
                    left_mm=left_mm, right_mm=right_mm, gap_mm=gap_mm,
                    header_mm=header_mm, key_mm=key_mm, stage_n=stage_n,
                    mode=mode, lineage_of=lineage_of)

    panel_letter(fig, 3.0, 3.0, "a", fig_w_mm, fig_h_mm)
    panel_letter(fig, 3.0, b_top - 3.5, "b", fig_w_mm, fig_h_mm)
    return fig


# ----------------------------------------------------------------------------
# 导出
# ----------------------------------------------------------------------------
def save_figure(fig: plt.Figure, stem: str, outdir="out",
                first_author: Optional[str] = None, fig_no: Optional[int] = None):
    """
    导出矢量 PDF（GR 首选，真矢量不受 dpi 约束）+ 600 dpi PNG 预览。

    注意：故意不传 bbox_inches='tight' —— 裁剪会改变实际栏宽，
    8 pt 在成品上就不再是 8 pt。画布宽度写死 174 mm 才有意义。

    GR 文件命名：FirstAuthor_FigN（拿到稿号后是 FirstAuthorNNNN_FigN）。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if first_author and fig_no:
        stem = f"{first_author}_Fig{fig_no}"
    pdf, png = outdir / f"{stem}.pdf", outdir / f"{stem}.png"
    fig.savefig(pdf)                 # 无 bbox_inches
    fig.savefig(png, dpi=600)
    return pdf, png


# ----------------------------------------------------------------------------
# 自检：把「配色是否合规」变成可跑的断言，而不是 docstring 里的一句声明
# ----------------------------------------------------------------------------
def verify_palette(cmap=None, verbose: bool = True) -> dict:
    """
    实测三件事，返回 {check: pass/fail}。需要 colorspacious。

    1. 序列色图的感知明度 J'（CAM02-UCS）是否单调 —— 不单调的色图会在数据里
       造出不存在的「亮带」，读者会当成结构。
    2. 分类色在 protan / deutan / tritan 下的两两 deltaE 是否 >15。
    3. **同色异义检查** —— 同一个 hex 有没有被派了两份工。
       v1 就是栽在这条：deltaE 恒为 0，任何调色板都救不了。
    """
    try:
        from colorspacious import cspace_convert, deltaE
    except ImportError:
        warnings.warn("verify_palette 需要 colorspacious：pip install colorspacious")
        return {}
    from matplotlib.colors import to_hex, to_rgb

    out = {}
    if cmap is not None:
        J = cspace_convert(cmap(np.linspace(0, 1, 32))[:, :3], "sRGB1", "CAM02-UCS")[:, 0]
        d = np.diff(J)
        out["cmap_lightness_monotonic"] = bool((d < 0).all() or (d > 0).all())
        out["cmap_step_std"] = float(d.std())

    cvds = {"normal": None,
            "protan": {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 100},
            "deutan": {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100},
            "tritan": {"name": "sRGB1+CVD", "cvd_type": "tritanomaly", "severity": 100}}
    cats = {k: v["color"] for k, v in GROUP_STYLE.items()}
    worst = np.inf
    for a in cats:
        for b in cats:
            if a >= b:
                continue
            for cvd in cvds.values():
                ca, cb = to_rgb(cats[a]), to_rgb(cats[b])
                if cvd:
                    ca = cspace_convert(ca, cvd, "sRGB1")
                    cb = cspace_convert(cb, cvd, "sRGB1")
                worst = min(worst, deltaE(np.clip(ca, 0, 1), np.clip(cb, 0, 1), "sRGB1"))
    out["categorical_min_deltaE"] = float(worst) if np.isfinite(worst) else None
    out["categorical_cvd_safe"] = bool(np.isfinite(worst) and worst > 15)

    # R1 同色异义：把每个 hex 映到它承担的所有语义角色
    jobs = {}
    for k, v in GROUP_STYLE.items():
        jobs.setdefault(to_hex(v["color"]).lower(), []).append(f"data:{k}")
    # 强调与标注在 v2 里已改为 achromatic，故不再登记任何 hex
    dup = {h: r for h, r in jobs.items() if len(r) > 1}
    out["one_color_one_job"] = not dup
    out["duplicate_jobs"] = dup

    if verbose:
        for k, v in out.items():
            print(f"  {k:<28} {v}")
    return out
