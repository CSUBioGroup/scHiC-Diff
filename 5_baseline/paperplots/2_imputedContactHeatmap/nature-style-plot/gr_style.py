"""
Genome Research (CSHL Press) 合规绘图配置 —— 热图网格专用
=========================================================

硬规范（来自 GR Digital Art 接收期要求）
---------------------------------------
* 图内字体 Helvetica / Arial，8–10 pt；同一张图内字号浮动 <= 2 pt
* 面板字母 A/B/C：12 pt 粗体大写（唯一允许超出 10 pt 的元素）
* 所有描边 >= 0.25 pt
* 优先提交矢量 PDF/EPS，字体嵌入（Type 42）；栅格组合图 600–900 dpi
* 避免浅色（浅黄/浅绿印刷会褪色），配色需对色盲（CVD）友好
* 色标（scale bar / colorbar）放在图内；描述性标题放图注，不放图上
* 双栏宽 174 mm ≈ 6.85 in；单栏宽 85 mm ≈ 3.35 in
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- 尺寸
MM = 1.0 / 25.4
GR_SINGLE_COL_IN = 85 * MM      # 3.346 in
GR_DOUBLE_COL_IN = 174 * MM     # 6.850 in

# ---------------------------------------------------------------- 字号
FS_TICK = 8      # 刻度 / colorbar 刻度
FS_ANNOT = 8     # 面板内 PCC 标注
FS_LABEL = 9     # 行标签 / 轴标签 / colorbar 标题
FS_TITLE = 10    # 列标题（方法名）
FS_PANEL = 12    # 面板字母，粗体大写

# ---------------------------------------------------------------- 线宽
LW_SPINE = 0.5     # 面板外框
LW_HERO = 1.2      # 本方法高亮框
LW_MIN = 0.25      # GR 下限

# ---------------------------------------------------------------- 颜色
# CVD-safe。hero 用朱红 (vermilion, Okabe-Ito)，是红绿色盲下最稳的强调色。
C_HERO = "#D55E00"
C_INK = "#1A1A1A"        # 正文黑（不用纯黑，印刷更柔和）
C_GRID = "#4D4D4D"       # 面板外框灰
C_MUTED = "#6E6E6E"      # 次级文字

# 接触图色图：单序列 白 -> 深红。零点=白，符合 Hi-C 惯例，避免浅黄。
CMAP_HIC = LinearSegmentedColormap.from_list(
    "hic_fall",
    ["#FFFFFF", "#FDE0D2", "#FBAF8B", "#F26B43", "#CB2114", "#67000D"],
    N=256,
)

# 备选：需要 log/发散语义时用（例如画 O/E 或差异图）
CMAP_DIVERGING = "RdBu_r"

CMAP_REGISTRY = {
    "hic_fall": CMAP_HIC,
    "Reds": plt.get_cmap("Reds"),
    "afmhot_r": plt.get_cmap("afmhot_r"),
    "icefire": "icefire",   # 兼容旧图，需 seaborn
    "RdBu_r": plt.get_cmap("RdBu_r"),
}


def get_cmap(name: str):
    cm = CMAP_REGISTRY.get(name, name)
    if isinstance(cm, str):
        import seaborn as sns
        return sns.color_palette(cm, as_cmap=True)
    return cm


# ---------------------------------------------------------------- rcParams
def apply_gr_style() -> None:
    """把 GR 规范固化进 matplotlib 全局配置。脚本开头调一次即可。"""
    mpl.rcParams.update({
        # 字体：Arial 优先，回退到度量兼容的 Liberation/Nimbus/Helvetica
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                            "Nimbus Sans", "DejaVu Sans"],
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,

        # 线宽全部 >= 0.25 pt
        "axes.linewidth": LW_SPINE,
        "xtick.major.width": LW_SPINE,
        "ytick.major.width": LW_SPINE,
        "lines.linewidth": 0.8,
        "patch.linewidth": LW_SPINE,

        # 颜色
        "text.color": C_INK,
        "axes.labelcolor": C_INK,
        "axes.edgecolor": C_GRID,
        "xtick.color": C_INK,
        "ytick.color": C_INK,

        # mathtext ($W$, $r$) 默认会拉 DejaVu 进来，破坏「全图 Arial」要求
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial:regular",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.default": "it",

        # 矢量输出：字体嵌入为 Type 42（TrueType），保持可编辑
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        # 图形背景透明，方便后期在 AI 里合版
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        # 不做 tight 裁剪：图幅严格等于 GR 栏宽，字号在成品上就是标称字号
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0.0,
    })


def save_gr(fig, stem, outdir=".", raster_dpi=600, formats=("pdf", "png")):
    """一次输出投稿用矢量 + 校样用高分辨率栅格。"""
    from pathlib import Path
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        p = outdir / f"{stem}.{ext}"
        # Vector formats retain vector text/lines; dpi controls embedded raster artists.
        fig.savefig(p, dpi=raster_dpi)
        paths.append(p)
    return paths


def robust_vmax(matrix, pct=99.0):
    """用分位数而非 max 定上限，避免个别极端 bin 把整张图压白。"""
    m = np.asarray(matrix)
    m = m[np.isfinite(m)]
    if m.size == 0:
        return 1.0
    v = np.percentile(m, pct)
    return float(v) if v > 0 else float(m.max() or 1.0)
