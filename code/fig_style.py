# -*- coding: utf-8 -*-
"""论文配图的统一样式：投稿规格的 rcParams、栏宽常量、配色。

全部绘图脚本共用（fig_kernel_comparison / fig_seed_density / fig_c1_threshold / fig_beta_compare / fig_collapse / fig_hetero），
改配色或字号只改这里一处。

三条硬约束，都是投稿要求，别改回去：
  1. pdf.fonttype = 42（TrueType）。matplotlib 默认 3，不少期刊直接拒收 Type 3 字体。
  2. 字体用 TeX Gyre Termes。本机没有 Times New Roman，写 "Times New Roman" 会静默
     fallback 到 DejaVu Serif，与正文 Times 系不搭。Termes 是 Times 的度量兼容克隆，
     与 核心机制及证明.tex 的 \\setmainfont 一致。
  3. 图按目标栏宽 1:1 出图，不靠 LaTeX 缩放。字号即最终字号，缩放会让 8pt 变 5pt。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 栏宽（inch）
COL1 = 3.375     # PRE / PRR 单栏
COL2 = 7.000     # PRE / PRR 双栏

# ---------------------------------------------------------------- 配色
# 临界质量 c 的三档（Fig 2）
C_COL = {1: "#0072B2", 2: "#E69F00", 3: "#C0392B"}
# 三个核（Fig 1）：两个母模型 + 本文
K_POWERLAW = "#E69F00"      # 式 (6) St-Onge
K_CRITMASS = "#0072B2"      # 式 (7) Arruda
K_THIS     = "#D55E00"      # 本文
# 非线性指数 beta 的五档（beta = 1.0, 1.5, 2.0, 2.5, 3.0），浅到深的单色系。
# 单色系而非彩虹色是有意的：Fig 2(b) 与 Fig 4 都要传达「不同 beta 的曲线彼此重合」，
# 同族深浅比多色更能读出这一点。Fig 2(b) 只有三档，取 [0]/[2]/[4]。
# 选蓝色系还因为 Fig 2(b) 全部是 c=1，与 (a) 中 c=1 的蓝一致。
BETA_SEQ = ["#A6C8E0", "#6BAED6", "#2E7EBB", "#12558E", "#062F5C"]
# 节点状态（Fig 1 示意）
N_INF, N_SUS = "#C44E52", "#9FBFDC"
GREY = "0.42"


def use(scale=1.0):
    """套用投稿规格 rcParams。scale 仅在需要整体放大字号时使用（默认 1:1）。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size":        8.0 * scale,
        "axes.labelsize":   9.0 * scale,
        "axes.titlesize":   9.0 * scale,
        "xtick.labelsize":  7.5 * scale,
        "ytick.labelsize":  7.5 * scale,
        "legend.fontsize":  7.0 * scale,
        "axes.linewidth":   0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # TrueType，非 Type 3
        "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
    })


def panel_label(ax, text, x=-0.155, y=1.02, size=10):
    """左上角的 (a)/(b) 标签，四张图统一位置与字号。"""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left")


def save(fig, stem, dpi=600):
    """同时出 PDF（投稿）与 PNG（预览 / 汇报）。"""
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=dpi)
    print(f"saved: {stem}.pdf / .png")
