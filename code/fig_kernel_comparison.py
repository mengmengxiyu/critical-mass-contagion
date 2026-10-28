# -*- coding: utf-8 -*-
"""论文 Fig. 1：核函数对照。投稿规格，双栏宽 7.0 in，1:1 出图不缩放。

(a) 模型示意：超边内感染人数 n 未达临界质量 c 时沉默，达到后释放 n^beta 压力。
(b) 三个核的取值对照：n<c 段与临界质量核相同，n>=c 段与幂律核相同（底色分区标出）。

对照的两个母模型（Nature Rev. Phys. 6, 468 (2024) 的统一形式）：
    式 (6) 幂律强化   f = n^beta         St-Onge et al., PRL 127, 158301 (2021)（原文记作 nu）
    式 (7) 临界质量   f = 1{n >= c}      Arruda et al., PRR 2, 023032 (2020)
本文为二者相乘。2026-10-09 起不再画「启动那一刻的压力 1 vs c^beta」inset：
c^beta 可并进传播率（Yamir 意见第一点），不再作为卖点。

运行： python fig_kernel_comparison.py   （秒级，无需数据文件）
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse
import fig_style as st

st.use()

M, C, BETA = 6, 3, 2.0           # 超边大小、临界质量、非线性指数

fig = plt.figure(figsize=(st.COL2, 2.62))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.20,
                      left=0.030, right=0.990, top=0.915, bottom=0.155)

# ================================================================ (a) 示意
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 10.0); ax.set_ylim(0, 5.0); ax.axis("off")


def draw_hyperedge(cx, cy, n_inf, active):
    """画一条 m=6 的超边：n_inf 个感染成员（红），其余易感（蓝）。"""
    ax.add_patch(Ellipse((cx, cy), 3.7, 1.30, facecolor="none",
                         edgecolor="0.30" if active else "0.68",
                         lw=1.4 if active else 0.8,
                         ls="-" if active else (0, (4, 2.4)), zorder=1))
    xs = np.linspace(cx - 1.32, cx + 1.32, M)
    ax.scatter(xs, np.full(M, cy), s=40, zorder=3, linewidths=0.5,
               c=[st.N_INF if k < n_inf else st.N_SUS for k in range(M)],
               edgecolors="0.25")


draw_hyperedge(3.05, 3.90, 2, active=False)
ax.text(3.05, 4.76, rf"$n=2 \;<\; c={C}$", ha="center", va="center", fontsize=7.2)
ax.text(5.15, 3.90, "silent\n(no pressure)", ha="left", va="center",
        fontsize=6.6, color="0.45", style="italic")

draw_hyperedge(3.05, 1.95, 3, active=True)
ax.text(3.05, 2.81, rf"$n=3 \;\geq\; c={C}$", ha="center", va="center", fontsize=7.2)
ax.add_patch(FancyArrowPatch((5.05, 1.95), (6.10, 1.95), lw=1.1,
                             color="0.25", arrowstyle="-|>", mutation_scale=9))
ax.text(5.57, 2.14, "activate", ha="center", va="bottom", fontsize=6.3, color="0.30")
ax.text(6.26, 1.95, rf"$n^{{\beta}} = {C}^{{\beta}}$",
        ha="left", va="center", fontsize=9.5, color=st.K_THIS)

ax.text(4.30, 0.52,
        r"$f(n)=n^{\beta}\,\mathbf{1}\{n \geq c\}$"
        r"$\qquad \Phi_i=\!\!\sum_{j\,:\,i\in e_j}\!\! f(n_j)$"
        r"$\qquad p_i=1-e^{-\lambda\Phi_i}$",
        ha="center", va="center", fontsize=7.4)

st.panel_label(ax, "(a)", x=-0.015, y=0.985)

# ================================================================ (b) 核对照
ax = fig.add_subplot(gs[1])
n = np.arange(0, M)
f_pl   = n.astype(float) ** BETA                      # 式 (6)
f_cm   = (n >= C).astype(float)                       # 式 (7)
f_this = np.where(n >= C, n.astype(float) ** BETA, 0.0)

# 底色分区：n<c 段与临界质量核重合，n>=c 段与幂律核重合（正文 §2.1 那句话的图示）
XL, XR = -0.28, M - 0.55
ax.axvspan(XL, C, color=st.K_CRITMASS, alpha=0.07, lw=0, zorder=0)
ax.axvspan(C, XR, color=st.K_POWERLAW, alpha=0.09, lw=0, zorder=0)
ax.axvline(C, color="0.70", lw=0.7, zorder=0)
ax.text(1.0, 12.0, "$n<c$\nsame as\ncritical mass", fontsize=7,
        color=st.K_CRITMASS, ha="center", va="center", linespacing=1.4)
ax.text(3.75, 21.5, "$n\\geq c$\nsame as\npower law", fontsize=7,
        color=st.K_POWERLAW, ha="center", va="center", linespacing=1.4)
ax.plot(n, f_pl, "o--", color=st.K_POWERLAW, ms=3.6, lw=1.0, mfc="w", mew=1.0,
        label=r"$n^{\beta}$")
ax.plot(n, f_this, "o-", color=st.K_THIS, ms=3.8, lw=1.2,
        label=r"$n^{\beta}\mathbf{1}\{n\geq c\}$", zorder=5)
# 空心方块画在最上层：n<c 处三核同为 0，套在实心圆外才看得出重合
ax.plot(n, f_cm, "s:", color=st.K_CRITMASS, ms=6.0, lw=1.0, mfc="none", mew=1.1,
        label=r"$\mathbf{1}\{n\geq c\}$", zorder=6)

ax.text(C + 0.06, 27.2, rf"$c={C}$", fontsize=6.5, color="0.45", ha="left", va="top")
ax.set_xlabel(r"$n$", labelpad=2)
ax.set_ylabel(r"$f(n)$", labelpad=2)
ax.set_xlim(XL, XR); ax.set_ylim(-1.4, 27.8)
ax.set_xticks(n)

h, l = ax.get_legend_handles_labels()          # 两个母模型在前，本文在后
order = [0, 2, 1]
ax.legend([h[i] for i in order], [l[i] for i in order],
          loc="upper left", framealpha=1.0, borderpad=0.4,
          handlelength=1.7, labelspacing=0.28,
          handletextpad=0.5, borderaxespad=0.35).set_zorder(20)
st.panel_label(ax, "(b)", x=-0.135, y=0.985)

st.save(fig, "fig_kernel_comparison")

print(f"  f(n) for n=0..{M-1}")
print(f"    Eq.(1) power law    : {f_pl.astype(int).tolist()}")
print(f"    Eq.(2) critical mass: {f_cm.astype(int).tolist()}")
print(f"    this work           : {f_this.astype(int).tolist()}")
