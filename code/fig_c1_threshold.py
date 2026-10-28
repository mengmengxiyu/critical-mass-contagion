# -*- coding: utf-8 -*-
"""论文图：c=1 时入侵阈值随 beta 降低，household 计算定量给出（§4.3 前半）。

(a) E4 单种子爆发比例 P_out 对 lambda（读 nucleation_figB.csv，4000 次/点，m=6、d=12、N=5000）
    灰色虚线是单节点闭合的阈值 -ln[1-mu/(d(m-1))]（与 beta 无关）；
    插图放大起飞区（线性横轴），横轴上三个彩色三角是 household 阈值 lambda_c(beta)。
    误差棒为二项分布的 Wilson 95% 区间。

(b)(c) E5 阈值以下单个种子的平均最终感染人数，d=3 与 d=12（读 household_check.csv，household.py mc 产出）。
    点为模拟，线为 1 + d E[Z] / (1 - R*)，零可调参数；竖线是各 beta 的 lambda_c，
    曲线在各自的 lambda_c 处发散，故 lambda_c 随 beta 增大而降低是直接可见的。

运行（在 god/data/ 下）： PYTHONPATH=../code python ../code/fig_c1_threshold.py   （秒级，读现成 csv）
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import fig_style as st
from household import lam_c, mean_size

st.use()

m, d, mu = 6, 12, 0.2
LAM_C = -np.log1p(-mu / (d * (m - 1)))     # 单节点闭合的线性阈值
D_BC = [3, 12]                             # (b)(c) 展示的超度
# 三个面板共用一套 beta 配色：(a) 用 beta=1,2,3，(b)(c) 用 beta=0,1,3
B_COL = {0.0: st.BETA_SEQ[0], 1.0: st.BETA_SEQ[1], 2.0: st.BETA_SEQ[2], 3.0: st.BETA_SEQ[4]}

fig, axes = plt.subplots(1, 3, figsize=(st.COL2, 2.5),
                         gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
fig.subplots_adjust(left=0.065, right=0.99, top=0.935, bottom=0.17, wspace=0.30)

# ================================================================ (a) E4 P_out
ax = axes[0]
B = pd.read_csv("nucleation_figB.csv")

ax.axvline(LAM_C, color="0.30", lw=0.7, ls=(0, (4, 2.4)), zorder=1)
ax.text(LAM_C * 0.93, 0.30, "linearization", fontsize=6.4,
        color="0.25", va="bottom", ha="right")
ins = ax.inset_axes([0.60, 0.125, 0.36, 0.36])        # 起飞区放大
ins.axvline(LAM_C, color="0.30", lw=0.7, ls=(0, (4, 2.4)), zorder=1)

for beta in sorted(B.beta.unique()):
    s = B[B.beta == beta].sort_values("lam")
    z, n, p = 1.96, s.n_run, s.P_out                 # Wilson 95% 区间
    mid = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    for a in (ax, ins):
        a.errorbar(s.lam, p, yerr=[p - (mid - half), mid + half - p], fmt="o-",
                   color=B_COL[beta], ms=2.6, lw=0.8, mfc="w", mew=0.7, elinewidth=0.6,
                   capsize=1.2, capthick=0.5, label=rf"$\beta={beta:g}$", zorder=3)
    ins.plot(lam_c(beta, m, d, mu), -0.02, "^", color=B_COL[beta], ms=4.2, mec="0.2",
             mew=0.4, clip_on=False, zorder=6)
ins.set_xlim(2.4e-3, 4.6e-3); ins.set_ylim(-0.02, 0.34)
ins.set_xticks([3e-3, 4e-3]); ins.set_xticklabels(["0.003", "0.004"])
ins.set_yticks([0, 0.1, 0.2, 0.3])
ins.tick_params(labelsize=5.8, length=2.0, pad=1.5)
ins.set_title("triangles: household " r"$\lambda_c(\beta)$", fontsize=5.8, color="0.25",
              pad=2.5)

ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$", labelpad=2)
ax.set_ylabel(r"fraction with $R(\infty)>0.05$", labelpad=2)
ax.set_xlim(0.9e-3, 3.4e-2); ax.set_ylim(-0.035, 1.05)
lg = ax.legend(loc="upper left", framealpha=0.95, handlelength=1.6,
               labelspacing=0.26, borderpad=0.4, handletextpad=0.5,
               borderaxespad=0.4, title=rf"single seed, $d={d}$", fontsize=6.4)
lg.get_title().set_fontsize(6.4)
st.panel_label(ax, "(a)", x=-0.17, y=0.99)

# ================================================================ (b)(c) E5 亚临界平均规模
H = pd.read_csv("household_check.csv")
for ax, d_bc, panel in zip(axes[1:], D_BC, ["(b)", "(c)"]):
    for beta, s in H[H.d == d_bc].groupby("beta"):
        lc = s.lam_c.iloc[0]
        lam = np.linspace(0.25, 0.96, 80) * lc
        col = B_COL[beta]
        ax.axvline(lc, color=col, lw=0.8, ls=(0, (4, 2.4)), zorder=1)
        ax.plot(lam, [mean_size(l, beta, m, d_bc, mu) for l in lam], color=col,
                lw=1.0, zorder=2)
        ax.errorbar(s.lam, s.size_mc, yerr=1.96 * s.size_se, fmt="o", color=col,
                    ms=2.8, mfc="w", mew=0.8, elinewidth=0.7, capsize=1.2, capthick=0.6,
                    label=rf"$\beta={beta:g}$", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$", labelpad=2)
    ax.set_ylim(1, 90)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.xaxis.get_offset_text().set_fontsize(6.5)
    lg = ax.legend(loc="upper left", framealpha=0.95, handlelength=1.2, labelspacing=0.26,
                   borderpad=0.4, handletextpad=0.4, borderaxespad=0.4,
                   title=rf"single seed, $d={d_bc}$", fontsize=6.4)
    lg.get_title().set_fontsize(6.4)
    st.panel_label(ax, panel, x=-0.19, y=0.99)
axes[1].set_ylabel("mean outbreak size", labelpad=2)

st.save(fig, "fig_c1_threshold")

# ---------------------------------------------------------------- 数值自检
print(f"\n单节点闭合 lambda_c = {LAM_C:.5f}；household lambda_c = "
      + " / ".join(f"{lam_c(b, m, d, mu):.5f}" for b in (1.0, 2.0, 3.0)))
