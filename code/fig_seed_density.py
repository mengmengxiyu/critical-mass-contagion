# -*- coding: utf-8 -*-
"""论文图：c>=2 时能否传开由种子密度决定（§4.2）。投稿规格，双栏宽 7.0 in，1:1 出图不缩放。

(a) E1 最终规模 R 对初始种子数（读 nucleation_figA.csv）
    c>=2 时无穷小种子无法触发传播。判据是曲线是否离开 R = rho_0 参考线 ——
    落在线上即「种子自己恢复，未传染任何人」。c=3 时即便 lambda=5、100 个种子仍完全贴线。

(b)(c) E3 种子密度边界的有限尺寸检验，c=2 与 c=3（读 seed_floor_curve.csv，seed_floor.py nscan 产出）
    lambda=5、beta=2、m=6、d=12、mu=0.2，10 网络 x 100 次；爆发判据 R(inf) > 1/2。
    每个 N 一条线，浅到深为 N 由小到大；误差棒为对网络 bootstrap（网络内再二项重抽）的 95% 区间。
    竖虚线为平均场估计 rho*_MF（增长条件在 lambda -> inf 的根）。

运行（在 god/data/ 下）： PYTHONPATH=../code python ../code/fig_seed_density.py   （秒级，读现成 csv）
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fig_style as st

st.use()

NA = 10000                                 # E1 的系统尺寸
LAM_MK = {0.2: ("o", "-"), 1.0: ("s", "--"), 5.0: ("^", ":")}
I_MIN = {2: 0.001675, 3: 0.043223}
LEG_POS = {2: dict(loc="upper left", bbox_to_anchor=(0.09, 1.0)),   # rho*_MF 线与曲线之间的空白
           3: dict(loc="lower right", bbox_to_anchor=(1.0, 0.08))}
XTICKS = {2: [0.002, 0.003, 0.004, 0.006], 3: [0.04, 0.05, 0.06, 0.08]}
N_SEQ = ["#A6C8E0", "#6BAED6", "#2E7EBB", "#062F5C"]     # N 由小到大，浅到深
N_BOOT = 2000

fig, axes = plt.subplots(1, 3, figsize=(st.COL2, 2.6),
                         gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
fig.subplots_adjust(left=0.065, right=0.99, top=0.93, bottom=0.17, wspace=0.30)

# ================================================================ (a) E1
ax = axes[0]
A = pd.read_csv("nucleation_figA.csv")
seeds = np.sort(A.n_seed.unique())

# 参考线 R = rho_0。画成浅灰粗带，曲线叠在其上时两侧仍露边，才看得出 c=3 是「完全贴线」
ax.plot(seeds, seeds / NA, color="0.80", lw=2.6, solid_capstyle="round", zorder=1)

for c in sorted(A.c.unique()):
    for lam in sorted(A.lam.unique()):
        s = A[(A.c == c) & (A.lam == lam)].sort_values("n_seed")
        mk, ls = LAM_MK[lam]
        ax.plot(s.n_seed, s.R, marker=mk, ls=ls, color=st.C_COL[c], ms=3.0,
                lw=0.7, mfc="w", mew=0.7, zorder=3)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\rho_0 N$", labelpad=2)
ax.set_ylabel(r"$R(\infty)$", labelpad=2)
ax.set_xlim(0.85, 160); ax.set_ylim(2.6e-5, 3.4)

# 曲线旁直接着色标注，省去第二个图例
ax.text(1.12, 1.55, r"$c=1$:  a single seed suffices",
        fontsize=6.6, color=st.C_COL[1], va="bottom")
ax.text(1.12, 0.125, r"$c=2$:  spreads" "\n" r"above $\sim\!20$ seeds",
        fontsize=6.6, color=st.C_COL[2], va="top")
ax.annotate(r"$c=3$:  never leaves the line," "\n" r"even at $\lambda=5$",
            xy=(47, 5.05e-3), xytext=(1.12, 9.0e-3), fontsize=6.6,
            color=st.C_COL[3], va="top",
            arrowprops=dict(arrowstyle="-|>", color=st.C_COL[3], lw=0.9,
                            mutation_scale=8,
                            connectionstyle="arc3,rad=-0.16", shrinkB=3))
ax.text(1.26, 5.3e-5, r"$R=\rho_0$: seeds recover without infecting anyone",
        fontsize=5.8, color="0.45", rotation=21, va="bottom")

# 图例两列：左列 c（颜色），右列 lambda（线型 + 标记）
leg_c = [Line2D([], [], color=st.C_COL[c], lw=1.1, label=rf"$c={c}$")
         for c in sorted(st.C_COL)]
leg_l = [Line2D([], [], color="0.35", lw=0.7, ls=LAM_MK[l][1], marker=LAM_MK[l][0],
                ms=3.0, mfc="w", mew=0.7, label=rf"$\lambda={l:g}$")
         for l in sorted(LAM_MK)]
ax.legend(handles=leg_c + leg_l, ncol=2, loc="lower right", framealpha=0.95,
          handlelength=2.0, labelspacing=0.26, borderpad=0.4, columnspacing=1.0,
          handletextpad=0.5, borderaxespad=0.4, fontsize=6.4)
st.panel_label(ax, "(a)", x=-0.16, y=0.99)

# ================================================================ (b)(c) E3
df = pd.read_csv("seed_floor_curve.csv")
rng = np.random.default_rng(0)
for ax, c, lab in zip(axes[1:], [2, 3], ["(b)", "(c)"]):
    sub = df[df.c == c]
    Ns = sorted(sub.N.unique())
    for Nv, col in zip(Ns, N_SEQ[-len(Ns):]):
        s = sub[sub.N == Nv]
        k = s.pivot(index="net", columns="rho0", values="n_out").to_numpy()
        n = s.n_run.iloc[0]
        x = np.sort(s.rho0.unique())
        P = k.sum(0) / (n * len(k))
        boot = np.array([rng.binomial(n, k[rng.integers(0, len(k), len(k))] / n).sum(0)
                         / (n * len(k)) for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        ax.errorbar(x, P, yerr=[P - lo, hi - P], color=col, marker="o", ms=2.8, lw=0.9,
                    mfc="w", mew=0.7, capsize=1.2, elinewidth=0.6,
                    label=rf"$N={Nv:,}$".replace(",", r"\,"))
    ax.axvline(I_MIN[c], color=st.GREY, ls=(0, (4, 2.4)), lw=0.8)
    ax.text(I_MIN[c] * 1.04, 0.08, r"$\rho^*_{\rm MF}$", color=st.GREY, fontsize=7.5)
    ax.axhline(0.5, color="0.8", lw=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(I_MIN[c] * 0.9, sub.rho0.max() * 1.1)
    ax.set_xticks(XTICKS[c]); ax.set_xticklabels([f"{t:g}" for t in XTICKS[c]])
    ax.set_xticks([], minor=True)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(r"$\rho_0$", labelpad=2)
    lg = ax.legend(frameon=False, **LEG_POS[c], handlelength=1.2, labelspacing=0.22,
                   title=rf"$c={c}$", fontsize=6.4)
    lg.get_title().set_fontsize(7.0)
    st.panel_label(ax, lab, x=-0.20, y=0.99)
axes[1].set_ylabel(r"fraction with $R(\infty)>1/2$", labelpad=2)

st.save(fig, "fig_seed_density")

# ---------------------------------------------------------------- 数值自检
print("\n各 c 下 R/rho_0 的最大放大倍数（=1 即完全无传播）：")
print((A.assign(amp=A.R / A.rho0).groupby(["c", "lam"]).amp.max()
       .unstack().round(1).to_string()))
