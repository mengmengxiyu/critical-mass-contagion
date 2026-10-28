# -*- coding: utf-8 -*-
r"""论文 Fig. 4：成核阈值的 c^beta 标度塌缩。投稿规格，双栏宽 7.0 in，1:1 出图。

三联对比，读 nucleation_collapse.csv（由 nucleation_run.py + nucleation_analysis.py 产出）
与 onestep_lambda.csv（第一步增长阈值，onestep_lambda.py 产出）：
  (a) 塌缩前的 lambda_grow          —— 未归一化
  (b) 最低阶幂律式（eq:nucleation）归一化
  (c) 数值精确解（eq:grow）归一化    —— 不做任何渐近近似；红色叉号是第一步增长阈值 / 同一理论值

误差棒为网络间几何标准误的 95% 区间（10 网络），多数小于符号。

图内不写公式编号：编号随正文增删公式而变，写死会静默失效。面板含义一律由 caption 里的
\eqref 交代（自动跟随编号）。

零自由参数：归一化因子完全由理论给出，没有任何拟合量。曲线塌不塌到 1 是可证伪的。

图上的 CV / 几何均值一律**从数据现算**，不得硬编码 —— 上一版因写死旧数字导致
图文不符（图 23.3/13.4/1.085，正文 24.4/15.8/1.168）。

运行： python fig_collapse.py   （秒级，读现成 csv）
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import fig_style as st

st.use()

N, m, d, mu = 1000, 6, 12, 0.2
g = pd.read_csv("nucleation_collapse.csv")
BET = sorted(g.beta.unique())
CS = sorted(g.c.unique())
col = {b: st.BETA_SEQ[i] for i, b in enumerate(BET)}   # beta 浅到深
mk = {2: "o", 3: "s"}
ls = {2: "-", 3: "--"}

cv = lambda v: v.std() / v.mean() * 100          # 与 nucleation_analysis.py 同口径
gm = lambda v: np.exp(np.log(v).mean())
g["err"] = g.sd ** (1.96 / np.sqrt(g.n)) - 1     # 相对误差（几何标准误的 95% 区间）

# 第一步增长阈值 / 理论值（理论值 = lam / Y_exact）
o = (pd.read_csv("onestep_lambda.csv").groupby(["c", "rho0", "beta"]).lam_1step
       .agg(gm).reset_index().merge(g, on=["c", "rho0", "beta"]))
o["Y_1step"] = o.lam_1step / (o.lam / o.Y_exact)

#  列名 / y 轴标签 / 标题 / 角注 / y 范围 / 角注位置（(a) 右上被 c=3 簇占用，故靠左）
PANELS = [
    ("lam",    r"$\lambda_{\rm g}$",
     "(a)",
     "CV $=%.1f\\%%$" % cv(g.lam.values),
     None,       (0.035, 0.955, "left")),
    ("Y_lead", r"$\lambda_{\rm g}\,d\,\binom{m-1}{c}c^{\beta}\rho_0^{\,c-1}/\mu$",
     "(b)",
     "CV $=%.1f\\%%$" % cv(g.Y_lead.values),
     (0.5, 3.0), (0.960, 0.950, "right")),
    ("Y_exact", r"$\lambda_{\rm g}/\lambda_{\rm g}^{\rm MF}$",
     "(c)",
     "mean\n" + "\n".join("$c=%d$: $%.2f$" % (c, g[g.c == c].Y_exact.mean())
                               for c in CS),
     (0.5, 3.0), (0.960, 0.950, "right")),
]

fig, axes = plt.subplots(1, 3, figsize=(st.COL2, 2.45))
fig.subplots_adjust(left=0.083, right=0.988, top=0.925, bottom=0.175, wspace=0.40)

for ax, (col_name, ylab, panel, note, ylim, npos) in zip(axes, PANELS):
    for c in CS:
        for b in BET:
            s = g[(g.c == c) & (g.beta == b)].sort_values("rho0")
            ax.errorbar(s.rho0, s[col_name], yerr=s[col_name] * s.err, marker=mk[c],
                        color=col[b], ms=3.0, lw=0.9, mfc="none", mew=0.9, ls=ls[c],
                        elinewidth=0.7, capsize=1.2, capthick=0.6)
    if col_name == "Y_exact":
        for c in CS:
            s = o[o.c == c]
            ax.plot(s.rho0, s.Y_1step, "x", color="#C0392B", ms=2.6, mew=0.7, zorder=5,
                    label="first step" if c == CS[0] else None)
        ax.legend(frameon=False, loc="lower left", fontsize=6.0, handletextpad=0.2,
                  borderaxespad=0.25)
    ax.set_xscale("log"); ax.set_yscale("log")
    if ylim is not None:                       # 理论预言：塌缩量恒为 1
        ax.axhline(1.0, color="0.15", ls=(0, (1.6, 1.6)), lw=1.0, zorder=0)
        ax.set_ylim(*ylim)
        # 对数轴默认标成 3x10^0，塌缩量在 1 附近，普通数字更好读
        ax.set_yticks([0.6, 0.8, 1.0, 1.5, 2.0, 3.0])
        ax.set_yticklabels(["0.6", "0.8", "1", "1.5", "2", "3"])
        ax.set_yticks([], minor=True)
    # 横轴不足一个量级，matplotlib 会给次刻度也标上 a×10^b 并互相重叠，改为固定的普通数字
    ax.set_xticks([0.02, 0.05, 0.1, 0.2]); ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel(r"$\rho_0$", labelpad=1.5)
    ax.set_ylabel(ylab, labelpad=2)
    ax.text(-0.055, 1.02, panel, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8.0, fontweight="bold")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.45)
    ax.text(npos[0], npos[1], note, transform=ax.transAxes, ha=npos[2], va="top",
            fontsize=6.4, bbox=dict(fc="w", ec="0.75", lw=0.5, alpha=0.95,
                                    boxstyle="round,pad=0.25"))

# beta 用图例（只放一次，在 (a)）；c 的两簇曲线在 rho_0 上完全分开，直接标注更清楚
h_b = [Line2D([], [], color=col[b], lw=1.4, label=rf"$\beta={b:g}$") for b in BET]
axes[0].legend(handles=h_b, frameon=False, loc="lower left", fontsize=6.0,
               labelspacing=0.20, handlelength=1.5, handletextpad=0.4,
               borderaxespad=0.25)
axes[0].text(0.032, 0.062, "$c=2$ (solid)", fontsize=6.3, color="0.30",
             ha="center", va="bottom")
axes[0].text(0.145, 0.115, "$c=3$ (dashed)", fontsize=6.3, color="0.30",
             ha="center", va="bottom")

st.save(fig, "fig_collapse")

# ---------------------------------------------------------------- 数值自检
for name, c in [("塌缩前 lambda_grow", "lam"),
                ("最低阶式 Y_lead   ", "Y_lead"),
                ("完整核  Y_full    ", "Y_full"),
                ("精确解  Y_exact   ", "Y_exact")]:
    v = g[c].values
    print(f"  {name}  几何均值={np.exp(np.log(v).mean()):8.4f}   "
          f"log10 跨度={np.log10(v.max()/v.min()):5.2f}   "
          f"CV={v.std()/v.mean()*100:6.1f}%")
