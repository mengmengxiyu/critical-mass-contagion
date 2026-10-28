# -*- coding: utf-8 -*-
"""论文 Fig. 5：异质超图上的有效阈值 c_eff，模拟对照无拟合的增长条件。

c_eff 的定义（模拟与理论同法）：ln lambda_grow 对 beta 的斜率 = -ln c_eff。

    点   模拟。读 hetero_raw.csv（A 与两个均匀对照）与 hetero2_raw.csv（B、C），
         对网络取几何平均后拟合；误差棒为对网络 bootstrap 的 95% 区间。
    线   异质增长条件的根，零可调参数。生成器（hetero_core.gen_hetero_hypergraph）随机抽成员，
         节点拥有的 m 尺寸超边条数近似 Poisson(d_m)，d_m = D p_m m / sum_m p_m m，故
             mu i0 = (1 - i0) [1 - exp(-sum_m d_m A_m(i0; lam))]
         其中 A_m 是一条 m 尺寸超边感染一个易感成员的概率（同均匀情形）。

均匀对照的 c_eff 略高于 c（2.1 / 3.2），理论曲线同样如此：来自 n > c 的项，不是方法偏差。

运行： python fig_hetero.py    （秒级，读现成 csv）
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from math import comb, ceil
from scipy.optimize import brentq
import fig_style as st

st.use()

D, MU, THETA = 12, 0.2, 0.5
BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
N_BOOT = 2000
SPEC = {                # 超边尺寸, 条数权重, 标签, 颜色, marker
    "uni_c2":      ([4], [1],                         r"uniform $c=2$", "0.45", "D"),
    "uni_c3":      ([6], [1],                         r"uniform $c=3$", "0.45", "v"),
    "hetero":      ([4, 5, 6, 7, 8], [1, 1, 1, 1, 1],   r"A: $\{2,3,3,4,4\}$", "#0072B2", "o"),
    "hetB_wide":   ([6, 7, 8, 9, 10], [1, 1, 1, 1, 1],  r"B: $\{3,4,4,5,5\}$", "#E69F00", "s"),
    "hetC_spread": ([4, 5, 6, 10, 12], [3, 3, 2, 1, 1], r"C: $\{2,3,3,5,6\}$", "#C0392B", "^"),
}


def lam_theory(ms, ps, i0, beta):
    """异质增长条件的根。"""
    norm = sum(p * m for p, m in zip(ps, ms))
    S = [(m, ceil(THETA * m - 1e-12), D * p * m / norm) for m, p in zip(ms, ps)]
    A = lambda m, c, lam: sum(comb(m - 1, k) * i0**k * (1 - i0)**(m - 1 - k)
                              * -np.expm1(-lam * k**beta) for k in range(c, m))
    f = lambda lam: (1 - i0) * -np.expm1(-sum(dm * A(m, c, lam) for m, c, dm in S)) - MU * i0
    return brentq(f, 1e-9, 1e6) if f(1e6) > 0 else np.nan


def c_eff(lams):
    return np.exp(-np.polyfit(BETAS, np.log(lams), 1)[0])


raw = pd.concat([pd.read_csv("hetero_raw.csv"), pd.read_csv("hetero2_raw.csv")])
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(st.COL1, 2.75))
fig.subplots_adjust(left=0.135, right=0.975, top=0.965, bottom=0.15)

print(f"{'config':>12} {'rho0':>5} {'模拟':>6} {'95% 区间':>15} {'理论':>6}")
for key, (ms, ps, lab, col, mk) in SPEC.items():
    L = np.log(raw[raw.config == key].pivot_table(index=["rho0", "net"], columns="beta",
                                                 values="lam_grow", dropna=False))
    pts = []
    for rho0 in L.index.get_level_values(0).unique():
        x = L.loc[rho0].values                           # (网络, beta)
        if np.isnan(x).any():                            # 有网络在上界仍不爆发：不拟合
            continue
        boot = [c_eff(np.exp(x[rng.integers(0, len(x), len(x))].mean(0)))
                for _ in range(N_BOOT)]
        ce, (lo, hi) = c_eff(np.exp(x.mean(0))), np.percentile(boot, [2.5, 97.5])
        th = c_eff([lam_theory(ms, ps, rho0, b) for b in BETAS])
        pts.append((rho0, ce, ce - lo, hi - ce))
        print(f"{key:>12} {rho0:>5.2f} {ce:>6.3f}   [{lo:.3f}, {hi:.3f}] {th:>6.3f}")
    r, ce, elo, ehi = np.array(pts).T
    rr = np.linspace(r.min(), r.max(), 60)
    ax.plot(rr, [c_eff([lam_theory(ms, ps, x, b) for b in BETAS]) for x in rr],
            color=col, lw=1.2, zorder=2)
    ax.errorbar(r, ce, yerr=[elo, ehi], fmt=mk, color=col, ms=3.8, mfc="w", mew=1.1,
                elinewidth=0.8, capsize=1.6, capthick=0.7, label=lab, zorder=3)

ax.set_xlabel(r"$\rho_0$", labelpad=2)
ax.set_ylabel(r"$c_{\rm eff}$", labelpad=2)
ax.set_xlim(0.075, 0.425); ax.set_ylim(1.6, 6.3)
ax.set_yticks([2, 3, 4, 5, 6])
ax.legend(loc="upper left", framealpha=0.95, handlelength=1.2, labelspacing=0.24,
          borderpad=0.4, handletextpad=0.4, borderaxespad=0.4, ncol=2, columnspacing=0.9)

st.save(fig, "fig_hetero")
