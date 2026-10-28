# -*- coding: utf-8 -*-
"""论文图：同一把尺子比较 c=1 与 c=2 中 beta 的作用（§4.3 后半）。

(a)(b) E6 lambda-beta 相图，c=1 与 c=2（读 fig2_phase_diagram.csv，fig2_phase_diagram.py 产出）。
    m=4、d=6、mu=0.2、rho0=0.05；底色为平均 R，白线为 R 等值线。
    (a) 红实线 household 阈值 lambda_c(beta)，青虚线单节点闭合阈值；(b) 红线增长条件的根。
(c) E7 beta 从 1 增到 3 使边界降低的倍数，对超度 d（读 dscan_raw.csv，dscan_run.py 产出；m=6）。
    黑线   household 精确阈值之比 lambda_c(1)/lambda_c(3)，c=1
    蓝点   c=1 的 R 等值线之比
    橙点   c=2 的 R 等值线之比，先除以 c^beta（即比较 lambda c^beta）
    橙虚线 c=2 的增长条件之比（同样除以 c^beta）
    误差棒为对网络 bootstrap 的 95% 区间。

运行（在 god/data/ 下）： PYTHONPATH=../code python ../code/fig_beta_compare.py   （约 10 秒）
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import fig_style as st
from math import comb
from scipy.optimize import brentq
from household import lam_c
import dscan_run as ds

st.use()   # 投稿规格：TeX Gyre Termes / TrueType 字体 / 双栏字号

N, m, d, mu, rho0 = 1000, 4, 6, 0.2, 0.05
LEVELS = [0.1, 0.3, 0.5, 0.7]
STYLES = [":", "-.", "-", "--"]

df = pd.read_csv("fig2_phase_diagram.csv")
BETAS = np.sort(df.beta.unique())
LAMS = np.sort(df.lam.unique())
CS = sorted(df.c.unique())


def lambda_c_theory():
    """c=1 时的线性入侵阈值，取自正文命题（无感染态处的 Jacobian）：

        J_I = (1-mu) I + (1-e^{-lam}) D_s B^(1),   rho(J_I) = 1
        =>  (1-e^{-lam}) * d(m-1) = mu
        =>  lam_c = -ln[1 - mu/(d(m-1))]

    2026-08-13 更正：此处原用 R_0 = d(m-1)p/[1-(1-mu)(1-p)] = 1 的启发式
    （分母为存活期内的累积传染概率），与正文闭式不是同一个量，m=4,d=6 时
    相差 4.7%。论文其余各处（含 Fig. 2 的 P_out 实验）用的都是上述 Jacobian
    闭式，故统一到它。c>=2 时该式不适用 —— 线性化恒稳定，临界点不存在。
    """
    return -np.log1p(-mu / (d * (m - 1)))


def lambda_grow_theory(beta, c):
    """完整增长条件 mu*i0 = (1-i0)*{1-[1-A(i0;lam)]^d} 的根（i0 = rho0），零可调参数。"""
    def f(lam):
        A = sum(comb(m-1, k) * rho0**k * (1-rho0)**(m-1-k) * -np.expm1(-lam * k**beta)
                for k in range(c, m))
        return (1-rho0) * (1 - (1-A)**d) - mu * rho0
    return brentq(f, 1e-9, 1e5)


def boundary(Rrow, level):
    k = np.where((Rrow[:-1] < level) & (Rrow[1:] >= level))[0]
    if len(k) == 0:
        return np.nan
    k = k[0]
    w = (level - Rrow[k]) / (Rrow[k+1] - Rrow[k])
    return np.exp(np.log(LAMS[k]) + w * (np.log(LAMS[k+1]) - np.log(LAMS[k])))


fig = plt.figure(figsize=(st.COL2, 2.6))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.05, 0.42, 1.05], wspace=0.07,
                      left=0.06, right=0.99, top=0.925, bottom=0.17)
axes = [fig.add_subplot(gs[0])]
axes.append(fig.add_subplot(gs[1], sharey=axes[0]))
cax = fig.add_subplot(gs[2])
axc = fig.add_subplot(gs[4])
plt.setp(axes[1].get_yticklabels(), visible=False)
for ax, c in zip(axes, CS):
    sub = df[df.c == c]
    theta = sub.theta.iloc[0]
    R = sub.pivot(index="beta", columns="lam", values="R").values
    pcm = ax.pcolormesh(LAMS, BETAS, R, cmap="viridis", shading="auto",
                        vmin=0, vmax=1, rasterized=True)

    for lv, ls in zip(LEVELS, STYLES):
        lam_b = np.array([boundary(R[i], lv) for i in range(len(BETAS))])
        ok = np.isfinite(lam_b)
        if ok.sum() < 2:
            continue
        ax.plot(lam_b[ok], BETAS[ok], ls, color="w",
                lw=1.5 if lv == 0.5 else 0.9, label=rf"$R={lv:g}$")

    if c == 1:
        ax.axvline(lambda_c_theory(), color="#64d2ff", lw=1.0, ls=(0, (4, 2.4)),
                   label=r"$\lambda_c$ (linearization)")
        ax.plot([lam_c(b, m, d, mu) for b in BETAS], BETAS, color="#ff3b30", lw=1.3,
                label=r"$\lambda_c$ (household)")
    else:
        ax.plot([lambda_grow_theory(b, c) for b in BETAS], BETAS, color="#ff3b30",
                lw=1.3, label="growth condition")

    ax.set_xscale("log")
    ax.set_xlim(LAMS[0], LAMS[-1])
    ax.set_xlabel(r"$\lambda$")
    # 面板只标字母，c 与 theta 的取值由 caption 交代，避免图文重复
    st.panel_label(ax, f"({'ab'[CS.index(c)]})", x=-0.02 if c > 1 else -0.22, y=0.99)
    # 图例落在高 R 的亮黄区，白字不可读 -> 加深色半透明底
    # (a) 图例放右上（高 R 亮区），(b) 放左下（低 R 暗区），都不压等值线
    lg = ax.legend(loc="upper right" if c == 1 else "lower left", labelcolor="w", ncol=1,
                   frameon=True, fontsize=6.4,
                   facecolor="0.15", edgecolor="none", framealpha=0.8)
    lg.get_frame().set_boxstyle("round,pad=0.3")
axes[0].set_ylabel(r"$\beta$")

fig.colorbar(pcm, cax=cax)
cax.set_title(r"$R$", fontsize=8, pad=3)

# ================================================================ (c) E7 beta 的作用对 d
ax = axc
dd = np.arange(2, 13)
ax.plot(dd, [lam_c(1.0, ds.m, d, ds.mu) / lam_c(3.0, ds.m, d, ds.mu) for d in dd],
        color="0.1", lw=1.2, label=r"$c=1$, household $\lambda_c$", zorder=2)


def lam_grow_d(d, c, beta):
    """E7 参数下（m=6）均匀超图完整增长条件的根（i0 = rho0）。"""
    m_, mu_, r0 = ds.m, ds.mu, ds.rho0
    def f(lam):
        A = sum(comb(m_-1, k) * r0**k * (1-r0)**(m_-1-k) * -np.expm1(-lam * k**beta)
                for k in range(c, m_))
        return (1-r0) * (1 - (1-A)**d) - mu_ * r0
    return brentq(f, 1e-9, 1e5)


ax.plot(dd, [lam_grow_d(d, 2, 1.0) * 2 / (lam_grow_d(d, 2, 3.0) * 8) for d in dd],
        color=st.C_COL[2], lw=1.0, ls=(0, (4, 2.4)), label=r"$c=2$, growth condition",
        zorder=2)

raw = pd.read_csv("dscan_raw.csv")
rng = np.random.default_rng(0)
print(f"{'d':>3} {'c':>2} {'R':>4} {'倍数':>6} {'95% 区间':>16}")
for c in (1, 2):
    for lv, mk in {0.1: "o", 0.5: "s"}.items():
        pts = []
        for d in ds.DS:
            # Rd[beta] 形状 (网络, lambda)
            Rd = {b: raw[(raw.d == d) & (raw.c == c) & (raw.beta == b)]
                  .pivot(index="net", columns="lam", values="R") for b in (1.0, 3.0)}
            lams = Rd[1.0].columns.values

            def ratio(idx):
                l1, l3 = (ds.boundary(lams, Rd[b].values[idx].mean(0), lv) for b in (1.0, 3.0))
                return l1 / l3 / c**2                    # c^beta: (c^1) / (c^3)

            n = len(Rd[1.0])
            r0 = ratio(np.arange(n))
            if np.isnan(r0):                             # 等值线不存在（R 达不到该水平）
                continue
            lo, hi = np.nanpercentile([ratio(rng.integers(0, n, n)) for _ in range(1000)],
                                      [2.5, 97.5])
            pts.append((d, r0, r0 - lo, hi - r0))
            print(f"{d:>3} {c:>2} {lv:>4} {r0:>6.3f}   [{lo:.3f}, {hi:.3f}]")
        x, y, elo, ehi = np.array(pts).T
        ax.errorbar(x, y, yerr=[elo, ehi], fmt=mk, color=st.C_COL[c], ms=3.4,
                    mfc="w" if lv == 0.1 else st.C_COL[c], mew=0.9, elinewidth=0.7,
                    capsize=1.2, capthick=0.6, zorder=3,
                    label=rf"$c={c}$, contour $R={lv:g}$")

ax.axhline(1.0, color="0.6", lw=0.7, ls=":", zorder=0)
ax.set_xscale("log")
ax.set_xticks(ds.DS); ax.set_xticklabels([str(d) for d in ds.DS]); ax.set_xticks([], minor=True)
ax.set_xlim(1.8, 13.5); ax.set_ylim(0.95, 2.75)
ax.set_xlabel(r"$d$", labelpad=2)
ax.set_ylabel(r"$\tilde\lambda(\beta{=}1)\,/\,\tilde\lambda(\beta{=}3)$", labelpad=2)
ax.legend(loc="upper right", framealpha=0.95, handlelength=1.6, labelspacing=0.22,
          borderpad=0.4, handletextpad=0.5, borderaxespad=0.4, fontsize=6.0)
st.panel_label(ax, "(c)", x=-0.20, y=0.99)

st.save(fig, "fig_beta_compare")
print(f"  (lambda_c theory = {lambda_c_theory():.5f})")

print("\n=== 各等值线随 beta 的移动倍数（beta: 0 -> 3）===")
print("c=2 的倍数里 c^beta = 8 可并进传播率（lambda c^beta），扣掉后约 1.14；c=1 的移动来自 household 阈值随 beta 降低")
for c in CS:
    R = df[df.c == c].pivot(index="beta", columns="lam", values="R").values
    print(f"\n--- c={c} ---")
    print(f"{'R':>5} {'beta=0':>9} {'beta=3':>9} {'移动倍数':>9}")
    for lv in LEVELS:
        lam_b = np.array([boundary(R[i], lv) for i in range(len(BETAS))])
        if np.isfinite(lam_b[0]) and np.isfinite(lam_b[-1]):
            print(f"{lv:>5.1f} {lam_b[0]:>9.4f} {lam_b[-1]:>9.4f} {lam_b[0]/lam_b[-1]:>9.2f}")
        else:
            print(f"{lv:>5.1f} {'--':>9} {'--':>9}      超范围")
