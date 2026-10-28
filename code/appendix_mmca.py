# -*- coding: utf-8 -*-
"""附录 B：MMCA 闭合近似与 MC 的定量对照（均匀超图）。

两点与旧的 `MC vs MMCA.py` 不同，都是有意为之：

1. **精确 leave-one-out DP，不用反卷积。** 旧实现由整条超边的 DP 反卷积出
   「排除节点 i」的分布，第 k 项误差被放大 (I_i/(1-I_i))^k，是数值病态的
   （见 CLAUDE.md）。这里对每个位置直接做 (m-1) 步 DP，跨超边向量化，
   代价 O(M m^3) 仍可接受，而误差回到浮点精度。

2. **MC 与 MMCA 跑同一批网络、同样多的网络。** 旧实现里 MMCA 只在 net_idx==0
   上跑而 MC 跑满，二者的差里混进了单网络涨落。

初始条件：MC 用 quenched（固定种子数），MMCA 用 annealed（乘积初值）。
两者在有限 N 下不是同一分布；均匀超图上差异很小（CLAUDE.md 实测 1.7e-3），
在异质/无标度结构上则不可忽略，故本附录只做均匀超图。

运行： python appendix_mmca.py    （约 3 分钟，产出 appendix_mmca.csv）
"""
import numpy as np, pandas as pd, time
from nucleation_core import gen_regular_hypergraph, critical_count, mc_final_size

N, M_SIZE, D, MU, BETA = 1000, 6, 12, 0.2, 2.0
THETA = 0.5                       # -> c = 3
RHO0S = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40]
LAM = 0.05
N_NET, N_RUN = 5, 400


def mmca_final_size(edges, N, rho0, beta, c, lam, mu, tol=1e-11, max_iter=5000):
    """MMCA 迭代到吸收态，返回 R(inf)。精确 leave-one-out DP。"""
    m = edges.shape[1]
    I = np.full(N, float(rho0))
    S = 1.0 - I
    R = np.zeros(N)
    rates = -np.expm1(-lam * np.arange(m + 1, dtype=float) ** beta)
    cols = [[j for j in range(m) if j != pos] for pos in range(m)]

    for _ in range(max_iter):
        Ie = I[edges]                                  # (M, m)
        q = np.ones(N)
        for pos in range(m):
            sub = Ie[:, cols[pos]]                     # (M, m-1)
            dp = np.zeros((edges.shape[0], m))
            dp[:, 0] = 1.0
            for j in range(m - 1):                     # 恰好 m-1 步 DP
                p = sub[:, j][:, None]
                dp[:, 1:] = dp[:, 1:] * (1 - p) + dp[:, :-1] * p
                dp[:, 0] *= (1 - p[:, 0])
            pinf = (dp[:, c:m] * rates[c:m]).sum(axis=1)
            np.multiply.at(q, edges[:, pos], 1.0 - pinf)

        S_new = S * q
        I_new = I * (1 - mu) + S * (1 - q)
        R_new = R + mu * I
        done = (np.max(np.abs(R_new - R)) < tol) and (np.max(I_new) < tol)
        S, I, R = S_new, I_new, R_new
        if done:
            break
    return float(R.mean() + I.mean())                  # 残余 I 并入最终规模


rows, t0 = [], time.time()
for net in range(N_NET):
    rng = np.random.default_rng(20260813 + 100 * net)
    edges = gen_regular_hypergraph(N, M_SIZE, D, rng)
    c = critical_count(THETA, M_SIZE)
    assert c == 3
    for rho0 in RHO0S:
        mc = np.mean([mc_final_size(edges, N, rho0, BETA, c, LAM, MU, rng)
                      for _ in range(N_RUN)])
        mf = mmca_final_size(edges, N, rho0, BETA, c, LAM, MU)
        rows.append(dict(net=net, rho0=rho0, R_mc=mc, R_mmca=mf, gap=mf - mc))
    print(f"[{time.time()-t0:5.0f}s] net={net} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("appendix_mmca_raw.csv", index=False)
g = df.groupby("rho0")[["R_mc", "R_mmca", "gap"]].agg(["mean", "std"])
g.columns = ["_".join(x) for x in g.columns]
g = g.reset_index()
g.to_csv("appendix_mmca.csv", index=False)

print(f"\n参数：N={N}, m={M_SIZE}, d={D}, mu={MU}, beta={BETA}, "
      f"theta={THETA} (c={c}), lambda={LAM}, {N_NET} 网络 x {N_RUN} 次\n")
print(f"{'rho_0':>7} {'R_MC':>9} {'R_MMCA':>9} {'差':>9}")
for _, r in g.iterrows():
    print(f"{r.rho0:>7.2f} {r.R_mc_mean:>9.4f} {r.R_mmca_mean:>9.4f} {r.gap_mean:>+9.4f}")
i = int(np.argmax(np.abs(g.gap_mean.values)))
print(f"\n最大偏差 {g.gap_mean.values[i]:+.4f} 出现在 rho_0={g.rho0.values[i]:.2f}")
ends = g[(g.rho0 <= 0.05) | (g.rho0 >= 0.30)]
print(f"两端（rho_0<=0.05 或 >=0.30）的最大 |偏差| = {ends.gap_mean.abs().max():.4f}")
