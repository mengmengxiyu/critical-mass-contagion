# -*- coding: utf-8 -*-
"""有限尺寸检验：c>=2 时的「不爆发」是否只是有限尺寸效应？

结论 A（c>=2 时任意有限 lambda 都不爆发）此前只在单一 N 下验证过，
审稿人必问「这会不会是 N 不够大造成的」。

设计要点：**固定种子数、扫 N**，而不是固定 rho_0。
理由：结论 A 说的是「无穷小种子无法触发」，对应 rho_0 -> 0；
固定 n_seed 时 rho_0 = n_seed/N 随 N 增大而减小，正是所需的极限方向。
（若固定 rho_0 扫 N，检验的是别的东西 —— 种子数也在同步增大。）

同时跑 c=2 作对照，可以一举两得：
    c=3 : 预期所有 N 都不爆发           -> 排除有限尺寸效应
    c=2 : 预期小 N（rho_0 大）爆发、
          大 N（rho_0 小）不爆发         -> 正面支持「成核由 rho_0 控制」

判据仍是放大倍数 amp = R/rho_0：amp = 1 表示种子自行恢复、未传染任何人。

运行： python finite_size.py     （约 30-45 分钟，产出 finite_size.csv）
随机种子固定为 20260813 + net，结果可完全复现。
"""
import numpy as np, pandas as pd, time
from nucleation_core import gen_regular_hypergraph, critical_count, mc_final_size

M_SIZE, D, MU, BETA = 6, 12, 0.2, 2.0
N_LIST = [1000, 2000, 5000, 10000, 20000, 50000]
N_SEED = 100                      # 固定种子数，故 rho_0 = 100/N 从 0.1 降到 0.002
CFG = {2: 1/3, 3: 0.5}            # c -> theta
LAMS = [1.0, 5.0]
N_NET, N_RUN = 3, 200

rows, t0 = [], time.time()
for N in N_LIST:
    for net in range(N_NET):
        rng = np.random.default_rng(20260813 + net)
        edges = gen_regular_hypergraph(N, M_SIZE, D, rng)
        for c, theta in CFG.items():
            assert critical_count(theta, M_SIZE) == c
            rho0 = N_SEED / N
            for lam in LAMS:
                Rs = np.array([mc_final_size(edges, N, rho0, BETA, c, lam, MU, rng)
                               for _ in range(N_RUN)])
                rows.append(dict(N=N, net=net, c=c, lam=lam, rho0=rho0,
                                 R=Rs.mean(), amp=Rs.mean() / rho0,
                                 frac_outbreak=float((Rs > 10 * rho0).mean())))
        print(f"[{time.time()-t0:6.0f}s] N={N} net={net} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("finite_size_raw.csv", index=False)
g = df.groupby(["N", "c", "lam", "rho0"])[["R", "amp", "frac_outbreak"]].mean().reset_index()
g.to_csv("finite_size.csv", index=False)

print(f"\n完成，耗时 {time.time()-t0:.0f}s")
print(f"参数：m={M_SIZE}, d={D}, mu={MU}, beta={BETA}, 固定 {N_SEED} 个种子，"
      f"{N_NET} 网络 x {N_RUN} 次\n")
print("放大倍数 amp = R/rho_0（=1 即完全无传播）")
print(f"{'N':>7} {'rho_0':>8} | " + " ".join(f"c={c},lam={l:g}".rjust(12)
                                             for c in CFG for l in LAMS))
for N in N_LIST:
    cells = []
    for c in CFG:
        for l in LAMS:
            r = g[(g.N == N) & (g.c == c) & (g.lam == l)]
            cells.append(f"{r.amp.iloc[0]:12.2f}" if len(r) else " " * 12)
    print(f"{N:>7} {N_SEED/N:>8.4f} | " + " ".join(cells))

print("\n爆发轨迹占比（R > 10*rho_0 计为爆发）")
print(f"{'N':>7} | " + " ".join(f"c={c},lam={l:g}".rjust(12) for c in CFG for l in LAMS))
for N in N_LIST:
    cells = []
    for c in CFG:
        for l in LAMS:
            r = g[(g.N == N) & (g.c == c) & (g.lam == l)]
            cells.append(f"{r.frac_outbreak.iloc[0]:12.3f}" if len(r) else " " * 12)
    print(f"{N:>7} | " + " ".join(cells))

c3 = g[g.c == 3]
print(f"\nc=3 的 amp：最小 {c3.amp.min():.4f}，最大 {c3.amp.max():.4f}"
      f"（若恒为 1.00 则「不爆发」与 N 无关，排除有限尺寸效应）")
