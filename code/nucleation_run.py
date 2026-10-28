# -*- coding: utf-8 -*-
"""
成核标度验证 —— 主程序。验证《核心机制及证明》式 (38) 的 c^beta 耦合与 rho_0^(1-c) 标度。

运行顺序（须在 code/ 目录下依次执行）：
    python nucleation_run.py       # 跑 MC，产出 nucleation_raw.csv / nucleation_avg.csv（约 12 分钟）
    python nucleation_analysis.py  # 计算两种塌缩量，产出 nucleation_collapse.csv
    python fig_collapse.py         # 出论文 Fig. 4 三联对比图

参数说明：m=6、d=12 与论文其余实验（m=4、d=6）不同，原因是
    m=4 时 c=3 即等于 m-1（需全部邻居感染），d=4 时 lambda=50 仍无法成核，
    该参数窗口下测不到 lambda_grow。详见正文「成核标度」一节。

随机种子固定为 20260807 + 1000*net，结果可完全复现；按网络并行，与串行结果一致。
2026-10-09 网络数 3 -> 10（Yamir：误差棒按网络 bootstrap），前 3 个网络复现旧值。

    python nucleation_run.py 5000   # 大 N 复核（只跑 c=3），产出 nucleation_raw_N5000.csv / _avg_N5000.csv
"""
import sys
import numpy as np, pandas as pd, time
from math import comb
from nucleation_core import *

N, m, d, mu = 1000, 6, 12, 0.2
BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
CFG = {2: (1/3,  [0.015, 0.02, 0.03, 0.04, 0.06, 0.08]),
       3: (0.5,  [0.08, 0.10, 0.12, 0.15, 0.20])}
N_NET, N_RUN, N_BIS = 10, 100, 13
SUFFIX = ""
if len(sys.argv) > 1:                      # 大 N 复核：c=3 的种子在 N=1000 时只有几十到两百个
    N, SUFFIX = int(sys.argv[1]), f"_N{sys.argv[1]}"
    CFG = {3: CFG[3]}


def run_net(net):
    rows = []
    rng = np.random.default_rng(20260807 + 1000*net)
    edges = gen_regular_hypergraph(N, m, d, rng)
    for c, (theta, rho_list) in CFG.items():
        assert critical_count(theta, m) == c
        for rho0 in rho_list:
            for beta in BETAS:
                lg, note = find_lambda_grow(edges, N, rho0, beta, c, mu, rng,
                                            n_run=N_RUN, n_bisect=N_BIS, hi=20.0)
                rows.append(dict(net=net, c=c, theta=theta, rho0=rho0, beta=beta,
                                 lam_grow=lg, note=note,
                                 sat=-np.expm1(-lg*c**beta) if np.isfinite(lg) else np.nan))
        print(f"[{time.time()-t0:6.0f}s] net={net} c={c} done", flush=True)
    return rows


t0 = time.time()
if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(N_NET) as pool:
        rows = sum(pool.map(run_net, range(N_NET)), [])

    df = pd.DataFrame(rows)
    df.to_csv(f"nucleation_raw{SUFFIX}.csv", index=False)
    ok = df[np.isfinite(df.lam_grow)]
    print(f"\n完成 {len(df)} 组，有效 {len(ok)} 组，耗时 {time.time()-t0:.0f}s")
    print(f"最大饱和度 {ok.sat.max():.3f}")

    # 网络平均（几何平均，因为量级跨度大）
    g = ok.groupby(['c','rho0','beta']).lam_grow.agg(
            lam=lambda s: np.exp(np.log(s).mean()),
            sd =lambda s: np.exp(np.log(s).std()) if len(s)>1 else np.nan,
            n='count').reset_index()
    g.to_csv(f"nucleation_avg{SUFFIX}.csv", index=False)
    print("\n=== c^beta 塌缩检验：同一 (c,rho0) 下 lam*c^beta 应与 beta 无关 ===")
    for c in sorted(g.c.unique()):
        print(f"\n--- c={c} ---")
        print(f"{'rho_0':>7} | " + " ".join(f"b={b:<4.1f}" for b in BETAS) + " | 变异系数")
        for rho0 in sorted(g[g.c==c].rho0.unique()):
            sub = g[(g.c==c)&(g.rho0==rho0)].sort_values('beta')
            y = (sub.lam * c**sub.beta).values
            cv = y.std()/y.mean() if len(y)>1 else np.nan
            print(f"{rho0:>7.3f} | " + " ".join(f"{v:<6.4f}" for v in y) + f" | {cv*100:>5.1f}%")
