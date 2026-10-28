# -*- coding: utf-8 -*-
"""
超度扫描（Yamir 意见第二点）：同一观测量、同一 beta 范围下对照 c=1 与 c=2。

观测量同 Fig. 3：固定 rho0，平均终态 R(lambda) 的等值线 R=0.1/0.3/0.5/0.7。
c=2 的等值线除以 c^beta 后比较（lambda~ = lambda c^beta，见正文 §2.2）。
c=1 的对照基准是 household 精确阈值（超边内离散时间 SIR 精确求解，(d-1)E[Z]=1）。

    python dscan_run.py     # 产出 ../data/dscan_raw.csv（每网络一行，留作 bootstrap）

随机种子 20261009 + 100*d + net，结果可复现；并行与否结果一致。
"""
import numpy as np, pandas as pd, time
from nucleation_core import gen_regular_hypergraph, mc_final_size

N, m, mu, rho0 = 3000, 6, 0.2, 0.05
DS = [2, 3, 6, 12]
CS = [1, 2]
BETAS = [0.0, 1.0, 3.0]
LAMS = np.logspace(-3, np.log10(5.0), 40)
N_NET, N_RUN = 10, 60
LEVELS = [0.1, 0.3, 0.5, 0.7]


def run_task(task):
    """一个 (d, net)：生成一张网络，扫完 c x beta x lambda，返回每格平均 R。"""
    d, net = task
    rng = np.random.default_rng(20261009 + 100 * d + net)
    edges = gen_regular_hypergraph(N, m, d, rng)
    rows = []
    for c in CS:
        for beta in BETAS:
            for lam in LAMS:
                R = np.mean([mc_final_size(edges, N, rho0, beta, c, lam, mu, rng)
                             for _ in range(N_RUN)])
                rows.append(dict(d=d, net=net, c=c, beta=beta, lam=lam, R=R))
    return rows


def boundary(lams, Rrow, level):
    """R(lambda) 首次越过 level 处的 lambda（对数尺度线性插值）。"""
    k = np.where((Rrow[:-1] < level) & (Rrow[1:] >= level))[0]
    if len(k) == 0:
        return np.nan
    k = k[0]
    w = (level - Rrow[k]) / (Rrow[k + 1] - Rrow[k])
    return np.exp(np.log(lams[k]) + w * (np.log(lams[k + 1]) - np.log(lams[k])))


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    tasks = [(d, net) for d in DS for net in range(N_NET)]
    with Pool(min(len(tasks), 30)) as pool:
        out = pool.map(run_task, tasks)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv("../data/dscan_raw.csv", index=False)
    print(f"完成 {len(tasks)} 个 (d, net)，耗时 {time.time()-t0:.0f}s")

    g = df.groupby(["d", "c", "beta", "lam"]).R.mean().reset_index()
    print("\n等值线移动倍数；c=2 已除以 c^beta")
    for d in DS:
        for c in CS:
            lam_b = {}
            for beta in BETAS:
                sub = g[(g.d == d) & (g.c == c) & (g.beta == beta)].sort_values("lam")
                lam_b[beta] = np.array([boundary(sub.lam.values, sub.R.values, lv)
                                        for lv in LEVELS]) * c**beta
            r13, r03 = lam_b[1.0] / lam_b[3.0], lam_b[0.0] / lam_b[3.0]
            print(f"d={d:2d} c={c}  R={LEVELS}  b1->3 " + " ".join(f"{x:.3f}" for x in r13)
                  + "   b0->3 " + " ".join(f"{x:.3f}" for x in r03))
