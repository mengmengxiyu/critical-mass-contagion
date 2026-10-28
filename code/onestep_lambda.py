# -*- coding: utf-8 -*-
"""
第一步增长阈值 lambda_1（Yamir 意见：与 Eq.(16) 直接可比的模拟量）。

Eq.(16) 给的是「平均感染比例在第一步开始增长」的 lambda，而 nucleation_run.py 测的是 P(爆发)=1/2。
本脚本在与 nucleation_run.py 相同的网络上求 E[I(1)] = I(0) 的 lambda：
种子位置给定后，第一步每个易感者的感染概率 1-exp(-lambda*Phi_i) 是确定的，
故 E[I(1)]-I(0) = sum_{i in S}(1-exp(-lambda*Phi_i)) - mu*n_seed 只需对种子摆放取平均，无需跑动力学。

    lambda_1 与 Eq.(16) 之差   -> 平均场（随机摆放 + 超边独立）本身的误差
    lambda_grow 与 lambda_1 之差 -> 第一步之后的效应（随机熄灭、新感染者之间的关联）

    python onestep_lambda.py     # 产出 ../data/onestep_lambda.csv（约 2 分钟）
"""
import numpy as np, pandas as pd, time
from scipy.optimize import brentq
from nucleation_core import gen_regular_hypergraph

N, m, d, mu = 1000, 6, 12, 0.2
BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
CFG = {2: [0.015, 0.02, 0.03, 0.04, 0.06, 0.08],
       3: [0.08, 0.10, 0.12, 0.15, 0.20]}
N_NET, N_DRAW = 10, 400


def run_net(net):
    rng = np.random.default_rng(20260807 + 1000 * net)     # 与 nucleation_run.py 同一张网络
    edges = gen_regular_hypergraph(N, m, d, rng)
    flat = edges.ravel()
    rng = np.random.default_rng(20261009 + net)            # 种子摆放用独立的流
    rows = []
    for c, rho_list in CFG.items():
        for rho0 in rho_list:
            n_seed = max(1, int(round(rho0 * N)))
            inf = np.zeros((N_DRAW, N), dtype=bool)
            for k in range(N_DRAW):
                inf[k, rng.choice(N, size=n_seed, replace=False)] = True
            cnt = inf[:, edges].sum(axis=2)                 # (draw, edge) 感染人数
            for beta in BETAS:
                edge_p = np.where(cnt >= c, cnt.astype(float) ** beta, 0.0)
                Phi = np.zeros((N_DRAW, N))
                for k in range(N_DRAW):
                    np.add.at(Phi[k], flat, np.repeat(edge_p[k], m))
                Phi = Phi[~inf]                             # 只留易感者
                f = lambda lam: -np.expm1(-lam * Phi).sum() / N_DRAW - mu * n_seed
                lam1 = brentq(f, 1e-8, 1e4) if f(1e4) > 0 else np.nan
                rows.append(dict(net=net, c=c, rho0=rho0, beta=beta, lam_1step=lam1))
    return rows


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(N_NET) as pool:
        df = pd.DataFrame(sum(pool.map(run_net, range(N_NET)), []))
    df.to_csv("../data/onestep_lambda.csv", index=False)
    print(f"完成，耗时 {time.time()-t0:.0f}s")
