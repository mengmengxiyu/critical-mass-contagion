# -*- coding: utf-8 -*-
"""
c=1 的入侵阈值：household（两层混合）计算（Yamir 意见第二点；Ball, Mollison & Scalia-Tomba 1997）。

局部树状超图上，把一条超边当作一个 household：
    1. 超边内从 1 个新感染者出发的离散时间 SIR 精确求解（状态 (s, i)，规则同 nucleation_core.mc_final_size）；
       Z = 超边内最终被感染的人数（不含初始者）。
    2. 每个新感染者把疫情带进另外 d-1 条超边，故 R* = (d-1) E[Z]，阈值 lambda_c 由 R* = 1 给出。
    3. 阈值以下，单个种子的平均最终感染人数 = 1 + d E[Z] / (1 - R*)，无可调参数，可直接与模拟对比。

    python household.py          # 打印 lambda_c 表（秒级）
    python household.py mc       # 亚临界平均规模的模拟核验，产出 ../data/household_check.csv（约 5 分钟）
"""
import sys
import numpy as np
from math import comb
from scipy.optimize import brentq


def _binom_pmf(n, p):
    return np.array([comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n + 1)])


def mean_Z(lam, beta, m, mu):
    """超边内（m 人、1 个初始感染者）最终被感染人数（不含初始者）的期望。"""
    states = [(s, i) for s in range(m) for i in range(m - s + 1)]
    idx = {st: k for k, st in enumerate(states)}
    A, b = np.eye(len(states)), np.zeros(len(states))
    for (s, i), k in idx.items():
        if i == 0:                                   # 吸收态
            b[k] = m - 1 - s
            continue
        px = _binom_pmf(s, -np.expm1(-lam * i**beta))    # 新感染
        py = _binom_pmf(i, mu)                           # 恢复（新感染者当步不恢复）
        for x in range(s + 1):
            for y in range(i + 1):
                A[k, idx[(s - x, i - y + x)]] -= px[x] * py[y]
    return np.linalg.solve(A, b)[idx[(m - 1, 1)]]


def lam_c(beta, m, d, mu):
    return brentq(lambda l: (d - 1) * mean_Z(l, beta, m, mu) - 1, 1e-9, 1e3)


def mean_size(lam, beta, m, d, mu):
    """亚临界时单个种子的平均最终感染人数。"""
    EZ = mean_Z(lam, beta, m, mu)
    return 1 + d * EZ / (1 - (d - 1) * EZ) if (d - 1) * EZ < 1 else np.inf


# ---------------------------------------------------------------- 模拟核验
M_SIZE, MU, N = 6, 0.2, 6000
DS, BETAS = [2, 3, 6, 12], [0.0, 1.0, 3.0]
FRACS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]          # lambda / lambda_c
N_RUN = 4000


def run_task(task):
    from nucleation_core import gen_regular_hypergraph, mc_final_size
    d, beta = task
    rng = np.random.default_rng(20261010 + 100 * d)      # 同一 d 的三个 beta 共用一张网络
    edges = gen_regular_hypergraph(N, M_SIZE, d, rng)
    rng = np.random.default_rng(20261010 + 100 * d + int(beta) + 1)
    lc, rows = lam_c(beta, M_SIZE, d, MU), []
    for f in FRACS:
        sz = np.array([mc_final_size(edges, N, 1 / N, beta, 1, f * lc, MU, rng)
                       for _ in range(N_RUN)]) * N
        rows.append(dict(d=d, beta=beta, lam_c=lc, frac=f, lam=f * lc,
                         size_mc=sz.mean(), size_se=sz.std(ddof=1) / np.sqrt(N_RUN),
                         size_theory=mean_size(f * lc, beta, M_SIZE, d, MU)))
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mc":
        import pandas as pd
        from multiprocessing import Pool
        with Pool(12) as pool:
            out = pool.map(run_task, [(d, b) for d in DS for b in BETAS])
        df = pd.DataFrame(sum(out, []))
        df.to_csv("../data/household_check.csv", index=False)
        df["z"] = (df.size_mc - df.size_theory) / df.size_se
        print(df.round(4).to_string(index=False))
    else:
        print(f"lambda_c (m={M_SIZE}, mu={MU})")
        for d in DS:
            L = {b: lam_c(b, M_SIZE, d, MU) for b in BETAS}
            print(f"  d={d:2d}  " + "  ".join(f"beta={b:.0f}: {L[b]:.5f}" for b in BETAS)
                  + f"   beta 1->3: {L[1.0]/L[3.0]:.3f}   beta 0->3: {L[0.0]/L[3.0]:.3f}")
