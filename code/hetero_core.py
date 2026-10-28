# -*- coding: utf-8 -*-
"""异质超图（超边大小 m_j 不同）上的向量化 MC。

与 nucleation_core.py 的区别只有一处：超边大小不再统一，因而
  - 每条超边有自己的临界质量 c_j = ceil(theta * m_j)；
  - 无法用单个 (M, m) 数组表示，改为按 m 分组，每组一个定长数组，逐组向量化。

模型其余部分完全不变：激活判据 n_j >= c_j，压力 n_j^beta，
感染概率 1 - exp(-lambda * Phi_i)，同步更新，新感染当步不恢复。
lambda*(|e_j|) 恒为 1（与本文模型一致，见 CLAUDE.md「创新点定位」）。
"""
import numpy as np
from math import ceil


def gen_hetero_hypergraph(N, m_list, m_prob, d, rng):
    """随机异质超图：超边大小按 m_prob 从 m_list 抽取，平均超度约为 d。

    返回 {m: (M_m, m) int 数组}。总的 sum(m_j) 定为 N*d，故平均超度即 d。
    """
    m_list = np.asarray(m_list)
    m_prob = np.asarray(m_prob, dtype=float)
    m_prob = m_prob / m_prob.sum()
    target_slots = N * d                       # sum_j m_j
    groups, slots = {m: [] for m in m_list}, 0
    seen = set()
    while slots < target_slots:
        m = int(rng.choice(m_list, p=m_prob))
        e = tuple(sorted(rng.choice(N, size=m, replace=False).tolist()))
        if e in seen:
            continue
        seen.add(e)
        groups[m].append(e)
        slots += m
    return {m: np.array(v, dtype=np.int64) for m, v in groups.items() if v}


def critical_counts(groups, theta):
    """每个 m 分组的整数阈值 c_j = ceil(theta*m)，用整数运算避开浮点边界。"""
    out = {}
    for m in groups:
        out[m] = next((k for k in range(m + 1) if k * 1.0 / m >= theta - 1e-12), m + 1)
    return out


def hetero_stats(groups, cs, N):
    """网络的结构摘要：各 m 的边数、c_j、平均超度、c_min 及其占比。"""
    M = {m: len(e) for m, e in groups.items()}
    tot_slots = sum(m * n for m, n in M.items())
    c_min = min(cs.values())
    # c_min 那一档占了多少条超边、多少个 incidence
    n_cmin = sum(n for m, n in M.items() if cs[m] == c_min)
    return dict(M=M, c=cs, d_avg=tot_slots / N, c_min=c_min,
                frac_edges_cmin=n_cmin / sum(M.values()),
                m_avg=tot_slots / sum(M.values()))


def mc_final_size_hetero(groups, cs, N, rho0, beta, lam, mu, rng, max_steps=2000):
    """单次 SIR 到灭绝，返回最终 R 比例。按 m 分组逐组累加压力。"""
    n_seed = max(1, int(round(rho0 * N)))
    state = np.zeros(N, dtype=np.int8)                 # 0=S, 1=I, 2=R
    state[rng.choice(N, size=n_seed, replace=False)] = 1
    flat = {m: e.ravel() for m, e in groups.items()}

    for _ in range(max_steps):
        inf_mask = state == 1
        if not inf_mask.any():
            break
        pressures = np.zeros(N)
        for m, edges in groups.items():
            cnt = inf_mask[edges].sum(axis=1)
            act = cnt >= cs[m]                          # 逐边阈值 c_j
            if not act.any():
                continue
            edge_p = np.where(act, cnt.astype(np.float64) ** beta, 0.0)
            np.add.at(pressures, flat[m], np.repeat(edge_p, m))

        sus = state == 0
        p_inf = -np.expm1(-lam * pressures)             # expm1 避免小量消减
        new_inf = sus & (rng.random(N) < p_inf)
        new_rec = inf_mask & (rng.random(N) < mu)
        state[new_inf] = 1
        state[new_rec] = 2

    return float((state == 2).sum()) / N


def outbreak_prob_hetero(groups, cs, N, rho0, beta, lam, mu, rng, n_run, thresh=0.5):
    hit = 0
    for _ in range(n_run):
        if mc_final_size_hetero(groups, cs, N, rho0, beta, lam, mu, rng) > thresh:
            hit += 1
    return hit / n_run


def find_lambda_grow_hetero(groups, cs, N, rho0, beta, mu, rng,
                            n_run=100, lo=1e-4, hi=50.0, n_bisect=13, target=0.5):
    """几何二分搜索使 P(爆发)=target 的 lambda。与均匀情形同一判据，可直接对比。"""
    p_lo = outbreak_prob_hetero(groups, cs, N, rho0, beta, lo, mu, rng, n_run)
    if p_lo >= target:
        return lo, f"下界已爆发 (P={p_lo:.2f})"
    p_hi = outbreak_prob_hetero(groups, cs, N, rho0, beta, hi, mu, rng, n_run)
    if p_hi < target:
        return np.nan, f"上界仍不爆发 (P={p_hi:.2f})"
    for _ in range(n_bisect):
        mid = np.sqrt(lo * hi)
        if outbreak_prob_hetero(groups, cs, N, rho0, beta, mid, mu, rng, n_run) >= target:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi), "ok"
