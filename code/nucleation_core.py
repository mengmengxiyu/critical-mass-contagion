# -*- coding: utf-8 -*-
"""
阈值门控非线性协同传染：有限种子成核阈值 lambda_grow 的测定与标度塌缩验证。

验证《核心机制及证明》式 (38):
    lambda_grow(rho_0) ~ mu / (d * C(m-1,c) * c^beta) * rho_0^(1-c)

向量化 MC（原 code/MC vs MMCA.py 是 Python 循环，跑不动这个规模）。
"""
import numpy as np

# ---------------------------------------------------------------- 网络生成
def gen_uniform_hypergraph(N, m, d, rng):
    """随机均匀超图：N 节点，超边大小 m，平均超度 d。返回 (M,m) 的 int 数组。"""
    M = int(round(N * d / m))
    seen, edges = set(), []
    while len(edges) < M:
        e = tuple(sorted(rng.choice(N, size=m, replace=False).tolist()))
        if e in seen:
            continue
        seen.add(e)
        edges.append(e)
    return np.array(edges, dtype=np.int64)


def gen_regular_hypergraph(N, m, d, rng, max_sweep=2000):
    """d-正则 m-均匀超图：每节点恰好 d 条超边（不是均值 d）。

    理论侧 Cor.(lambda_c) 与 Prop.(lambda_grow) 都假定 d-regular，取
    rho(D_s B^(1)) = d(m-1)。gen_uniform_hypergraph 给的是二项超度，
    实测 rho(B^(1)) = (d+1)(m-1)，m=6,d=12 时 65 而非 60，lambda_c 差 8%。
    本函数用配置模型 + stub 交换修复，实测 rho(B^(1)) 精确等于 d(m-1)。
    """
    M = N * d // m
    if N * d != M * m:
        raise ValueError("N*d 必须被 m 整除：N=%d d=%d m=%d" % (N, d, m))
    stubs = np.repeat(np.arange(N), d)
    rng.shuffle(stubs)
    E = stubs.reshape(M, m).copy()

    for _ in range(max_sweep):
        S = np.sort(E, axis=1)
        dup_node = (np.diff(S, axis=1) == 0).any(axis=1)          # 超边内重复节点
        _, inv, cnt = np.unique(S, axis=0, return_inverse=True, return_counts=True)
        dup_edge = cnt[inv] > 1                                    # 超边间重复
        bad = np.flatnonzero(dup_node | dup_edge)
        if bad.size == 0:
            break
        partner = rng.integers(M, size=bad.size)
        a = rng.integers(m, size=bad.size)
        b = rng.integers(m, size=bad.size)
        for i, j, ai, bi in zip(bad, partner, a, b):
            if i != j:
                E[i, ai], E[j, bi] = E[j, bi], E[i, ai]
    else:
        raise RuntimeError("stub 交换在 %d 轮内未收敛" % max_sweep)

    if not np.all(np.bincount(E.ravel(), minlength=N) == d):
        raise AssertionError("超度不是正则的")
    return np.sort(E, axis=1).astype(np.int64)


def critical_count(theta, m):
    """整数阈值 k_c = ceil(theta*m)，用整数运算避开浮点边界。"""
    return next((k for k in range(m + 1) if k * 1.0 / m >= theta - 1e-12), m + 1)


# ---------------------------------------------------------------- 向量化 MC
def mc_final_size(edges, N, rho0, beta, k_c, lam, mu, rng, max_steps=2000):
    """单次 SIR 到灭绝，返回最终 R 比例。同步更新，新感染当步不恢复。"""
    m = edges.shape[1]
    n_seed = max(1, int(round(rho0 * N)))
    state = np.zeros(N, dtype=np.int8)             # 0=S, 1=I, 2=R
    state[rng.choice(N, size=n_seed, replace=False)] = 1
    flat = edges.ravel()

    for _ in range(max_steps):
        inf_mask = state == 1
        if not inf_mask.any():
            break
        # 每条超边内的感染人数
        cnt = inf_mask[edges].sum(axis=1)
        act = cnt >= k_c                            # 激活判据（整数比较）
        if act.any():
            edge_p = np.where(act, cnt.astype(np.float64) ** beta, 0.0)
            pressures = np.zeros(N)
            np.add.at(pressures, flat, np.repeat(edge_p, m))
        else:
            pressures = np.zeros(N)

        sus = state == 0
        p_inf = -np.expm1(-lam * pressures)         # expm1 避免小量消减
        new_inf = sus & (rng.random(N) < p_inf)
        new_rec = inf_mask & (rng.random(N) < mu)
        state[new_inf] = 1
        state[new_rec] = 2

    return float((state == 2).sum()) / N


def outbreak_prob(edges, N, rho0, beta, k_c, lam, mu, rng, n_run, thresh=0.5):
    """P(大爆发)，爆发判据 R > thresh。"""
    hit = 0
    for _ in range(n_run):
        if mc_final_size(edges, N, rho0, beta, k_c, lam, mu, rng) > thresh:
            hit += 1
    return hit / n_run


# ---------------------------------------------------------------- 阈值测定
def find_lambda_grow(edges, N, rho0, beta, k_c, mu, rng,
                     n_run=60, lo=1e-4, hi=50.0, n_bisect=14, target=0.5):
    """二分搜索使 P(爆发)=target 的 lambda。返回 (lambda_grow, 诊断信息)。"""
    p_lo = outbreak_prob(edges, N, rho0, beta, k_c, lo, mu, rng, n_run)
    p_hi = outbreak_prob(edges, N, rho0, beta, k_c, hi, mu, rng, n_run)
    if p_lo >= target:
        return lo, f"下界已爆发 (P={p_lo:.2f})"
    if p_hi < target:
        return np.nan, f"上界仍不爆发 (P={p_hi:.2f})"

    for _ in range(n_bisect):
        mid = np.sqrt(lo * hi)                      # 几何二分（量级跨度大）
        if outbreak_prob(edges, N, rho0, beta, k_c, mid, mu, rng, n_run) >= target:
            hi = mid
        else:
            lo = mid
    return np.sqrt(lo * hi), "ok"
