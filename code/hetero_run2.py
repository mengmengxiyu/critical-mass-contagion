# -*- coding: utf-8 -*-
"""异质超图第二轮：判别 c_eff 由什么决定。

第一轮（hetero_run.py）已确立：异质时 beta 的效力仍以 c_eff^beta 的形式起作用，
但 c_eff 不是 c_min，且随 rho_0 从 2.81 单调升到 3.64（第一轮分布 A 的
c_j={2,3,3,4,4}，c_min=2，c_max=4，边权平均 <c>=3.20）。

现有数据分不开两个解释：
    (I)  <c> 支配      —— c_eff 跟着边权平均走
    (II) 区间内滑动    —— c_eff 在 [c_min, c_max] 内随 rho_0 滑动，<c> 只是路过

本轮加两个分布，二者的预测在 C 上直接冲突：

    B: m in {6,7,8,9,10} 等权     -> c_j={3,4,4,5,5}, c_min=3, c_max=5, <c>=4.20
       整体相对 A 平移 +1。(I) 预期 c_eff 上移约 1；(II) 预期在 [3,5] 内滑动。

    C: m in {4,5,6,10,12} 权 3:3:2:1:1 -> c_j={2,3,3,5,6}, c_min=2, c_max=6, <c>=3.20
       **<c> 与 A 完全相同，但区间从 [2,4] 拓宽到 [2,6]。**
       (I) 预期 c_eff 与 A 重合；(II) 预期滑动幅度明显大于 A。

rho_0 范围按预跑确定：B 在 rho_0<0.15 不爆发，C 与 A 一致可从 0.10 起；
取共同区间 [0.15, 0.40] 并保留 0.10（C 有、B 记为 nan），便于与第一轮对齐。

运行： python hetero_run2.py    （约 25 分钟，产出 hetero2_raw/avg/ceff.csv）
随机种子固定为 20260813 + 1000*net，结果可完全复现。
"""
import numpy as np, pandas as pd, time
from hetero_core import (gen_hetero_hypergraph, critical_counts, hetero_stats,
                         find_lambda_grow_hetero)

N, D, MU, THETA = 1500, 12, 0.2, 0.5
BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
RHO0S = [0.10, 0.15, 0.20, 0.30, 0.40]
N_NET, N_RUN, N_BIS = 10, 100, 13   # 2026-10-09 由 3 加到 10，误差棒按网络 bootstrap

CONFIGS = {
    "hetB_wide":  ([6, 7, 8, 9, 10],  [1, 1, 1, 1, 1]),   # <c>=4.20, [3,5]
    "hetC_spread": ([4, 5, 6, 10, 12], [3, 3, 2, 1, 1]),  # <c>=3.20, [2,6]
}

t0 = time.time()


def run_net(net):
    rows = []
    for name, (m_list, m_prob) in CONFIGS.items():
        rng = np.random.default_rng(20260813 + 1000 * net)
        groups = gen_hetero_hypergraph(N, m_list, m_prob, D, rng)
        cs = critical_counts(groups, THETA)
        stt = hetero_stats(groups, cs, N)
        # 边权平均阈值（与 <c> 的定义一致：按超边条数加权）
        M = {m: len(e) for m, e in groups.items()}
        c_mean = sum(cs[m] * n for m, n in M.items()) / sum(M.values())
        c_max = max(cs.values())
        if net == 0:
            print(f"[{name}] c_j={dict(stt['c'])}  c_min={stt['c_min']}  c_max={c_max}  "
                  f"<c>={c_mean:.2f}  d={stt['d_avg']:.2f}  m_avg={stt['m_avg']:.2f}",
                  flush=True)
        for rho0 in RHO0S:
            for beta in BETAS:
                lg, note = find_lambda_grow_hetero(
                    groups, cs, N, rho0, beta, MU, rng,
                    n_run=N_RUN, n_bisect=N_BIS, hi=50.0)
                rows.append(dict(config=name, net=net, c_min=stt["c_min"],
                                 c_max=c_max, c_mean=c_mean, rho0=rho0,
                                 beta=beta, lam_grow=lg, note=note))
            print(f"[{time.time()-t0:6.0f}s] net={net} {name} rho0={rho0:.3f} done",
                  flush=True)
    return rows


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(N_NET) as pool:           # 一网络一进程，种子只依赖 net，与串行结果一致
        rows = sum(pool.map(run_net, range(N_NET)), [])

    df = pd.DataFrame(rows)
    df.to_csv("hetero2_raw.csv", index=False)
    ok = df[np.isfinite(df.lam_grow)]
    print(f"\n完成 {len(df)} 组，有效 {len(ok)} 组，耗时 {time.time()-t0:.0f}s")

    # c_mean 是逐网络的浮点数，放进 key 会让 3 个网络永远合不到一起（旧版 n 恒为 1）。
    # 只按离散标签分组，c_mean 事后取平均带回。
    g = (ok.groupby(["config", "rho0", "beta"]).lam_grow
           .agg(lam=lambda s: np.exp(np.log(s).mean()), n="count").reset_index())
    meta = ok.groupby("config")[["c_min", "c_max", "c_mean"]].mean().reset_index()
    g = g.merge(meta, on="config")
    assert (g.n == N_NET).all(), "每组应恰好聚合 %d 个网络" % N_NET
    g.to_csv("hetero2_avg.csv", index=False)

    # ---------------------------------------------------------------- 判别
    print("\n=== c_eff 拟合（ln lam_grow 对 beta 的斜率 = -ln c_eff）===")
    fit = []
    for name in CONFIGS:
        s0 = g[g.config == name]
        if s0.empty:
            continue
        cmin, cmax, cmean = s0.c_min.iloc[0], s0.c_max.iloc[0], s0.c_mean.iloc[0]
        print(f"\n--- {name}: c_min={cmin}, c_max={cmax}, <c>={cmean:.2f} ---")
        print(f"{'rho_0':>7} {'c_eff':>7}")
        for rho0 in RHO0S:
            s = s0[s0.rho0 == rho0].sort_values("beta")
            if len(s) < 3:
                continue
            k, _ = np.polyfit(s.beta, np.log(s.lam), 1)
            ce = np.exp(-k)
            fit.append(dict(config=name, rho0=rho0, c_eff=ce,
                            c_min=cmin, c_max=cmax, c_mean=cmean))
            print(f"{rho0:>7.3f} {ce:>7.3f}")
    fit = pd.DataFrame(fit)
    fit.to_csv("hetero2_ceff.csv", index=False)

    print("\n=== 判别（第一轮分布 A：c_min=2, c_max=4, <c>=3.20，c_eff 2.81 -> 3.64）===")
    print("  (I)  <c> 支配   : B 应比 A 高约 1.0；C 应与 A 重合")
    print("  (II) 区间内滑动 : B 在 [3,5] 滑；C 的滑动幅度应明显大于 A（区间 [2,6] vs [2,4]）")
    if not fit.empty:
        for name in CONFIGS:
            s = fit[fit.config == name]
            if len(s) >= 2:
                span = s.c_eff.max() - s.c_eff.min()
                print(f"    {name}: c_eff {s.c_eff.min():.2f} -> {s.c_eff.max():.2f}"
                      f"   幅度 {span:.2f}   (A 的幅度为 0.84)")
