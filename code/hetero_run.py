# -*- coding: utf-8 -*-
"""异质超图上的成核阈值：检验「c_min 支配」预言。

背景。式 (38) 的推导假设所有超边同大小、故 c_j 恒等于 c。异质时每条超边有自己的
c_j = ceil(theta*m_j)，成核条件里每条超边贡献 A_j ~ i^{c_j}，小 i 下最低阶项由
**最小的 c_j** 主导。因此理论预言：

    异质超图的成核由最容易点燃的那一档小超边支配，
    式 (38) 中的 c^beta 应替换为 c_min^beta，rho_0 指数应为 1-c_min。

2026-08-12 预跑已**证伪该预言的朴素版本**：取 m in {4..8}、theta=0.5（c_j={2,3,3,4,4}，
c_min=2），实测 lambda_grow 对 rho_0 的标度斜率为 -2.28，接近 c=3 而非 c_min=2 的 -1。
原因是 c_min 那一档只占约 13% 的 incidence（每节点仅 1.58 条，均匀对照是 12 条），
被稀释到单独撑不起成核，必须靠更高 c 档协同；且 rho_0 < 0.10 时 lambda -> 无穷 也不爆发，
说明存在**结构性的 bootstrap percolation 阈值**（压力已饱和，再增大 lambda 无效）。

因此本实验的重点改为真正关乎主结论的那一个量 —— **beta 的效力由哪个有效 c 决定**：
    ln(lambda_grow) 对 beta 的斜率 = -ln(c_eff)
这与 rho_0 标度无关，是单参数、可证伪的检验。三个候选：
    c_eff = c_min = 2      （最易点燃的一档支配）
    c_eff = c_avg ~ 3.2    （按边数加权的平均阈值）
    介于两者之间          （多档协同，需给出经验值）
均匀对照的 c_eff 必须回到各自的 c（2 和 3），否则说明拟合方法本身有问题。

三组网络（全部走同一份 hetero_core 代码，避免实现差异混入对比）：
    hetero  : m in {4,5,6,7,8} 等权          -> c_j={2,3,3,4,4}, c_min=2
    uni_c2  : 全 m=4                          -> c=2   （= c_min，预言应贴这条）
    uni_c3  : 全 m=6（= 异质的平均边大小）      -> c=3   （朴素「用平均值代入」的预期）

运行： python hetero_run.py     （约 1-2 小时，产出 hetero_raw.csv / hetero_avg.csv）
随机种子固定为 20260812 + 1000*net，结果可完全复现。
"""
import numpy as np, pandas as pd, time
from hetero_core import (gen_hetero_hypergraph, critical_counts, hetero_stats,
                         find_lambda_grow_hetero)

N, D, MU, THETA = 1500, 12, 0.2, 0.5
BETAS = [1.0, 1.5, 2.0, 2.5, 3.0]
RHO0S = [0.10, 0.15, 0.20, 0.30]   # 三组共同可测区间（hetero 的 lambda_grow 在 rho_0<0.10 测不出，爆发概率达不到 1/2；见 hetero_floor.py）
N_NET, N_RUN, N_BIS = 10, 100, 13   # 2026-10-09 由 3 加到 10，误差棒按网络 bootstrap

CONFIGS = {
    "hetero": ([4, 5, 6, 7, 8], [1, 1, 1, 1, 1]),
    "uni_c2": ([4],             [1]),
    "uni_c3": ([6],             [1]),
}

t0 = time.time()


def run_net(net):
    rows = []
    for name, (m_list, m_prob) in CONFIGS.items():
        rng = np.random.default_rng(20260812 + 1000 * net)
        groups = gen_hetero_hypergraph(N, m_list, m_prob, D, rng)
        cs = critical_counts(groups, THETA)
        stt = hetero_stats(groups, cs, N)
        if net == 0:
            print(f"[{name}] c_j={dict(stt['c'])}  c_min={stt['c_min']}  "
                  f"d={stt['d_avg']:.2f}  m_avg={stt['m_avg']:.2f}  "
                  f"c_min 档占边数 {stt['frac_edges_cmin']:.1%}", flush=True)
        for rho0 in RHO0S:
            for beta in BETAS:
                lg, note = find_lambda_grow_hetero(
                    groups, cs, N, rho0, beta, MU, rng,
                    n_run=N_RUN, n_bisect=N_BIS, hi=50.0)
                rows.append(dict(config=name, net=net, c_min=stt["c_min"],
                                 m_avg=stt["m_avg"], rho0=rho0, beta=beta,
                                 lam_grow=lg, note=note))
            print(f"[{time.time()-t0:6.0f}s] net={net} {name} rho0={rho0:.3f} done",
                  flush=True)
    return rows


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(N_NET) as pool:           # 一网络一进程，种子只依赖 net，与串行结果一致
        rows = sum(pool.map(run_net, range(N_NET)), [])

    df = pd.DataFrame(rows)
    df.to_csv("hetero_raw.csv", index=False)
    ok = df[np.isfinite(df.lam_grow)]
    print(f"\n完成 {len(df)} 组，有效 {len(ok)} 组，耗时 {time.time()-t0:.0f}s")

    g = (ok.groupby(["config", "c_min", "rho0", "beta"]).lam_grow
           .agg(lam=lambda s: np.exp(np.log(s).mean()), n="count").reset_index())
    g.to_csv("hetero_avg.csv", index=False)

    # ---------------------------------------------------------------- 关键对比
    print("\n=== lambda_grow 对比（几何平均）：异质贴近哪个均匀对照？ ===")
    piv = g.pivot_table(index=["rho0", "beta"], columns="config", values="lam")
    piv["het/uni_c2"] = piv["hetero"] / piv["uni_c2"]
    piv["het/uni_c3"] = piv["hetero"] / piv["uni_c3"]
    print(piv.round(4).to_string())
    print("\n若「c_min 支配」成立，het/uni_c2 应接近 1，且明显小于 het/uni_c3。")
    print(f"  het/uni_c2 几何均值 = {np.exp(np.log(piv['het/uni_c2']).mean()):.3f}")
    print(f"  het/uni_c3 几何均值 = {np.exp(np.log(piv['het/uni_c3']).mean()):.3f}")

    print("\n=== 核心检验：拟合有效阈值 c_eff（ln lam_grow 对 beta 的斜率 = -ln c_eff）===")
    print(f"{'config':>8} {'rho_0':>7} {'slope':>8} {'c_eff':>7}")
    fit = []
    for name in CONFIGS:
        s0 = g[g.config == name]
        for rho0 in RHO0S:
            s = s0[s0.rho0 == rho0].sort_values("beta")
            if len(s) < 3:
                continue
            k, _ = np.polyfit(s.beta, np.log(s.lam), 1)
            fit.append(dict(config=name, rho0=rho0, slope=k, c_eff=np.exp(-k)))
            print(f"{name:>8} {rho0:>7.3f} {k:>8.4f} {np.exp(-k):>7.3f}")
    fit = pd.DataFrame(fit)
    fit.to_csv("hetero_ceff.csv", index=False)
    print("\n各 config 的 c_eff（对 rho_0 取平均）：")
    print(fit.groupby("config").c_eff.agg(["mean", "std"]).round(3).to_string())
    print("\n判读：uni_c2 / uni_c3 的 c_eff 应分别回到 2 / 3（方法自检）；")
    print("      hetero 的 c_eff 若接近 2 则 c_min 支配，接近 3.2 则平均阈值支配，居中则多档协同。")

    print("\n=== 用各候选 c 归一化后的 CV（越小越好）===")
    for name in CONFIGS:
        s0 = g[g.config == name]
        print(f"\n--- {name} ---")
        for cand in [2.0, 2.5, 3.0, 3.2, 3.5]:
            cvs = []
            for rho0 in RHO0S:
                s = s0[s0.rho0 == rho0].sort_values("beta")
                if len(s) < 3:
                    continue
                y = (s.lam * cand ** s.beta).values
                cvs.append(y.std() / y.mean())
            if cvs:
                print(f"    c={cand:<4} -> CV = {np.mean(cvs)*100:5.1f}%")
