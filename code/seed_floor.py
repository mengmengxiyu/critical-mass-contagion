# -*- coding: utf-8 -*-
"""种子密度下限 i_min 的直接测量（正文 §4.2 / §4.4）。

Eq.(18) 的右端在 lambda -> inf 时有上界，故 c>=2 存在与 beta 无关的 i_min，
低于它任何 lambda 都无解。理论值（m=6, d=12, mu=0.2）：
    c=2 -> 0.001675      c=3 -> 0.043223
finite_size.py 的 rho_0 网格（100/N）太粗，只能把 i_min 框在 2.5 倍宽的区间里。
本脚本在饱和 lambda 下细扫 rho_0，直接定位 P(爆发)=1/2 的位置。

lambda 取 5.0：finite_size.py 实测 lambda 从 1 加到 5 只在转变点附近改变结果，
说明压力已饱和；i_min 是 lambda -> inf 的极限，故取饱和值测量最接近它。

运行： python seed_floor.py          （约 20 分钟，产出 seed_floor.csv）
      python seed_floor.py nscan    （有限尺寸检验，32 进程约数分钟，产出 seed_floor_curve.csv / seed_floor_N.csv）
随机种子固定为 20260901 + net，结果可完全复现。
"""
import numpy as np, pandas as pd, time
from nucleation_core import gen_regular_hypergraph, mc_final_size

N, M_SIZE, D, MU, BETA, LAM = 20000, 6, 12, 0.2, 2.0, 5.0
N_NET, N_RUN = 2, 100
GRID = {2: np.geomspace(0.0010, 0.0050, 9),
        3: np.geomspace(0.0250, 0.0700, 9)}
THEORY = {2: 0.001675, 3: 0.043223}

def fine_scan():
    rows, t0 = [], time.time()
    for net in range(N_NET):
        rng = np.random.default_rng(20260901 + net)
        edges = gen_regular_hypergraph(N, M_SIZE, D, rng)
        for c, grid in GRID.items():
            for rho0 in grid:
                Rs = np.array([mc_final_size(edges, N, rho0, BETA, c, LAM, MU, rng)
                               for _ in range(N_RUN)])
                rows.append(dict(net=net, c=c, rho0=rho0,
                                 P_out=float((Rs > 0.5).mean()), R=Rs.mean()))
            print(f"[{time.time()-t0:6.0f}s] net={net} c={c} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("seed_floor.csv", index=False)

    g = df.groupby(["c", "rho0"])[["P_out", "R"]].mean().reset_index()
    print(f"\n参数：N={N}, m={M_SIZE}, d={D}, mu={MU}, beta={BETA}, lambda={LAM}, "
          f"{N_NET} 网络 x {N_RUN} 次\n")
    for c in sorted(g.c.unique()):
        s = g[g.c == c].sort_values("rho0")
        print(f"--- c={c} ---")
        print(f"{'rho_0':>9} {'P_out':>7} {'R':>7}")
        for _, r in s.iterrows():
            print(f"{r.rho0:>9.5f} {r.P_out:>7.2f} {r.R:>7.3f}")
        # 对 log(rho_0) 线性插值定位 P_out = 1/2
        x, y = np.log(s.rho0.values), s.P_out.values
        hit = np.nan
        for i in range(len(y) - 1):
            if y[i] < 0.5 <= y[i + 1]:
                hit = np.exp(x[i] + (0.5 - y[i]) * (x[i+1] - x[i]) / (y[i+1] - y[i]))
                break
        print(f"  P_out=1/2 处 rho_0 = {hit:.5f}   理论 i_min = {THEORY[c]:.5f}   "
              f"比值 = {hit/THEORY[c]:.2f}\n")


# ---------------------------------------------------------------- 有限尺寸检验
# 关键问题：上面的 floor 是真的种子密度下限，还是有限尺寸效应？
# 若是下限，固定 rho_0 扫 N 时 floor 位置应不随 N 移动（只是转变变陡）。
# 2026-10-12 网络数 2 -> 10、按 (c, N, 网络) 并行，并把每个网格点的爆发次数存进
# seed_floor_curve.csv（旧版只存边界位置，正文引用的边界两侧爆发比例无法复核）。
# 随机种子仍为 20260901 + net，前 2 个网络逐值复现旧结果。
N_NET_SCAN = 10
NSCAN = {2: ([5000, 20000, 50000], np.geomspace(0.0020, 0.0080, 7)),
         3: ([2000, 5000, 20000, 50000], np.geomspace(0.030, 0.090, 7))}


def scan_net(task):
    c, Nv, net, n_run = task
    rng = np.random.default_rng(20260901 + net)
    edges = gen_regular_hypergraph(Nv, M_SIZE, D, rng)
    rows = []
    for r in NSCAN[c][1]:
        Rs = np.array([mc_final_size(edges, Nv, r, BETA, c, LAM, MU, rng)
                       for _ in range(n_run)])
        rows.append(dict(c=c, N=Nv, net=net, rho0=r, n_out=int((Rs > 0.5).sum()),
                         n_run=n_run, R=Rs.mean()))
    return rows


def crossing(grid, P):
    """对 log(rho_0) 线性插值定位 P = 1/2。"""
    x = np.log(grid)
    for i in range(len(P) - 1):
        if P[i] < 0.5 <= P[i + 1]:
            return np.exp(x[i] + (0.5 - P[i]) * (x[i+1] - x[i]) / (P[i+1] - P[i]))
    return np.nan


def n_scan(n_run=100, n_boot=2000):
    from multiprocessing import Pool
    tasks = [(c, Nv, net, n_run) for c, (Ns, _) in NSCAN.items()
             for Nv in Ns for net in range(N_NET_SCAN)]
    with Pool(min(32, len(tasks))) as pool:
        df = pd.DataFrame(sum(pool.map(scan_net, tasks), []))
    df.to_csv("seed_floor_curve.csv", index=False)

    # 边界的 95% 区间：对网络有放回重抽，网络内爆发次数再按二项分布重抽
    print("=== 有限尺寸检验：固定 rho_0 网格扫 N，看 floor 位置是否随 N 移动 ===")
    rng, frows = np.random.default_rng(0), []
    for (c, Nv), s in df.groupby(["c", "N"]):
        grid = NSCAN[c][1]
        k = s.pivot(index="net", columns="rho0", values="n_out").to_numpy()
        P = k.sum(0) / (n_run * len(k))
        f = crossing(grid, P)
        boot = []
        for _ in range(n_boot):
            kb = rng.binomial(n_run, k[rng.integers(0, len(k), len(k))] / n_run)
            boot.append(crossing(grid, kb.sum(0) / (n_run * len(k))))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        frows.append(dict(c=c, N=Nv, floor=f, lo=lo, hi=hi))
        print(f"c={c} N={Nv:>6} floor={f:.5f} [{lo:.5f},{hi:.5f}] floor/i_min={f/THEORY[c]:.2f}   "
              f"P=[{' '.join('%.3f' % v for v in P)}]")
    pd.DataFrame(frows).to_csv("seed_floor_N.csv", index=False)


if __name__ == "__main__":
    import sys
    if "nscan" in sys.argv:          # python seed_floor.py nscan  -> seed_floor_curve.csv / seed_floor_N.csv
        n_scan()
    else:
        fine_scan()
