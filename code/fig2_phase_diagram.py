# -*- coding: utf-8 -*-
"""
lambda-beta 相图（论文 Fig. 2）：对照 c=1 与 c=2 两种阈值状态下协同指数的作用。

    python fig2_phase_diagram.py    # 跑 MC，产出 fig2_phase_diagram.csv（约 10 分钟）
    python fig_beta_compare.py      # 出图 fig_beta_compare.png（(a)(b) 为相图）

相对 code/均匀超图beta_lambda热图.py 的三处改动：
    1. 补 theta=0.25（c=1）的对照 panel —— 原图只有 c=2，看不出 beta 何时失效；
    2. beta 下延到 0（c^0=1 即 Arruda PRR 2020 的固定压力模型），原为 [1,2]；
    3. lambda 改对数扫描，因为 c=1 的相界比 c=2 低一个量级，线性轴放不下。

等值线取多条（R=0.1/0.3/0.5/0.7）而非只取 R=0.5：R=0.5 深在超临界区，那里
超边内常有 k>=2 使 k^beta>1，beta 仍起作用；只有靠近 R_0=1 的低等值线才反映
临界点本身。多条一起画，才看得出「越接近阈值越不依赖 beta」。

随机种子固定为 20260807 + net，结果可完全复现。
"""
import numpy as np, pandas as pd, time
from nucleation_core import gen_regular_hypergraph, critical_count, mc_final_size

N, m, d, mu, rho0 = 1000, 4, 6, 0.2, 0.05
THETAS = [0.25, 0.5]                                  # -> c = 1, 2
BETAS = np.linspace(0.0, 3.0, 13)
LAMS = np.logspace(np.log10(0.004), np.log10(0.8), 30)
N_NET, N_RUN = 10, 100      # 2026-09-04 由 10x10 加密到 10x100，压低 R=0.1 等值线的抖动

CS = [critical_count(t, m) for t in THETAS]
assert CS == [1, 2], CS
print(f"theta={THETAS} -> c={CS}；网格 {len(BETAS)}x{len(LAMS)}，"
      f"每格 {N_NET*N_RUN} 次 MC", flush=True)

def run_net(net):
    """一个网络上的整张网格；随机种子只依赖 net 编号，并行与否结果一致。"""
    rng = np.random.default_rng(20260807 + net)
    edges = gen_regular_hypergraph(N, m, d, rng)      # 两个 panel 共用同一批网络
    Rn = np.zeros((len(THETAS), len(BETAS), len(LAMS)))
    for ci, c in enumerate(CS):
        for i, beta in enumerate(BETAS):
            for j, lam in enumerate(LAMS):
                Rn[ci, i, j] = sum(
                    mc_final_size(edges, N, rho0, beta, c, lam, mu, rng)
                    for _ in range(N_RUN)) / N_RUN
    return Rn


if __name__ == "__main__":
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(N_NET) as pool:                          # 一网络一进程
        R = sum(pool.map(run_net, range(N_NET))) / N_NET
    print(f"[{time.time()-t0:6.0f}s] {N_NET} nets done", flush=True)

    rows = [dict(theta=THETAS[ci], c=c, beta=b, lam=l, R=R[ci, i, j])
            for ci, c in enumerate(CS)
            for i, b in enumerate(BETAS)
            for j, l in enumerate(LAMS)]
    pd.DataFrame(rows).to_csv("fig2_phase_diagram.csv", index=False)
    print(f"\n完成，耗时 {time.time()-t0:.0f}s，已存 fig2_phase_diagram.csv")


    def boundary(Rrow, level=0.5):
        """在一行 R(lambda) 上线性插值出 R=level 的 lambda（log 尺度插值）。"""
        k = np.where((Rrow[:-1] < level) & (Rrow[1:] >= level))[0]
        if len(k) == 0:
            return np.nan
        k = k[0]
        w = (level - Rrow[k]) / (Rrow[k+1] - Rrow[k])
        return np.exp(np.log(LAMS[k]) + w * (np.log(LAMS[k+1]) - np.log(LAMS[k])))


    LEVELS = [0.1, 0.3, 0.5, 0.7]
    print("\n=== 各等值线随 beta 的移动倍数（beta: 0 -> 3）===")
    print("理论：c=1 时临界点与 beta 无关（倍数 -> 1），c=2 时按 c^beta 移动（倍数 -> 8）")
    for ci, c in enumerate(CS):
        print(f"\n--- c={c}（theta={THETAS[ci]}）---")
        print(f"{'R':>5} {'beta=0':>9} {'beta=3':>9} {'移动倍数':>9}")
        for lv in LEVELS:
            lam_b = np.array([boundary(R[ci, i], lv) for i in range(len(BETAS))])
            if np.isfinite(lam_b[0]) and np.isfinite(lam_b[-1]):
                print(f"{lv:>5.1f} {lam_b[0]:>9.4f} {lam_b[-1]:>9.4f} {lam_b[0]/lam_b[-1]:>9.2f}")
            else:
                print(f"{lv:>5.1f} {'--':>9} {'--':>9} {'超出扫描范围':>9}")
