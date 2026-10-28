# -*- coding: utf-8 -*-
"""
成核标度验证 —— 图 A / 图 B。
图 A：验证推论 6.1（k_c>=2 时不存在线性入侵阈值）
图 B：验证推论 6.2（k_c=1 时 lambda_c 与 beta 无关，等于闭式 (42)）
运行： python nucleation_fig_ab.py      # A + B 都跑
      python nucleation_fig_ab.py B    # 只跑 B（4000 次/点，3 个 beta 并行，约 25 分钟）
      python fig_seed_density.py       # (a) 用 A；python fig_c1_threshold.py  # (a) 用 B
"""
import sys, numpy as np, pandas as pd, time
from multiprocessing import Pool
from nucleation_core import gen_regular_hypergraph, critical_count, mc_final_size

PART = sys.argv[1].upper() if len(sys.argv) > 1 else "AB"

m, d, mu = 6, 12, 0.2
LAM_C = -np.log(1 - mu / (d * (m - 1)))          # 闭式 (42)
print(f"理论线性阈值 lambda_c = {LAM_C:.5f}\n")

# ---------------- 图 A ----------------
def run_A():
    NA, N_RUN_A = 10000, 400
    SEEDS = [1, 2, 3, 5, 10, 20, 50, 100]
    LAMS_A = [0.2, 1.0, 5.0]
    rows = []
    rng = np.random.default_rng(20260807)
    edges_A = gen_regular_hypergraph(NA, m, d, rng)
    t0 = time.time()
    for theta, c in [(0.10, 1), (1/3, 2), (0.5, 3)]:
        assert critical_count(theta, m) == c
        for lam in LAMS_A:
            for ns in SEEDS:
                rho0 = ns / NA
                R = np.mean([mc_final_size(edges_A, NA, rho0, 2.0, c, lam, mu, rng)
                             for _ in range(N_RUN_A)])
                rows.append(dict(fig='A', c=c, lam=lam, n_seed=ns, rho0=rho0,
                                 R=R, amp=R / rho0))
            print(f"[{time.time()-t0:5.0f}s] A: c={c} lam={lam} done", flush=True)
    pd.DataFrame(rows).to_csv("nucleation_figA.csv", index=False)

# ---------------- 图 B ----------------
# 2026-09-02 加密：400 -> 4000 次/点，网格 18 点(3e-4..3e-2) -> 26 点(1e-3..3e-2)。
# 400 次时 P~0.5 处二项标准差 0.025，三条 beta 曲线在 0.006-0.01 区间的交叉全在噪声内。
# 每个 beta 一个进程；网络由固定种子在各进程内重建，故三个 beta 用的是同一张网络。
NB, N_RUN_B = 5000, 4000
LAMS_B = np.logspace(np.log10(1e-3), np.log10(3e-2), 26)
theta_B = 0.10

def run_B_beta(beta):
    rng = np.random.default_rng(20260808)
    edges_B = gen_regular_hypergraph(NB, m, d, rng)
    assert critical_count(theta_B, m) == 1
    rng = np.random.default_rng(20260902 + int(beta))
    out, t0 = [], time.time()
    for lam in LAMS_B:
        Rs = np.array([mc_final_size(edges_B, NB, 1/NB, beta, 1, lam, mu, rng)
                       for _ in range(N_RUN_B)])
        n_out = int((Rs > 0.05).sum())
        out.append(dict(fig='B', beta=beta, lam=lam, n_run=N_RUN_B, n_out=n_out,
                        P_out=n_out / N_RUN_B, R=Rs.mean()))
        print(f"[{time.time()-t0:5.0f}s] B: beta={beta} lam={lam:.5f} "
              f"P_out={n_out/N_RUN_B:.3f}", flush=True)
    return out

def run_B():
    with Pool(3) as pool:
        parts = pool.map(run_B_beta, [1.0, 2.0, 3.0])
    pd.DataFrame(sum(parts, [])).to_csv("nucleation_figB.csv", index=False)

if __name__ == "__main__":
    if "A" in PART:
        run_A()
    if "B" in PART:
        run_B()
    print("\n数据已保存")
