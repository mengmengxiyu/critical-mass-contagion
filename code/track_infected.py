# -*- coding: utf-8 -*-
"""
追踪 i_min < rho_0 < rho* 时感染人数随时间的变化（§4.2 第五段）。
c=3、rho_0=0.052 与 c=2、rho_0=0.0032，lambda=5、beta=2、m=6、d=12、mu=0.2、N=5e4。
10 张网络各 1 次，记录 I(t)/I(0) 与最终 R(inf)。

运行（在 god/data/ 下）： PYTHONPATH=../code python ../code/track_infected.py  -> track_infected.csv
"""
import numpy as np, pandas as pd
from nucleation_core import gen_regular_hypergraph

N, m, d, mu, beta, lam, N_NET, T = 50000, 6, 12, 0.2, 2.0, 5.0, 10, 12
rows = []
for c, rho0 in [(3, 0.052), (2, 0.0032)]:
    for net in range(N_NET):
        rng = np.random.default_rng(1000 * c + net)
        E = gen_regular_hypergraph(N, m, d, rng)
        state = np.zeros(N, np.int8)
        state[rng.choice(N, int(round(rho0 * N)), replace=False)] = 1
        I0, t = (state == 1).sum(), 0
        while (state == 1).any():
            inf = state == 1
            if t <= T:
                rows.append(dict(c=c, rho0=rho0, net=net, t=t, I_ratio=inf.sum() / I0))
            cnt = inf[E].sum(1)
            p = np.zeros(N)
            np.add.at(p, E.ravel(), np.repeat(np.where(cnt >= c, cnt ** beta, 0.0), m))
            new_inf = (state == 0) & (rng.random(N) < -np.expm1(-lam * p))
            state[inf & (rng.random(N) < mu)] = 2
            state[new_inf] = 1
            t += 1
        rows.append(dict(c=c, rho0=rho0, net=net, t=-1, I_ratio=(state == 2).sum() / N))  # t=-1 存最终 R
df = pd.DataFrame(rows)
df.to_csv("track_infected.csv", index=False)
traj = df[df.t >= 0].groupby(["c", "t"]).I_ratio.mean().unstack()
print(traj.round(2).to_string())
fin = df[df.t == -1].groupby("c").I_ratio
print("最终 R 平均:", fin.mean().round(4).to_dict(), " 最大:", fin.max().round(4).to_dict())
