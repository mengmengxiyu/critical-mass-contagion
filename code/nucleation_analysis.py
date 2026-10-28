import numpy as np, pandas as pd
from math import comb
from scipy.optimize import brentq
N,m,d,mu = 1000,6,12,0.2
g = pd.read_csv("nucleation_avg.csv")

def B_full(rho, c, beta):
    return sum(comb(m-1,k)*rho**k*(1-rho)**(m-1-k)*k**beta for k in range(c, m))

def lam_exact(rho, c, beta):
    """数值精确解 Eq.(18) mu*i = (1-i)*{1-[1-A(i;lam)]^d}，不做任何渐近近似。

    Eq.(19) 的幂律式额外用了 1-exp(-lam*c^beta) ~ lam*c^beta 与 dA<<1，两者在
    rho_0 不够小、beta 不够小时都会失效；直接解 Eq.(18) 把这两步一起去掉，
    归一化仍然零自由参数。lam -> inf 时右端有上界，故 rho_0 低于某个 i_min 时
    方程无解（返回 nan）——这正是正文所说的种子密度硬下限。
    """
    def f(lam):
        A = sum(comb(m-1,k)*rho**k*(1-rho)**(m-1-k)*(-np.expm1(-lam*k**beta))
                for k in range(c, m))
        return (1-rho)*(1-(1-A)**d) - mu*rho
    try:
        return brentq(f, 1e-9, 1e5)
    except ValueError:
        return np.nan

g['Y_lead'] = g.lam * d*np.array([comb(m-1,int(c)) for c in g.c])/mu * g.c**g.beta * g.rho0**(g.c-1)
g['Y_full'] = g.lam * (1-g.rho0) * d * np.array(
    [B_full(r,int(c),b) for r,c,b in zip(g.rho0,g.c,g.beta)]) / (mu*g.rho0)
g['Y_exact'] = g.lam / np.array(
    [lam_exact(r,int(c),b) for r,c,b in zip(g.rho0,g.c,g.beta)])

print("=== 完整式塌缩：理论预言 Y_full = 1 (所有 c、beta、rho_0 共同塌缩到同一常数) ===")
print(f"{'c':>3} {'rho_0':>7} | " + " ".join(f"b={b:<4.1f}" for b in sorted(g.beta.unique())) + " |    CV")
for c in sorted(g.c.unique()):
    for rho0 in sorted(g[g.c==c].rho0.unique()):
        s = g[(g.c==c)&(g.rho0==rho0)].sort_values('beta')
        y = s.Y_full.values
        print(f"{c:>3.0f} {rho0:>7.3f} | " + " ".join(f"{v:<6.3f}" for v in y) + f" | {y.std()/y.mean()*100:>5.1f}%")

print("\n=== 三种塌缩变量的全局优度对比（全部 55 个点一起统计）===")
for name, col in [("塌缩前 λ_grow", g.lam),
                  ("最低阶式 (19)  Y_lead", g.Y_lead),
                  ("完整核 B_c,β   Y_full", g.Y_full),
                  ("精确解 Eq.(18) Y_exact", g.Y_exact)]:
    v = col.values
    print(f"  {name:<24} 几何均值={np.exp(np.log(v).mean()):>8.4f}   "
          f"log10 跨度={np.log10(v.max()/v.min()):>5.2f}   CV={v.std()/v.mean()*100:>6.1f}%")

print("\n=== 残余的 beta 依赖（对 log Y 关于 beta 做线性回归，斜率应为 0）===")
for name,col in [("Y_lead",'Y_lead'),("Y_full",'Y_full'),("Y_exact",'Y_exact')]:
    for c in sorted(g.c.unique()):
        s = g[g.c==c]
        k,_ = np.polyfit(s.beta, np.log(s[col]), 1)
        print(f"  {name}  c={c:.0f}: d(lnY)/d(beta) = {k:+.4f}")
print("\n=== 精确解归一化后按 c 分组（c=2 与 c=3 系统偏差不同）===")
for c in sorted(g.c.unique()):
    v = g[g.c==c].Y_exact.values
    print(f"  c={c:.0f}: 几何均值={np.exp(np.log(v).mean()):.3f}  "
          f"CV={v.std()/v.mean()*100:.1f}%  范围 {v.min():.3f}-{v.max():.3f}")

g.to_csv("nucleation_collapse.csv", index=False)
