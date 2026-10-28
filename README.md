# Critical mass and nonlinear reinforcement in higher-order contagion

Code and data for the paper

> M. Li, A. Zhang, A. Li, Z. Di, Y. Moreno, *Critical mass shifts control of nonlinear higher-order contagion from transmission rate to seed density*.

The model is a discrete-time SIR process on hypergraphs with infection kernel
f(n) = n^β 1{n ≥ c_j}, c_j = ⌈θ m_j⌉. All Monte Carlo results come from direct
simulation of the microscopic rules; the mean-field closure is never used inside the simulations.

## Layout

```
code/      simulation, analysis and plotting scripts (comments are partly in Chinese)
data/      CSV files produced by the simulation scripts and read by the plotting scripts
figures/   the six figures of the paper (PDF as used in the manuscript, PNG for preview)
```

## Requirements

Python 3 with `numpy`, `scipy`, `pandas` and `matplotlib`. The results in the paper were
produced with numpy 2.5, scipy 1.18, pandas 3.0 and matplotlib 3.11. Figures use the
TeX Gyre Termes font if it is installed.

## How to run

All scripts are run from `data/`, with `code/` on the Python path, so that CSV files are
read from and written to `data/`:

```bash
cd data
PYTHONPATH=../code python ../code/<script>.py [mode]
```

The plotting scripts write `<figure>.pdf` and `<figure>.png` into the current directory;
copy them to `figures/`. Random seeds are fixed in every script, so the results are reproducible.

## Figures and tables

| Item in the paper | Plotting script | Data | Produced by |
|---|---|---|---|
| Fig. 1, kernel | `fig_kernel_comparison.py` | none | — |
| Fig. 2(a), final size against number of seeds | `fig_seed_density.py` | `nucleation_figA.csv` | `nucleation_fig_ab.py` |
| Fig. 2(b,c), critical seed density for several N | `fig_seed_density.py` | `seed_floor_curve.csv`, `seed_floor_N.csv` | `seed_floor.py nscan` |
| Table 2, 100 seeds as N grows | — | `finite_size.csv`, `finite_size_raw.csv` | `finite_size.py` |
| Sec. 4.2, growth and decay after the first update near ρ* | — | `track_infected.csv` | `track_infected.py` |
| Fig. 3, growth threshold and its collapse | `fig_collapse.py` | `nucleation_collapse.csv`, `onestep_lambda.csv` | `nucleation_run.py` → `nucleation_analysis.py`; `onestep_lambda.py` |
| Sec. 4.2, c = 3 check at N = 5000 | — | `nucleation_raw_N5000.csv`, `nucleation_avg_N5000.csv` | `nucleation_run.py 5000` |
| Fig. 4, invasion threshold for c = 1 (household calculation) | `fig_c1_threshold.py` | `nucleation_figB.csv`, `household_check.csv` | `nucleation_fig_ab.py`; `household.py mc` |
| Fig. 5(a,b), λ–β plane for c = 1 and c = 2 | `fig_beta_compare.py` | `fig2_phase_diagram.csv` | `fig2_phase_diagram.py` |
| Fig. 5(c), effect of β against hyperdegree | `fig_beta_compare.py` | `dscan_raw.csv` | `dscan_run.py` |
| Fig. 6, effective threshold on heterogeneous hypergraphs | `fig_hetero.py` | `hetero_raw.csv` (A and uniform controls), `hetero2_raw.csv` (B and C) | `hetero_run.py`, `hetero_run2.py` |
| Table 3 (Appendix C), accuracy of the mean-field closure | — | `appendix_mmca.csv`, `appendix_mmca_raw.csv` | `appendix_mmca.py` |

Shared modules:

- `nucleation_core.py`: d-regular, m-uniform hypergraph generator (configuration model with stub exchange) and vectorised Monte Carlo.
- `hetero_core.py`: generator and Monte Carlo for hypergraphs with heterogeneous hyperedge sizes.
- `household.py`: exact within-hyperedge Markov chain and the household reproduction number R_* = (d − 1) E[Z] for c = 1.
- `fig_style.py`: common figure style.

The mean-field growth condition and its heterogeneous version are solved inside
`nucleation_analysis.py`, `fig_beta_compare.py` and `fig_hetero.py`.

## Contact

Aobo Zhang, aobozhang@btbu.edu.cn
