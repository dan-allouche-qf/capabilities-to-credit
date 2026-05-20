# From Capabilities to Credit

**Development KPIs as Causal Drivers, Priced Risk, and Tradable EM Factors**

A quantitative-finance study of 23 emerging economies (1990–2024) across three lenses: causal macro identification, sovereign-credit pricing, and emerging-market factor construction. The empirical headline: development KPIs price into the cross-sectional level of sovereign ratings (out-of-sample AUC 0.81 on the investment-grade binary) but do not generate tradable returns in either equity or credit at monthly horizons.

— Dan Allouche, 2026

---

## Headline results

| Block | Result |
|---|---|
| **Sovereign credit** | Pooled ordered probit on six sector scores reaches MAE 1.84 notches and AUC 0.81 OOS, 8 pp above macro-only baseline |
| **Bayesian posterior** | Localises signal to energy (+0.92, 90% HDI [+0.10, +1.74]) and health (−0.92, HDI [−1.38, −0.45]); four other sectors cover zero |
| **Causal identification** | Dumitrescu-Hurlin energy → poverty p = 0.027, health → Gini p = 0.046; Westerlund cointegration composite ↔ log GDPpc p = 0.009 |
| **Equity factor** | KPI long-only Sharpe 0.15 vs EEM 0.12, sitting at the **0th percentile** of 5 000 random EM portfolios (mean random Sharpe 0.28) |
| **Credit factor** | Aggregate KPI YoY does not predict EMB returns (β = 0.10, t = 0.73, R² = 0.003, n = 204 months) |

KPIs are slow-moving fundamentals: they price into rating levels, not into monthly returns. The paper documents that distinction over 61 pages, 47 figures, 39 pytest tests, and a factual-claims audit returning 0 orphan tokens.

---

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make all          # data → analysis → figures (~4 min cold)
make paper        # rebuild paper/main.pdf
make dashboard    # http://localhost:8501  (6 pages)
make tests        # 39 pytest tests
```

The `data/raw/` cache pulls from free APIs only (World Bank Open Data, WGI, WHO GHO, SIPRI, FRED, Yahoo Finance). The Bayesian step is the only stochastic block; convergence is gated by `max R̂ < 1.05, min ESS > 200` in the build.

---

## Repository

```
config/              Country, indicator, palette, ratings, verdict configs
data/                Raw API caches + harmonised long-format panel
src/newperformers/   io · etl · analysis · viz · pipeline
paper/main.pdf       Compiled paper (61 pages)
dashboard/           Streamlit app — 6 pages, interactive rating scorecard
outputs/             47 CSV tables + 47 PDF/PNG figures
tools/               Factual-claims auditor + ratings cross-checker
tests/               39 pytest tests
```

---

## Methods

PCA-based composite scoring (jackknife + bootstrap CI + Lasso-CV alternative).
Jordà (2005) local projections with Hamilton (2018) cyclical shocks and Driscoll-Kraay SE.
Abadie-Diamond-Hainmueller (2010) synthetic control with placebo permutation p-values.
Pesaran (2007) CIPS panel unit root + Westerlund (2007) cointegration.
Dumitrescu-Hurlin (2012) panel Granger.
Pooled ordered probit + Bayesian hierarchical ordered logistic (PyMC NUTS).
Cox proportional hazards + Kaplan-Meier on time-to-investment-grade.
Stationary-bootstrap Sharpe CI (Politis-Romano 1994) + 5 000-replication random-portfolio benchmark.

---

## Data

All sources are free APIs. The hand-compiled S&P sovereign ratings CSV (`config/ratings.csv`) covers 23 countries × 30 years and carries an audit trail with 17 cross-checked rating transitions.

Limitations are documented in §4 and §9 of the paper: the panel is purposive rather than random; PISA / IEA balances / UN Comtrade are not in the headline build; the Rwanda synthetic-control gap is economically meaningful but not statistically significant under in-space placebo; survival analysis is underpowered at n = 21 with 16 events; the rating-change OOS classifier collapses to the random-walk baseline; tradable performance fails in both equity and credit at monthly horizons.

---

## License

MIT for the code. CC-BY-4.0 for the hand-compiled ratings CSV.
