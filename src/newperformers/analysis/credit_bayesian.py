"""Bayesian hierarchical sovereign-credit model.

Latent Gaussian + ordered-threshold likelihood:

    α_i ~ Normal(μ_α, σ_α)
    β_i ~ Normal(μ_β, σ_β)
    R*_{i,t} = α_i + β_i · F_{i,t} + γ · M_{i,t}
    R_{i,t} = k  if  τ_{k-1} < R*_{i,t} ≤ τ_k
    cutpoints τ ~ ordered Normal(0, 4)

We sample with NUTS, 4 chains, 1500 draws + 750 warmup. The point of the
exercise is two-fold: (i) get a *posterior* on each sector loading
β_i (and the population mean μ_β) so we can quote credible intervals
rather than point estimates; (ii) demonstrate that the OOS predictive
performance of the Bayesian posterior predictive matches the pooled
ordered probit, validating the simpler model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ..utils.paths import OUT_RESULTS, OUT_TABLES, ensure_dirs

log = get_logger(__name__)


@dataclass
class BayesianFit:
    posterior_mean_loadings: pd.Series  # sector -> posterior mean
    posterior_hdi_low: pd.Series        # sector -> 5% HDI
    posterior_hdi_high: pd.Series       # sector -> 95% HDI
    rhat: pd.Series                     # convergence diagnostic per parameter
    ess_bulk: pd.Series                 # effective sample size per parameter


def _rhat_per_sector(samples: np.ndarray, sectors: list[str]) -> np.ndarray:
    """Gelman-Rubin R̂ for an (n_chains, n_draws, n_params) tensor.

    Computed from scratch to avoid the arviz-version drift across releases
    (1.0 ↔ 1.1 changed APIs and DataTree semantics).
    """
    m, n, k = samples.shape
    chain_means = samples.mean(axis=1)                        # (m, k)
    chain_vars = samples.var(axis=1, ddof=1)                  # (m, k)
    grand_mean = chain_means.mean(axis=0)                     # (k,)
    B = (n / (m - 1)) * ((chain_means - grand_mean) ** 2).sum(axis=0)
    W = chain_vars.mean(axis=0)
    var_hat = ((n - 1) / n) * W + B / n
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / W)
    rhat = np.where(np.isfinite(rhat), rhat, np.nan)
    return rhat


def _autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation up to lag max_lag for a 1-D array."""
    x = x - x.mean()
    var = (x ** 2).mean()
    if var == 0:
        return np.zeros(max_lag + 1)
    n = len(x)
    out = np.empty(max_lag + 1)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        out[lag] = (x[lag:] * x[:-lag]).mean() / var
    return out


def _ess_per_sector(samples: np.ndarray, sectors: list[str],
                     max_lag: int = 200) -> np.ndarray:
    """Effective sample size — Geyer's initial monotone sequence estimator,
    aggregated across chains.
    """
    m, n, k = samples.shape
    ess = np.empty(k)
    for j in range(k):
        per_chain_ess = []
        for c in range(m):
            x = samples[c, :, j]
            ac = _autocorr(x, min(max_lag, n - 2))
            # Sum pairs ρ_{2t} + ρ_{2t+1} until they go negative (Geyer).
            tau = 1.0
            t = 1
            while 2 * t + 1 < len(ac):
                pair = ac[2 * t] + ac[2 * t + 1]
                if pair < 0:
                    break
                tau += 2 * pair
                t += 1
            per_chain_ess.append(n / max(tau, 1e-6))
        ess[j] = float(np.sum(per_chain_ess))
    return ess


def _make_feature_frame(composite: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    sectors = ["education", "energy", "research_innovation",
               "health", "housing_living", "security_stability"]
    base = composite[["iso3", "year"] + sectors].copy()
    macro_codes = ["GC.DOD.TOTL.GD.ZS", "BN.CAB.XOKA.GD.ZS",
                   "FP.CPI.TOTL.ZG", "NY.GDP.PCAP.CD", "SP_RATING"]
    pivot = (panel[panel["indicator"].isin(macro_codes)]
             .pivot_table(index=["iso3", "year"], columns="indicator", values="value")
             .reset_index())
    df = base.merge(pivot, on=["iso3", "year"], how="left")
    df["log_gdppc"] = np.log(df["NY.GDP.PCAP.CD"].clip(lower=1))
    df = df.dropna(subset=["SP_RATING"] + sectors)
    return df


def fit(composite: pd.DataFrame, panel: pd.DataFrame, *,
        draws: int = 1500, tune: int = 750, chains: int = 4,
        target_accept: float = 0.95, seed: int = 20260519) -> BayesianFit:
    try:
        import arviz as az
        import pymc as pm
        import pytensor.tensor as pt
    except ImportError as exc:  # pragma: no cover
        log.warning("PyMC not available: %s", exc)
        return BayesianFit(*([pd.Series(dtype=float)] * 5))

    ensure_dirs()
    sectors = ["education", "energy", "research_innovation",
               "health", "housing_living", "security_stability"]
    df = _make_feature_frame(composite, panel)
    if df.empty:
        log.warning("Bayesian credit: empty feature frame, skipping.")
        return BayesianFit(*([pd.Series(dtype=float)] * 5))

    countries = sorted(df["iso3"].unique())
    country_idx = pd.Categorical(df["iso3"], categories=countries).codes
    F = df[sectors].to_numpy(dtype=float)
    macro = df[["GC.DOD.TOTL.GD.ZS", "BN.CAB.XOKA.GD.ZS",
                "FP.CPI.TOTL.ZG", "log_gdppc"]].fillna(df[[
                "GC.DOD.TOTL.GD.ZS", "BN.CAB.XOKA.GD.ZS",
                "FP.CPI.TOTL.ZG", "log_gdppc"]].mean()).to_numpy(dtype=float)
    macro = (macro - macro.mean(0)) / (macro.std(0) + 1e-9)
    R = df["SP_RATING"].astype(int).to_numpy()
    n_classes = int(R.max() - R.min() + 1)
    R_idx = R - int(R.min())

    log.info("Bayesian credit: %d obs, %d countries, %d classes, sampling...",
             len(R), len(countries), n_classes)

    with pm.Model() as model:
        mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0, shape=len(sectors))
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0, shape=len(sectors))
        z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0,
                            shape=(len(countries), len(sectors)))
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * z_beta)

        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=2.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.5)
        z_alpha = pm.Normal("z_alpha", mu=0.0, sigma=1.0, shape=len(countries))
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * z_alpha)

        gamma = pm.Normal("gamma", mu=0.0, sigma=1.0, shape=macro.shape[1])

        cutpoints = pm.Normal(
            "cutpoints",
            mu=np.linspace(-2.0, 2.0, n_classes - 1),
            sigma=1.0,
            shape=n_classes - 1,
            transform=pm.distributions.transforms.ordered,
        )

        eta = (alpha[country_idx]
               + pt.sum(beta[country_idx, :] * F, axis=1)
               + pt.dot(macro, gamma))
        pm.OrderedLogistic("R", eta=eta, cutpoints=cutpoints, observed=R_idx)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            idata = pm.sample(draws=draws, tune=tune, chains=chains,
                              target_accept=target_accept, random_seed=seed,
                              progressbar=False)

    out_nc = OUT_RESULTS / "credit_bayesian_trace.nc"
    try:
        idata.to_netcdf(str(out_nc))
        log.info("Bayesian credit trace saved -> %s", out_nc.name)
    except Exception as exc:  # noqa: BLE001
        log.warning("trace.to_netcdf failed: %s — skipping trace persistence", exc)

    # Pull the mu_beta samples directly from the inference object rather
    # than relying on az.summary, whose API has shifted across versions.
    mu_beta_samples = None
    if hasattr(idata, "posterior"):
        try:
            mu_beta_samples = np.asarray(idata.posterior["mu_beta"])
        except Exception:  # noqa: BLE001
            mu_beta_samples = None
    if mu_beta_samples is None:
        post = idata["posterior"] if "posterior" in idata.children else idata.posterior  # type: ignore[index]
        mu_beta_samples = np.asarray(post["mu_beta"])
    flat = mu_beta_samples.reshape(-1, mu_beta_samples.shape[-1])
    posterior_mean = pd.Series(flat.mean(axis=0), index=sectors)
    posterior_hdi_low = pd.Series(np.quantile(flat, 0.05, axis=0), index=sectors)
    posterior_hdi_high = pd.Series(np.quantile(flat, 0.95, axis=0), index=sectors)

    rhat_vals = _rhat_per_sector(mu_beta_samples, sectors)
    ess_vals = _ess_per_sector(mu_beta_samples, sectors)
    rhat = pd.Series(rhat_vals, index=sectors)
    ess = pd.Series(ess_vals, index=sectors)
    log.info("Convergence: max R̂ = %.3f, min ESS = %.0f",
             rhat.max(), ess.min())
    if rhat.max() > 1.05:
        log.warning("R̂ exceeds 1.05 — consider rerunning with more draws.")
    if ess.min() < 200:
        log.warning("ESS below 200 — posterior summary may be noisy.")

    out_csv = pd.DataFrame({
        "sector": sectors,
        "posterior_mean": posterior_mean.values,
        "hdi_5%": posterior_hdi_low.values,
        "hdi_95%": posterior_hdi_high.values,
        "r_hat": rhat.values,
        "ess_bulk": ess.values,
    })
    out_csv.to_csv(OUT_TABLES / "credit_bayesian_loadings.csv", index=False)

    log.info("Bayesian posterior mu_beta (90%% HDI) + diagnostics:\n%s",
             out_csv.round(3).to_string(index=False))

    return BayesianFit(
        posterior_mean_loadings=pd.Series(posterior_mean.values, index=sectors),
        posterior_hdi_low=pd.Series(posterior_hdi_low.values, index=sectors),
        posterior_hdi_high=pd.Series(posterior_hdi_high.values, index=sectors),
        rhat=pd.Series(rhat.values, index=sectors),
        ess_bulk=pd.Series(ess.values, index=sectors),
    )
