"""Paper-grade vector PDF figures.

Each function consumes pre-computed parquet / CSV outputs and writes one PDF
to ``outputs/figures/``. Figures are saved with ``bbox_inches="tight"`` and
matplotlib's pdf backend (``pdf.fonttype=42``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import palette as pal
from . import theme
from ..utils.config import countries
from ..utils.logging import get_logger
from ..utils.paths import OUT_FIGURES, OUT_TABLES, ensure_dirs

log = get_logger(__name__)

HIGHLIGHT_DEFAULT = ("SGP", "CHN", "BRA", "IND", "RUS", "NER")


def _save(fig, name: str) -> Path:
    ensure_dirs()
    out = OUT_FIGURES / f"{name}.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    log.info("Figure saved -> %s", out.relative_to(OUT_FIGURES.parents[1]))
    return out


def _highlight_lines(ax, df: pd.DataFrame, value_col: str,
                     highlight: tuple[str, ...] = HIGHLIGHT_DEFAULT,
                     x_col: str = "year") -> None:
    cs = countries()
    cycle = pal.cycle()
    others = [i for i in df["iso3"].unique() if i not in highlight]
    for iso3 in others:
        s = df[df["iso3"] == iso3].sort_values(x_col)
        ax.plot(s[x_col], s[value_col], color=pal.hex("neutral"),
                linewidth=0.6, alpha=0.35, zorder=1)
    for i, iso3 in enumerate(highlight):
        s = df[df["iso3"] == iso3].sort_values(x_col)
        if s.empty:
            continue
        ax.plot(s[x_col], s[value_col], color=cycle[i % len(cycle)],
                linewidth=1.9, zorder=3, label=cs[iso3].name)


def composite_rank_journey(composite: pd.DataFrame) -> Path:
    """Each country's cross-sectional rank on the composite over 1990–2024.

    The y-axis is inverted so that rank 1 (best) is at the top. Focal
    countries are highlighted; extension EMs sit behind in light grey.
    """
    theme.apply()
    cs = countries()
    df = composite[["iso3", "year", "composite"]].copy()
    df["rank"] = df.groupby("year")["composite"].rank(ascending=False, method="min")
    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    # Grey context for non-highlighted countries.
    others = [i for i in df["iso3"].unique() if i not in HIGHLIGHT_DEFAULT]
    for iso3 in others:
        s = df[df["iso3"] == iso3].sort_values("year")
        ax.plot(s["year"], s["rank"], color=pal.hex("neutral"),
                linewidth=0.5, alpha=0.35, zorder=1)

    cycle = pal.cycle()
    end_year = int(df["year"].max())
    for i, iso3 in enumerate(HIGHLIGHT_DEFAULT):
        s = df[df["iso3"] == iso3].sort_values("year")
        if s.empty:
            continue
        color = cycle[i % len(cycle)]
        ax.plot(s["year"], s["rank"], color=color, linewidth=2.0, zorder=3)
        last = s[s["year"] == end_year]
        if not last.empty:
            ax.annotate(f"{iso3}", xy=(end_year, last["rank"].iloc[0]),
                         xytext=(4, 0), textcoords="offset points",
                         color=color, fontsize=9, va="center")

    ax.invert_yaxis()
    ax.set_yticks([1, 5, 10, 15, 20, 23])
    ax.set_title("Composite rank journey, 1990–2024 (1 = best)", loc="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rank within 23-country panel")
    return _save(fig, "f02b_composite_rank_journey")


def improvement_vs_starting_point(composite: pd.DataFrame) -> Path:
    """Composite-1990 (x) vs composite-change-1990-to-2024 (y) scatter."""
    theme.apply()
    cs = countries()
    df = composite[["iso3", "year", "composite"]].copy()
    start = df[df["year"] == 1990].set_index("iso3")["composite"]
    end = df[df["year"] == int(df["year"].max())].set_index("iso3")["composite"]
    common = start.index.intersection(end.index)
    change = (end.loc[common] - start.loc[common]).rename("change")
    plot_df = pd.concat([start.loc[common].rename("start"), change], axis=1).dropna()

    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    median_start = plot_df["start"].median()
    median_change = plot_df["change"].median()
    ax.axvline(median_start, color=pal.hex("neutral"), linewidth=0.6,
               linestyle="--", alpha=0.5)
    ax.axhline(median_change, color=pal.hex("neutral"), linewidth=0.6,
               linestyle="--", alpha=0.5)

    # Quadrant labels.
    xmin, xmax = plot_df["start"].min() - 0.4, plot_df["start"].max() + 0.4
    ymin, ymax = plot_df["change"].min() - 0.3, plot_df["change"].max() + 0.3
    ax.text(xmax - 0.05, ymax - 0.1, "started high, gained more",
            ha="right", va="top", fontsize=8.5, style="italic",
            color=pal.hex("neutral"))
    ax.text(xmin + 0.05, ymax - 0.1, "started low, improved fast",
            ha="left", va="top", fontsize=8.5, style="italic",
            color=pal.hex("neutral"))
    ax.text(xmax - 0.05, ymin + 0.1, "started high, stagnated",
            ha="right", va="bottom", fontsize=8.5, style="italic",
            color=pal.hex("neutral"))
    ax.text(xmin + 0.05, ymin + 0.1, "started low, stagnated",
            ha="left", va="bottom", fontsize=8.5, style="italic",
            color=pal.hex("neutral"))

    for iso3, row in plot_df.iterrows():
        is_focal = iso3 in HIGHLIGHT_DEFAULT
        color = pal.hex("primary") if row["change"] > 0 else pal.hex("accent_1")
        size = 60 if is_focal else 28
        fontweight = "bold" if is_focal else "normal"
        ax.scatter(row["start"], row["change"], color=color, s=size,
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(iso3, xy=(row["start"], row["change"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8.5, color=pal.hex("dark"),
                    fontweight=fontweight)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Composite score in 1990")
    ax.set_ylabel("Composite change, 1990 → 2024")
    ax.set_title("Starting position vs subsequent improvement",
                 loc="left")
    return _save(fig, "f02c_improvement_vs_starting_point")


def sector_change_heatmap(composite: pd.DataFrame) -> Path:
    """Countries (rows) × sectors (cols) heatmap of Δ score 2024 − 1990."""
    theme.apply()
    sectors = ["education", "energy", "research_innovation",
               "health", "housing_living", "security_stability"]
    end_year = int(composite["year"].max())
    start = composite[composite["year"] == 1990].set_index("iso3")[sectors]
    end = composite[composite["year"] == end_year].set_index("iso3")[sectors]
    common = start.index.intersection(end.index)
    delta = end.loc[common] - start.loc[common]
    delta = delta.assign(total=delta.sum(axis=1)).sort_values("total", ascending=False)
    delta_plot = delta[sectors]

    vmax = max(abs(delta_plot.min().min()), abs(delta_plot.max().max()))
    fig, ax = plt.subplots(figsize=(8.6, 7.5))
    im = ax.imshow(delta_plot.values, aspect="auto",
                   cmap="np_diverging", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(sectors)))
    ax.set_xticklabels([s.replace("_", " ") for s in sectors],
                        rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(delta_plot.index)))
    ax.set_yticklabels(delta_plot.index, fontsize=9)
    # Cell annotations
    for i in range(delta_plot.shape[0]):
        for j in range(delta_plot.shape[1]):
            v = delta_plot.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > vmax * 0.55
                                       else pal.hex("dark"))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Δ sector score (2024 − 1990)", fontsize=9)
    ax.set_title("Where each country improved, 1990 → 2024",
                 loc="left")
    return _save(fig, "f06b_sector_change_heatmap")


def composite_vs_rating_scatter(composite: pd.DataFrame,
                                  panel: pd.DataFrame) -> Path:
    """Latest-year composite (x) vs S&P rating (y), with OLS fit and R²."""
    from scipy.stats import linregress

    theme.apply()
    end_year = int(composite["year"].max())
    comp = composite[composite["year"] == end_year][["iso3", "composite"]]
    rating = (panel[(panel["indicator"] == "SP_RATING")
                    & (panel["year"] == end_year)][["iso3", "value"]]
              .rename(columns={"value": "rating"}))
    merged = comp.merge(rating, on="iso3", how="inner")
    # Some countries might be unrated in the latest year — use the most
    # recent available rating in that case.
    if len(merged) < 15:
        latest_rating = (panel[panel["indicator"] == "SP_RATING"]
                         .sort_values("year")
                         .groupby("iso3").tail(1)[["iso3", "value"]]
                         .rename(columns={"value": "rating"}))
        merged = comp.merge(latest_rating, on="iso3", how="inner")

    x = merged["composite"].to_numpy(dtype=float)
    y = merged["rating"].to_numpy(dtype=float)
    res = linregress(x, y)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    # Rating buckets shaded.
    ax.axhspan(20, 22, color=pal.hex("primary"), alpha=0.06, zorder=1)
    ax.axhspan(13, 20, color=pal.hex("positive"), alpha=0.06, zorder=1)
    ax.axhspan(1, 13, color=pal.hex("accent_1"), alpha=0.06, zorder=1)
    ax.axhline(20.5, color=pal.hex("neutral"), linewidth=0.4, linestyle=":")
    ax.axhline(12.5, color=pal.hex("neutral"), linewidth=0.4, linestyle=":")
    ax.text(x.min() - 0.2, 21, "AAA / AA tier", fontsize=8,
            style="italic", color=pal.hex("dark"))
    ax.text(x.min() - 0.2, 16, "Investment grade",
            fontsize=8, style="italic", color=pal.hex("dark"))
    ax.text(x.min() - 0.2, 7, "Sub-investment grade",
            fontsize=8, style="italic", color=pal.hex("dark"))

    # OLS line.
    xline = np.linspace(x.min() - 0.1, x.max() + 0.1, 50)
    yline = res.intercept + res.slope * xline
    ax.plot(xline, yline, color=pal.hex("primary"), linewidth=1.4,
            linestyle="--", zorder=2)
    ax.text(0.02, 0.97, f"OLS:  $R^2 = {res.rvalue**2:.2f}$, "
                         f"slope $= {res.slope:.2f}$",
            transform=ax.transAxes, va="top", fontsize=10,
            color=pal.hex("primary"))

    ax.scatter(x, y, color=pal.hex("primary"), s=42, zorder=3,
               edgecolor="white", linewidth=0.7)
    for _, row in merged.iterrows():
        ax.annotate(row["iso3"], xy=(row["composite"], row["rating"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8.5, color=pal.hex("dark"))

    ax.set_xlabel(f"Composite KPI score ({end_year})")
    ax.set_ylabel("S&P sovereign rating (1 = SD, 22 = AAA)")
    ax.set_title(f"Development KPIs map onto sovereign ratings ({end_year})",
                 loc="left")
    return _save(fig, "f17b_composite_vs_rating_scatter")


def energy_health_credit_map(composite: pd.DataFrame,
                              panel: pd.DataFrame) -> Path:
    """Countries placed in (energy, health) score space, coloured by rating."""
    theme.apply()
    end_year = int(composite["year"].max())
    sub = composite[composite["year"] == end_year][
        ["iso3", "energy", "health"]]
    rating = (panel[panel["indicator"] == "SP_RATING"]
              .sort_values("year").groupby("iso3").tail(1)[["iso3", "value"]]
              .rename(columns={"value": "rating"}))
    merged = sub.merge(rating, on="iso3", how="left")

    fig, ax = plt.subplots(figsize=(8.4, 6.0))

    def _bucket(r):
        if pd.isna(r):
            return "NR"
        r = int(r)
        if r >= 20:
            return "AAA / AA"
        if r >= 13:
            return "IG"
        if r >= 7:
            return "Sub-IG"
        return "Distressed"

    bucket_color = {
        "AAA / AA": pal.hex("primary"),
        "IG": pal.hex("positive"),
        "Sub-IG": pal.hex("accent_1"),
        "Distressed": pal.hex("negative"),
        "NR": pal.hex("neutral"),
    }
    merged["bucket"] = merged["rating"].apply(_bucket)

    # Bayesian-posterior IG decision boundary (heuristic): the IG threshold
    # in the posterior is roughly the line energy * 0.92 - health * 0.92 = c
    # for some constant c. We solve for c by requiring the line to pass
    # through the empirical mean of IG-vs-Sub-IG.
    if (merged["bucket"].isin({"IG", "AAA / AA"})).any() and (merged["bucket"] == "Sub-IG").any():
        ig = merged[merged["bucket"].isin({"IG", "AAA / AA"})]
        sub_ig = merged[merged["bucket"] == "Sub-IG"]
        mid = 0.5 * (ig[["energy", "health"]].mean() + sub_ig[["energy", "health"]].mean())
        c = 0.92 * mid["energy"] - 0.92 * mid["health"]
        xs = np.linspace(merged["energy"].min() - 0.2, merged["energy"].max() + 0.2, 50)
        ys = (0.92 * xs - c) / 0.92
        ax.plot(xs, ys, color=pal.hex("neutral"), linewidth=1.2,
                linestyle="--", alpha=0.7,
                label="IG boundary (Bayesian posterior)")

    for bucket, color in bucket_color.items():
        b = merged[merged["bucket"] == bucket]
        if b.empty:
            continue
        ax.scatter(b["energy"], b["health"], color=color, s=70,
                   edgecolor="white", linewidth=0.8, label=bucket, zorder=3)
        for _, row in b.iterrows():
            ax.annotate(row["iso3"], xy=(row["energy"], row["health"]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=8.5, color=pal.hex("dark"))

    ax.set_xlabel(f"Energy sector score ({end_year})")
    ax.set_ylabel(f"Health sector score ({end_year})")
    ax.set_title("Credit signal in (energy, health) space", loc="left")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False, ncol=1)
    return _save(fig, "f18b_energy_health_credit_map")


def drawdown_overlap(panel: pd.DataFrame) -> Path:
    """KPI long-only drawdown + EEM drawdown side by side, crisis bands shaded."""
    theme.apply()
    from ..utils.paths import DATA_RAW

    nav = pd.read_csv(OUT_TABLES / "portfolio_long_only_nav.csv",
                       parse_dates=[0])
    nav.columns = ["date", "nav"]
    nav["date"] = pd.to_datetime(nav["date"])
    nav = nav.sort_values("date").set_index("date")
    dd_kpi = (nav["nav"] / nav["nav"].cummax() - 1.0)

    yfin_path = DATA_RAW / "yfinance.parquet"
    eem_dd = None
    if yfin_path.exists():
        prices = pd.read_parquet(yfin_path)
        eem = prices[prices["ticker"] == "EEM"].copy()
        eem["date"] = pd.to_datetime(eem["date"])
        eem = eem.sort_values("date").set_index("date")
        monthly_eem = eem["adj_close"].resample("ME").last()
        monthly_eem = monthly_eem.loc[(monthly_eem.index >= dd_kpi.index.min())
                                        & (monthly_eem.index <= dd_kpi.index.max())]
        eem_nav = (1 + monthly_eem.pct_change().fillna(0)).cumprod()
        eem_dd = (eem_nav / eem_nav.cummax() - 1.0)

    fig, ax = plt.subplots(figsize=(9.0, 4.6))

    crisis_bands = [
        ("2008-09-01", "2009-06-30", "GFC"),
        ("2013-05-01", "2013-09-30", "Taper"),
        ("2015-06-01", "2016-02-29", "Commodities"),
        ("2020-02-01", "2020-04-30", "COVID"),
        ("2022-01-01", "2022-12-31", "Rates"),
    ]
    for start, end, name in crisis_bands:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=pal.hex("neutral"), alpha=0.15, zorder=0)
        ax.text(pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2,
                -0.02, name, ha="center", va="bottom", fontsize=7,
                style="italic", color=pal.hex("dark"))

    ax.fill_between(dd_kpi.index, dd_kpi, 0, color=pal.hex("primary"),
                     alpha=0.30, label="KPI long-only")
    ax.plot(dd_kpi.index, dd_kpi, color=pal.hex("primary"), linewidth=1.0)
    if eem_dd is not None:
        ax.plot(eem_dd.index, eem_dd, color=pal.hex("accent_1"),
                linewidth=1.4, label="MSCI EM (EEM)")
    ax.axhline(0, color=pal.hex("neutral"), linewidth=0.5)
    ax.set_title("Drawdowns aligned across crisis regimes, 2007–2024",
                 loc="left")
    ax.set_xlabel(""); ax.set_ylabel("Drawdown")
    ax.legend(loc="lower right")
    return _save(fig, "f24b_drawdown_overlap")


def survival_km() -> Path:
    """Kaplan–Meier survival curves split by energy and health medians."""
    theme.apply()
    km = pd.read_csv(OUT_TABLES / "survival_km.csv")
    if km.empty:
        return Path()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, split_by in zip(axes, ["energy", "health"], strict=True):
        sub = km[km["split_by"] == split_by]
        for arm, color in [("high", pal.hex("primary")),
                           ("low", pal.hex("accent_1"))]:
            arm_df = sub[sub["arm"] == arm].sort_values("duration")
            if arm_df.empty:
                continue
            ax.step(arm_df["duration"], arm_df["survival"], where="post",
                    linewidth=2.0, color=color,
                    label=f"{arm} {split_by} (n={int(arm_df['n'].iloc[0])})")
        ax.set_xlabel("Years from first rating to IG attainment")
        ax.set_title(f"Split by {split_by}", loc="left", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color=pal.hex("neutral"), linewidth=0.4,
                   linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("Survival = P(not yet IG)")
    fig.suptitle("Kaplan–Meier: time-to-investment-grade attainment",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "f30_survival_km")


def cox_hazards_forest() -> Path:
    """Forest plot of Cox PH hazard ratios."""
    theme.apply()
    cox = pd.read_csv(OUT_TABLES / "survival_cox.csv")
    if cox.empty:
        return Path()
    cox = cox.sort_values("hazard_ratio")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    y = np.arange(len(cox))
    lo = cox["hazard_ratio"] - cox["hr_lower"]
    hi = cox["hr_upper"] - cox["hazard_ratio"]
    ax.errorbar(cox["hazard_ratio"], y, xerr=[lo, hi],
                fmt="o", color=pal.hex("primary"), ecolor=pal.hex("primary"),
                elinewidth=1.4, capsize=4, markersize=7)
    ax.axvline(1.0, color=pal.hex("neutral"), linewidth=0.6, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in cox["sector"]])
    ax.set_xlabel("Hazard ratio (95% CI)")
    ax.set_title("Cox PH on time-to-IG: per-sector hazard ratios",
                 loc="left")
    return _save(fig, "f31_cox_hazards")


def rating_change_confusion() -> Path:
    """3×3 confusion matrix for rating-change OOS predictions."""
    theme.apply()
    cm = pd.read_csv(OUT_TABLES / "rating_change_confusion.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(cm.values, cmap="np_sequential",
                   aspect="equal", interpolation="nearest")
    ax.set_xticks(range(cm.shape[1])); ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(cm.shape[0])); ax.set_yticklabels(cm.index)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm.values[i, j])
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=11,
                    color="white" if v > cm.values.max() / 2
                                       else pal.hex("dark"))
    ax.set_xlabel("Predicted rating change")
    ax.set_ylabel("Actual rating change")
    ax.set_title("Rating-change OOS: confusion matrix", loc="left")
    return _save(fig, "f32_rating_change_confusion")


def selection_flowchart() -> Path:
    """CONSORT-style sample-selection flowchart."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    boxes = [
        (5, 9.0, "23 emerging economies in panel\n(13 focal + 10 extension)"),
        (5, 7.2, "21 rated by S&P\n(BFA from 2004; MLI, NER unrated)"),
        (5, 5.4, "17 with liquid country ETF\n(used in Part-III equity backtest)"),
        (5, 3.6, "16 in expanding-window credit OOS\n(WHO UHC coverage from mid-2000s)"),
        (5, 1.8, "13 in country-dashboards\n(focal group only)"),
    ]
    for x, y, txt in boxes:
        ax.add_patch(plt.Rectangle((x - 2.6, y - 0.6), 5.2, 1.2,
                                     facecolor=pal.hex("panel"),
                                     edgecolor=pal.hex("primary"),
                                     linewidth=1.0))
        ax.text(x, y, txt, ha="center", va="center", fontsize=10,
                color=pal.hex("dark"))

    arrows = [(9.0, 7.8), (7.2, 6.0), (5.4, 4.2), (3.6, 2.4)]
    for top, bot in arrows:
        ax.annotate("", xy=(5, bot), xytext=(5, top),
                    arrowprops=dict(arrowstyle="->", lw=1.0,
                                     color=pal.hex("primary")))

    ax.text(8.5, 8.1, "−2 unrated", fontsize=8.5, style="italic",
            color=pal.hex("accent_1"))
    ax.text(8.5, 6.3, "−4 no liquid ETF", fontsize=8.5, style="italic",
            color=pal.hex("accent_1"))
    ax.text(8.5, 4.5, "−1 coverage gap", fontsize=8.5, style="italic",
            color=pal.hex("accent_1"))
    ax.text(8.5, 2.7, "−3 extension", fontsize=8.5, style="italic",
            color=pal.hex("accent_1"))

    fig.suptitle("Sample selection flow", x=0.02, ha="left", fontsize=12)
    return _save(fig, "f33_selection_flow")


def coverage_heatmap(coverage: pd.DataFrame) -> Path:
    theme.apply()
    order = sorted(coverage.index, key=lambda c: -coverage.loc[c].mean())
    cov = coverage.loc[order]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(cov.values, aspect="auto", cmap="np_sequential",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xticks(range(len(cov.columns)))
    ax.set_xticklabels(cov.columns, rotation=80, fontsize=6.5)
    ax.set_title("Indicator coverage, 1990–2024 (share of years observed)",
                 loc="left", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Share of years observed", fontsize=9)
    return _save(fig, "f01_coverage_heatmap")


def composite_evolution(composite: pd.DataFrame) -> Path:
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    _highlight_lines(ax, composite, "composite")
    ax.axhline(0, color=pal.hex("neutral"), linewidth=0.6, linestyle="--", alpha=0.6)

    # Dated event annotations on the focal countries' trajectories.
    def _ann(year, iso3, text, dx, dy):
        s = composite[(composite["iso3"] == iso3) & (composite["year"] == year)]
        if s.empty:
            return
        y = float(s["composite"].iloc[0])
        ax.annotate(text, xy=(year, y), xytext=(year + dx, y + dy),
                    fontsize=8, style="italic", color=pal.hex("dark"),
                    arrowprops=dict(arrowstyle="-", lw=0.5,
                                    color=pal.hex("neutral"), alpha=0.7))

    _ann(2001, "CHN", "China WTO accession", -8, -0.8)
    _ann(2014, "RUS", "Crimea sanctions", -10, 0.7)
    _ann(2000, "RWA", "Vision 2020 launched", -10, -0.7)

    ax.set_title("Composite KPI score, 1990–2024", loc="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Composite (standardised)")
    ax.legend(ncol=3, fontsize=8, loc="lower right", frameon=False)
    return _save(fig, "f02_composite_evolution")


def composite_dispersion(composite: pd.DataFrame) -> Path:
    theme.apply()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    g = composite.groupby("year")["composite"].agg(["mean", "std"]).reset_index()
    ax.plot(g["year"], g["std"], color=pal.hex("primary"), linewidth=1.8)
    ax.fill_between(g["year"], g["std"] * 0.95, g["std"] * 1.05,
                    color=pal.hex("primary"), alpha=0.10)
    ax.set_title("Cross-country dispersion of the composite score", loc="left")
    ax.set_xlabel("Year")
    ax.set_ylabel("Std. dev. across the 23-country panel")
    return _save(fig, "f03_composite_dispersion")


def sector_scores_small_multiples(composite: pd.DataFrame) -> Path:
    theme.apply()
    sectors = ["education", "energy", "research_innovation",
               "health", "housing_living", "security_stability"]
    titles = {"education": "Education", "energy": "Energy",
              "research_innovation": "Research & Innovation",
              "health": "Health", "housing_living": "Housing & Living",
              "security_stability": "Security & Stability"}
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), sectors, strict=True):
        sub = composite[["iso3", "year", s]].rename(columns={s: "value"})
        _highlight_lines(ax, sub, "value")
        ax.axhline(0, color=pal.hex("neutral"), linewidth=0.6,
                   linestyle="--", alpha=0.5)
        ax.set_title(titles[s], loc="left", fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    for ax in axes[:, 0]:
        ax.set_ylabel("Score (std.)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), fontsize=8.5, frameon=False)
    fig.suptitle("Sector scores by country, 1990–2024", x=0.01, ha="left",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return _save(fig, "f04_sector_scores")


def gdp_slope_chart(panel: pd.DataFrame) -> Path:
    """Slope chart with non-overlapping labels via greedy y-offset packing."""
    theme.apply()
    gdppc = panel[panel["indicator"] == "NY.GDP.PCAP.CD"]
    sub = (gdppc[gdppc["year"].isin([1990, 2024])]
           .pivot_table(index="iso3", columns="year", values="value")
           .dropna())
    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    for iso3, row in sub.iterrows():
        gain = row[2024] / row[1990]
        color = pal.hex("primary") if gain >= 3 else pal.hex("accent_1") if gain < 1.5 else pal.hex("neutral")
        ax.plot([0, 1], [row[1990], row[2024]], color=color, alpha=0.55,
                linewidth=1.0)
        ax.scatter([0, 1], [row[1990], row[2024]], color=color, s=20,
                   zorder=3)
    ax.set_yscale("log")

    # Label packing — separate by at least min_log_sep on the log axis.
    def _pack_labels(values: pd.Series, min_log_sep: float = 0.045) -> dict[str, float]:
        order = values.sort_values().index.tolist()
        y_pos: dict[str, float] = {}
        last = -np.inf
        for iso3 in order:
            yv = max(float(values[iso3]), 1.0)
            log_y = np.log10(yv)
            if log_y - last < min_log_sep:
                log_y = last + min_log_sep
            y_pos[iso3] = 10 ** log_y
            last = log_y
        return y_pos

    left_pos = _pack_labels(sub[1990])
    right_pos = _pack_labels(sub[2024])
    for iso3, row in sub.iterrows():
        gain = row[2024] / row[1990]
        color = pal.hex("primary") if gain >= 3 else pal.hex("accent_1") if gain < 1.5 else pal.hex("neutral")
        ax.annotate(iso3, xy=(0, row[1990]),
                     xytext=(-0.08, left_pos[iso3]), ha="right", va="center",
                     fontsize=8, color=color,
                     arrowprops=dict(arrowstyle="-", lw=0.4, color=color, alpha=0.4))
        ax.annotate(f"{iso3} (×{gain:.1f})", xy=(1, row[2024]),
                     xytext=(1.08, right_pos[iso3]), ha="left", va="center",
                     fontsize=8, color=color,
                     arrowprops=dict(arrowstyle="-", lw=0.4, color=color, alpha=0.4))

    ax.set_xticks([0, 1]); ax.set_xticklabels(["1990", "2024"])
    ax.set_ylabel("GDP per capita, current USD (log)")
    ax.set_title("Real divergence, 1990 vs 2024 ($\\times$N = GDPpc multiplier)",
                 loc="left")
    ax.set_xlim(-0.3, 1.4)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)
    return _save(fig, "f05_gdp_slope")


def lp_irf_panel(outcome: str = "log_gdppc_x100",
                  outcome_label: str = "Log GDP per capita ×100") -> Path:
    theme.apply()
    df = pd.read_csv(OUT_TABLES / "lp_irf.csv")
    df = df[df["outcome"] == outcome]
    sectors = sorted(df["shock"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4), sharex=True)
    for ax, s in zip(axes.ravel(), sectors, strict=False):
        sub = df[df["shock"] == s].sort_values("horizon")
        beta = sub["beta"].to_numpy()
        se = sub["se"].to_numpy()
        h = sub["horizon"].to_numpy()
        ax.plot(h, beta, color=pal.hex("primary"), linewidth=1.6)
        ax.fill_between(h, beta - 1.645 * se, beta + 1.645 * se,
                        color=pal.hex("primary"), alpha=0.18,
                        label="90% DK CI")
        ax.axhline(0, color=pal.hex("neutral"), linewidth=0.6,
                   linestyle="--", alpha=0.6)
        # Per-panel h=1 summary annotation.
        h1 = sub[sub["horizon"] == 1]
        if not h1.empty:
            b1 = float(h1["beta"].iloc[0]); p1 = float(h1["pvalue"].iloc[0])
            sig = h1["sig_flag"].iloc[0] if "sig_flag" in h1.columns else ""
            label = f"$h{{=}}1$: $\\beta{{=}}{b1:+.1f}$, $p{{=}}{p1:.3f}${sig}"
            ax.text(0.97, 0.97, label, transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5,
                    color=pal.hex("dark"),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                               edgecolor=pal.hex("gridline"), alpha=0.85))
        ax.set_title(s.replace("_", " "), loc="left", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("Horizon (years)")
    for ax in axes[:, 0]:
        ax.set_ylabel(outcome_label)
    fig.suptitle("Local-projection IRFs by sector shock (Driscoll–Kraay lag=2)",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "f09_lp_irfs_gdppc")


def macro_panel(panel: pd.DataFrame) -> Path:
    theme.apply()
    macros = [
        ("GC.DOD.TOTL.GD.ZS", "Central govt. debt (% GDP)"),
        ("BN.CAB.XOKA.GD.ZS", "Current account (% GDP)"),
        ("FI.RES.TOTL.CD",    "Total reserves (USD)"),
        ("BX.KLT.DINV.WD.GD.ZS", "FDI net inflows (% GDP)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.0), sharex=True)
    for ax, (code, label) in zip(axes.ravel(), macros, strict=True):
        sub = panel[panel["indicator"] == code].pivot_table(
            index="year", columns="iso3", values="value")
        if code == "FI.RES.TOTL.CD":
            sub = sub / 1e9
            label = label.replace("(USD)", "(USD bn)")
        _df = sub.reset_index().melt(id_vars="year", var_name="iso3",
                                       value_name="value").dropna()
        _highlight_lines(ax, _df, "value")
        ax.axhline(0, color=pal.hex("neutral"), linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_title(label, loc="left", fontsize=10.5)
    for ax in axes[-1]:
        ax.set_xlabel("Year")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), fontsize=8.5, frameon=False)
    fig.suptitle("Macroeconomic backdrop: four dimensions", x=0.01, ha="left",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return _save(fig, "f06_macro_panel")


def pca_biplots(panel: pd.DataFrame) -> Path:
    """One subplot per sector: indicators projected onto PC1-PC2."""
    from sklearn.decomposition import PCA

    from ..utils.config import kpi_indicators

    theme.apply()
    sectors = ["education", "energy", "research_innovation",
               "health", "housing_living", "security_stability"]
    titles = {"education": "Education", "energy": "Energy",
              "research_innovation": "Research & Innovation",
              "health": "Health", "housing_living": "Housing & Living",
              "security_stability": "Security & Stability"}
    inds_by_sector: dict[str, list[str]] = {}
    for ind in kpi_indicators():
        if ind.role != "core":
            continue
        inds_by_sector.setdefault(ind.sector, []).append(ind.code)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for ax, sector in zip(axes.ravel(), sectors, strict=True):
        codes = [c for c in inds_by_sector.get(sector, [])
                 if c in panel["indicator"].unique()]
        sub = (panel[panel["indicator"].isin(codes)]
               .pivot_table(index=["iso3", "year"], columns="indicator", values="value")
               .dropna())
        if sub.shape[1] < 2 or sub.empty:
            ax.axis("off"); continue
        z = (sub - sub.mean()) / (sub.std() + 1e-9)
        pca = PCA(n_components=2, random_state=20260519).fit(z)
        coords = pca.transform(z)
        ax.scatter(coords[:, 0], coords[:, 1], s=6, color=pal.hex("neutral"),
                   alpha=0.2)
        loadings = pca.components_.T  # (n_features, 2)
        scale = np.max(np.abs(coords)) / (np.max(np.abs(loadings)) + 1e-9)
        for k, code in enumerate(sub.columns):
            ax.arrow(0, 0, loadings[k, 0] * scale * 0.7, loadings[k, 1] * scale * 0.7,
                     head_width=0.3, color=pal.hex("primary"), alpha=0.9,
                     length_includes_head=True)
            ax.text(loadings[k, 0] * scale * 0.75, loadings[k, 1] * scale * 0.75,
                    code.split(".")[-1][:8], fontsize=7,
                    color=pal.hex("dark"))
        ax.axhline(0, color=pal.hex("neutral"), linewidth=0.4, alpha=0.6)
        ax.axvline(0, color=pal.hex("neutral"), linewidth=0.4, alpha=0.6)
        ax.set_title(titles[sector] + f"  (PC1={pca.explained_variance_ratio_[0]:.0%})",
                     loc="left", fontsize=10)
    fig.suptitle("PCA biplots — indicator loadings on PC1 vs PC2 per sector",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "f07_pca_biplots")


def rating_heatmap(ratings_panel: pd.DataFrame) -> Path:
    theme.apply()
    wide = (ratings_panel[ratings_panel["indicator"] == "SP_RATING"]
            .pivot_table(index="iso3", columns="year", values="value"))
    wide = wide.sort_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(wide.values, aspect="auto", cmap="np_sequential", vmin=1,
                   vmax=22, interpolation="nearest")
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels(wide.index, fontsize=9)
    ax.set_xticks(range(0, len(wide.columns), 2))
    ax.set_xticklabels([str(y) for y in wide.columns[::2]], rotation=60, fontsize=8)
    ax.set_title("S&P sovereign rating (year-end, 22 = AAA, 13 = BBB-, 1 = SD)",
                 loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Numeric rating", fontsize=9)
    return _save(fig, "f17_rating_heatmap")


def credit_oos_calibration() -> Path:
    theme.apply()
    df = pd.read_csv(OUT_TABLES / "credit_oos_predictions.csv")
    score = pd.read_csv(OUT_TABLES / "credit_oos_scorecard.csv")
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["actual"], df["predicted"], alpha=0.25,
               color=pal.hex("primary"), s=14)
    lims = [int(df["actual"].min()) - 1, int(df["actual"].max()) + 1]
    ax.plot(lims, lims, color=pal.hex("neutral"), linewidth=0.8, linestyle="--")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual S&P rating (numeric 1–22)")
    ax.set_ylabel("Out-of-sample predicted rating")
    full_mae = float(score[score["model"] == "full"]["mae"].iloc[0])
    ax.set_title(f"Out-of-sample rating calibration  |  full-model MAE = {full_mae:.2f} notches",
                 loc="left")
    return _save(fig, "f19_credit_calibration")


def credit_oos_scorecard() -> Path:
    theme.apply()
    df = pd.read_csv(OUT_TABLES / "credit_oos_scorecard.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    order = df.sort_values("auc_ig")["model"].tolist()
    df = df.set_index("model").reindex(order).reset_index()
    axes[0].barh(df["model"], df["mae"], color=pal.hex("primary"))
    axes[0].set_xlabel("Out-of-sample MAE (notches)")
    axes[0].set_title("Prediction error", loc="left", fontsize=11)
    axes[1].barh(df["model"], df["auc_ig"], color=pal.hex("accent_1"))
    axes[1].set_xlabel("AUC — investment-grade classification")
    axes[1].set_title("Discrimination", loc="left", fontsize=11)
    axes[1].axvline(0.5, color=pal.hex("neutral"), linewidth=0.6, linestyle="--")
    fig.tight_layout()
    return _save(fig, "f20_credit_scorecard")


def portfolio_nav() -> Path:
    theme.apply()
    nav_lo = pd.read_csv(OUT_TABLES / "portfolio_long_only_nav.csv", parse_dates=[0])
    nav_lo.columns = ["date", "nav"]
    nav_lo["date"] = pd.to_datetime(nav_lo["date"])
    nav_ls = pd.read_csv(OUT_TABLES / "portfolio_long_short_nav.csv", parse_dates=[0])
    nav_ls.columns = ["date", "nav"]
    nav_ls["date"] = pd.to_datetime(nav_ls["date"])
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(nav_lo["date"], nav_lo["nav"], color=pal.hex("primary"),
            linewidth=1.8, label="Long-only top quintile")
    ax.plot(nav_ls["date"], nav_ls["nav"], color=pal.hex("accent_1"),
            linewidth=1.4, label="Long-short top − bottom")
    ax.axhline(1.0, color=pal.hex("neutral"), linewidth=0.6, linestyle="--", alpha=0.7)

    # Dated annotations on key inflection points.
    gfc_trough_date = nav_lo.loc[nav_lo["date"] < "2010-01-01"]["nav"].idxmin()
    gfc_row = nav_lo.iloc[gfc_trough_date] if isinstance(gfc_trough_date, int) else nav_lo.loc[gfc_trough_date]
    ax.annotate("GFC trough\n(long-only −56%)",
                 xy=(gfc_row["date"], gfc_row["nav"]),
                 xytext=(20, -25), textcoords="offset points",
                 fontsize=8, style="italic", color=pal.hex("dark"),
                 arrowprops=dict(arrowstyle="->", lw=0.6,
                                  color=pal.hex("neutral")))

    rsx_collapse = nav_ls[(nav_ls["date"] > "2015-01-01") & (nav_ls["date"] < "2016-06-30")]
    if not rsx_collapse.empty:
        worst = rsx_collapse.loc[rsx_collapse["nav"].idxmin()]
        ax.annotate("Long-short cliff\n(2015–16 commodities)",
                     xy=(worst["date"], worst["nav"]),
                     xytext=(-90, 20), textcoords="offset points",
                     fontsize=8, style="italic", color=pal.hex("dark"),
                     arrowprops=dict(arrowstyle="->", lw=0.6,
                                      color=pal.hex("neutral")))

    ax.set_title("KPI-sorted EM portfolio NAV, 2007–2024 (after 20 bps tcosts)",
                 loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative NAV (start = 1.0)")
    ax.legend(loc="upper left")
    return _save(fig, "f23_portfolio_nav")


def bayesian_loadings() -> Path:
    theme.apply()
    df = pd.read_csv(OUT_TABLES / "credit_bayesian_loadings.csv")
    df = df.sort_values("posterior_mean")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    y = np.arange(len(df))
    lo = df["posterior_mean"] - df["hdi_5%"]
    hi = df["hdi_95%"] - df["posterior_mean"]
    ax.errorbar(df["posterior_mean"], y, xerr=[lo, hi],
                fmt="o", color=pal.hex("primary"), ecolor=pal.hex("primary"),
                elinewidth=1.6, capsize=4, markersize=7)
    ax.axvline(0, color=pal.hex("neutral"), linewidth=0.6, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ") for s in df["sector"]])
    ax.set_xlabel(r"Posterior mean of $\mu_\beta$ (90% HDI)")
    ax.set_title("Bayesian hierarchical credit model — sector loadings",
                 loc="left")
    return _save(fig, "f18_bayesian_loadings")


def synthetic_control_plot(tag: str = "rwanda_2000",
                            title: str = "Rwanda Vision 2020 (treated 2000)",
                            treatment_year: int = 2000) -> Path:
    theme.apply()
    df = pd.read_csv(OUT_TABLES / f"synthcontrol_{tag}.csv")
    placebos = pd.read_csv(OUT_TABLES / f"synthcontrol_{tag}_placebos.csv",
                            index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    ax.plot(df["year"], df["actual"], color=pal.hex("accent_1"),
            linewidth=2.0, label="Actual")
    ax.plot(df["year"], df["synthetic"], color=pal.hex("primary"),
            linewidth=2.0, linestyle="--", label="Synthetic")
    ax.axvline(treatment_year, color=pal.hex("neutral"), linewidth=0.6,
               linestyle="--")
    ax.legend(loc="upper left")
    ax.set_title(f"{title} — actual vs synthetic GDP/capita (PPP)",
                 loc="left", fontsize=11)
    ax.set_xlabel("Year"); ax.set_ylabel("GDPpc (PPP, 2017 USD)")

    ax = axes[1]
    years = placebos.index.astype(int)
    for col in placebos.columns:
        ax.plot(years, placebos[col], color=pal.hex("neutral"),
                linewidth=0.6, alpha=0.4)
    if "actual" in df.columns and "synthetic" in df.columns:
        gap = df["actual"] - df["synthetic"]
        ax.plot(df["year"], gap, color=pal.hex("accent_1"), linewidth=2.0,
                label="Treated gap")
    ax.axvline(treatment_year, color=pal.hex("neutral"), linewidth=0.6,
               linestyle="--")
    ax.axhline(0, color=pal.hex("neutral"), linewidth=0.4)
    ax.set_title("Placebo gap distribution (in-space permutation)",
                 loc="left", fontsize=11)
    ax.set_xlabel("Year"); ax.set_ylabel("Gap, GDPpc (PPP USD)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, f"f13_synth_{tag}")


def quintile_navs() -> Path:
    theme.apply()
    df = pd.read_csv(OUT_TABLES / "portfolio_quintile_nav.csv", index_col=0,
                      parse_dates=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    cycle = pal.cycle()
    for k, col in enumerate(df.columns):
        ax.plot(df.index, df[col], color=cycle[k % len(cycle)],
                linewidth=1.6, label=col)
    ax.set_yscale("log")
    ax.axhline(1.0, color=pal.hex("neutral"), linewidth=0.6,
               linestyle="--", alpha=0.7)
    ax.set_title("Quintile portfolio NAVs (Q1 = bottom KPI quintile, Q5 = top)",
                 loc="left")
    ax.set_ylabel("Cumulative NAV (log)"); ax.set_xlabel("")
    ax.legend(loc="upper left", ncol=5)
    return _save(fig, "f25_quintile_navs")


def bootstrap_sharpe_vs_random() -> Path:
    theme.apply()
    arr = pd.read_csv(OUT_TABLES / "robustness_random_portfolios_distribution.csv")[
        "random_sharpe"].to_numpy()
    rand_summary = pd.read_csv(OUT_TABLES / "robustness_random_portfolios.csv").iloc[0]
    kpi_sharpe = float(rand_summary["kpi_sharpe"])
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.hist(arr, bins=50, color=pal.hex("primary"), alpha=0.75,
            edgecolor="white")
    ax.axvline(kpi_sharpe, color=pal.hex("accent_1"), linewidth=2.4,
               label=f"KPI strategy = {kpi_sharpe:.2f}")
    ax.axvline(float(arr.mean()), color=pal.hex("neutral"), linewidth=1.0,
               linestyle="--", label=f"Random mean = {arr.mean():.2f}")
    ax.set_title("Stationary-bootstrap distribution of Sharpe ratios — "
                 "5 000 random 4-country baskets",
                 loc="left", fontsize=11)
    ax.set_xlabel("Annualised Sharpe")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    return _save(fig, "f26_bootstrap_sharpe_vs_random")


def risk_decomposition_bars() -> Path:
    theme.apply()
    df = pd.read_csv(OUT_TABLES / "portfolio_risk_decomposition.csv")
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    strategies = df["strategy"].unique()
    factors = ["alpha", "EEM", "URTH", "UUP", "USO"]
    width = 0.38
    x = np.arange(len(factors))
    for i, strat in enumerate(strategies):
        sub = df[df["strategy"] == strat].set_index("factor")
        vals = [float(sub.loc[f, "coefficient"]) if f in sub.index else 0.0 for f in factors]
        color = pal.hex("primary") if strat == "long_only" else pal.hex("accent_1")
        ax.bar(x + i * width - width/2, vals, width=width, label=strat,
               color=color, alpha=0.85)
    ax.axhline(0, color=pal.hex("neutral"), linewidth=0.6, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels([f.upper() if f != "alpha" else r"$\alpha$"
                                          for f in factors])
    ax.set_title("Risk-factor decomposition of monthly returns", loc="left")
    ax.set_ylabel("OLS coefficient (monthly)")
    ax.legend()
    return _save(fig, "f27_risk_decomposition")


def auc_curve() -> Path:
    theme.apply()
    df = pd.read_csv(OUT_TABLES / "credit_oos_predictions.csv")
    y_true = (df["actual"] >= 13).astype(int).to_numpy()
    y_score = df["predicted"].to_numpy(dtype=float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tpr = np.cumsum(y_sorted) / max(y_true.sum(), 1)
    fpr = np.cumsum(1 - y_sorted) / max((1 - y_true).sum(), 1)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, color=pal.hex("primary"), linewidth=2.0,
            label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=pal.hex("neutral"), linewidth=0.7,
            linestyle="--", alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — investment-grade out-of-sample classification",
                 loc="left", fontsize=11)
    ax.legend(loc="lower right")
    return _save(fig, "f21_auc_curve")


def marginal_effects() -> Path:
    theme.apply()
    df = pd.read_csv(OUT_TABLES / "credit_coefficients.csv")
    df = df[df["model"] == "sectors_only"].copy().sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = [pal.hex("accent_1") if v < 0 else pal.hex("primary")
              for v in df["coefficient"]]
    ax.barh(df["variable"], df["coefficient"], color=colors)
    ax.axvline(0, color=pal.hex("neutral"), linewidth=0.6, linestyle="--")
    ax.set_xlabel("Ordered-probit coefficient")
    ax.set_title("Marginal effect of sector scores on rating index (frequentist)",
                 loc="left", fontsize=11)
    return _save(fig, "f22_marginal_effects")


def policy_timeline() -> Path:
    from ..utils.config import policy_events as _events

    theme.apply()
    events = _events()
    countries_order = list(events.keys())
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    palette_map = {
        "fiscal": pal.hex("primary"),
        "monetary": pal.hex("accent_1"),
        "reform": pal.hex("accent_2"),
        "conflict": pal.hex("negative"),
        "sanctions": pal.hex("accent_3"),
        "treaty": pal.hex("positive"),
    }
    for i, iso3 in enumerate(countries_order):
        for ev in events[iso3]:
            color = palette_map.get(ev.get("category"), pal.hex("neutral"))
            ax.scatter(ev["year"], i, color=color, s=18, alpha=0.9, edgecolor="white",
                       linewidth=0.4)
    ax.set_yticks(range(len(countries_order)))
    ax.set_yticklabels(countries_order, fontsize=9)
    ax.set_xlim(1989, 2025)
    ax.set_title("Curated policy / event timeline by focal country", loc="left")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=c, markersize=7, label=lab)
               for lab, c in palette_map.items()]
    ax.legend(handles=handles, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.10), fontsize=8.5, frameon=False)
    return _save(fig, "f08_policy_timeline")


def granger_heatmap() -> Path:
    theme.apply()
    df = pd.read_csv(OUT_TABLES / "granger_panel.csv")
    pivot = df.pivot(index="shock", columns="outcome", values="p_value")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    im = ax.imshow(pivot.values, cmap="np_sequential_r" if "np_sequential_r"
                    in plt.colormaps() else "np_sequential", aspect="auto",
                    vmin=0, vmax=0.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([s.replace("_", " ") for s in pivot.index], fontsize=9)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.25 else pal.hex("dark"),
                        fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("p-value")
    ax.set_title("Dumitrescu-Hurlin panel Granger causality (p-values)",
                 loc="left", fontsize=11)
    return _save(fig, "f16_granger_heatmap")


def rolling_sharpe() -> Path:
    """36-month rolling Sharpe of the long-only KPI strategy vs EEM."""
    theme.apply()
    nav_lo = pd.read_csv(OUT_TABLES / "portfolio_long_only_nav.csv",
                          parse_dates=[0])
    nav_lo.columns = ["date", "nav"]
    nav_lo["date"] = pd.to_datetime(nav_lo["date"])
    nav_lo = nav_lo.set_index("date").sort_index()
    rets = nav_lo["nav"].pct_change().dropna()
    roll_sharpe = (rets.rolling(36).mean() * 12) / (
        rets.rolling(36).std() * np.sqrt(12))

    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.plot(roll_sharpe.index, roll_sharpe, color=pal.hex("primary"),
            linewidth=1.6, label="Long-only top quintile")
    ax.axhline(0, color=pal.hex("neutral"), linewidth=0.6, linestyle="--")
    ax.axhline(0.28, color=pal.hex("accent_1"), linewidth=0.8, linestyle=":",
               label="Random-portfolio mean (0.28)")
    ax.set_title("Rolling 36-month Sharpe ratio", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Annualised Sharpe (36m window)")
    ax.legend(loc="upper left")
    return _save(fig, "f28_rolling_sharpe")


def country_dashboards(composite: pd.DataFrame, panel: pd.DataFrame) -> list[Path]:
    """One PDF per focal country: composite + 6 sectors + 4 macro panels."""
    from ..utils.config import focal_iso3, countries as _cs

    theme.apply()
    cs = _cs()
    sectors = ["education", "energy", "research_innovation",
               "health", "housing_living", "security_stability"]
    macros = [("NY.GDP.PCAP.CD", "GDP per capita (USD)"),
              ("GC.DOD.TOTL.GD.ZS", "Central govt. debt (% GDP)"),
              ("BN.CAB.XOKA.GD.ZS", "Current account (% GDP)"),
              ("FP.CPI.TOTL.ZG", "CPI inflation (%)")]
    out_paths: list[Path] = []
    for iso3 in focal_iso3():
        c_name = cs[iso3].name
        comp_sub = composite[composite["iso3"] == iso3].sort_values("year")
        fig, axes = plt.subplots(3, 4, figsize=(11.5, 7.5))
        fig.suptitle(f"{c_name} ({iso3}) — country dashboard, 1990–2024",
                     x=0.01, ha="left", fontsize=13)

        # Row 1, col 0: composite
        ax = axes[0, 0]
        ax.plot(comp_sub["year"], comp_sub["composite"], color=pal.hex("primary"),
                linewidth=1.8)
        ax.axhline(0, color=pal.hex("neutral"), linewidth=0.5, linestyle="--")
        ax.set_title("Composite KPI", loc="left", fontsize=10)

        # Row 1, cols 1-3 and row 2: six sector mini-plots
        slots = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
        for (r, c), s in zip(slots, sectors, strict=True):
            ax = axes[r, c]
            ax.plot(comp_sub["year"], comp_sub[s], color=pal.hex("accent_1"),
                    linewidth=1.4)
            ax.axhline(0, color=pal.hex("neutral"), linewidth=0.5,
                       linestyle="--", alpha=0.6)
            ax.set_title(s.replace("_", " "), loc="left", fontsize=9.5)

        # Row 2 col 3 + row 3: macro snapshot
        macro_slots = [(1, 3), (2, 0), (2, 1), (2, 2)]
        for (r, c), (code, label) in zip(macro_slots, macros, strict=True):
            ax = axes[r, c]
            m = panel[(panel["iso3"] == iso3) & (panel["indicator"] == code)]
            m = m.sort_values("year")
            if m.empty:
                ax.axis("off"); continue
            ax.plot(m["year"], m["value"], color=pal.hex("primary"),
                    linewidth=1.4)
            ax.set_title(label, loc="left", fontsize=9.5)

        # Row 3 col 3: rating
        ax = axes[2, 3]
        r = panel[(panel["iso3"] == iso3) & (panel["indicator"] == "SP_RATING")]
        if not r.empty:
            r = r.sort_values("year")
            ax.step(r["year"], r["value"], color=pal.hex("primary"), where="post",
                    linewidth=1.6)
            ax.axhline(13, color=pal.hex("accent_1"), linewidth=0.6,
                       linestyle="--", alpha=0.7)
            ax.set_ylim(0, 23)
            ax.set_title("S&P rating (1=SD, 22=AAA)", loc="left", fontsize=9.5)
        else:
            ax.axis("off")

        for ax_row in axes:
            for ax in ax_row:
                ax.tick_params(labelsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = _save(fig, f"f30_dashboard_{iso3}")
        out_paths.append(out)
    return out_paths


def portfolio_drawdown() -> Path:
    theme.apply()
    nav_lo = pd.read_csv(OUT_TABLES / "portfolio_long_only_nav.csv", parse_dates=[0])
    nav_lo.columns = ["date", "nav"]
    dd = nav_lo["nav"] / nav_lo["nav"].cummax() - 1.0
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    ax.fill_between(nav_lo["date"], dd, 0, color=pal.hex("accent_1"), alpha=0.6)
    ax.plot(nav_lo["date"], dd, color=pal.hex("accent_1"), linewidth=0.9)
    ax.set_title("Drawdown — long-only top quintile", loc="left")
    ax.set_xlabel(""); ax.set_ylabel("Drawdown")
    return _save(fig, "f24_portfolio_drawdown")
