import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# Two slide figures from analysis/rbp_crisis.py's outputs: the rank of each predictor by Impact on Fit
# in each crisis window and in other times, and the past periods whose loss months pushed RBP's
# forecasts up in each window. Written by Claude Code at Flynn's request (2026-10-20).
# Run from the repository root, after analysis/rbp_crisis.py: python analysis/rbp_crisis_figures.py

NAMES = {
    "logret_past_3m_tr": "Past 3-month return", "vol_trailing63_tr": "Trailing volatility",
    "ip_growth_12m": "Industrial production", "cpi_infl_12m": "Inflation",
    "unrate_chg_12m": "Unemployment change", "yc_long": "Yield-curve slope", "turb17": "Turbulence",
    "ar17_shift": "Absorption-ratio shift", "credit_baa_aaa": "Credit spread", "cape": "CAPE",
    "sentiment_umich": "Consumer sentiment", "hhi": "Concentration (HHI)", "gjr_skewt_vol_q": "GARCH volatility",
}
# Past loss months grouped into eras, by the year their episode starts
ERAS = [(1929, 1934, "Depression 1931-34"), (1937, 1939, "Recession 1937-39"), (1940, 1946, "War years 1940-46"),
        (1957, 1966, "1957-62"), (1969, 1975, "1969-75"), (1981, 1990, "1981-90 (incl. 1987)"),
        (1998, 1998, "1998 (LTCM)"), (2000, 2002, "Dot-com 2000-02"), (2007, 2009, "GFC 2007-08"),
        (2010, 2019, "2010-19"), (2020, 2025, "2022 (inflation)")]
WINDOWS = ["Dot-com", "GFC", "COVID", "Post-COVID"]
INK, MUTED, SURFACE, BAR = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7",
        "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]  # rank 1 (darkest) to 13
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})


def era(year):
    return next((name for a, b, name in ERAS if a <= year <= b), "Other")


def predictor_ranks():
    t = pd.read_csv("data/rbp/rbp13_crisis_predictors.csv", header=[0, 1], index_col=0)["Impact on fit"]
    ranks = t[["Full sample", *WINDOWS]].rank(ascending=False).astype(int)
    ranks = ranks.sort_values("Full sample")
    fig, ax = plt.subplots(figsize=(8.4, 6.2), facecolor=SURFACE)
    ax.imshow(ranks.to_numpy() - 1, cmap=ListedColormap(RAMP), vmin=0, vmax=12, aspect="auto")
    for (i, j), r in pd.DataFrame(ranks.to_numpy()).stack().items():
        ax.text(j, i, r, ha="center", va="center", fontsize=10, color="white" if r <= 7 else INK)
    ax.set_xticks(range(5), ["Full sample\n1991-2025","Dot-com\n2000-01 to\n2002-10", "GFC\n2007-07 to\n2009-03",
                             "COVID\n2019-11 to\n2020-06", "Post-COVID\n2020-07 to\n2025-12"])
    ax.set_yticks(range(len(ranks)), [NAMES[p] for p in ranks.index])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x + 0.5 for x in range(4)], minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(ranks) - 1)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.suptitle("Which predictors mattered: rank by Impact on Fit (1 = most important)", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.02, 0.01, "Mean Impact on Fit over each window's monthly forecasts. RBP, 13 predictors.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig("deck/figures/rbp_crisis_predictors.png", dpi=200, facecolor=SURFACE)


def past_periods():
    o = pd.read_csv("data/rbp/rbp13_crisis_observations.csv", parse_dates=["Forecast date", "Past month"])
    o = o[o["Outcome"] == 1]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharex=True, facecolor=SURFACE)
    for ax, w in zip(axes, WINDOWS):
        g = o[o["Window"] == w]
        share = g.groupby(g["Past month"].dt.year.map(era))["Contribution"].sum()
        share = (share / share.sum()).sort_values()
        share = share[share >= 0.03]
        ax.barh(range(len(share)), share.to_numpy(), color=BAR, height=0.7)
        ax.set_yticks(range(len(share)), share.index)
        for i, v in enumerate(share):
            ax.text(v + 0.008, i, f"{v:.0%}", va="center", fontsize=9, color=INK)
        ax.set_title(w, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(0, 0.62)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
        ax.tick_params(length=0)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#d8d7d2")
    fig.suptitle("Which past periods drove the warnings: share of the push above the base rate",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "Each past loss month's weight times (1 - base rate), summed by era over the window's "
             "forecasts; eras under 3% not shown.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig("deck/figures/rbp_crisis_past_periods.png", dpi=200, facecolor=SURFACE)


def rbp_against_logit():
    """RBP's rank by Impact on Fit beside the logit's full-sample Newey-West t-statistics
    (analysis/logit_tstats.py), predictors in the same order. The logit is not split by regime (Flynn)."""
    t = pd.read_csv("data/rbp/rbp13_crisis_predictors.csv", header=[0, 1], index_col=0)["Impact on fit"]
    ranks = t[["Full sample", *WINDOWS]].rank(ascending=False).astype(int).sort_values("Full sample")
    logit = pd.read_csv("data/rbp/logit13_tstats.csv", header=[0, 1], index_col=0).xs("t (Newey-West)", axis=1, level=1)
    logit = logit.loc[ranks.index, ["Full sample"]]
    diverging = LinearSegmentedColormap.from_list("bgr", ["#184f95", "#86b6ef", "#f0efec", "#f0a3a2", "#b8302f"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 6.4), facecolor=SURFACE,
                               gridspec_kw={"wspace": 0.06, "width_ratios": [5, 1.25]})
    a.imshow(ranks.to_numpy() - 1, cmap=ListedColormap(RAMP), vmin=0, vmax=12, aspect="auto")
    for (i, j), r in pd.DataFrame(ranks.to_numpy()).stack().items():
        a.text(j, i, r, ha="center", va="center", fontsize=10, color="white" if r <= 7 else INK)
    b.imshow(logit.to_numpy(), cmap=diverging, vmin=-3.5, vmax=3.5, aspect="auto")
    for (i, j), v in pd.DataFrame(logit.to_numpy()).stack().items():
        b.text(j, i, f"{v:+.1f}" + ("*" if abs(v) >= 1.96 else ""), ha="center", va="center", fontsize=10,
               color="white" if abs(v) >= 2.6 else INK, fontweight="bold" if abs(v) >= 1.96 else "normal")
    a.set_xticks(range(5), ["Full sample", "Dot-com", "GFC", "COVID", "Post-COVID"])
    b.set_xticks(range(1), ["Full sample\n1960-2025"])
    a.set_yticks(range(len(ranks)), [NAMES[p] for p in ranks.index])
    b.set_yticks([])
    for ax, title, ncol in ((a, "RBP: rank by Impact on Fit (1 = most)", 5), (b, "Logit: t-statistic", 1)):
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([x + 0.5 for x in range(ncol - 1)], minor=True)
        ax.set_yticks([y + 0.5 for y in range(len(ranks) - 1)], minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=44)
    fig.suptitle("RBP's crisis-specific predictors against the logistic regression's", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "Logit on the same 13 predictors, standardized, Newey-West t (2 lags for overlapping 3-month "
             "outcomes), 1960-2025.\nRed: a higher value raises the probability of a loss; blue: lowers it. "
             "* |t| >= 1.96.", color=MUTED, fontsize=8.5)
    fig.subplots_adjust(left=0.2, right=0.99, top=0.80, bottom=0.09)
    fig.savefig("deck/figures/rbp_vs_logit_predictors.png", dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    predictor_ranks()
    past_periods()
    rbp_against_logit()
