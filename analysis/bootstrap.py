import numpy as np
import pandas as pd
from scipy.stats import rankdata

from backtest import STATIC_STOCKS, monthly_returns

# Is it noise? A block bootstrap of the differences between methods in the like-for-like comparison.
# Written by Claude Code at Maxime's request (2026-10-18).
#
# Reads the 420 month-ends of data/compare/forecasts_all.csv (analysis/compare.py) and resamples
# whole blocks of BLOCK consecutive months, so that the overlap of 3-month outcomes and the
# clustering of loss months survive inside each block. Blocks are circular (a block may run from
# 2025 back into 1991), so every month is equally likely to be drawn. Each draw recomputes every
# method's score exactly as compare.py does:
#   - Brier skill against the base rate on the drawn months;
#   - AUC;
#   - the Sharpe ratio of the scaled trading rule, with k re-solved so that the drawn months
#     average 75% in stocks (scaled_weights in analysis/backtest.py).
# It then reports, for each method against RBP and against the logistic regression, the difference on the
# actual sample, its 90% interval over the draws, and the share of draws where it is positive.
#
# Writes data/compare/bootstrap.csv. Run from the repository root:
#   .venv/bin/python analysis/bootstrap.py

BLOCK = 12                       # months per block: 3-month outcomes overlap, losses cluster
SENSITIVITY = [6, 24]            # other block lengths, to see whether the answer depends on it
DRAWS = 5000
SEED = 15451
AGAINST = ["RBP", "Logistic regression"]  # GJR-GARCH became a predictor (Kritzman, 2026-10-19)
SCORES = ["Brier skill", "AUC", "Sharpe, scaled"]


def scaled_weights(p, mean_weight=STATIC_STOCKS, steps=60):
    """scaled_weights of analysis/backtest.py for every method at once: p is months x methods."""
    lo, hi = np.zeros(p.shape[1]), np.full(p.shape[1], 1e4)
    for _ in range(steps):  # the average weight falls as k rises: bisect, one k per method
        k = (lo + hi) / 2
        above = np.clip(1 - k * p, 0, 1).mean(axis=0) > mean_weight
        lo, hi = np.where(above, k, lo), np.where(above, hi, k)
    w = np.clip(1 - (lo + hi) / 2 * p, 0, 1)
    flat = np.ptp(p, axis=0) < 1e-12  # nothing to scale on
    w[:, flat] = mean_weight
    return w


def scores(p, y, base, stocks, bills):
    """Brier skill, AUC and scaled Sharpe of every method (columns of p) on one set of months.

    stocks and bills are NaN where the next month's return is not known yet: those months are
    scored for Brier and AUC but left out of the trading rule, as in trading_rule.
    """
    brier = ((p - y[:, None]) ** 2).sum(axis=0)
    skill = 1 - brier / ((base - y) ** 2).sum()

    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        auc = np.full(p.shape[1], np.nan)
    else:  # Mann-Whitney, with ties at average rank: the same as roc_auc_score
        ranks = rankdata(p, axis=0)
        auc = (ranks[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)

    known = ~np.isnan(stocks)
    w = scaled_weights(p[known])
    excess = w * (stocks[known] - bills[known])[:, None]
    sharpe = excess.mean(axis=0) / excess.std(axis=0, ddof=1) * np.sqrt(12)
    return np.vstack([skill, auc, sharpe])  # SCORES x methods


def draw_index(n, block, rng):
    """Row numbers of one circular block-bootstrap sample of length n."""
    starts = rng.integers(0, n, size=-(-n // block))
    return ((starts[:, None] + np.arange(block)) % n).ravel()[:n]


def bootstrap(forecasts, returns, block, draws=DRAWS, seed=SEED):
    """Every method's scores on the actual months and on each draw: arrays SCORES x methods."""
    y = forecasts["Event_A"].to_numpy()
    methods = [c for c in forecasts.columns if c != "Event_A"]
    p = forecasts[methods].to_numpy()
    base = forecasts["Empirical frequency"].to_numpy()  # the reference of compare.py's Brier skill
    stocks, bills = returns["Stocks"].to_numpy(), returns["Bills"].to_numpy()

    actual = scores(p, y, base, stocks, bills)
    rng = np.random.default_rng(seed)
    sample = np.empty((draws,) + actual.shape)
    for i in range(draws):
        j = draw_index(len(y), block, rng)
        sample[i] = scores(p[j], y[j], base[j], stocks[j], bills[j])
    return methods, actual, sample


def differences(methods, actual, sample, block):
    """Each method minus each reference in AGAINST: actual difference, 90% interval, share > 0."""
    rows = []
    for ref in AGAINST:
        r = methods.index(ref)
        for m, name in enumerate(methods):
            if name == ref:
                continue
            for s, score in enumerate(SCORES):
                d = sample[:, s, m] - sample[:, s, r]
                d = d[~np.isnan(d)]
                rows.append({
                    "Block": block, "Against": ref, "Method": name, "Score": score,
                    "Difference": actual[s, m] - actual[s, r],
                    "Low 5%": np.percentile(d, 5), "High 95%": np.percentile(d, 95),
                    "Share positive": (d > 0).mean(),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    forecasts = pd.read_csv("data/compare/forecasts_all.csv", parse_dates=["Date"]).set_index("Date")
    returns = monthly_returns().reindex(forecasts.index)

    tables = []
    for block in [BLOCK] + SENSITIVITY:
        methods, actual, sample = bootstrap(forecasts, returns, block)
        tables.append(differences(methods, actual, sample, block))
    out = pd.concat(tables, ignore_index=True)

    # The actual-sample scores must be compare.py's
    saved = pd.read_csv("data/compare/scores.csv", index_col=0)
    for s, score in enumerate(SCORES):
        assert np.allclose(actual[s], saved.loc[score, methods].astype(float), atol=1e-4), score

    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:.3f}"
    main = out[out["Block"] == BLOCK]
    print(f"{len(forecasts)} month-ends, {int(forecasts['Event_A'].sum())} loss months; "
          f"{DRAWS} draws of {BLOCK}-month circular blocks. "
          f"Difference = method minus reference; 90% interval; share of draws where it is positive.")
    for ref in AGAINST:
        for score in SCORES:
            t = main[(main["Against"] == ref) & (main["Score"] == score)]
            print(f"\n{score}, against {ref}")
            print(t.set_index("Method")[["Difference", "Low 5%", "High 95%", "Share positive"]]
                  .to_string(float_format=fmt))

    print("\nShare positive by block length (months)")
    wide = out.pivot_table(index=["Against", "Score", "Method"], columns="Block",
                           values="Share positive", sort=False)
    print(wide.to_string(float_format=fmt))

    out.to_csv("data/compare/bootstrap.csv", index=False, float_format="%.6f")
