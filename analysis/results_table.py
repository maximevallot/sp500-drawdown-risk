import numpy as np
import pandas as pd
from scipy.stats import norm

import rbp
import trading_rule as tr

# The results table as Kritzman asked on 2026-10-19:
# columns, left to right, the empirical frequency, the normal (theoretical) frequency, the logistic
# regression, the machine-learning models (neural net, gradient boosting, penalized logit), and RBP (all
# forecasts, top 50% and top 25% by fit) furthest right;
# plain labels; "true positive" for the share of warnings followed by a 10% loss; the fraction of 10%
# losses preceded by a warning over the whole test sample; and the trading rule's return and maximum
# drawdown as two rows. Written by Claude Code at Flynn's request (2026-10-20).
#
# Every column is scored on the same 420 month-ends, 1991-01 to 2025-12 (data/rbp/rbp13_compare_forecasts.csv).
# Gradient boosting and the penalized logit are analysis/ml.py's on the same thirteen predictors and dates
# (data/ml/ml_forecasts_13.csv, `analysis/ml.py thirteen`), added at Maxime's request (2026-10-21).
# A warning is a forecast probability of a 10% loss at or above the threshold. RBP's top-50% and top-25%
# columns warn only when the forecast's fit is in that top share of the previous 60 forecasts' fits
# (rbp.high_fit); they are scored on all 420 months, like the other columns, so the fraction of losses
# preceded by a warning has the same 21 losses below it in every column.
#   Empirical: the share of past three-month windows, known at the forecast date, that lost 10%.
#   Normal: the probability of a 10% loss under a normal distribution with the arithmetic mean and
#   standard deviation of the rolling three-month total returns known at the forecast date, converted to
#   log terms (sigma^2 = ln(1 + s^2 / (1 + m)^2), mu = ln(1 + m) - sigma^2 / 2), as analysis/baseline.py
#   does on the whole sample (9.0% over 1928-2026).
# Trading rule (analysis/trading_rule.py): 80% stocks, 50% in the month after a warning, the rest in
# 10-year Treasuries, rebalanced monthly, no costs; annual return and maximum drawdown, February 1991 to
# December 2025, beside a static mix at the rule's own average stock weight.
#
# Writes data/rbp/results_table.csv. Run from the repository root: python analysis/results_table.py

THRESHOLDS = [0.10, 0.15, 0.20]
ROWS = ["Warnings", "Average 3-month return after a warning", "Average 3-month return otherwise",
        "True positives: share of warnings followed by a 10% loss", "False positives: warnings not followed by a loss",
        "Share of 10% losses preceded by a warning", "Trading rule: annual return", "Trading rule: maximum drawdown",
        "Static mix, same average stocks: annual return", "Static mix, same average stocks: maximum drawdown"]


def empirical(events, dates):
    """Share of three-month windows that lost 10%, among those whose outcome is known at each date."""
    y = events["Event_A_TR"].dropna()
    return pd.Series([y.iloc[: y.index.get_loc(t) - 2].mean() for t in dates], index=dates)


def normal(events, dates):
    """Normal probability of a 10% loss from the three-month total returns known at each date."""
    r = events["Return_3m_TR"].dropna()
    out = []
    for t in dates:
        x = r.iloc[: r.index.get_loc(t) - 2]  # windows that have ended by t
        m, s = x.mean(), x.std()
        sig2 = np.log(1 + s ** 2 / (1 + m) ** 2)
        mu = np.log(1 + m) - sig2 / 2
        out.append(norm.cdf((np.log(0.9) - mu) / np.sqrt(sig2)))
    return pd.Series(out, index=dates)


def column(warn, f, stocks, bonds):
    """The table's rows for one column, given its warnings (a boolean Series on the forecast dates)."""
    y, ret = f["Event_A"], f["Return_3m"]
    w = pd.Series(np.where(warn, tr.LOW, tr.HIGH), index=f.index).shift(1).dropna()
    s_, b_ = stocks.reindex(w.index), bonds.reindex(w.index)
    rule, static = w * s_ + (1 - w) * b_, w.mean() * s_ + (1 - w.mean()) * b_
    mr, ms = tr.metrics(rule), tr.metrics(static)
    n = int(warn.sum())
    return [n, ret[warn].mean() if n else np.nan, ret[~warn].mean(), y[warn].mean() if n else np.nan,
            int(n - y[warn].sum()), y[warn].sum() / y.sum(), mr["Annual return"], mr["Max drawdown"],
            ms["Annual return"], ms["Max drawdown"]]


if __name__ == "__main__":
    events = pd.read_csv("data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]).set_index("Date")
    f = pd.read_csv("data/rbp/rbp13_compare_forecasts.csv", parse_dates=["Date"]).set_index("Date")
    ml = pd.read_csv("data/ml/ml_forecasts_13.csv", parse_dates=["Date"]).set_index("Date").reindex(f.index)
    assert ml[["Boosting", "Logit, 13"]].notna().all(axis=None), "the ML forecasts must cover the 420 dates"
    tbill = pd.read_csv("data/rates/TB3MS_raw.csv", parse_dates=["observation_date"]).set_index("observation_date")["TB3MS"]
    tbill.index = tbill.index + pd.offsets.MonthEnd(0)
    tr.RF = tbill / 1200
    stocks, bonds = events["Return_1m_TR"], tr.bond_returns()

    probs = {"Empirical frequency": empirical(events, f.index), "Normal distribution": normal(events, f.index),
             "Logistic regression": f["Logistic regression"], "Neural net": f["Neural net"],
             "Gradient boosting": ml["Boosting"], "Penalized logit": ml["Logit, 13"], "RBP, all": f["RBP"]}
    top50, top25 = rbp.high_fit(f["Fit"], 0.50), rbp.high_fit(f["Fit"], 0.25)
    print("Empirical", probs["Empirical frequency"].agg(["min", "max"]).round(4).to_dict(),
          "Normal", probs["Normal distribution"].agg(["min", "max"]).round(4).to_dict())
    table = {}
    for t in THRESHOLDS:
        warns = {name: p >= t for name, p in probs.items()}
        warns["RBP, top 50% by fit"] = (f["RBP"] >= t) & top50
        warns["RBP, top 25% by fit"] = (f["RBP"] >= t) & top25
        for name, w in warns.items():
            table[(f"{t:.0%}", name)] = column(w, f, stocks, bonds)
    out = pd.DataFrame(table, index=ROWS).T
    out.index.names = ["Threshold", "Method"]
    out.to_csv("data/rbp/results_table.csv", float_format="%.6f")
    pd.set_option("display.width", 250)
    for t in THRESHOLDS:
        print(f"\n{t:.0%}")
        print(out.xs(f"{t:.0%}").T.to_string(float_format=lambda v: f"{v:.3f}"))
