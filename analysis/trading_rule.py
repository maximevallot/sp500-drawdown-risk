import numpy as np
import pandas as pd

import rbp
from backtest import load_panel, predictors, run_backtest
from logistic import logistic

# The trading rule (Kritzman, 2026-10-09): move from stocks to bonds when a model warns of a 10%
# loss, and compare the result with a static mix holding the same average weight in stocks.
# Written by Claude Code at Flynn's request (2026-10-18).
#
# Rules (Flynn, 2026-10-18), for each threshold in THRESHOLDS: at each month-end t, a forecast at or
# above the threshold is a warning, and sets the stock weight for month t + 1 (the rest in bonds).
#   Plain: LOW after a warning, otherwise HIGH.
#   Scaled by fit (RBP only): a warning moves further when RBP's fit is higher. Fit in the top 25%
#   of the previous 60 forecasts' fits: FIT_TOP25 in stocks; in the top 50%: FIT_TOP50; lower
#   fit: the warning is ignored (HIGH). The fit cut-offs use only earlier forecasts (rbp.high_fit).
# Rebalanced every month-end, no transaction costs. The forecast at t uses only data known at t.
#
# Models: RBP (data/rbp/rbp11_forecasts.csv, from analysis/rbp.py) and the logistic regression
# (analysis/logistic.py), both on RBP's eleven predictors, forecasts from 1991-01.
#
# Stocks: S&P 500 total return of each month (Return_1m_TR, data/equity-index/sp500_tr_events.csv).
# Bonds: the 10-year Treasury, from FRED's daily constant-maturity yield (data/rates/DGS10_raw.csv):
# each month, a 10-year par bond bought at the previous month-end yield, repriced at this month-end
# yield, plus a month of coupon. Month-end to month-end, like the stock returns.
#
# Benchmark: a static mix rebalanced monthly at each rule's own average stock weight (Kritzman: the
# same average exposure).
#
# Run from the repository root, after analysis/rbp.py: python analysis/trading_rule.py

THRESHOLDS = [0.10, 0.15, 0.20]
HIGH, LOW = 0.80, 0.50
FIT_TOP25, FIT_TOP50 = 0.40, 0.60
FIRST_FORECAST, LAST_FORECAST = "1991-01-31", "2026-06-30"
OUT = "data/rbp/rbp11_trading_rule_1991_2026.csv"


def bond_returns():
    """Monthly total return of a 10-year Treasury par bond, from month-end yields."""
    raw = pd.read_csv("data/rates/DGS10_raw.csv", parse_dates=["observation_date"])
    y = pd.to_numeric(raw.set_index("observation_date")["DGS10"], errors="coerce").dropna() / 100
    y = y.resample("ME").last()
    coupon, new = y.shift(1), y
    # price of a 10-year bond with semiannual coupon `coupon`, at yield `new`
    n = 20
    price = coupon / new * (1 - (1 + new / 2) ** -n) + (1 + new / 2) ** -n
    return (price - 1 + coupon / 12).rename("bond")


def logistic_forecasts():
    """The logistic regression's forecasts on RBP's predictors, through the shared back-test."""
    panel = rbp.total_return_inputs(rbp.total_return_target(load_panel()))
    panel = rbp.sentiment(rbp.cape(rbp.yield_curve_long(rbp.credit_spread(rbp.fragility(panel)))))
    X = predictors(panel, rbp.PREDICTORS)
    return run_backtest(logistic, X, panel, FIRST_FORECAST, LAST_FORECAST)["Probability"]


def metrics(r, weight=None):
    """Performance of a monthly return series."""
    nav = (1 + r).cumprod()
    years = len(r) / 12
    out = {
        "Final NAV": nav.iloc[-1],
        "Annual return": nav.iloc[-1] ** (1 / years) - 1,
        "Annual volatility": r.std() * np.sqrt(12),
        "Sharpe (vs T-bills)": (r - RF.reindex(r.index)).mean() / r.std() * np.sqrt(12),
        "Max drawdown": (nav / nav.cummax() - 1).min(),
    }
    if weight is not None:
        out["Avg stock weight"] = weight.mean()
        out["Months below HIGH"] = int((weight < HIGH).sum())
        out["Switches"] = int((weight.diff().abs() > 0).sum())
    return out


def plain(p, threshold):
    return pd.Series(np.where(p >= threshold, LOW, HIGH), index=p.index)


def scaled_by_fit(p, fit, threshold):
    top25, top50 = rbp.high_fit(fit, 0.25), rbp.high_fit(fit, 0.50)
    w = np.where(p < threshold, HIGH, np.where(top25, FIT_TOP25, np.where(top50, FIT_TOP50, HIGH)))
    return pd.Series(w, index=p.index)


if __name__ == "__main__":
    events = pd.read_csv("data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]).set_index("Date")
    stocks = events["Return_1m_TR"]
    bonds = bond_returns()
    tbill = pd.read_csv("data/rates/TB3MS_raw.csv", parse_dates=["observation_date"]).set_index("observation_date")["TB3MS"]
    tbill.index = tbill.index + pd.offsets.MonthEnd(0)
    RF = tbill / 1200  # monthly risk-free return, for the Sharpe ratio

    rbp_f = pd.read_csv("data/rbp/rbp11_forecasts.csv", parse_dates=["Date"]).set_index("Date")
    forecasts = pd.DataFrame({"RBP": rbp_f["Probability"], "Fit": rbp_f["Fit"],
                              "Logistic": logistic_forecasts()}).loc[FIRST_FORECAST:LAST_FORECAST].dropna()

    rows, out = {}, {}
    for t in THRESHOLDS:
        rules = {
            "RBP, plain": plain(forecasts["RBP"], t),
            "RBP, scaled by fit": scaled_by_fit(forecasts["RBP"], forecasts["Fit"], t),
            "Logistic, plain": plain(forecasts["Logistic"], t),
        }
        for name, w in rules.items():
            held = w.shift(1).dropna()  # weights chosen at month-end t apply to month t + 1
            held = held.loc[held.index.isin(stocks.dropna().index) & held.index.isin(bonds.dropna().index)]
            s_, b_ = stocks.loc[held.index], bonds.loc[held.index]
            r = held * s_ + (1 - held) * b_
            static = held.mean() * s_ + (1 - held.mean()) * b_
            rows[(f"{t:.0%}", name, "Rule")] = metrics(r, held)
            rows[(f"{t:.0%}", name, f"Static at {held.mean():.0%}")] = metrics(static)
            out[f"{t:.0%} {name} NAV"] = (1 + r).cumprod()
            out[f"{t:.0%} {name} weight"] = held

    table = pd.DataFrame(rows).T
    pd.set_option("display.width", 250)
    print(f"Months held: {held.index[0]:%Y-%m} to {held.index[-1]:%Y-%m} ({len(held)}); NAV starts at 1. "
          f"Plain: {LOW:.0%} stocks after a warning, otherwise {HIGH:.0%}. Scaled by fit: {FIT_TOP25:.0%} "
          f"(top 25% fit), {FIT_TOP50:.0%} (top 50%), otherwise {HIGH:.0%}.")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    pd.DataFrame(out).to_csv(OUT, float_format="%.6f")
    print(f"Saved {OUT}")
