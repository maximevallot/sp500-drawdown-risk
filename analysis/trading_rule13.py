import pandas as pd

import trading_rule as tr

# The trading rule of analysis/trading_rule.py on the thirteen-predictor forecasts: RBP (plain and scaled
# by fit), the logistic regression and the neural net, all from data/rbp/rbp13_compare_forecasts.csv
# (analysis/rbp13_compare.py), forecasts 1991-01 to 2025-12. Same rules, bonds and benchmark: 80% stocks,
# 50% after a warning; each rule against a static mix at its own average stock weight.
# Written by Claude Code at Flynn's request (2026-10-20).
#
# Writes data/rbp/rbp13_trading_rule.csv (summary) and data/rbp/rbp13_trading_rule_nav.csv (NAV, weight).
# Run from the repository root, after analysis/rbp13_compare.py: python analysis/trading_rule13.py

if __name__ == "__main__":
    events = pd.read_csv("data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]).set_index("Date")
    stocks, bonds = events["Return_1m_TR"], tr.bond_returns()
    tbill = pd.read_csv("data/rates/TB3MS_raw.csv", parse_dates=["observation_date"]).set_index("observation_date")["TB3MS"]
    tbill.index = tbill.index + pd.offsets.MonthEnd(0)
    tr.RF = tbill / 1200  # trading_rule.metrics reads RF from its own module

    f = pd.read_csv("data/rbp/rbp13_compare_forecasts.csv", parse_dates=["Date"]).set_index("Date")
    rows, out = {}, {}
    for t in tr.THRESHOLDS:
        rules = {
            "RBP": tr.plain(f["RBP"], t),
            "RBP, scaled by fit": tr.scaled_by_fit(f["RBP"], f["Fit"], t),
            "Logistic regression": tr.plain(f["Logistic regression"], t),
            "Neural net": tr.plain(f["Neural net"], t),
        }
        for name, w in rules.items():
            held = w.shift(1).dropna()  # weights chosen at month-end t apply to month t + 1
            held = held.loc[held.index.isin(stocks.dropna().index) & held.index.isin(bonds.dropna().index)]
            s_, b_ = stocks.loc[held.index], bonds.loc[held.index]
            r = held * s_ + (1 - held) * b_
            static = held.mean() * s_ + (1 - held.mean()) * b_
            rows[(f"{t:.0%}", name, "Rule")] = tr.metrics(r, held)
            rows[(f"{t:.0%}", name, "Static, same average")] = tr.metrics(static)
            out[f"{t:.0%} {name} NAV"] = (1 + r).cumprod()
            out[f"{t:.0%} {name} static NAV"] = (1 + static).cumprod()
            out[f"{t:.0%} {name} weight"] = held

    table = pd.DataFrame(rows).T
    table.index.names = ["Threshold", "Model", "Portfolio"]
    table.to_csv("data/rbp/rbp13_trading_rule.csv", float_format="%.6f")
    pd.DataFrame(out).to_csv("data/rbp/rbp13_trading_rule_nav.csv", float_format="%.6f")
    pd.set_option("display.width", 250)
    print(f"Months held {held.index[0]:%Y-%m} to {held.index[-1]:%Y-%m} ({len(held)})")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
