import os

import pandas as pd

from backtest import run_backtest
from ml import EXTENDED, HISTORY_START, TEST_START, load
from rbp import RECORDS, rbp_grid_full_history

# RBP with the CSA engine on the extended predictor set: RBP's nine plus log CAPE, the credit
# spread and the drawdown from the 12-month high, which forward selection on 1961-1990 chose
# (analysis/select_predictors.py). Everything else as in analysis/rbp.py: Grid Prediction on
# default settings, trained on every month from 1930-10 with gaps left to the engine.
# Written by Claude Code at Maxime's request (2026-10-18).
#
# Needs a licensed machine (README.md). Writes data/rbp/rbp_forecasts_extended.csv,
# which analysis/compare.py picks up when it exists.
#
# Run from the repository root: .venv/bin/python analysis/rbp_extended.py

if __name__ == "__main__":
    panel, X_all = load(EXTENDED)
    gate = panel.loc[HISTORY_START:, ["turb17"]]  # never missing, so no training month is dropped
    RECORDS.clear()
    results = run_backtest(rbp_grid_full_history(X_all), gate, panel, TEST_START)
    info = pd.DataFrame(RECORDS).set_index("Date")
    results = results.join(info[["Fit", "Adjusted fit", "Raw"]])
    os.makedirs("data/rbp", exist_ok=True)
    results.to_csv("data/rbp/rbp_forecasts_extended.csv")
    print(f"{len(results)} forecasts, {results.index.min():%Y-%m} to {results.index.max():%Y-%m}")
