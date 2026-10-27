import os

import numpy as np
import pandas as pd
import rbp_engine

import rbp
from backtest import HORIZON, load_panel, predictors

# What drove RBP's forecasts around the dot-com bust, the global financial crisis and COVID
# (Kritzman, 2026-10-20). Written by Claude Code at Flynn's request (2026-10-20).
#
# Predictors: the mean Impact on Fit and Impact on Prediction of each of the thirteen predictors over
# each crisis window's forecasts and over every other forecast, from data/rbp/rbp13_importance.csv.
#
# Observations: Grid Prediction's forecast is a weighted sum of past outcomes, yhat = sum_i w_i y_i,
# with weights summing to one (result.prediction_weights). So yhat = ybar + sum_i w_i (y_i - ybar),
# and w_i (y_i - ybar) is how far past month i pushed the forecast above or below the base rate of the
# training months. Each forecast in a window is recomputed as in analysis/rbp.py (same inputs, median
# of REPEATS calls) to get its weights; contributions are averaged over the window's forecasts.
#
# Writes data/rbp/rbp13_crisis_predictors.csv and data/rbp/rbp13_crisis_observations.csv.
# Run from the repository root, after analysis/rbp.py: python analysis/rbp_crisis.py

WINDOWS = {  # forecast month-ends, from just before each fall to its bottom
    "Dot-com": ("2000-01-31", "2002-10-31"),
    "GFC": ("2007-07-31", "2009-03-31"),
    "COVID": ("2019-11-30", "2020-06-30"),
    "Post-COVID": ("2020-07-31", "2025-12-31"),  # a regime rather than a fall: inflation, rate hikes, 2022 (Flynn)
}


def window_of(dates):
    label = pd.Series("Other times", index=dates)
    for name, (a, b) in WINDOWS.items():
        label[(dates >= a) & (dates <= b)] = name
    return label


def predictor_table():
    imp = pd.read_csv("data/rbp/rbp13_importance.csv", parse_dates=["Date"]).set_index(["Date", "Measure"])
    out = {}
    for measure in ("Impact on fit", "Impact on prediction"):
        m = imp.xs(measure, level="Measure")
        out[measure] = m.groupby(window_of(m.index)).mean().T.assign(**{"Full sample": m.mean()})
    return pd.concat(out, axis=1)


def weights(X_all, y, t):
    """Past months' weights in the forecast at t, from the median of REPEATS engine calls."""
    last_known = y.index[y.index.get_loc(t) - HORIZON]
    y_train = y.loc[:last_known]
    X_train = X_all.loc[y_train.index]
    outs = [rbp_engine.predict_grid(y_train.to_numpy(float), X_train.to_numpy(float), X_all.loc[[t]].to_numpy(float))
            for _ in range(rbp.REPEATS)]
    yhats = [float(o.yhat[0]) for o in outs]
    out = outs[int(np.argsort(yhats)[len(yhats) // 2])]
    w = pd.Series(np.asarray(out.prediction_weights.weights)[:, 0], index=y_train.index)
    return w, y_train, float(out.yhat[0])


if __name__ == "__main__":
    panel = rbp.total_return_inputs(rbp.total_return_target(load_panel()))
    panel = rbp.concentration(rbp.sentiment(rbp.cape(rbp.yield_curve_long(rbp.credit_spread(rbp.fragility(panel))))))
    X_all = predictors(panel, rbp.PREDICTORS).loc[rbp.HISTORY_START:]
    y = panel["Event_A"].loc[rbp.HISTORY_START:]
    saved = pd.read_csv("data/rbp/rbp13_forecasts.csv", parse_dates=["Date"]).set_index("Date")

    table = predictor_table()
    os.makedirs("data/rbp", exist_ok=True)
    table.to_csv("data/rbp/rbp13_crisis_predictors.csv", float_format="%.6f")

    rows = []
    for name, (a, b) in WINDOWS.items():
        for t in saved.loc[a:b].index:
            w, y_train, yhat = weights(X_all, y, t)
            contribution = w * (y_train - y_train.mean())
            rows.append(pd.DataFrame({"Window": name, "Forecast date": t, "Past month": w.index, "Weight": w.values,
                                      "Outcome": y_train.values, "Contribution": contribution.values,
                                      "Forecast": yhat, "Saved forecast": saved.loc[t, "Raw"]}))
    obs = pd.concat(rows, ignore_index=True)
    obs.to_csv("data/rbp/rbp13_crisis_observations.csv", index=False, float_format="%.6g")
    gap = (obs.groupby("Forecast date")[["Forecast", "Saved forecast"]].first().diff(axis=1).iloc[:, 1].abs())
    print(f"Recomputed against saved forecasts: largest gap {gap.max():.4f}")
