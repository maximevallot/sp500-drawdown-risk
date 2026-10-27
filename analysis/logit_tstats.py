import os
import warnings

import pandas as pd
import statsmodels.api as sm

import rbp
from backtest import HORIZON, load_panel, predictors
from rbp_crisis import WINDOWS

# t-statistics of the logistic regression on RBP's thirteen predictors, to set beside RBP's Impact on
# Fit around the three crises (analysis/rbp_crisis.py). Written by Claude Code at Flynn's request
# (2026-10-20).
#
# The logit as in analysis/logistic.py: predictors standardized on the training sample, trained on the
# months with every predictor (from 1960). Fitted without a penalty here (statsmodels), so the
# coefficients have standard errors. Two standard errors: the usual ones, and Newey-West with
# HORIZON - 1 lags, because consecutive months' 3-month outcomes overlap, which makes the usual ones
# too small.
#
# Samples: every month whose outcome is known (through 2025-09), and, for each crisis, the months the
# back-test's logit was trained on at the window's first forecast (outcomes known by then).
#
# Writes data/rbp/logit13_tstats.csv. Run from the repository root: python analysis/logit_tstats.py


def fit(X, y):
    Z = (X - X.mean()) / X.std()
    model = sm.Logit(y, sm.add_constant(Z))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = model.fit(disp=0)
        hac = model.fit(disp=0, cov_type="HAC", cov_kwds={"maxlags": HORIZON - 1})
    return pd.DataFrame({"Coefficient": plain.params, "t": plain.tvalues, "t (Newey-West)": hac.tvalues}).drop("const")


if __name__ == "__main__":
    panel = rbp.total_return_inputs(rbp.total_return_target(load_panel()))
    panel = rbp.concentration(rbp.sentiment(rbp.cape(rbp.yield_curve_long(rbp.credit_spread(rbp.fragility(panel))))))
    data = predictors(panel, rbp.PREDICTORS).join(panel["Event_A"]).dropna()

    samples = {"Full sample": data.index[-1]}
    for name, (start, _) in WINDOWS.items():
        y = panel["Event_A"]
        samples[f"Known at {name} start"] = y.index[y.index.get_loc(pd.Timestamp(start)) - HORIZON]
    out = {}
    for label, end in samples.items():
        d = data.loc[:end]
        out[label] = fit(d[rbp.PREDICTORS], d["Event_A"])
        print(f"{label}: {d.index[0]:%Y-%m} to {d.index[-1]:%Y-%m}, {len(d)} months, {int(d['Event_A'].sum())} loss months")
    table = pd.concat(out, axis=1)
    os.makedirs("data/rbp", exist_ok=True)
    table.to_csv("data/rbp/logit13_tstats.csv", float_format="%.4f")
    pd.set_option("display.width", 250)
    print(table.xs("t (Newey-West)", axis=1, level=1).round(2).to_string())
