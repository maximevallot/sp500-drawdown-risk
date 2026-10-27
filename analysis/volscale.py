import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from backtest import LOSS, base_rate, monthly_returns, score
from ml import HISTORY_START, LAST_13, TEST_START, VALIDATION, filled, load_13, walk_forward

# Volatility-scaled model on the thirteen predictors: instead of classifying the rare 10%-loss
# months directly, forecast how volatile the next three months will be, then read the probability
# of a 10% loss off the distribution of past returns scaled by their forecast volatility.
# Written by Claude Code at Maxime's request (2026-10-22), after a search over about thirty
# models on the same dates.
#
# Why: a logit or a net learns from about 70 loss months since 1930, a third of them in the 1930s.
# Every month, though, tells us how volatile the following three months were. A regression of
# that realized volatility on the predictors learns from all 1,100 months, and volatility is
# what carries the signal.
#
# At each month-end t, on the training months s <= t - 3, as in analysis/ml.py:
#   1. Volatility: ridge regression of the log of the realized 63-day volatility at s + 3
#      (vol_trailing63_tr three rows later, measured by t) on the thirteen predictors.
#   2. Standardized returns: z_s = (r_s - m) / sigma_s, where r_s is the 3-month log total return,
#      m its expanding mean, and sigma_s the fitted volatility over three months (annual / 2).
#   3. Probability: the share of training z_s at or below (log 0.9 - m) / sigma_t.
# The predictors are those of analysis/ml.py's net, transformed so that a regression on them is
# stable: skewed ones in logs, and the ones whose level drifts over decades (CAPE, HHI, the credit
# spread, sentiment, the yield curve, inflation) as z-scores against their own past ten years.
# The ridge penalty is chosen on 1961-1990 by Brier score, before the test run, as in ml.py.
#
# Writes data/ml/volscale_validation_13.csv and data/ml/volscale_forecasts_13.csv.
# Run from the repository root: .venv/bin/python analysis/volscale.py

ALPHAS = (1.0, 10.0, 100.0, 1000.0)  # fixed before any run
LOGGED = ["vol_trailing63_tr", "gjr_skewt_vol_q", "turb17", "credit_baa_aaa", "cape"]
DRIFTING = ["cape", "hhi", "credit_baa_aaa", "sentiment_umich", "yc_long", "cpi_infl_12m"]
Z_WINDOW, Z_MIN = 120, 36  # months: ten years, at least three


def features(X):
    """The thirteen, logged where skewed, and drifting levels against their trailing ten years."""
    out = X.copy()
    out[LOGGED] = np.log(out[LOGGED])
    for c in DRIFTING:
        s = out[c]
        roll = s.rolling(Z_WINDOW, min_periods=Z_MIN)
        out[c] = (s - roll.mean()) / roll.std()
    return out


def volscale(alpha, panel):
    """fit_predict for analysis/ml.py's walk-forward: the probability of a 10% loss for ridge penalty alpha."""
    r = np.log1p(panel["Return_3m"])
    future_vol = panel["vol_trailing63_tr"].shift(-3)  # realized over the three months after s, known at s + 3

    def fit_predict(X_train, y_train, X_now):
        r_train, v_train = r.loc[X_train.index], future_vol.loc[X_train.index]
        known = v_train.notna()
        model = filled(Ridge(alpha=alpha)).fit(X_train[known], np.log(v_train[known]))
        sigma_train = np.exp(model.predict(X_train)) / 2  # annual to 3-month: sqrt(1/4)
        sigma_now = np.exp(model.predict(X_now)[0]) / 2
        m = r_train.mean()
        z = (r_train - m) / sigma_train
        return float((z <= (LOSS - m) / sigma_now).mean())
    return fit_predict


if __name__ == "__main__":
    from joblib import Parallel, delayed

    panel, X13 = load_13()
    X_all = features(X13).loc[HISTORY_START:]
    fmt = lambda x: f"{x:.3f}"
    pd.set_option("display.width", 250)

    # 1. Validation, 1961-1990: every penalty, keep the lowest Brier
    reference = walk_forward(base_rate, X_all, panel, *VALIDATION)
    runs = Parallel(n_jobs=len(ALPHAS))(delayed(walk_forward)(volscale(a, panel), X_all, panel, *VALIDATION) for a in ALPHAS)
    validation = pd.DataFrame({f"alpha={a:g}": score(res, reference) for a, res in zip(ALPHAS, runs)}).T
    chosen = ALPHAS[int(np.argmin(validation["Brier"].to_numpy()))]
    print("Validation, 1961-1990:")
    print(validation.to_string(float_format=fmt))
    print("Chosen: alpha =", chosen)

    # 2. Test, 1991-01 to 2025-12, RBP's 420 dates: the chosen penalty, once
    reference = walk_forward(base_rate, X_all, panel, TEST_START, LAST_13)
    test = walk_forward(volscale(chosen, panel), X_all, panel, TEST_START, LAST_13)
    print(f"\nTest, {test.index[0]:%Y-%m} to {test.index[-1]:%Y-%m}, {len(test)} forecasts:")
    print(pd.DataFrame({"Base rate": score(reference, reference), f"Vol-scaled (alpha={chosen:g})": score(test, reference)})
          .T.to_string(float_format=fmt))

    os.makedirs("data/ml", exist_ok=True)
    validation.to_csv("data/ml/volscale_validation_13.csv", float_format="%.6f")
    test.rename(columns={"Probability": "Vol-scaled"}).to_csv("data/ml/volscale_forecasts_13.csv", float_format="%.6f")

    # 3. Against every method on the thirteen, scored as analysis/compare.py scores them
    from compare import COMMON, runs as compare_runs, saved, table
    common = saved(COMMON)
    others = compare_runs(common)
    others["Vol-scaled"] = pd.DataFrame({"Probability": test["Probability"].reindex(common.index), "Event_A": common["Event_A"]})
    assert others["Vol-scaled"].notna().all(axis=None)
    names = ["Empirical frequency", "Logistic regression", "Neural net", "Boosting", "Logit, 13", "Timestral, daily", "RBP", "Vol-scaled"]
    scores = table({n: others[n] for n in names}, others["Empirical frequency"], monthly_returns(), common)
    print(f"\nEvery method on the same {len(common)} dates (analysis/compare.py's rows):")
    print(scores.drop(columns=["Static 75/25", "100% stocks"]).to_string(float_format=fmt))
