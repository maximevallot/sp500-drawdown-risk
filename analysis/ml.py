import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtest import HORIZON, base_rate, load_panel, predictors, run_backtest, score

# Machine learning stream: a neural network, gradient boosting and a regularized logit, as
# fit_predict functions for the shared back-test (analysis/backtest.py), built to compare
# with RBP like for like. Written by Claude Code at Maxime's request (2026-10-18).
#
# Same as RBP (analysis/rbp.py): the nine predictors, trained on every month from 1930-10
# with gaps left in, the total-return target, monthly forecasts from 1991. RBP's engine
# handles a missing predictor itself; here the net and the logit fill it with the training
# median and add a 0/1 "missing" column, and boosting splits on missing values natively.
#
# Against overfitting: the data are split by time, never at random.
#   - Walk-forward: at each month-end t, refit on every month up to t - 3 (the last outcome
#     known at t) and predict t. So every forecast is out of sample.
#   - Settings: each model has a small grid of settings, fixed before any run. Each is
#     walked forward over VALIDATION (1961-1990); the one with the lowest Brier score is
#     frozen and only then run on TEST (1991-2026), once. Every setting's validation
#     score is written out, not only the winner's.
#   - Small models with heavy regularization, and no class reweighting, so the
#     probabilities stay comparable with the base rate under the Brier score.
#
# Writes data/ml/ml_validation.csv (validation scores of every setting) and
# data/ml/ml_forecasts.csv (test forecasts of the chosen setting of each model).
# With the argument "extended", the same on RBP's nine plus the predictors chosen on 1961-1990
# by analysis/select_predictors.py, to data/ml/ml_validation_extended.csv and
# data/ml/ml_forecasts_extended.csv.
# With the argument "thirteen", the same on the team's thirteen predictors (analysis/rbp.py, built by
# analysis/rbp13_compare.py, total-return target) and RBP's 420 test dates, 1991-01 to 2025-12, to
# data/ml/ml_validation_13.csv and data/ml/ml_forecasts_13.csv (Maxime, 2026-10-21). It needs the
# RBP engine installed, as rbp.py imports it.
#
# Run from the repository root: .venv/bin/python analysis/ml.py [extended | thirteen]

# RBP's nine predictors (analysis/rbp.py), in the same order
PREDICTORS = [
    "logret_past_3m", "vol_rolling63",
    "ip_growth_12m", "cpi_infl_12m", "unrate_chg_12m", "yc_10y_3m", "m2_real_growth_12m",
    "turb17", "ar17_shift",
]
# RBP's nine plus what forward selection on 1961-1990 added (data/ml/selection.csv, 2026-10-18)
EXTENDED = PREDICTORS + ["log_cape", "credit_spread", "dd_12m_high"]
PREDICTOR_SETS = {"nine": PREDICTORS, "extended": EXTENDED}
HISTORY_START = "1930-10-31"  # turbulence, the last of the long-history predictors, starts
VALIDATION = ("1961-01-31", "1990-12-31")
TEST_START = "1991-01-31"     # RBP's first forecast
LAST_13 = "2025-12-31"        # RBP's last forecast on the thirteen: HHI ends there with CRSP
SEEDS = range(5)              # the net is averaged over these starting points

# Before 1962 the yield curve and M2 have no value at all: the imputer drops them, as intended
warnings.filterwarnings("ignore", message="Skipping features without any observed values")


def fragility(panel):
    """Olivia's absorption ratio shift and turbulence (data/fragility), joined by month-end.
    A copy of rbp.py's, which cannot be imported without the licensed engine."""
    ar = pd.read_csv("data/fragility/absorption_ratio.csv", parse_dates=["Date"]).set_index("Date")
    turb = pd.read_csv("data/fragility/turbulence.csv", parse_dates=["Date"]).set_index("Date")
    return panel.join(ar[["ar17_shift"]]).join(turb[["turb17"]])


def full_history(fit_predict, X_all):
    """Hand the model every training month, gaps included, as rbp_grid_full_history does.
    Run the back-test on a column of X_all that is never missing, so it drops no month."""
    def wrapped(X_train, y_train, X_now):
        t = X_now.index[0]
        assert y_train.index.max() <= t - pd.offsets.MonthEnd(HORIZON), "look-ahead"
        return fit_predict(X_all.loc[y_train.index], y_train, X_all.loc[X_now.index])
    return wrapped


def filled(model):
    """Median for a missing predictor, plus a missing-value flag, then standardized."""
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), model)


def neural_net(alpha):
    """One hidden layer of 8 units, L2 penalty alpha, averaged over SEEDS."""
    def fit_predict(X_train, y_train, X_now):
        if y_train.nunique() < 2:
            return float(y_train.mean())
        probs = []
        for seed in SEEDS:
            net = MLPClassifier(hidden_layer_sizes=(8,), alpha=alpha, solver="lbfgs",
                                max_iter=2000, random_state=seed)
            probs.append(filled(net).fit(X_train, y_train).predict_proba(X_now)[0, 1])
        return float(np.mean(probs))
    return fit_predict


def boosting(depth):
    """Shallow, slow gradient boosting; missing values go down their own branch."""
    def fit_predict(X_train, y_train, X_now):
        if y_train.nunique() < 2:
            return float(y_train.mean())
        model = HistGradientBoostingClassifier(
            max_depth=depth, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=False, random_state=0)
        return float(model.fit(X_train, y_train).predict_proba(X_now)[0, 1])
    return fit_predict


def logit(C):
    """Logistic regression with an L2 penalty (smaller C, stronger), on the same predictors."""
    def fit_predict(X_train, y_train, X_now):
        if y_train.nunique() < 2:
            return float(y_train.mean())
        model = filled(LogisticRegression(C=C, max_iter=1000))
        return float(model.fit(X_train, y_train).predict_proba(X_now)[0, 1])
    return fit_predict


def grid(logit_name):
    """The settings each model may take, fixed before any run; the logit is named after its predictor count."""
    return {
        "Neural net": {f"alpha={a}": neural_net(a) for a in (0.1, 1.0, 10.0)},
        "Boosting": {f"depth={d}": boosting(d) for d in (1, 2)},
        logit_name: {f"C={c}": logit(c) for c in (0.01, 0.1, 1.0)},
    }


GRID = grid("Logit, 9")


def load(columns=PREDICTORS):
    """The panel with the fragility signals, and the given predictors from HISTORY_START."""
    panel = fragility(load_panel())
    X_all = predictors(panel, columns).loc[HISTORY_START:]
    return panel, X_all


def load_13():
    """The panel and the team's thirteen predictors from HISTORY_START, as analysis/rbp13_compare.py
    builds them for RBP, the logit and the net."""
    import rbp
    from rbp13_compare import panel_13
    panel = panel_13()
    return panel, predictors(panel, rbp.PREDICTORS).loc[HISTORY_START:]


def walk_forward(fit_predict, X_all, panel, first, last=None):
    """Walk-forward forecasts with the full-history wrapper, gated on turbulence, which is
    never missing from HISTORY_START, so no training month is dropped whatever X_all holds."""
    gate = panel.loc[HISTORY_START:, ["turb17"]]
    return run_backtest(full_history(fit_predict, X_all), gate, panel, first, last)


if __name__ == "__main__":
    from joblib import Parallel, delayed

    set_name = sys.argv[1] if len(sys.argv) > 1 else "nine"
    if set_name == "thirteen":
        suffix, last, GRID = "_13", LAST_13, grid("Logit, 13")
        panel, X_all = load_13()
    else:
        suffix, last = ("" if set_name == "nine" else f"_{set_name}"), None
        panel, X_all = load(PREDICTOR_SETS[set_name])
    fmt = lambda x: f"{x:.3f}"
    pd.set_option("display.width", 250)

    # 1. Validation, 1961-1990: score every setting, keep the lowest Brier (settings run in parallel)
    reference = walk_forward(base_rate, X_all, panel, *VALIDATION)
    jobs = [(model, label, f) for model, settings in GRID.items() for label, f in settings.items()]
    runs = Parallel(n_jobs=len(jobs))(delayed(walk_forward)(f, X_all, panel, *VALIDATION) for _, _, f in jobs)
    rows, chosen = [], {}
    for (model, label, _), results in zip(jobs, runs):
        rows.append({"Model": model, "Setting": label, **score(results, reference)})
    validation = pd.DataFrame(rows).set_index(["Model", "Setting"])
    for model in GRID:
        chosen[model] = validation.loc[model, "Brier"].idxmin()
    print("\nValidation, 1961-1990:")
    print(validation.to_string(float_format=fmt))
    print("Chosen:", chosen)

    # 2. Test, 1991-2026 (to 2025-12 on the thirteen): run each chosen setting once
    reference = walk_forward(base_rate, X_all, panel, TEST_START, last)
    forecasts = {"Event_A": reference["Event_A"]}
    table = {"Base rate": score(reference, reference)}
    tested = Parallel(n_jobs=len(chosen))(
        delayed(walk_forward)(GRID[model][label], X_all, panel, TEST_START, last) for model, label in chosen.items())
    for (model, label), results in zip(chosen.items(), tested):
        forecasts[model] = results["Probability"]
        table[f"{model} ({label})"] = score(results, reference)
    print(f"\nTest, {reference.index[0]:%Y-%m} to {reference.index[-1]:%Y-%m}, {len(reference)} forecasts:")
    print(pd.DataFrame(table).T.to_string(float_format=fmt))

    os.makedirs("data/ml", exist_ok=True)
    validation.to_csv(f"data/ml/ml_validation{suffix}.csv", float_format="%.6f")
    pd.DataFrame(forecasts).to_csv(f"data/ml/ml_forecasts{suffix}.csv", float_format="%.6f")
