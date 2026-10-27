import os

import pandas as pd

from backtest import base_rate, score
from logistic import logistic
from ml import PREDICTORS, VALIDATION, boosting, load, logit, walk_forward

# Which of the new predictors to add, chosen on 1961-1990 only, never on the test period.
# Written by Claude Code at Maxime's request (2026-10-18).
#
# Candidates: the credit spread and its 3-month change, log CAPE, the 10-month trend and the
# drawdown from the 12-month high (analysis/build_panel.py). Greedy forward selection: start
# from a base set, try adding each remaining candidate, keep the one that lowers the
# validation Brier score most, and stop when no candidate lowers it.
#
# Two base sets, each scored with the models that use it:
#   - RBP's nine predictors, for RBP and ML: the mean Brier score of the regularized logit and
#     boosting, the two ML models fast enough to refit 360 times per try, at the settings
#     validation chose for them in analysis/ml.py (C=0.01, depth=1);
#   - the two of Pir's logit (past 3-month return and 63-day volatility): Pir's logistic.
#
# Writes data/ml/selection.csv. Run from the repository root:
#   .venv/bin/python analysis/select_predictors.py

CANDIDATES = ["credit_spread", "credit_spread_chg_3m", "log_cape", "trend_10m", "dd_12m_high"]
BASES = {"nine": PREDICTORS, "two": ["logret_past_3m", "vol_rolling63"]}
MODELS = {
    "nine": {"Logit": logit(0.01), "Boosting": boosting(1)},
    "two": {"Logit": logistic},
}


def validation_brier(columns, models, panel, X_all, reference):
    """Mean validation Brier score over the models, with each model's own score."""
    briers = {}
    for name, fit_predict in models.items():
        results = walk_forward(fit_predict, X_all[columns], panel, *VALIDATION)
        briers[name] = score(results, reference)["Brier"]
    return sum(briers.values()) / len(briers), briers


def forward(base, models, panel, X_all, reference):
    chosen, steps = list(base), []
    best, parts = validation_brier(chosen, models, panel, X_all, reference)
    steps.append({"Step": 0, "Added": "(base)", "Brier": best, **parts})
    remaining = list(CANDIDATES)
    while remaining:
        tries = {c: validation_brier(chosen + [c], models, panel, X_all, reference) for c in remaining}
        for c, (b, p) in tries.items():
            steps.append({"Step": len(chosen) - len(base) + 1, "Added": c, "Brier": b, **p, "Tried": True})
        c, (b, p) = min(tries.items(), key=lambda kv: kv[1][0])
        if b >= best:
            break
        chosen.append(c)
        remaining.remove(c)
        best = b
        print(f"  keep {c}: validation Brier {b:.5f}", flush=True)
    return chosen, pd.DataFrame(steps)


if __name__ == "__main__":
    panel, X_all = load(PREDICTORS + CANDIDATES)
    reference = walk_forward(base_rate, X_all, panel, *VALIDATION)
    tables, sets = [], {}
    for name, base in BASES.items():
        print(f"Base: {name}", flush=True)
        chosen, steps = forward(base, MODELS[name], panel, X_all, reference)
        sets[name] = chosen[len(base):]
        tables.append(steps.assign(Base=name))
        print(f"  added to {name}: {sets[name] or 'nothing'}")
    os.makedirs("data/ml", exist_ok=True)
    pd.concat(tables).to_csv("data/ml/selection.csv", index=False, float_format="%.6f")
