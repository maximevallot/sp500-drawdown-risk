import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import ml

# Which of the thirteen predictors each machine-learning model relies on, to set beside the logit's
# t-statistics (analysis/logit_tstats.py). Written by Claude Code at Maxime's request (2026-10-21), for
# the predictor pills on the regression and machine-learning slides of the 10-22 deck.
#
# Each model with the setting chosen on 1961-1990 (analysis/ml.py, "thirteen"): the net with alpha 10,
# averaged over the same five seeds; boosting with depth 1; the logit with an L2 penalty, C = 0.01.
# Fitted once on every month with a known outcome from 1930-10, then each predictor shuffled 20 times
# and the rise in log-loss recorded. In sample, like the logit's full-sample t-statistics, so it says
# what each model leans on, not what predicts out of sample; read beyond the top few as rough.
#
# Writes data/ml/ml_importance_13.csv. Run from the repository root: .venv/bin/python analysis/ml_importance.py

MODELS = {
    "Neural net": VotingClassifier(
        [(f"seed {s}", ml.filled(MLPClassifier(hidden_layer_sizes=(8,), alpha=10.0, solver="lbfgs",
                                               max_iter=2000, random_state=s))) for s in ml.SEEDS],
        voting="soft"),
    "Boosting": HistGradientBoostingClassifier(
        max_depth=1, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=0),
    "Penalized logit": ml.filled(LogisticRegression(C=0.01, max_iter=1000)),
}

if __name__ == "__main__":
    panel, X = ml.load_13()
    y = panel["Event_A"].reindex(X.index)
    X, y = X[y.notna()], y[y.notna()].astype(int)
    print(f"{X.index[0]:%Y-%m} to {X.index[-1]:%Y-%m}, {len(X)} months, {y.sum()} loss months")
    rise = {}
    for name, model in MODELS.items():
        model.fit(X, y)
        r = permutation_importance(model, X, y, scoring="neg_log_loss", n_repeats=20, random_state=0)
        rise[name] = pd.Series(r.importances_mean, index=X.columns)
    table = pd.DataFrame(rise)
    ranks = table.rank(ascending=False).astype(int).add_suffix(" rank")
    os.makedirs("data/ml", exist_ok=True)
    table.join(ranks).to_csv("data/ml/ml_importance_13.csv", float_format="%.6f")
    print(ranks.sort_values("Neural net rank").to_string())
