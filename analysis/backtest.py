import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

# Shared back-test: every model plugs into run_backtest and is scored the same way by score.
# Written by Claude Code at Maxime's request (2026-10-13), walked through with Maxime.
#
# Timing rule: the outcome of the forecast made at month-end s is only known at s + 3 months.
# So at month-end t, a model trains on rows dated t - 3 months or earlier, and predicts
# with the predictors as of t.
#
# Every model reads the same monthly panel, data/panel/monthly.csv (analysis/build_panel.py).
# Run from the repository root: python3 analysis/backtest.py

HORIZON = 3          # months ahead
T = HORIZON / 12     # horizon in years
LOSS = np.log(0.9)   # a 10% loss, as a log return
THRESHOLD = 0.10     # probability above which a forecast counts as a warning (placeholder)


# ---------------------------------------------------------------------------
# 1. Data: the one monthly panel, built by analysis/build_panel.py
# ---------------------------------------------------------------------------

TARGET_COLUMNS = ["Return_3m", "Event_A"]  # the future: never a predictor


def load_panel():
    """The monthly panel: the target and every predictor, indexed by month-end."""
    return pd.read_csv("data/panel/monthly.csv", parse_dates=["Date"]).set_index("Date")


def predictors(panel, columns):
    """The predictor columns a model uses, refusing the target."""
    assert not set(columns) & set(TARGET_COLUMNS), "the target cannot be a predictor"
    return panel[columns]


def normal_probability(mu, sigma):
    """P(log return over T < log 0.9) for annualized mean mu and volatility sigma of log returns."""
    return norm.cdf((LOSS - mu * T) / (sigma * np.sqrt(T)))


# ---------------------------------------------------------------------------
# 3. The back-test loop
# ---------------------------------------------------------------------------

def run_backtest(fit_predict, X, panel, first_forecast, last_forecast=None):
    """Rolling monthly forecasts with an expanding training window, refitted every month.

    fit_predict(X_train, y_train, X_now) -> probability of a 10% loss over the next 3 months
    X: monthly predictors indexed by month-end, known at that date (None for no predictors)
    panel: the monthly panel (its Event_A column is the target)
    first_forecast, last_forecast: the forecast window, as month-ends

    Returns a DataFrame indexed by forecast date with columns Probability and Event_A.
    """
    y = panel["Event_A"]
    dates = y.loc[first_forecast:last_forecast].index
    rows = []

    for t in dates:
        # The last month-end whose outcome is known at t: three rows earlier
        last_known = y.index[y.index.get_loc(t) - HORIZON]
        y_train = y.loc[:last_known]

        if X is None:
            X_train, X_now = None, None
        else:
            X_train = X.loc[:last_known].dropna()
            y_train = y_train.loc[X_train.index]
            X_now = X.loc[[t]]
            if X_now.isna().any(axis=None):
                continue  # predictors not available at t: no forecast

        p = fit_predict(X_train, y_train, X_now)
        rows.append({"Date": t, "Probability": p, "Event_A": y.loc[t]})

    return pd.DataFrame(rows).set_index("Date")


# ---------------------------------------------------------------------------
# 4. Models: each is a fit_predict function
# ---------------------------------------------------------------------------

def normal_10_20(X_train, y_train, X_now):
    """Normal model with Kritzman's 10% mean and 20% volatility (arithmetic, so log mean 8%)."""
    return float(normal_probability(0.10 - 0.5 * 0.20**2, 0.20))


def always_no(X_train, y_train, X_now):
    """Never warns: probability 0."""
    return 0.0


def base_rate(X_train, y_train, X_now):
    """Expanding base rate: how often the event happened in the training months."""
    return y_train.mean()


def signal(column):
    """A model that takes a ready-made probability from the panel, with nothing to fit."""
    def fit_predict(X_train, y_train, X_now):
        return float(X_now[column].iloc[0])
    return fit_predict


# ---------------------------------------------------------------------------
# 5. Scoring
# ---------------------------------------------------------------------------

def count_episodes(event_dates):
    """Events closer than 3 months to the previous one belong to the same episode."""
    episodes, last = 0, None
    for d in event_dates:
        if last is None or (d - last).days > 100:
            episodes += 1
        last = d
    return episodes


def score(results, reference=None):
    """Scores of one model's forecasts. reference: the base rate's results on the same dates."""
    p, y = results["Probability"], results["Event_A"]
    brier = ((p - y) ** 2).mean()
    out = {
        "Forecasts": len(y),
        "Events": int(y.sum()),
        "Episodes": count_episodes(y[y == 1].index),
        "Mean prob": p.mean(),
        "Brier": brier,
        "AUC": roc_auc_score(y, p) if p.nunique() > 1 else 0.5,
        "Hit rate": (p[y == 1] >= THRESHOLD).mean(),
        "False alarms": (p[y == 0] >= THRESHOLD).mean(),
    }
    # Onsets: the first loss month of an episode, with no loss month in the three months before.
    # Later loss months in the same episode are easier: the crash is already under way.
    earlier = sum(y.shift(k, fill_value=0) for k in range(1, HORIZON + 1))
    onset = (y == 1) & (earlier == 0)
    out["Onset prob"] = p[onset].mean()
    out["Onsets caught"] = (p[onset] >= THRESHOLD).mean()
    if reference is not None:
        ref = reference.loc[results.index, "Probability"]
        out["Brier skill"] = 1 - brier / ((ref - y) ** 2).mean()
    # Robustness: the three quarterly series that do not overlap, by month of the year mod 3
    for k in range(3):
        sub = results[results.index.month % 3 == k]
        out[f"Brier q{k}"] = ((sub["Probability"] - sub["Event_A"]) ** 2).mean()
    return out


# ---------------------------------------------------------------------------
# 6. Trading rule: the economic test (Kritzman, 2026-10-09)
# ---------------------------------------------------------------------------
# The weight in stocks is set at each forecast date t from that date's probability and
# earns the S&P 500 total return of the following month; the rest earns the 3-month bill.
# Added by Claude Code at Maxime's request (2026-10-18).

STATIC_STOCKS = 0.75  # the static mix every dynamic rule is held to, on average


def monthly_returns():
    """Stocks and bills for the month after each month-end, as simple returns.

    Stocks: Michal's monthly total return (Return_1m_TR dated m is the return of month m),
    moved back one row. Bills: the 3-month T-bill rate on the last day of the month
    (discount basis, as published, close enough for a monthly cash return), over 12.
    """
    tr = pd.read_csv("data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]).set_index("Date")
    bill = pd.read_csv("data/rates/DTB3_raw.csv", parse_dates=["observation_date"]).set_index("observation_date")["DTB3"]
    bill = bill.dropna().groupby(bill.dropna().index + pd.offsets.MonthEnd(0)).last()
    return pd.DataFrame({
        "Stocks": tr["Return_1m_TR"].shift(-1),
        "Bills": (bill / 100 / 12).reindex(tr.index),
    })


def scaled_weights(p, mean_weight=STATIC_STOCKS):
    """w = clip(1 - k p, 0, 1), with k chosen so the average weight is mean_weight.

    k uses the whole sample's probabilities: that is what holding the rule to the same
    average exposure means, and it is the one place the rule looks ahead.
    """
    if p.max() - p.min() < 1e-12:
        return pd.Series(mean_weight, index=p.index)  # nothing to scale on
    lo, hi = 0.0, 1e4
    for _ in range(200):  # the average weight falls as k rises: bisect
        k = (lo + hi) / 2
        lo, hi = (k, hi) if (1 - k * p).clip(0, 1).mean() > mean_weight else (lo, k)
    return (1 - k * p).clip(0, 1)


def portfolio_stats(weights, returns):
    """Annual return, volatility, Sharpe ratio (over bills) and maximum drawdown."""
    r = weights * returns["Stocks"] + (1 - weights) * returns["Bills"]
    excess = r - returns["Bills"]
    wealth = (1 + r).cumprod()
    return {
        "Mean stocks": weights.mean(),
        "Return": wealth.iloc[-1] ** (12 / len(r)) - 1,
        "Volatility": r.std() * np.sqrt(12),
        "Sharpe": excess.mean() / excess.std() * np.sqrt(12),
        "Max drawdown": (wealth / wealth.cummax() - 1).min(),
    }


def trading_rule(results, returns=None):
    """One model's forecasts through the scaled rule, the simpler switch and the static mix.

    Scaled: stocks scale down as the probability rises, 75% on average. Switch: 100% stocks,
    50/50 once the probability reaches THRESHOLD. Static: 75/25 every month.
    """
    returns = monthly_returns() if returns is None else returns
    returns = returns.reindex(results.index).dropna()
    p = results.loc[returns.index, "Probability"]
    switch = pd.Series(np.where(p >= THRESHOLD, 0.5, 1.0), index=p.index)
    static = pd.Series(STATIC_STOCKS, index=p.index)
    return {
        "Scaled": portfolio_stats(scaled_weights(p), returns),
        "Switch": portfolio_stats(switch, returns),
        "Static 75/25": portfolio_stats(static, returns),
    }


def compare(models, panel, first_forecast, last_forecast=None):
    """Run every model on the same dates and return one table. models: name -> (fit_predict, X)."""
    runs = {name: run_backtest(f, X, panel, first_forecast, last_forecast) for name, (f, X) in models.items()}
    common = sorted(set.intersection(*(set(r.index) for r in runs.values())))
    runs = {name: r.loc[common] for name, r in runs.items()}
    table = pd.DataFrame({name: score(r, runs["Base rate"]) for name, r in runs.items()}).T
    return table, runs


if __name__ == "__main__":
    panel = load_panel()
    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:.3f}"

    # Models with nothing to fit, so no training window is needed: each is scored
    # from the first date on which every model in the table has a forecast.
    models = {
        "Always no": (always_no, None),
        "Base rate": (base_rate, None),
        "Normal 10/20": (normal_10_20, None),
    }
    for name, column in [("Normal hist", "p_normal_hist"),
                         ("Vol rolling", "p_rolling63"), ("Vol EWMA", "p_ewma94")]:
        models[name] = (signal(column), predictors(panel, [column]))

    print("Long window: price-only models")
    table, _ = compare(models, panel, "1935-01-31")
    print(table.to_string(float_format=fmt))

    for name, column in [("VIX raw", "p_vix_raw"), ("VIX adjusted", "p_vix_adjusted")]:
        models[name] = (signal(column), predictors(panel, [column]))

    print("\nVIX window: every model, from the first month the adjusted VIX exists")
    table, _ = compare(models, panel, "1991-01-31")
    print(table.to_string(float_format=fmt))
