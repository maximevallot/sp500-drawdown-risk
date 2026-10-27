import os
import sys

import numpy as np
import pandas as pd
import torch

from ml import TEST_START, load

# Timestral, Mistral AI's time-series foundation model (huggingface.co/mistralai/Timestral-209M),
# as a zero-shot forecaster of a 10% loss. Written by Claude Code at Maxime's request (2026-10-19).
#
# Nothing is fitted on our data. At each month-end t the model sees only the history up to t
# and returns 21 quantiles of the path ahead; the probability of a 10% loss is where a
# cumulative log return of ln(0.9) falls in the forecast distribution of the 3-month horizon.
# Two variants:
#   - daily: the US market's cumulative log total return (Fama-French Mkt-RF + RF, CRSP's
#     value-weighted market, a stand-in for the S&P 500) on every trading day since 1926-07,
#     up to the day before the month's last trading day (Kritzman's day-before rule), and a
#     horizon to the last trading day three months later;
#   - monthly: the S&P 500 total-return index at each month-end (sp500_tr_events.csv, the
#     target's own series), as known at t like logret_past_3m, and a horizon of 3 months.
#
# Caveat: Mistral does not say what the model was trained on. If its training data included
# US stock indices up to recent years, the 1991-2026 forecasts are not a clean out-of-sample
# test, unlike every other method here. Read the results as an upper bound.
#
# The weights (about 0.8 GB) are cached in .cache/huggingface, which is gitignored.
# Needs: pip install git+https://github.com/mistralai/timestral.git
# Writes data/timestral/forecasts.csv. Run from the repository root:
#   .venv/bin/python analysis/timestral_forecast.py [first last]   (default: TEST_START to the end)

LOSS = np.log(0.9)
REPO = "mistralai/Timestral-209M"
BATCH = 8


def daily_market():
    """Cumulative log total return of the US market, by trading day."""
    raw = pd.read_csv("data/equity-factors/F-F_Research_Data_Factors_daily_raw.csv", skiprows=4,
                      skipfooter=2, engine="python")
    raw = raw.rename(columns={raw.columns[0]: "date"}).dropna()
    raw = raw[raw["date"].astype(str).str.fullmatch(r"\d{8}")]
    date = pd.to_datetime(raw["date"].astype(str), format="%Y%m%d")
    ret = (raw["Mkt-RF"].astype(float) + raw["RF"].astype(float)) / 100
    return pd.Series(np.log1p(ret).cumsum().to_numpy(), index=date, name="level")


def monthly_sp500():
    """Cumulative log total return of the S&P 500 at each month-end."""
    tr = pd.read_csv("data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]).set_index("Date")
    return np.log1p(tr["Return_1m_TR"].fillna(0)).cumsum().rename("level")


def loss_probability(quantiles, levels):
    """P(cumulative log return <= ln 0.9) from one horizon's quantiles, linear between knots
    and extrapolated linearly from the two lowest knots below the first, within [0.001, 0.999]."""
    q, lv = np.asarray(quantiles, float), np.asarray(levels, float)
    if LOSS < q[0]:
        slope = (lv[1] - lv[0]) / max(q[1] - q[0], 1e-12)
        return float(np.clip(lv[0] - slope * (q[0] - LOSS), 0.001, 0.999))
    return float(np.clip(np.interp(LOSS, q, lv), 0.001, 0.999))


def forecast_batch(model, contexts, horizons):
    """Loss probabilities for a batch of contexts, each relative to its last value."""
    length = max(len(c) for c in contexts)
    padded = np.full((len(contexts), length), np.nan, dtype=np.float32)
    for i, c in enumerate(contexts):
        padded[i, length - len(c):] = c - c[-1]
    out = model.forecast(padded, prediction_length=max(horizons), clip_nonnegative=False)
    q = out.quantiles.float().cpu().numpy()  # (B, V, H, Q) or (B, H, Q)
    q = q[:, 0] if q.ndim == 4 else q
    return [loss_probability(q[i, h - 1], out.quantile_levels) for i, h in enumerate(horizons)]


def daily_inputs(dates, series):
    """For each month-end t: the context to the day before the month's last trading day, and the
    horizon from there to the last trading day of month t + 3."""
    days = series.index
    contexts, horizons = [], []
    for t in dates:
        in_month = days[(days <= t) & (days > t - pd.offsets.MonthBegin(1))]
        cut = days.get_loc(in_month[-1])           # the month's last trading day
        end = t + pd.offsets.MonthEnd(3)
        end_days = days[days <= end]
        # past the data's last day, count business days: the calendar is known in advance
        extra = np.busday_count((days[-1] + pd.Timedelta(days=1)).date(), (end + pd.Timedelta(days=1)).date()) \
            if end > days[-1] else 0
        contexts.append(series.iloc[:cut].to_numpy())  # up to the day before it
        horizons.append(days.get_loc(end_days[-1]) - cut + 1 + extra)
    return contexts, horizons


def run(model, dates, contexts, horizons, label):
    probs = []
    for i in range(0, len(dates), BATCH):
        probs += forecast_batch(model, contexts[i:i + BATCH], horizons[i:i + BATCH])
        print(f"{label}: {min(i + BATCH, len(dates))} of {len(dates)}", flush=True)
    return pd.Series(probs, index=dates, name=label)


if __name__ == "__main__":
    from timestral import TimestralForecaster

    panel, _ = load()
    first, last = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else (TEST_START, None)
    dates = panel["Event_A"].dropna().loc[first:last].index
    device = "cpu"  # the scaler works in float64, which Apple's GPU (mps) does not support
    model = TimestralForecaster.from_pretrained(REPO, device=device, cache_dir=".cache/huggingface")

    daily = daily_market()
    d_ctx, d_h = daily_inputs(dates, daily)
    monthly = monthly_sp500()
    m_ctx = [monthly.loc[:t].to_numpy() for t in dates]
    with torch.no_grad():
        out = pd.concat([
            run(model, dates, d_ctx, d_h, "Timestral, daily"),
            run(model, dates, m_ctx, [3] * len(dates), "Timestral, monthly"),
        ], axis=1)
    out.insert(0, "Event_A", panel["Event_A"].reindex(out.index))
    out.index.name = "Date"
    os.makedirs("data/timestral", exist_ok=True)
    path = "data/timestral/forecasts.csv" if len(sys.argv) <= 2 else "data/timestral/forecasts_test.csv"
    out.to_csv(path, float_format="%.6f")
    print(out.describe().round(3).to_string())
