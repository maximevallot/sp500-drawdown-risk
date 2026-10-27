from math import erf

import numpy as np
import pandas as pd


def day_before_month_end(series):
    """Value on each month's second-to-last trading day (Kritzman, 2026-10-13).

    A month-end forecast can only use what is known before the month's last
    close. The trading days are the series' own dates. The current month is
    dropped until it is complete, so the latest row is never a partial month.
    Added by Claude (2026-10-15).
    """
    last = series.index[-1]
    if last < last + pd.offsets.BMonthEnd(0):
        series = series[series.index.to_period("M") < last.to_period("M")]

    months = series.index.to_period("M")
    out = series.groupby(months).apply(
        lambda x: x.iloc[-2] if len(x) >= 2 else np.nan
    )
    out.index = out.index.to_timestamp() + pd.offsets.MonthEnd(0)
    return out

# 1. Load the S&P 500 closes
px = pd.read_csv("data/licensed/sp500_yahoo_raw.csv")
px["Date"] = pd.to_datetime(px["Date"], format="%m/%d/%Y")
px = px.set_index("Date").sort_index()[["Close"]]

# 2. Daily log returns
r = np.log(px["Close"] / px["Close"].shift(1)).dropna()

# 3. Rolling 63-day volatility, annualized
vol = r.rolling(63).std() * np.sqrt(252)
vol.name = "Volatility"

# 4. Keep the value known at each month-end: the day before the last trading day
monthly_vol = day_before_month_end(vol).rename("Volatility")

# 5. Join with the team's event series and compare
# The target is total return since 2026-10-14 (Michal's sp500_tr_events.csv).
ev = pd.read_csv(
    "data/equity-index/sp500_tr_events.csv",
    parse_dates=["Date"],
).set_index("Date").rename(
    columns={"Return_3m_TR": "Return_3m", "Event_A_TR": "Event_A"}
)[["Return_3m", "Event_A"]]

# Keep every month-end, including the latest ones whose outcome is not known
# yet: they are needed for today's forecast. Event statistics skip them.
comparison = monthly_vol.to_frame().join(ev, how="left")
comparison = comparison.dropna(subset=["Volatility"])
average_vol = comparison.groupby("Event_A")["Volatility"].mean()

print("Average annualized volatility:")
print(average_vol.reindex([1, 0]).to_string(float_format=lambda x: f"{x:.2%}"))

# Define calm markets using the median in non-event months.
threshold = comparison.loc[
    comparison["Event_A"] == 0, "Volatility"
].median()

events = comparison.loc[comparison["Event_A"] == 1]
calm_events = events.loc[events["Volatility"] < threshold]

print(f"\nCalm-market threshold: {threshold:.2%}")
print(f"Events below threshold: {len(calm_events)} of {len(events)}")

print("\nEvent month-end dates and annualized volatility:")
print(
    calm_events[["Volatility"]].to_string(
        formatters={"Volatility": lambda x: f"{x:.2%}"}
    )
)

# 6. Probability of finishing a quarter down 10% or more
T = 0.25
loss_threshold = np.log(0.9)

def normal_cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

# The formula uses an annualized mean of log returns. To avoid look-ahead,
# each month-end uses only the returns up to that date (an expanding mean),
# and needs at least one year of data.
mu_expanding = day_before_month_end(r.expanding(min_periods=252).mean() * 252)
comparison["Mu_expanding"] = mu_expanding.reindex(comparison.index)

mean_assumptions = {
    "P_mu_10pct": 0.10,
    "P_mu_expanding": comparison["Mu_expanding"],
}

for column, mu in mean_assumptions.items():
    sigma = comparison["Volatility"].where(
        comparison["Volatility"] > 0
    )
    z = (loss_threshold - mu * T) / (sigma * np.sqrt(T))
    comparison[column] = z.map(normal_cdf)

# 7. Inspect the previously identified calm-market event dates
probability_columns = list(mean_assumptions)
calm_date_results = comparison.loc[
    calm_events.index,
    ["Volatility"] + probability_columns,
]

print(
    "\nExpanding annualized mean log return at the last month-end: "
    f"{comparison['Mu_expanding'].iloc[-1]:.2%}"
)
print(f"\nProbabilities at {len(calm_date_results)} calm-market event dates:")
print(
    calm_date_results.to_string(
        formatters={
            column: lambda x: f"{x:.2%}"
            for column in calm_date_results.columns
        }
    )
)

print("\nAverage predicted probability at these dates:")
print(
    calm_date_results[probability_columns]
    .mean()
    .to_string(float_format=lambda x: f"{x:.2%}")
)

# 8. EWMA volatility: lambda = 0.94
ewma_variance = r.pow(2).ewm(alpha=0.06, adjust=False).mean()
ewma_vol = np.sqrt(ewma_variance) * np.sqrt(252)

comparison["EWMA_Volatility"] = day_before_month_end(ewma_vol)

# 9. Probabilities using the same expanding mean for both methods
# (Mu_expanding from step 6: past returns only, at least one year of data)
T = 0.25
loss_threshold = np.log(0.9)

for vol_column, probability_column in [
    ("Volatility", "P_rolling"),
    ("EWMA_Volatility", "P_ewma"),
]:
    sigma = comparison[vol_column].where(comparison[vol_column] > 0)
    z = (
        loss_threshold - comparison["Mu_expanding"] * T
    ) / (sigma * np.sqrt(T))

    comparison[probability_column] = z.map(normal_cdf)

# Use identical available months for the comparison.
paired = comparison.dropna(
    subset=["Event_A", "P_rolling", "P_ewma"]
)
prob_columns = ["P_rolling", "P_ewma"]
percent_formatters = {
    column: lambda x: f"{x:.2%}"
    for column in [
        "Volatility", "EWMA_Volatility", "P_rolling", "P_ewma"
    ]
}

# 10. Average probability before falls versus other months
print("\nAverage predicted probability by event:")
print(
    paired.groupby("Event_A")[prob_columns]
    .mean()
    .reindex([1, 0])
    .to_string(formatters=percent_formatters)
)

# 11. Keep the original 27 calm-market dates for both methods
calm_results = paired.reindex(calm_events.index)

print(f"\nAverage probability at the {len(calm_results)} calm-market dates:")
print(
    calm_results[prob_columns]
    .mean()
    .to_string(float_format=lambda x: f"{x:.2%}")
)
print("Available observations per method:")
print(calm_results[prob_columns].count().to_string())

# 12. Inspect selected month-ends
selected_dates = pd.to_datetime([
    "1987-08-31",
    "1987-10-31",
    "2008-09-30",
    "2019-12-31",
    "2020-03-31",
])

selected_results = comparison.reindex(selected_dates)[[
    "Volatility",
    "EWMA_Volatility",
    "P_rolling",
    "P_ewma",
]]

print("\nSelected month-end estimates:")
print(selected_results.to_string(formatters=percent_formatters))

# 13. VIX and VIX3M probabilities
def load_option_volatility(path, value_column):
    data = pd.read_csv(path)
    data["observation_date"] = pd.to_datetime(
        data["observation_date"]
    )
    # Handles blank holidays and any "." missing-value markers.
    data[value_column] = pd.to_numeric(
        data[value_column], errors="coerce"
    )

    daily = (
        data.set_index("observation_date")[value_column]
        .sort_index()
        .dropna()
        / 100.0
    )
    return day_before_month_end(daily)


vix_monthly = load_option_volatility(
    "data/volatility/VIXCLS_raw.csv", "VIXCLS"
)
vix3m_monthly = load_option_volatility(
    "data/volatility/VXVCLS_raw.csv", "VXVCLS"
)

# Actual volatility within each completed calendar month.
realized_monthly = r.resample("ME").std() * np.sqrt(252)


def add_option_probabilities(monthly_sigma, prefix, horizon_months):
    # Match each old forecast with its subsequently realized volatility.
    # All calibration outcomes end on or before the current month-end.
    if horizon_months == 1:
        realized = realized_monthly
    else:
        # Volatility over the three completed calendar months.
        monthly_periods = r.index.to_period("M")
        realized = pd.Series(
            {
                date: r.loc[
                    (monthly_periods > date.to_period("M") - 3)
                    & (monthly_periods <= date.to_period("M"))
                ].std() * np.sqrt(252)
                for date in monthly_sigma.index
            },
            dtype=float,
        )

    calibration = pd.concat(
        [
            realized.rename("Realized"),
            monthly_sigma.shift(horizon_months).rename("Forecast"),
        ],
        axis=1,
    ).dropna()

    # Start adjustment after 12 matched observations.
    scale = (
        calibration["Realized"].expanding(min_periods=12).mean()
        / calibration["Forecast"].expanding(min_periods=12).mean()
    )
    # The realized volatility of month m ends on its last day, after the
    # day-before signal date, so month m's forecast uses the scale known at
    # the end of month m - 1.
    scale = scale.shift(1)

    comparison[f"{prefix}_Volatility"] = monthly_sigma
    comparison[f"{prefix}_Scale"] = scale
    comparison[f"{prefix}_Adjusted_Volatility"] = (
        comparison[f"{prefix}_Volatility"]
        * comparison[f"{prefix}_Scale"]
    )

    for suffix, vol_column in [
        ("raw", f"{prefix}_Volatility"),
        ("adjusted", f"{prefix}_Adjusted_Volatility"),
    ]:
        sigma = comparison[vol_column].where(
            comparison[vol_column] > 0
        )
        z = (
            np.log(0.9) - comparison["Mu_expanding"] * 0.25
        ) / (sigma * np.sqrt(0.25))

        comparison[f"P_{prefix.lower()}_{suffix}"] = z.map(normal_cdf)


add_option_probabilities(vix_monthly, "VIX", horizon_months=1)
add_option_probabilities(vix3m_monthly, "VIX3M", horizon_months=3)


# 14. Compare methods on identical available months
def report_option_comparison(prefix):
    columns = [
        "P_rolling",
        "P_ewma",
        f"P_{prefix.lower()}_raw",
        f"P_{prefix.lower()}_adjusted",
    ]
    common = comparison.dropna(subset=["Event_A"] + columns)
    formatters = {
        column: lambda x: f"{x:.2%}" for column in columns
    }

    print(f"\n{prefix}: {len(common)} common months")
    if common.empty:
        return

    print(f"Period: {common.index.min():%Y-%m} to "
          f"{common.index.max():%Y-%m}")

    print("\nAverage probability before falls versus other months:")
    print(
        common.groupby("Event_A")[columns]
        .mean()
        .reindex([1, 0])
        .to_string(formatters=formatters)
    )

    calm_common = common.loc[
        common.index.isin(calm_events.index)
    ]
    print(f"\nCalm-market events available: {len(calm_common)}")
    print(calm_common[columns].to_string(formatters=formatters))

    print("\nAverage probability at those calm-market dates:")
    print(
        calm_common[columns].mean().to_string(
            float_format=lambda x: f"{x:.2%}"
        )
    )

    print("\nAverage volatility calibration multiplier:")
    print(f"{common[f'{prefix}_Scale'].mean():.3f}")


report_option_comparison("VIX")
report_option_comparison("VIX3M")


# 15. Save the monthly series for the other streams and the shared back-test.
# Derived values only (no index prices), so the file can be committed.
output_columns = {
    "Volatility": "vol_rolling63",
    "EWMA_Volatility": "vol_ewma94",
    "VIX_Volatility": "vol_vix",
    "VIX_Adjusted_Volatility": "vol_vix_adjusted",
    "VIX3M_Volatility": "vol_vix3m",
    "VIX3M_Adjusted_Volatility": "vol_vix3m_adjusted",
    "Mu_expanding": "mu_expanding",
    "P_rolling": "p_rolling63",
    "P_ewma": "p_ewma94",
    "P_vix_raw": "p_vix_raw",
    "P_vix_adjusted": "p_vix_adjusted",
    "P_vix3m_raw": "p_vix3m_raw",
    "P_vix3m_adjusted": "p_vix3m_adjusted",
}
output = comparison[list(output_columns)].rename(columns=output_columns)
output.index.name = "Date"
output.to_csv(
    "data/volatility/vol_probabilities.csv",
    float_format="%.6f",
    date_format="%Y-%m-%d",
)
print(f"\nSaved data/volatility/vol_probabilities.csv: {len(output)} month-ends")
