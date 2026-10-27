from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd

from absorption_ratio import get_calm_events, load_events


ROOT = Path(__file__).resolve().parents[1]

LOOKBACK = 1260       # Previous ~5 years
SMOOTH = 21          # Average the last 21 observations
MIN_EVENTS = 10      # Minimum known pre-crash months
MIN_THRESHOLD = 60   # Prior observations for diagnostic percentiles


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

def french_daily(path, marker=None):
    """Read the first daily block, optionally following a marker."""
    lines = Path(path).read_text().splitlines()

    start = (
        next(i for i, line in enumerate(lines) if marker in line)
        if marker else 0
    )

    header = next(
        i for i in range(start, len(lines))
        if lines[i].startswith(",")
    )

    rows = []
    for line in lines[header + 1:]:
        if re.match(r"^\s*\d{8}\s*,", line):
            rows.append(line)
        elif rows:
            break

    frame = pd.read_csv(
        StringIO("\n".join([lines[header]] + rows)),
        index_col=0,
        na_values=[-99.99, -999],
    )

    frame.index = pd.to_datetime(
        frame.index.astype(str), format="%Y%m%d"
    )
    frame.columns = frame.columns.str.strip()

    if frame.index.has_duplicates:
        raise ValueError("Duplicate French-data dates.")

    return frame.apply(pd.to_numeric).sort_index() / 100.0


def fred(relative_path, column):
    frame = pd.read_csv(
        ROOT / "data" / relative_path,
        parse_dates=["observation_date"],
    )

    series = frame.set_index("observation_date")[column]

    return (
        pd.to_numeric(series, errors="coerce")
        .dropna()
        .sort_index()
    )


def load_data():
    # A: 17 value-weighted industry returns
    industries = french_daily(
        ROOT
        / "data/industry-returns/17_Industry_Portfolios_Daily_raw.csv",
        marker="Average Value Weighted Returns -- Daily",
    )

    if industries.shape[1] != 17:
        raise ValueError("Expected exactly 17 industries.")

    # B: five multi-asset variables
    factors = french_daily(
        ROOT
        / "data/equity-factors/F-F_Research_Data_Factors_daily_raw.csv"
    )

    stocks = (
        factors["Mkt-RF"] + factors["RF"]
    ).rename("US_stocks")

    yields = fred("rates/DGS10_raw.csv", "DGS10")
    yield_change = yields.diff().rename("Yield_change")

    aaa = fred("credit/DAAA_raw.csv", "DAAA")
    baa = fred("credit/DBAA_raw.csv", "DBAA")

    credit_levels = pd.concat(
        [baa.rename("Baa"), aaa.rename("Aaa")],
        axis=1,
    ).dropna()

    spread_change = (
        credit_levels["Baa"] - credit_levels["Aaa"]
    ).diff().rename("Spread_change")

    oil = fred("commodities/DCOILWTICO_raw.csv", "DCOILWTICO")
    oil_return = oil.pct_change(fill_method=None)

    # Exclude nonpositive prices and the following observation.
    # This removes April 20 and April 21, 2020.
    oil_return = oil_return.where(
        oil.gt(0) & oil.shift(1).gt(0)
    ).rename("Oil")

    old_dollar = fred("fx/DTWEXM_raw.csv", "DTWEXM")
    new_dollar = fred("fx/DTWEXBGS_raw.csv", "DTWEXBGS")

    old_return = np.log(old_dollar / old_dollar.shift(1))
    new_return = np.log(new_dollar / new_dollar.shift(1))

    # Compute returns separately BEFORE joining the two indices.
    dollar_return = pd.concat([
        old_return.loc[:"2005-12-31"],
        new_return.loc["2006-01-01":],
    ]).rename("Dollar")

    multi = pd.concat(
        [
            stocks,
            yield_change,
            spread_change,
            oil_return,
            dollar_return,
        ],
        axis=1,
        join="inner",
    )

    multi = (
        multi.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )

    return industries, multi


# ------------------------------------------------------------
# 2. Normalized squared Mahalanobis distance
# ------------------------------------------------------------

def distance(x, mean, covariance):
    """
    Return (x - mean)' covariance^-1 (x - mean) / asset count.

    Standardizing first improves numerical stability without
    changing the Mahalanobis distance.
    """
    scales = np.sqrt(np.diag(covariance))

    if not np.isfinite(scales).all() or (scales <= 0).any():
        return np.nan

    correlation = covariance / np.outer(scales, scales)

    # Small regularization only if the matrix is ill-conditioned.
    if np.linalg.cond(correlation) > 1e12:
        correlation = (
            correlation + np.eye(len(scales)) * 1e-8
        )

    delta = (np.asarray(x) - mean) / scales
    solved = np.linalg.solve(correlation, delta)

    return max(float(delta @ solved), 0.0) / len(scales)


# ------------------------------------------------------------
# 3. Version 1: daily classic turbulence
# ------------------------------------------------------------

def daily_turbulence(returns):
    result = pd.Series(
        np.nan, index=returns.index, dtype=float
    )

    for _, current_month in returns.groupby(
        returns.index.to_period("M")
    ):
        # Estimate once at the start of each month.
        # Exclude every observation from the current month.
        history = returns.loc[
            returns.index < current_month.index[0]
        ].tail(LOOKBACK)

        if len(history) < LOOKBACK:
            continue

        if history.isna().any().any():
            continue

        mean = history.mean().to_numpy()
        covariance = history.cov().to_numpy()

        for date, row in current_month.iterrows():
            if row.notna().all():
                result.loc[date] = distance(
                    row.to_numpy(), mean, covariance
                )

    return result


# ------------------------------------------------------------
# 4. Monthly turbulence and monthly condition vectors
# ------------------------------------------------------------

def monthly_conditions(returns, signal_dates):
    smoothed_turbulence = (
        daily_turbulence(returns)
        .rolling(SMOOTH, min_periods=SMOOTH)
        .mean()
    )

    # Simple initial condition vector: 21-day mean per asset.
    conditions = (
        returns.rolling(SMOOTH, min_periods=SMOOTH).mean()
    )

    turbulence_values = []
    condition_vectors = []

    for month_end, cutoff in signal_dates.items():
        available = returns.index[
            returns.index <= cutoff
        ]

        if (
            len(available) == 0
            or available[-1].to_period("M")
            != cutoff.to_period("M")
        ):
            turbulence_values.append(np.nan)
            condition_vectors.append(
                np.full(returns.shape[1], np.nan)
            )
            continue

        # Last available observation on or before the common cutoff.
        date = available[-1]

        turbulence_values.append(
            smoothed_turbulence.loc[date]
        )
        condition_vectors.append(
            conditions.loc[date].to_numpy()
        )

    monthly_turbulence = pd.Series(
        turbulence_values,
        index=signal_dates.index,
        dtype=float,
    )

    monthly_vectors = pd.DataFrame(
        condition_vectors,
        index=signal_dates.index,
        columns=returns.columns,
    )

    return monthly_turbulence, monthly_vectors


# ------------------------------------------------------------
# 5. Version 2: distance from known pre-crash conditions
# ------------------------------------------------------------

def crash_distance(conditions, signal_dates, events):
    """
    Return two series:
    - the distance from the average of known pre-crash months;
    - the relative score: that distance minus the distance from the
      average of all months, with the same covariance. Negative means
      closer to pre-crash months than to an ordinary month, which
      removes how unusual the month is and keeps only the similarity.
    """
    result = pd.Series(
        np.nan, index=conditions.index, dtype=float
    )
    relative = pd.Series(
        np.nan, index=conditions.index, dtype=float
    )

    labels = events["Event_A"].reindex(conditions.index)

    # A label for month s becomes available after its
    # forward-three-month horizon has ended.
    outcome_known_dates = (
        conditions.index + pd.offsets.MonthEnd(3)
    )

    asset_count = conditions.shape[1]

    for month_end, cutoff in signal_dates.items():
        current = conditions.loc[month_end]

        if current.isna().any():
            continue

        # Covariance uses all available monthly conditions,
        # including the current condition vector.
        history = conditions.loc[:month_end].dropna()

        known_event_mask = (
            (outcome_known_dates <= cutoff)
            & labels.eq(1)
        )

        known_crashes = conditions.loc[
            known_event_mask
        ].dropna()

        if len(known_crashes) < MIN_EVENTS:
            continue

        if len(history) < max(24, asset_count + 2):
            continue

        crash_mean = known_crashes.mean().to_numpy()
        normal_mean = history.mean().to_numpy()
        covariance = history.cov().to_numpy()

        to_crash = distance(current.to_numpy(), crash_mean, covariance)
        to_normal = distance(current.to_numpy(), normal_mean, covariance)

        result.loc[month_end] = to_crash
        relative.loc[month_end] = to_crash - to_normal

    return result, relative


# ------------------------------------------------------------
# 6. Printed diagnostics; continuous scores remain the output
# ------------------------------------------------------------

def diagnostics(output, events, calm_dates):
    joined = output.join(
        events[["Event_A"]], how="inner"
    )

    columns = [
        "turb17",
        "turb_multi",
        "crash_dist17",
        "crash_dist_multi",
        "crash_rel17",
        "crash_rel_multi",
    ]

    for column in columns:
        series = output[column]
        is_crash_distance = column.startswith("crash_")

        # High turbulence is a warning.
        # LOW crash distance is a warning.
        quantile = 0.20 if is_crash_distance else 0.80

        # Shift first: today's score cannot set its own threshold.
        threshold = (
            series.shift(1)
            .expanding(min_periods=MIN_THRESHOLD)
            .quantile(quantile)
        )

        valid = joined.dropna(
            subset=[column, "Event_A"]
        )

        print(f"\n{column}: continuous predictor by Event_A")
        print(
            valid.groupby("Event_A")[column]
            .agg(["count", "mean", "median"])
        )

        scored = joined.join(
            threshold.rename("threshold")
        ).dropna(
            subset=[column, "threshold", "Event_A"]
        )

        if is_crash_distance:
            flag = scored[column].le(scored["threshold"])
            print("\nLow-distance warning diagnostics:")
        else:
            flag = scored[column].ge(scored["threshold"])
            print("\nHigh-turbulence warning diagnostics:")

        event = scored["Event_A"].eq(1)

        print({
            "scored_months": len(scored),
            "hits": int((flag & event).sum()),
            "misses": int((~flag & event).sum()),
            "false_alarms": int((flag & ~event).sum()),
            "correct_negatives": int((~flag & ~event).sum()),
        })

        print("\nOriginal calm-market dates; NaN = unavailable:")
        print(
            output.reindex(calm_dates)[
                ["signal_date", column]
            ].to_string()
        )

    # Compare datasets over identical labeled months as well.
    for columns in [
        ["turb17", "turb_multi"],
        ["crash_dist17", "crash_dist_multi"],
        ["crash_rel17", "crash_rel_multi"],
    ]:
        common = joined.dropna(
            subset=columns + ["Event_A"]
        )

        print(f"\nCommon-month comparison: {columns}")
        print(
            common.groupby("Event_A")[columns].mean()
        )


# ------------------------------------------------------------
# 7. Run and save
# ------------------------------------------------------------

def main():
    industries, multi = load_data()

    print(f"Industry observations: {len(industries):,}")
    print(f"Multi-asset observations: {len(multi):,}")

    # Use one common cutoff calendar for both datasets.
    rows = []

    for month, group in industries.groupby(
        industries.index.to_period("M")
    ):
        if len(group) >= 2:
            rows.append((
                month.to_timestamp("M"),
                group.index[-2],
            ))

    signal_dates = pd.Series(
        dict(rows), name="signal_date"
    ).rename_axis("Date")

    events = load_events()

    output = signal_dates.to_frame()

    for returns, suffix in [
        (industries, "17"),
        (multi, "_multi"),
    ]:
        turbulence, conditions = monthly_conditions(
            returns, signal_dates
        )

        output[f"turb{suffix}"] = turbulence

        (
            output[f"crash_dist{suffix}"],
            output[f"crash_rel{suffix}"],
        ) = crash_distance(conditions, signal_dates, events)

    output = output[[
        "signal_date",
        "turb17",
        "turb_multi",
        "crash_dist17",
        "crash_dist_multi",
        "crash_rel17",
        "crash_rel_multi",
    ]]

    # Reuse the original 27 dates, defined in absorption_ratio.py.
    _, calm_events = get_calm_events(events)
    calm_dates = calm_events.index

    diagnostics(output, events, calm_dates)

    folder = ROOT / "data/fragility"
    folder.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        folder / "turbulence.csv",
        float_format="%.6f",
        date_format="%Y-%m-%d",
    )

    print(f"\nSaved {(folder / 'turbulence.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
