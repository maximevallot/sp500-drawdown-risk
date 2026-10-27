import numpy as np
import pandas as pd

# M2 money supply growth as it was known at each month-end, with no look-ahead.
# Written by Claude Code at Samantha's request (2026-10-15), the same way as analysis/indpro_pit.py.
#
# Input: data/macro/M2SL_raw_part1.csv and M2SL_raw_part2.csv, every ALFRED vintage of M2
# (seasonally adjusted, billions of dollars) in the real-time-period layout. ALFRED allows 1,500
# vintages per download and M2 has more (it is published weekly), so it came in two parts:
# part 1 has the vintages 1980-02-08 to 2004-12-30, part 2 those from 2005-01-06.
# Each part is right only on the days its own vintages cover, so part 1 is used before
# 2005-01-06 and part 2 from then on. ALFRED's vintages start in 1980, when M2 was redefined.
#
# For each month-end t, the series as published the day before t (Kritzman's timing rule).
# Growth is computed inside that one vintage, because M2 is revised every week.
#
# Output: data/macro/m2_pit.csv, one row per calendar month-end.
# Run from the repository root: python3 analysis/m2_pit.py

PARTS = ["data/macro/M2SL_raw_part1.csv", "data/macro/M2SL_raw_part2.csv"]
CUT = pd.Timestamp("2005-01-06")  # part 2's first vintage
OUT = "data/macro/m2_pit.csv"


def load_part(path):
    """Every published value in one part, with the dates it was the current figure."""
    raw = pd.read_csv(
        path, parse_dates=["period_start_date", "realtime_start_date", "realtime_end_date"]
    )
    raw = raw.dropna(subset=["M2SL"])
    raw["realtime_end_date"] = raw["realtime_end_date"].fillna(pd.Timestamp.max)
    return raw


def as_known_on(parts, day):
    """The series as published on day: observation month -> value, from the part covering day."""
    raw = parts[0] if day < CUT else parts[1]
    valid = (raw["realtime_start_date"] <= day) & (raw["realtime_end_date"] >= day)
    return raw.loc[valid].set_index("period_start_date")["M2SL"].sort_index()


def growth(series, months):
    """Change over the last `months` months of one vintage, as a fraction; NaN if not published."""
    last = series.index[-1]
    earlier = last - pd.DateOffset(months=months)
    if earlier not in series.index:
        return np.nan
    return series.loc[last] / series.loc[earlier] - 1


def point_in_time(parts):
    # from the first vintage to the end of the month of the last one downloaded
    first = parts[0]["realtime_start_date"].min()
    last_vintage = parts[1]["realtime_start_date"].max()
    month_ends = pd.date_range(first, last_vintage + pd.offsets.MonthEnd(0), freq="ME")
    month_ends = month_ends[month_ends <= pd.Timestamp.today()]  # not a month-end still to come
    rows = []
    for t in month_ends:
        series = as_known_on(parts, t - pd.Timedelta(days=1))
        if series.empty:
            continue
        last = series.index[-1]
        rows.append({
            "Date": t,
            "m2_obs_month": last.strftime("%Y-%m"),
            # how stale the latest figure is: 1 means last month's figure is out
            "m2_months_old": (t.year - last.year) * 12 + t.month - last.month,
            "m2_growth_3m": growth(series, 3),
            "m2_growth_12m": growth(series, 12),
        })
    return pd.DataFrame(rows).set_index("Date")


if __name__ == "__main__":
    pit = point_in_time([load_part(path) for path in PARTS])
    pit.to_csv(OUT, float_format="%.6f")
    print(f"{OUT}: {len(pit)} month-ends, {pit.index[0].date()} to {pit.index[-1].date()}")
    print("How old the latest figure is, in months:")
    print(pit["m2_months_old"].value_counts().sort_index().to_string())
