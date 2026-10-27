import numpy as np
import pandas as pd

# CAPE as known at each month-end, from Shiller's monthly data (data/valuation/ie_data_raw.xls).
# Written by Claude Code at Maxime's request (2026-10-18).
#
# Shiller's own CAPE column divides this month's price by the last ten years of earnings,
# but recent earnings are estimated and revised and come out months after the quarter. So at
# month-end t, this uses:
#   - the price of month t - 1 (Shiller's P is the month's average of daily closes, so the
#     whole of month t is not known before its last day), deflated by that month's CPI;
#   - the average real earnings of the 120 months ending at t - EARNINGS_LAG, each deflated by
#     its own month's CPI (data/README.md: lag CAPE by three to six months).
# The ratio is the cyclically adjusted P/E as it could have been computed at t.
#
# Reading the .xls needs xlrd (pip install xlrd); build_panel.py reads the CSV, so it does not.
# Writes data/valuation/cape_pit.csv. Run from the repository root: .venv/bin/python analysis/cape_pit.py

EARNINGS_LAG = 6   # months between the end of an earnings month and its use
YEARS = 10


def shiller_monthly():
    """Price, earnings and CPI by calendar month-end, from the Data sheet."""
    raw = pd.read_excel("data/valuation/ie_data_raw.xls", sheet_name="Data", header=None, skiprows=8)
    raw = raw.iloc[:, [0, 1, 3, 4]]
    raw.columns = ["date", "P", "E", "CPI"]
    raw = raw[pd.to_numeric(raw["date"], errors="coerce").notna()].astype(float)
    # Shiller writes the month as hundredths of the year: 1871.01 is January, 1871.1 October
    year = np.floor(raw["date"]).astype(int)
    month = np.rint((raw["date"] - year) * 100).astype(int)
    raw.index = pd.to_datetime(dict(year=year, month=month, day=1)) + pd.offsets.MonthEnd(0)
    return raw[["P", "E", "CPI"]]


def cape_pit(shiller):
    real_price = (shiller["P"] / shiller["CPI"]).shift(1)
    real_earnings = shiller["E"] / shiller["CPI"]
    avg_earnings = real_earnings.rolling(12 * YEARS, min_periods=12 * YEARS).mean().shift(EARNINGS_LAG)
    return (real_price / avg_earnings).rename("cape_pit").to_frame()


if __name__ == "__main__":
    out = cape_pit(shiller_monthly()).dropna()
    out.index.name = "Date"
    out.to_csv("data/valuation/cape_pit.csv", float_format="%.4f")
    print(f"{len(out)} month-ends, {out.index.min():%Y-%m} to {out.index.max():%Y-%m}")
    print(out.loc[["1929-08-31", "1999-12-31", "2009-02-28", "2021-12-31"]].round(1).to_string())
