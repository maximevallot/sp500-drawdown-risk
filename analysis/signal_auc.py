"""How well each of Olivia's signals ranks loss months above other months.

Written by Claude (2026-10-15) at Olivia's request, to test Michal's finding
that volatility "finds risky eras, not risky months" on the other signals of
the stream. Reads the saved monthly files; prints only, saves nothing.

- AUC: the chance that a random month followed by a 10% loss (total return,
  Event_A_TR) gets a higher signal than a random month that was not. 0.5 is
  chance, 1 is perfect.
- Within-decade AUC: the same, comparing only months of the same decade, then
  pooled over decades (weighted by the number of pairs). It removes the effect
  of whole decades being both riskier and more volatile.
- Overlapping three-month windows make neighbouring months share an outcome,
  so the effective sample is the number of episodes, not of months.

Run from the repository root: python3 analysis/signal_auc.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# (file, column, sign): sign -1 when a LOW value is the warning.
SIGNALS = [
    ("volatility/vol_probabilities.csv", "p_rolling63", 1),
    ("volatility/vol_probabilities.csv", "p_ewma94", 1),
    ("volatility/vol_probabilities.csv", "p_vix_adjusted", 1),
    ("fragility/absorption_ratio.csv", "ar17_shift", 1),
    ("fragility/absorption_ratio.csv", "ar49_shift", 1),
    ("fragility/turbulence.csv", "turb17", 1),
    ("fragility/turbulence.csv", "turb_multi", 1),
    ("fragility/turbulence.csv", "crash_rel17", -1),
    ("fragility/turbulence.csv", "crash_rel_multi", -1),
]

ERAS = [("1930-1959", 1930, 1959), ("1960-1989", 1960, 1989), ("1990-2026", 1990, 2026)]


def auc_pairs(score, event):
    """AUC and the number of (event, non-event) pairs it is based on."""
    pos = score[event == 1].to_numpy()
    neg = score[event == 0].to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, 0
    # Mann-Whitney: rank all scores, ties get the average rank.
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return auc, len(pos) * len(neg)


def episodes(event):
    """Separate loss episodes: runs of consecutive event months."""
    e = event.astype(int)
    return int(((e == 1) & (e.shift(1, fill_value=0) == 0)).sum())


def main():
    events = pd.read_csv(
        ROOT / "data/equity-index/sp500_tr_events.csv", parse_dates=["Date"]
    ).set_index("Date")["Event_A_TR"]

    rows = []
    for file, column, sign in SIGNALS:
        signal = pd.read_csv(ROOT / "data" / file, parse_dates=["Date"]).set_index("Date")[column]
        data = pd.concat([sign * signal, events], axis=1, join="inner").dropna()
        data.columns = ["score", "event"]

        overall, _ = auc_pairs(data["score"], data["event"])

        weighted, pairs = 0.0, 0
        for _, decade in data.groupby(data.index.year // 10):
            a, n = auc_pairs(decade["score"], decade["event"])
            if n:
                weighted += a * n
                pairs += n
        within = weighted / pairs if pairs else np.nan

        row = {
            "signal": column,
            "from": f"{data.index.min():%Y-%m}",
            "months": len(data),
            "loss months": int(data["event"].sum()),
            "episodes": episodes(data["event"]),
            "AUC": overall,
            "within decade": within,
        }
        for name, start, end in ERAS:
            era = data[(data.index.year >= start) & (data.index.year <= end)]
            row[name], _ = auc_pairs(era["score"], era["event"])
        rows.append(row)

    table = pd.DataFrame(rows).set_index("signal")
    print("Target: S&P 500 total return, loss of 10% or more over the next three months (Event_A_TR).")
    print("crash_rel scores are flipped, so that higher always means more warning.\n")
    print(table.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
