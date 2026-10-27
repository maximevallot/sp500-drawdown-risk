"""HMC SPX options: terminal-loss and intraperiod-touch backtests.

Usage (Spyder IPython Console):
%cd <folder holding the two CSV files>
%run hmc_option_pricing_updated.py --features 'spx_monthly_option_features_1996_2025(1).csv' --events sp500_events.csv

The options feature file contains first-trading-day observations. The event
file's dates represent month-end forecast origins. We join each event origin
to the most recent *prior* option observation, report signal age, and label
this as a STALE-SIGNAL exploratory test, not same-day forecast validation.
A proper synchronized backtest needs month-end options observations.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T = 91 / 365
MIN_TRAIN = 120

def find_col(df, options):
    norm_names = {str(c).strip().lower().replace(' ', '_'): c for c in df.columns}
    for name in options:
        if name.lower() in norm_names:
            return norm_names[name.lower()]
    return None

def binary_col(df, names, label):
    c = find_col(df, names)
    if c is None:
        raise ValueError(f'Missing {label}. Available columns: {list(df.columns)}')
    x = pd.to_numeric(df[c], errors='coerce')
    bad = x.notna() & ~x.isin([0, 1])
    if bad.any():
        raise ValueError(f'{label} ({c}) must contain 0/1 values. Please verify definitions.')
    return x

def read_features(path):
    df = pd.read_csv(path)
    date_col = find_col(df, ['date', 'option_date', 'month'])
    atm_col = find_col(df, ['atm_iv_91d_put50'])
    put_col = find_col(df, ['put25_iv_91d'])
    if not all([date_col, atm_col, put_col]):
        raise ValueError(f'Expected date, atm_iv_91d_put50, put25_iv_91d; got {list(df.columns)}')
    df['signal_date'] = pd.to_datetime(df[date_col], errors='coerce')
    df['atm_iv'] = pd.to_numeric(df[atm_col], errors='coerce')
    df['put_iv'] = pd.to_numeric(df[put_col], errors='coerce')
    units = find_col(df, ['iv_units'])
    if units is not None:
        vals = df[units].dropna().astype(str).str.lower().unique()
        print('IV units metadata:', list(vals)[:10])
    # IV should be decimal (e.g. 0.20). Convert percentages if consistently > 1.
    if df['atm_iv'].median() > 1:
        df[['atm_iv', 'put_iv']] /= 100
        print('Converted IV from percentage units to decimals.')
    df['put_skew'] = df['put_iv'] - df['atm_iv']
    df = df.dropna(subset=['signal_date', 'atm_iv', 'put_iv'])
    df = df[(df.atm_iv > 0) & (df.atm_iv < 3) & (df.put_iv > 0) & (df.put_iv < 3)]
    return df.sort_values('signal_date').drop_duplicates('signal_date', keep='last')[
        ['signal_date', 'atm_iv', 'put_iv', 'put_skew']]

def read_events(path):
    df = pd.read_csv(path)
    date_col = find_col(df, ['date', 'month_end', 'month', 'observation_date'])
    if date_col is None:
        raise ValueError(f'No event-origin date column found: {list(df.columns)}')
    df['origin_date'] = pd.to_datetime(df[date_col], errors='coerce')
    # Samantha's case a: >=10% down at 3m end; case b: touched -10% anytime.
    df['terminal_event'] = binary_col(df, ['Event_A', 'event_a', 'terminal_event'], 'terminal event')
    df['touch_event'] = binary_col(df, ['Event_B', 'event_b', 'touch_event'], 'intraperiod touch event')
    df = df.dropna(subset=['origin_date', 'terminal_event', 'touch_event'])
    df = df.sort_values('origin_date').drop_duplicates('origin_date', keep='last')
    if (df.terminal_event > df.touch_event).any():
        raise ValueError('Some terminal events are 1 while touch events are 0. Check Event_A/Event_B definitions and dates.')
    return df[['origin_date', 'terminal_event', 'touch_event']]

def join_point_in_time(features, events, max_signal_age_days=45):
    merged = pd.merge_asof(events.sort_values('origin_date'),
                           features.sort_values('signal_date'),
                           left_on='origin_date', right_on='signal_date',
                           direction='backward', tolerance=pd.Timedelta(days=max_signal_age_days))
    merged['signal_age_days'] = (merged.origin_date - merged.signal_date).dt.days
    merged = merged.dropna(subset=['atm_iv', 'put_skew']).copy()
    if merged.empty:
        raise ValueError('No observations after date matching. Check event date format and source dates.')
    if (merged.signal_age_days < 0).any():
        raise ValueError('Look-ahead error: signal date later than event origin.')
    print(f'Joined {len(merged)} observations; signal age: '
          f'median {merged.signal_age_days.median():.0f} days, '
          f'max {merged.signal_age_days.max():.0f} days.')
    print('WARNING: Earlier-month option signals paired with later month-end event origins.')
    print('This is an exploratory STALE-SIGNAL backtest, NOT a synchronized month-end options backtest.')
    return merged

def bs_probabilities(sigma):
    """Risk-neutral illustrative r-q=0 GBM; Q probabilities, not calibrated P."""
    s = np.asarray(sigma, dtype=float)
    a = np.log(0.9)
    mu = -0.5 * s**2
    sd = s * np.sqrt(T)
    terminal = norm.cdf((a - mu*T) / sd)
    touch = norm.cdf((a - mu*T) / sd) + np.exp(2*mu*a / s**2) * norm.cdf((a + mu*T) / sd)
    return np.clip(terminal, 0, 1), np.clip(touch, 0, 1)

def score(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return {'n_forecasts': len(y), 'n_events': int(y.sum()),
            'event_rate': float(y.mean()),
            'brier': float(brier_score_loss(y, p)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan}

def expanding_backtest(df, event_col, min_train):
    y = df[event_col].to_numpy(dtype=int)
    terminal, touch = bs_probabilities(df['atm_iv'].to_numpy())
    q = terminal if event_col == 'terminal_event' else touch
    X1 = df[['atm_iv']].to_numpy()
    X2 = df[['atm_iv', 'put_skew']].to_numpy()
    rows = []
    for i in range(min_train, len(df)):
        hist = y[:i]
        if len(np.unique(hist)) < 2:
            continue
        row = {'origin_date': df.iloc[i]['origin_date'],
               'signal_date': df.iloc[i]['signal_date'],
               'signal_age_days': df.iloc[i]['signal_age_days'],
               'event': event_col, 'actual': int(y[i]),
               'historical_baseline': float(hist.mean()),
               'bs_risk_neutral_uncalibrated': float(q[i])}
        for name, X in [('atm_iv_logit', X1), ('atm_iv_put_skew_logit', X2)]:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            model.fit(X[:i], hist)
            row[name] = float(model.predict_proba(X[i:i+1])[0, 1])
        # Calibrate BS Q probability to observed P via training-only logistic fit.
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(q[:i].reshape(-1, 1), hist)
        row['bs_calibrated'] = float(model.predict_proba(q[i:i+1].reshape(-1, 1))[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)

def evaluate(predictions, label):
    names = ['historical_baseline', 'bs_risk_neutral_uncalibrated',
             'bs_calibrated', 'atm_iv_logit', 'atm_iv_put_skew_logit']
    out = []
    for model in names:
        if predictions.empty:
            continue
        out.append({'event_definition': label, 'model': model, **score(predictions.actual, predictions[model])})
    return out

def plot_preds(pred, event, dest):
    if pred.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(pred.origin_date, pred.atm_iv_put_skew_logit, label='ATM IV + put skew (P estimate)')
    ax.plot(pred.origin_date, pred.historical_baseline, label='Historical baseline', alpha=.8)
    ax.scatter(pred.loc[pred.actual.eq(1), 'origin_date'],
               pred.loc[pred.actual.eq(1), 'atm_iv_put_skew_logit'],
               marker='x', s=65, label='Realized event')
    ax.set_ylabel('Forecast probability')
    ax.set_title(f'{event}: expanding-window forecasts (stale signals; exploratory)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=170)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--events', required=True)
    ap.add_argument('--output', default='hmc_option_results_updated')
    ap.add_argument('--min-train', type=int, default=MIN_TRAIN)
    args = ap.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    df = join_point_in_time(read_features(args.features), read_events(args.events))
    df.to_csv(dest / 'option_aligned_features_and_events.csv', index=False)
    print('Event counts:', {k: int(df[k].sum()) for k in ['terminal_event', 'touch_event']})
    all_pred, metrics = [], []
    for target in ['terminal_event', 'touch_event']:
        p = expanding_backtest(df, target, args.min_train)
        if p.empty:
            print(f'No valid forecasts for {target}; reduce min-train or inspect events.')
            continue
        all_pred.append(p)
        metrics += evaluate(p, f'{target}_monthly_overlapping')
        plot_preds(p, target, dest / f'{target}_probability_chart.png')
        # Non-overlapping 3-month forecasts: 3 possible calendar phases.
        # This is a sensitivity analysis, NOT 3 independent datasets.
        for phase in range(3):
            sub = p.iloc[phase::3]
            metrics += evaluate(sub, f'{target}_nonoverlap_phase_{phase}')
    if all_pred:
        pd.concat(all_pred, ignore_index=True).to_csv(dest / 'option_backtest_predictions.csv', index=False)
        m = pd.DataFrame(metrics)
        m.to_csv(dest / 'option_backtest_metrics.csv', index=False)
        print('\nMONTHLY OVERLAPPING RESULTS:')
        print(m[m.event_definition.str.endswith('monthly_overlapping')].to_string(index=False))
    print('\nIMPORTANT: Verify Event_A/Event_B represent terminal and touch events at the SAME month-end origin.')
    print('Do not report these exploratory results as synchronized until you rebuild options at month-end.')
    print('Saved to:', dest.resolve())

if __name__ == '__main__':
    main()
