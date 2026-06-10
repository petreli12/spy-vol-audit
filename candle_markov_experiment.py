"""candle_markov_experiment.py

Statistical audit of whether candlestick states predict next-bar direction on
SPY, unconditionally and conditioned on a volatility regime decoded by an HMM.

IMPORTANT: This is an IN-SAMPLE statistical audit, NOT a backtest. Every number
below (terciles, chi-square tests, HMM fit, regime labels) is computed on the
full sample with no train/test separation and no walk-forward. The goal is to
measure whether structure *exists* in the data, not to estimate tradable edge.
Treat p-values as descriptive, and read the economic-significance column in the
final comparison table before believing any of it matters after costs.

Pipeline (per timeframe in {"1d", "1h", "15m"}):
  Part 1  candle-state discretization (direction x body-tercile x wick) -> 12 states
  Part 2  unconditional chi-square: state vs next-bar direction (+ Wilson CIs)
  Part 3  3-state GaussianHMM on [log_return, log(GK vol)] -> volatility regimes
  Part 4  regime-conditioned chi-square, multiple-comparisons context
  Part 5  three matplotlib figures saved to ./output/

Data sources:
  1d            yfinance (auto_adjust=True), 10 years
  1h / 15m      alpaca-py StockHistoricalDataClient (IEX feed), max history;
                falls back to Massive aggregates if Alpaca returns nothing.

Downloaded data is cached to ./data/*.parquet so reruns do not re-pull.

Run:
  python candle_markov_experiment.py --timeframes 1d
  python candle_markov_experiment.py --timeframes 1d 1h 15m
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Headless backend so figures save without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.stats import chi2_contingency  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "output"

SYMBOL = "SPY"
TRADING_DAYS_PER_YEAR = 252
DAILY_VOL_WINDOW = 21  # trading days for realized vol
ALPHA = 0.05
ROUND_TRIP_COST_BP = 2.0  # basis points per round trip, for economic-significance column

# Body-ratio terciles -> bucket names.
BODY_BUCKETS = ("small", "medium", "large")

# Per-timeframe constants. `vol_window_bars` mirrors "21 trading days" at each
# timeframe; `bars_per_year` scales per-bar vol to annualized.
@dataclass(frozen=True)
class TimeframeConfig:
    label: str
    intraday: bool
    bars_per_day: int
    yf_interval: Optional[str] = None  # only for daily
    alpaca_amount: Optional[int] = None
    alpaca_unit: Optional[str] = None  # "Hour" / "Min"
    massive_multiplier: Optional[int] = None
    massive_timespan: Optional[str] = None  # "hour" / "minute"

    @property
    def vol_window_bars(self) -> int:
        return DAILY_VOL_WINDOW * self.bars_per_day

    @property
    def bars_per_year(self) -> int:
        return TRADING_DAYS_PER_YEAR * self.bars_per_day

    @property
    def annualization(self) -> float:
        return math.sqrt(self.bars_per_year)


TIMEFRAMES: Dict[str, TimeframeConfig] = {
    # Regular session = 6.5h. Hour bars -> 7/day (09:30..15:30 + 16:00 stub);
    # 15-min bars -> 26/day (6.5h * 4). Daily -> 1/day.
    "1d": TimeframeConfig("1d", intraday=False, bars_per_day=1, yf_interval="1d"),
    "1h": TimeframeConfig(
        "1h", intraday=True, bars_per_day=7,
        alpaca_amount=1, alpaca_unit="Hour",
        massive_multiplier=1, massive_timespan="hour",
    ),
    "15m": TimeframeConfig(
        "15m", intraday=True, bars_per_day=26,
        alpaca_amount=15, alpaca_unit="Min",
        massive_multiplier=15, massive_timespan="minute",
    ),
}

EASTERN = "US/Eastern"
SESSION_OPEN = (9, 30)
SESSION_CLOSE = (16, 0)


# ---------------------------------------------------------------------------
# Environment / secrets
# ---------------------------------------------------------------------------

def _load_dotenv_into_environ() -> None:
    """Backward-compatible alias; loads ./.env via research_utils."""
    from research_utils import load_dotenv_if_present

    load_dotenv_if_present()


def _alpaca_keys() -> Tuple[Optional[str], Optional[str]]:
    """Spec wants ALPACA_API_KEY / ALPACA_SECRET_KEY; fall back to the repo's paper keys."""
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_PAPER_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_PAPER_SECRET_KEY")
    return key, secret


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

def _cache_path(tf: TimeframeConfig) -> Path:
    return DATA_DIR / f"{SYMBOL}_{tf.label}.parquet"


def load_data(tf: TimeframeConfig, *, refresh: bool = False) -> pd.DataFrame:
    """Return a tz-naive (Eastern wall-clock for intraday) OHLCV frame indexed by timestamp.

    Columns: open, high, low, close, volume. Cached to ./data/<sym>_<tf>.parquet.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(tf)
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        print(f"[{tf.label}] loaded {len(df):,} cached bars from {cache.name}")
        return df

    if tf.intraday:
        df = _load_intraday(tf)
    else:
        df = _load_daily()

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_parquet(cache)
    print(f"[{tf.label}] downloaded {len(df):,} bars -> cached to {cache.name}")
    return df


def _load_daily() -> pd.DataFrame:
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=365 * 10 + 5)
    raw = yf.download(
        SYMBOL, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
        interval="1d", auto_adjust=True, progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no daily data for SPY")
    # yfinance may return a MultiIndex (field, ticker) for single symbols.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "timestamp"
    return raw


def _load_intraday(tf: TimeframeConfig) -> pd.DataFrame:
    df = _load_intraday_alpaca(tf)
    if df is None or df.empty:
        print(f"[{tf.label}] Alpaca returned nothing — falling back to Massive aggregates")
        df = _load_intraday_massive(tf)
    if df is None or df.empty:
        raise RuntimeError(f"No intraday data available for {tf.label} from Alpaca or Massive")
    return _restrict_regular_session(df)


def _load_intraday_alpaca(tf: TimeframeConfig) -> Optional[pd.DataFrame]:
    key, secret = _alpaca_keys()
    if not key or not secret:
        print(f"[{tf.label}] no Alpaca keys in environment — skipping Alpaca")
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed
    except Exception as exc:  # pragma: no cover - import guard
        print(f"[{tf.label}] alpaca-py import failed ({exc}) — skipping Alpaca")
        return None

    unit = TimeFrameUnit.Hour if tf.alpaca_unit == "Hour" else TimeFrameUnit.Minute
    timeframe = TimeFrame(amount=tf.alpaca_amount, unit=unit)
    client = StockHistoricalDataClient(key, secret)
    start = datetime.now(timezone.utc) - timedelta(days=365 * 8)
    req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=timeframe,
        start=start,
        feed=DataFeed.IEX,
    )
    try:
        bars = client.get_stock_bars(req)
    except Exception as exc:
        print(f"[{tf.label}] Alpaca request failed: {exc}")
        return None

    df = bars.df
    if df is None or len(df) == 0:
        return None
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(SYMBOL, level="symbol")
    df = df.rename(columns=str.lower)
    idx = pd.to_datetime(df.index)
    # Alpaca timestamps are UTC; convert to Eastern wall-clock, then drop tz.
    idx = idx.tz_convert(EASTERN).tz_localize(None)
    df.index = idx
    df.index.name = "timestamp"
    return df[["open", "high", "low", "close", "volume"]]


def _load_intraday_massive(tf: TimeframeConfig) -> Optional[pd.DataFrame]:
    api_key = os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print(f"[{tf.label}] no MASSIVE_API_KEY — cannot fall back to Massive")
        return None
    try:
        from massive import RESTClient
    except Exception as exc:  # pragma: no cover
        print(f"[{tf.label}] massive import failed ({exc})")
        return None

    client = RESTClient(api_key=api_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * 5)
    rows: List[dict] = []
    try:
        for bar in client.list_aggs(
            ticker=SYMBOL,
            multiplier=tf.massive_multiplier,
            timespan=tf.massive_timespan,
            from_=start.strftime("%Y-%m-%d"),
            to=end.strftime("%Y-%m-%d"),
            limit=50000,
        ):
            if bar.timestamp is None:
                continue
            ts = datetime.fromtimestamp(bar.timestamp / 1000.0, tz=timezone.utc)
            rows.append({
                "timestamp": ts,
                "open": bar.open, "high": bar.high, "low": bar.low,
                "close": bar.close, "volume": bar.volume,
            })
    except Exception as exc:
        print(f"[{tf.label}] Massive aggregates failed: {exc}")
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    idx = pd.to_datetime(df.index, utc=True).tz_convert(EASTERN).tz_localize(None)
    df.index = idx
    df.index.name = "timestamp"
    return df[["open", "high", "low", "close", "volume"]]


def _restrict_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-session bars [09:30, 16:00) Eastern (wall-clock index)."""
    t = df.index
    minutes = t.hour * 60 + t.minute
    open_m = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]
    close_m = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
    mask = (minutes >= open_m) & (minutes < close_m) & (t.dayofweek < 5)
    return df.loc[mask]


# ---------------------------------------------------------------------------
# Features: returns + volatility (close-to-close and Garman-Klass)
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame, tf: TimeframeConfig) -> pd.DataFrame:
    """Add log returns and three realized-vol estimators (+ session metadata).

    Realized-variance estimators, all over the same window `w`:
      yz_var / yz_vol  Yang-Zhang (DEFAULT realized variance downstream)
      gk_var / gk_vol  Garman-Klass (kept for the efficiency comparison)
      cc_vol           close-to-close (kept for the efficiency comparison)

    For intraday timeframes the first bar of each session has its log return set
    to NaN (overnight gap is not a within-session move). GK variance uses only a
    bar's own OHLC, so it is overnight-clean by construction.
    """
    out = df.copy()
    out["session_date"] = out.index.normalize()

    prev_close = out["close"].shift(1)
    log_ret = np.log(out["close"] / prev_close)
    if tf.intraday:
        overnight = out["session_date"].ne(out["session_date"].shift(1))
        log_ret = log_ret.mask(overnight)
    out["log_return"] = log_ret

    w = tf.vol_window_bars
    ann = tf.annualization

    # --- Garman-Klass per-bar variance (intrabar; overnight-clean) ---
    ln_hl = np.log(out["high"] / out["low"])
    ln_co = np.log(out["close"] / out["open"])
    gk_var = 0.5 * ln_hl**2 - (2.0 * math.log(2.0) - 1.0) * ln_co**2
    gk_var = gk_var.clip(lower=1e-12)
    out["gk_var"] = gk_var
    out["gk_vol"] = np.sqrt(gk_var.rolling(w, min_periods=w).mean()) * ann

    # --- Close-to-close vol (intraday masks the overnight bar to NaN, so relax
    #     min_periods there; daily keeps the strict full-window warmup) ---
    cc_min = w if not tf.intraday else max(2, int(w * 0.6))
    out["cc_vol"] = out["log_return"].rolling(w, min_periods=cc_min).std() * ann

    # --- Yang-Zhang (default) ---
    # Components: overnight o = ln(open/prev_close), open-to-close c = ln(close/open),
    # and per-bar Rogers-Satchell rs. k weights the open-to-close variance.
    o = np.log(out["open"] / prev_close)
    c = np.log(out["close"] / out["open"])
    rs = (np.log(out["high"] / out["close"]) * np.log(out["high"] / out["open"])
          + np.log(out["low"] / out["close"]) * np.log(out["low"] / out["open"]))
    k = 0.34 / (1.34 + (w + 1.0) / (w - 1.0))

    # yz_vol: the textbook windowed YZ estimator (demeaned variances of o and c,
    # mean of RS), one rolling pass -> w warmup. Used for the HMM and comparison.
    var_on = o.rolling(w, min_periods=w).var(ddof=1)
    var_oc = c.rolling(w, min_periods=w).var(ddof=1)
    var_rs = rs.rolling(w, min_periods=w).mean()
    yz_window_var = (var_on + k * var_oc + (1.0 - k) * var_rs).clip(lower=1e-12)
    out["yz_vol"] = np.sqrt(yz_window_var) * ann

    # yz_var: a PER-BAR Yang-Zhang variance contribution (overnight + open-to-close
    # demeaned-squared per spec, plus per-bar RS), so HAR has a daily RV series whose
    # rolling mean tracks the windowed YZ above. Used as the realized variance fed to
    # HAR / VRP / the short-vol proxy (the GK-replacement).
    o_dm = o - o.rolling(w, min_periods=w).mean()
    c_dm = c - c.rolling(w, min_periods=w).mean()
    yz_var = (o_dm**2 + k * c_dm**2 + (1.0 - k) * rs).clip(lower=1e-12)
    out["yz_var"] = yz_var

    return out


def vol_estimator_efficiency(feat: pd.DataFrame) -> Dict[str, float]:
    """Pairwise correlation + estimation-noise ratios for YZ / GK / close-to-close.

    Estimation noise is proxied by the std of the bar-over-bar change in each rolling
    vol series (a less efficient estimator jitters more). Ratios > 1 vs YZ mean the
    other estimator is noisier, i.e. YZ is the tighter (more efficient) estimator.
    """
    sub = feat[["yz_vol", "gk_vol", "cc_vol"]].dropna()
    yz_se = float(sub["yz_vol"].diff().std())
    gk_se = float(sub["gk_vol"].diff().std())
    cc_se = float(sub["cc_vol"].diff().std())
    return {
        "n": len(sub),
        "corr_yz_gk": float(sub["yz_vol"].corr(sub["gk_vol"])),
        "corr_yz_cc": float(sub["yz_vol"].corr(sub["cc_vol"])),
        "corr_gk_cc": float(sub["gk_vol"].corr(sub["cc_vol"])),
        "yz_se": yz_se, "gk_se": gk_se, "cc_se": cc_se,
        "gk_over_yz": gk_se / yz_se if yz_se > 0 else float("nan"),
        "cc_over_yz": cc_se / yz_se if yz_se > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Part 1: candle-state discretization
# ---------------------------------------------------------------------------

def add_candle_states(feat: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Add direction / body bucket / wick / state columns. Returns (frame, tercile_edges)."""
    out = feat.copy()
    o, h, l, c = out["open"], out["high"], out["low"], out["close"]

    out["direction"] = np.where(c > o, "up", "down")

    rng = (h - l)
    body = (c - o).abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        body_ratio = np.where(rng > 0, body / rng, np.nan)
    out["body_ratio"] = body_ratio

    valid = out["body_ratio"].dropna()
    q1, q2 = np.quantile(valid, [1.0 / 3.0, 2.0 / 3.0])
    edges = np.array([q1, q2])

    def _bucket(x: float) -> str:
        if not np.isfinite(x):
            return "medium"  # high == low: assign middle bucket
        if x <= q1:
            return "small"
        if x <= q2:
            return "medium"
        return "large"

    out["body_bucket"] = out["body_ratio"].apply(_bucket)

    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    out["wick"] = np.where(upper_wick > lower_wick, "upper", "lower")

    out["state"] = out["direction"] + "_" + out["body_bucket"] + "_" + out["wick"]
    return out, edges


def all_state_labels() -> List[str]:
    labels = []
    for d in ("up", "down"):
        for b in BODY_BUCKETS:
            for w in ("upper", "lower"):
                labels.append(f"{d}_{b}_{w}")
    return labels


# ---------------------------------------------------------------------------
# Next-bar direction target + transition validity
# ---------------------------------------------------------------------------

def add_next_direction(feat: pd.DataFrame, tf: TimeframeConfig) -> pd.DataFrame:
    """Target = sign of next bar's close-to-close return. Mark cross-session transitions invalid."""
    out = feat.copy()
    next_close = out["close"].shift(-1)
    next_ret = np.log(next_close / out["close"])
    out["next_ret"] = next_ret
    out["next_dir"] = np.where(next_ret > 0, "up", "down")

    if tf.intraday:
        next_session = out["session_date"].shift(-1)
        same_session = out["session_date"].eq(next_session)
        out["valid_trans"] = same_session & next_ret.notna()
    else:
        out["valid_trans"] = next_ret.notna()
    return out


def add_session_bucket(feat: pd.DataFrame) -> pd.DataFrame:
    """open = first hour, close = last hour, midday = everything between."""
    out = feat.copy()
    minutes = out.index.hour * 60 + out.index.minute
    open_start = SESSION_OPEN[0] * 60 + SESSION_OPEN[1]
    close_end = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
    bucket = np.full(len(out), "midday", dtype=object)
    bucket[minutes < open_start + 60] = "open"
    bucket[minutes >= close_end - 60] = "close"
    out["session_bucket"] = bucket
    return out


# ---------------------------------------------------------------------------
# Part 2 / 4: chi-square + Wilson CIs
# ---------------------------------------------------------------------------

@dataclass
class ChiResult:
    chi2: float
    p_value: float
    dof: int
    n: int
    baseline_p_up: float
    per_state: pd.DataFrame  # index=state; cols count, p_up, ci_low, ci_high
    label: str = ""


def run_chi_square(states: pd.Series, next_dir: pd.Series, *, label: str = "") -> Optional[ChiResult]:
    """State vs next-bar direction chi-square + per-state Wilson 95% CIs."""
    mask = states.notna() & next_dir.notna()
    s = states[mask].astype(str)
    d = next_dir[mask].astype(str)
    if len(s) < 2 or d.nunique() < 2:
        return None

    table = pd.crosstab(s, d).reindex(columns=["down", "up"]).fillna(0).astype(int)
    table = table[table.sum(axis=1) > 0]
    if table.shape[0] < 2:
        return None

    chi2, p, dof, _ = chi2_contingency(table.values)
    n = int(table.values.sum())
    baseline_p_up = float(table["up"].sum() / n)

    rows = []
    for state, row in table.iterrows():
        cnt = int(row.sum())
        n_up = int(row["up"])
        p_up = n_up / cnt if cnt > 0 else float("nan")
        lo, hi = proportion_confint(n_up, cnt, alpha=ALPHA, method="wilson") if cnt > 0 else (np.nan, np.nan)
        rows.append({"state": state, "count": cnt, "p_up": p_up, "ci_low": lo, "ci_high": hi})
    per_state = pd.DataFrame(rows).set_index("state").reindex(all_state_labels()).dropna(how="all")

    return ChiResult(
        chi2=float(chi2), p_value=float(p), dof=int(dof), n=n,
        baseline_p_up=baseline_p_up, per_state=per_state, label=label,
    )


def print_chi_result(res: ChiResult) -> None:
    print(f"\n  chi2={res.chi2:.3f}  p={res.p_value:.4g}  dof={res.dof}  n={res.n:,}  baseline P(up)={res.baseline_p_up:.4f}")
    disp = res.per_state.copy()
    disp["p_up"] = disp["p_up"].map(lambda x: f"{x:.4f}")
    disp["ci_low"] = disp["ci_low"].map(lambda x: f"{x:.4f}")
    disp["ci_high"] = disp["ci_high"].map(lambda x: f"{x:.4f}")
    print(disp.to_string())


def flag_ci_excludes_baseline(res: ChiResult) -> pd.DataFrame:
    """States whose Wilson CI for P(up) excludes the (regime) baseline P(up)."""
    base = res.baseline_p_up
    ps = res.per_state
    excl = ps[(ps["ci_high"] < base) | (ps["ci_low"] > base)].copy()
    excl["baseline_p_up"] = base
    return excl


# ---------------------------------------------------------------------------
# Part 3: HMM volatility regimes
# ---------------------------------------------------------------------------

@dataclass
class HMMResult:
    regimes: pd.Series              # index aligned to feature frame, values 0..2
    frame: pd.DataFrame             # feature frame restricted to decoded rows
    trans_matrix: np.ndarray        # 3x3 empirical, row-normalized
    per_regime: pd.DataFrame
    log_likelihood: float
    seed: int


def fit_hmm(feat: pd.DataFrame, tf: TimeframeConfig, *, n_seeds: int = 10, n_iter: int = 500) -> HMMResult:
    """3-state GaussianHMM on standardized [log_return, log(yz_vol)]; best of n_seeds restarts."""
    from hmmlearn.hmm import GaussianHMM

    cols = ["log_return", "yz_vol"]
    sub = feat.dropna(subset=cols).copy()
    obs = np.column_stack([sub["log_return"].to_numpy(), np.log(sub["yz_vol"].to_numpy())])
    mu = obs.mean(axis=0)
    sigma = obs.std(axis=0)
    sigma[sigma == 0] = 1.0
    z = (obs - mu) / sigma

    best_model = None
    best_ll = -np.inf
    best_seed = -1
    for seed in range(n_seeds):
        model = GaussianHMM(
            n_components=3, covariance_type="full",
            n_iter=n_iter, random_state=seed, tol=1e-4,
        )
        try:
            model.fit(z)
            ll = model.score(z)
        except Exception:
            continue
        if ll > best_ll:
            best_ll, best_model, best_seed = ll, model, seed
    if best_model is None:
        raise RuntimeError("HMM failed to fit on all seeds")

    raw_states = best_model.predict(z)

    # Relabel so regime 0 = lowest mean realized (Yang-Zhang) vol, 2 = highest.
    tmp = pd.DataFrame({"raw": raw_states, "yz_vol": sub["yz_vol"].to_numpy()})
    order = tmp.groupby("raw")["yz_vol"].mean().sort_values().index.tolist()
    remap = {raw: new for new, raw in enumerate(order)}
    regimes = pd.Series([remap[s] for s in raw_states], index=sub.index, name="regime")

    sub = sub.assign(regime=regimes.values)
    trans = _empirical_transition_matrix(regimes.to_numpy(), 3)
    per_regime = _per_regime_stats(sub, tf)

    return HMMResult(
        regimes=regimes, frame=sub, trans_matrix=trans,
        per_regime=per_regime, log_likelihood=float(best_ll), seed=best_seed,
    )


def _empirical_transition_matrix(seq: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        m[a, b] += 1.0
    row_sums = m.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return m / row_sums


def _avg_run_length(seq: np.ndarray, state: int) -> float:
    runs, cur = [], 0
    for s in seq:
        if s == state:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


def _per_regime_stats(sub: pd.DataFrame, tf: TimeframeConfig) -> pd.DataFrame:
    seq = sub["regime"].to_numpy()
    rows = []
    for r in range(3):
        m = sub[sub["regime"] == r]
        rets = m["log_return"].dropna()
        p_up = float((rets > 0).mean()) if len(rets) else float("nan")
        dur_bars = _avg_run_length(seq, r)
        rows.append({
            "regime": r,
            "n_days": int(len(m)),
            "mean_ret": float(rets.mean()) if len(rets) else float("nan"),
            "ann_vol": float(rets.std() * tf.annualization) if len(rets) else float("nan"),
            "p_up": p_up,
            "avg_duration_bars": dur_bars,
            "avg_duration_days": dur_bars / tf.bars_per_day,
        })
    return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Part 5: figures
# ---------------------------------------------------------------------------

def fig_state_heatmap(res: ChiResult, tf: TimeframeConfig) -> Path:
    states = all_state_labels()
    ps = res.per_state.reindex(states)
    p_up = ps["p_up"].to_numpy(dtype=float)
    mat = np.column_stack([p_up, 1.0 - p_up])

    fig, ax = plt.subplots(figsize=(5.5, 7.0), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.35, vmax=0.65)
    ax.set_xticks([0, 1], ["P(next up)", "P(next down)"])
    ax.set_yticks(range(len(states)), states)
    for i in range(len(states)):
        for j in range(2):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"SPY {tf.label}: candle state -> next-bar direction\n(baseline P(up)={res.baseline_p_up:.3f})")
    fig.colorbar(im, ax=ax, shrink=0.6, label="probability")
    path = OUTPUT_DIR / f"state_nextdir_heatmap_{tf.label}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_price_by_regime(hmm: HMMResult, tf: TimeframeConfig) -> Path:
    frame = hmm.frame
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    sc = ax.scatter(frame.index, frame["close"], c=frame["regime"], cmap="viridis", s=4)
    ax.set_title(f"SPY {tf.label} close colored by HMM volatility regime (0=low vol, 2=high vol)")
    ax.set_xlabel("date")
    ax.set_ylabel("close")
    cbar = fig.colorbar(sc, ax=ax, ticks=[0, 1, 2], shrink=0.8)
    cbar.set_label("regime")
    path = OUTPUT_DIR / f"price_by_regime_{tf.label}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def fig_transition_matrix(hmm: HMMResult, tf: TimeframeConfig) -> Path:
    m = hmm.trans_matrix
    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)
    im = ax.imshow(m, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(3), [f"to {i}" for i in range(3)])
    ax.set_yticks(range(3), [f"from {i}" for i in range(3)])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                    color="white" if m[i, j] > 0.5 else "black")
    ax.set_title(f"SPY {tf.label}: regime transition matrix")
    fig.colorbar(im, ax=ax, shrink=0.7, label="P(transition)")
    path = OUTPUT_DIR / f"regime_transition_{tf.label}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Per-timeframe driver
# ---------------------------------------------------------------------------

@dataclass
class TimeframeSummary:
    label: str
    n_obs: int
    uncond_p: float
    best_regime_p: float
    strongest_dev_pp: float
    strongest_dev_state: str
    mean_abs_next_ret: float
    net_edge_bp: float
    regime_vol_spread: float  # ann vol ratio high/low regime


def run_timeframe(label: str, *, refresh: bool = False) -> TimeframeSummary:
    tf = TIMEFRAMES[label]
    print("\n" + "=" * 78)
    print(f"TIMEFRAME = {label}  (bars/day={tf.bars_per_day}, vol window={tf.vol_window_bars} bars, "
          f"bars/year={tf.bars_per_year})")
    print("=" * 78)

    raw = load_data(tf, refresh=refresh)
    feat = compute_features(raw, tf)
    feat_full = feat.copy()
    # Drop only the vol-warmup rows. We deliberately keep rows whose log_return is
    # NaN (the first bar of each intraday session): their candle state and next-bar
    # target are still valid, and dropping them would erase the entire "open"
    # session bucket. The HMM drops NaN-return rows internally where it needs them.
    feat = feat.dropna(subset=["yz_vol", "gk_vol", "cc_vol"])
    print(f"[{label}] usable rows after feature/NaN drop: {len(feat):,}")
    if len(feat):
        print(f"[{label}] date range: {feat.index.min()}  ->  {feat.index.max()}")

    # Volatility estimator comparison (Yang-Zhang is the default downstream measure)
    eff = vol_estimator_efficiency(feat_full)
    print(f"\n--- Volatility estimators: Yang-Zhang (default) vs Garman-Klass vs close-to-close ---")
    print(f"  n={eff['n']:,}   corr(YZ,GK)={eff['corr_yz_gk']:.4f}  "
          f"corr(YZ,CC)={eff['corr_yz_cc']:.4f}  corr(GK,CC)={eff['corr_gk_cc']:.4f}")
    print(f"  rolling SE (jitter proxy):  YZ={eff['yz_se']:.5f}  GK={eff['gk_se']:.5f}  CC={eff['cc_se']:.5f}")
    print(f"  SE ratio vs YZ:  GK/YZ={eff['gk_over_yz']:.3f}x  CC/YZ={eff['cc_over_yz']:.3f}x  "
          f"(>1 => noisier than YZ)")

    # Part 1
    feat, edges = add_candle_states(feat)
    feat = add_next_direction(feat, tf)
    if tf.intraday:
        feat = add_session_bucket(feat)

    print("\n--- PART 1: candle-state frequency (body-ratio terciles "
          f"@ {edges[0]:.3f}, {edges[1]:.3f}) ---")
    freq = feat["state"].value_counts().reindex(all_state_labels()).fillna(0).astype(int)
    freq_tbl = pd.DataFrame({"count": freq, "pct": (freq / freq.sum() * 100).round(2)})
    print(freq_tbl.to_string())

    # Part 2 (unconditional). For intraday only valid (within-session) transitions.
    valid = feat[feat["valid_trans"]]
    print("\n--- PART 2: unconditional chi-square (state vs next-bar direction) ---")
    uncond = run_chi_square(valid["state"], valid["next_dir"], label=f"{label}/unconditional")
    if uncond is None:
        raise RuntimeError("unconditional chi-square could not be computed")
    print_chi_result(uncond)

    # Strongest state deviation from baseline (statistical + economic)
    dev = (uncond.per_state["p_up"] - uncond.baseline_p_up)
    strongest_state = dev.abs().idxmax()
    strongest_dev = float(dev.loc[strongest_state])
    mean_abs_next_ret = float(valid["next_ret"].abs().mean())
    # Economic proxy: shifting P(up) by |d| relative to baseline moves expected
    # per-trade return by ~2*|d|*E|next move|. Compare to a 2bp round-trip cost.
    gross_edge_bp = 2.0 * abs(strongest_dev) * mean_abs_next_ret * 1e4
    net_edge_bp = gross_edge_bp - ROUND_TRIP_COST_BP
    print(f"\n  strongest deviation: {strongest_state}  "
          f"P(up)={uncond.per_state.loc[strongest_state, 'p_up']:.4f}  "
          f"dev={strongest_dev * 100:+.2f} pp")
    print(f"  mean |next-bar return| = {mean_abs_next_ret * 1e4:.1f} bp")
    print(f"  gross edge proxy = {gross_edge_bp:.2f} bp  ->  net of {ROUND_TRIP_COST_BP:.0f}bp cost = {net_edge_bp:+.2f} bp")

    # Intraday: pooled vs per session bucket (time-of-day confound)
    if tf.intraday:
        print("\n--- PART 2b: intraday session-bucket chi-square (time-of-day confound) ---")
        print("  [pooled] (same as Part 2 above):"
              f" chi2={uncond.chi2:.3f} p={uncond.p_value:.4g}")
        for bucket in ("open", "midday", "close"):
            sub = valid[valid["session_bucket"] == bucket]
            res = run_chi_square(sub["state"], sub["next_dir"], label=f"{label}/{bucket}")
            if res is None:
                why = "no within-session successors" if len(sub) == 0 else "too few/degenerate"
                print(f"  [{bucket:6}] n={len(sub):,} ({why}) — chi-square skipped")
                continue
            print(f"  [{bucket:6}] chi2={res.chi2:.3f} p={res.p_value:.4g} "
                  f"n={res.n:,} baseline P(up)={res.baseline_p_up:.4f}")

    # Part 3: HMM
    print("\n--- PART 3: HMM volatility regimes (3-state GaussianHMM) ---")
    hmm = fit_hmm(feat, tf)
    print(f"  best log-likelihood={hmm.log_likelihood:.1f} (seed={hmm.seed})")
    pr = hmm.per_regime.copy()
    disp = pr.copy()
    disp["mean_ret"] = disp["mean_ret"].map(lambda x: f"{x:+.6f}")
    disp["ann_vol"] = disp["ann_vol"].map(lambda x: f"{x:.4f}")
    disp["p_up"] = disp["p_up"].map(lambda x: f"{x:.4f}")
    disp["avg_duration_bars"] = disp["avg_duration_bars"].map(lambda x: f"{x:.1f}")
    disp["avg_duration_days"] = disp["avg_duration_days"].map(lambda x: f"{x:.2f}")
    print(disp.to_string())
    print("\n  regime transition matrix (rows=from, cols=to):")
    print(np.array2string(hmm.trans_matrix, precision=3, suppress_small=True))
    vol_lo = float(hmm.per_regime.loc[0, "ann_vol"])
    vol_hi = float(hmm.per_regime.loc[2, "ann_vol"])
    regime_vol_spread = vol_hi / vol_lo if vol_lo > 0 else float("nan")

    # Part 4: regime-conditioned chi-square
    print("\n--- PART 4: regime-conditioned chi-square ---")
    feat_reg = feat.join(hmm.regimes, how="inner")
    feat_reg = feat_reg[feat_reg["valid_trans"]]
    regime_results: List[ChiResult] = []
    total_state_tests = 0
    total_flagged = 0
    for r in range(3):
        sub = feat_reg[feat_reg["regime"] == r]
        res = run_chi_square(sub["state"], sub["next_dir"], label=f"{label}/regime{r}")
        if res is None:
            print(f"\n  [regime {r}] insufficient data")
            continue
        regime_results.append(res)
        print(f"\n  [regime {r}]  chi2={res.chi2:.3f}  p={res.p_value:.4g}  "
              f"dof={res.dof}  n={res.n:,}  baseline P(up)={res.baseline_p_up:.4f}")
        flagged = flag_ci_excludes_baseline(res)
        total_state_tests += len(res.per_state)
        total_flagged += len(flagged)
        if len(flagged):
            ft = flagged[["count", "p_up", "ci_low", "ci_high", "baseline_p_up"]].copy()
            for col in ("p_up", "ci_low", "ci_high", "baseline_p_up"):
                ft[col] = ft[col].map(lambda x: f"{x:.4f}")
            print(f"    CI excludes regime baseline P(up) in {len(flagged)} state(s):")
            print(ft.to_string().replace("\n", "\n    "))
        else:
            print("    no state CI excludes the regime baseline P(up)")

    expected_by_chance = ALPHA * total_state_tests
    print(f"\n  multiple-comparisons context: {total_flagged} flagged across "
          f"{total_state_tests} state-regime tests; "
          f"~{expected_by_chance:.1f} expected by chance at alpha={ALPHA}")

    best_regime_p = min((r.p_value for r in regime_results), default=float("nan"))

    # Unconditional vs per-regime contrast
    print("\n--- Part 2 vs Part 4 contrast (chi-square p-values) ---")
    print(f"  unconditional        p = {uncond.p_value:.4g}")
    for r in regime_results:
        print(f"  {r.label:22} p = {r.p_value:.4g}")

    # Part 5: figures
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = fig_state_heatmap(uncond, tf)
    p2 = fig_price_by_regime(hmm, tf)
    p3 = fig_transition_matrix(hmm, tf)
    print(f"\n--- PART 5: figures saved ---\n  {p1.name}\n  {p2.name}\n  {p3.name}")

    # Plain-English summary
    print("\n--- SUMMARY (" + label + ") ---")
    _print_summary_block(uncond, best_regime_p, regime_vol_spread, net_edge_bp)

    return TimeframeSummary(
        label=label, n_obs=uncond.n, uncond_p=uncond.p_value,
        best_regime_p=best_regime_p, strongest_dev_pp=strongest_dev * 100,
        strongest_dev_state=strongest_state, mean_abs_next_ret=mean_abs_next_ret,
        net_edge_bp=net_edge_bp, regime_vol_spread=regime_vol_spread,
    )


def _print_summary_block(uncond: ChiResult, best_regime_p: float,
                         regime_vol_spread: float, net_edge_bp: float) -> None:
    dir_sig = "significant" if uncond.p_value < ALPHA else "NOT significant"
    reg_sig = "significant" if (best_regime_p == best_regime_p and best_regime_p < ALPHA) else "NOT significant"
    lines = [
        f"Unconditional candle->direction chi-square p = {uncond.p_value:.4g} ({dir_sig} at {ALPHA}).",
        f"Best per-regime chi-square p = {best_regime_p:.4g} ({reg_sig}).",
        f"HMM separated vol regimes by a factor of {regime_vol_spread:.2f}x "
        f"(high-vol regime annualized vol / low-vol regime).",
    ]
    if regime_vol_spread > 2.0 and uncond.p_value > ALPHA:
        verdict = ("Volatility-regime STRUCTURE is far stronger than directional structure: "
                   "candles barely move next-bar direction, but the vol regimes are large, "
                   "persistent, and cleanly separated. The tradable signal here (if any) is "
                   "in volatility state, not candle-implied direction.")
    elif uncond.p_value < ALPHA and net_edge_bp > 0:
        verdict = ("Directional structure is statistically significant AND survives the 2bp "
                   "round-trip cost proxy — worth a proper out-of-sample test before believing it.")
    elif uncond.p_value < ALPHA:
        verdict = ("Directional structure is statistically significant but does NOT survive the "
                   "2bp cost proxy — real but economically negligible (classic in-sample mirage).")
    else:
        verdict = ("Neither candle direction nor (necessarily) regime shows convincing structure "
                   "at this timeframe; treat as noise.")
    lines.append(verdict)
    for ln in lines:
        print("  " + ln)


# ---------------------------------------------------------------------------
# Cross-timeframe comparison
# ---------------------------------------------------------------------------

def print_comparison(summaries: List[TimeframeSummary]) -> None:
    if not summaries:
        return
    print("\n" + "=" * 78)
    print("CROSS-TIMEFRAME COMPARISON (statistical vs economic significance)")
    print("=" * 78)
    rows = []
    for s in summaries:
        gross = 2.0 * abs(s.strongest_dev_pp / 100.0) * s.mean_abs_next_ret * 1e4
        rows.append({
            "timeframe": s.label,
            "n_obs": s.n_obs,
            "uncond_chi2_p": f"{s.uncond_p:.4g}",
            "strongest_dev_pp": f"{s.strongest_dev_pp:+.2f}",
            "dev_state": s.strongest_dev_state,
            "gross_edge_bp": f"{gross:.2f}",
            "net_edge_bp(@2bp)": f"{s.net_edge_bp:+.2f}",
            "regime_vol_spread": f"{s.regime_vol_spread:.2f}x",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  Read: a small p-value with a negative net_edge_bp means the pattern is")
    print("  statistically real but economically dead after a 2bp round-trip cost.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_dotenv_into_environ()
    parser = argparse.ArgumentParser(description="Candle-state / HMM-regime statistical audit for SPY.")
    parser.add_argument(
        "--timeframes", nargs="+", default=["1d"],
        choices=list(TIMEFRAMES.keys()),
        help="Which timeframes to run (default: 1d).",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cached parquet and re-download.")
    args = parser.parse_args()

    summaries: List[TimeframeSummary] = []
    for label in args.timeframes:
        try:
            summaries.append(run_timeframe(label, refresh=args.refresh))
        except Exception as exc:
            print(f"\n[{label}] FAILED: {type(exc).__name__}: {exc}")

    print_comparison(summaries)


if __name__ == "__main__":
    main()
