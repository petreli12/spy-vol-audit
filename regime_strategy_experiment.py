"""regime_strategy_experiment.py

Does volatility-regime gating of a short-vol position survive *causal* decoding?

The VRP audit (vrp_experiment.py) showed short vol is a positive-carry, fat-left-
tail trade whose premium concentrates in (and whose losses explode in) the high-
vol regime. The obvious idea: turn the position OFF when we are in the high-vol
regime. But that earlier regime label was an IN-SAMPLE Viterbi path — it peeks at
the future. This script asks whether the idea still works when the regime must be
decoded online, with a filter that only sees the past.

Design:
  PART 1  causal regime filter: walk-forward HMM (refit every 63 trading days,
          expanding window, 3y burn-in), FILTERED probabilities via the forward
          algorithm (no future leakage), vs in-sample Viterbi (agreement + lag).
  PART 2  regime-gated short-vol proxy: hold only when filtered P(high-vol) < thr;
          regime-dependent friction (3x in high-vol) charged on EVERY gate entry/
          exit and on scheduled monthly rolls (whipsaw is real).
  PART 3  honesty: deflated Sharpe for the best of 4 thresholds, refit-cadence
          sensitivity (21/126d), cumulative-P&L plot with flat-gate shading.
  PART 4  plain-English verdict.

NOT a backtest of a tradable system: the short-vol proxy is the same stylized,
costless-upper-bound variance position from vrp_experiment (now with regime-
dependent friction). The HMM is the only thing made causal here.

Run:  python regime_strategy_experiment.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.special import logsumexp  # noqa: E402
from scipy.stats import kurtosis, multivariate_normal, norm, skew  # noqa: E402

# Reused utilities (do not duplicate)
from candle_markov_experiment import (  # noqa: E402
    DATA_DIR,
    OUTPUT_DIR,
    SYMBOL,
    TIMEFRAMES,
    compute_features,
    fit_hmm,
    load_data,
)
from vrp_experiment import (  # noqa: E402
    BURN_IN_YEARS,
    HIGH_VOL_REGIME,
    HORIZON,
    TRADING_DAYS_PER_YEAR,
    _assumed_friction,
    load_spy_vix,
    max_drawdown_additive,
    sharpe,
    short_vol_proxy,
)

REFIT_EVERY = 63           # trading days (~quarterly)
N_SEEDS = 10
N_ITER = 500
ROLL_DAYS = HORIZON        # variance horizon for friction notional
THRESHOLDS = (0.2, 0.4, 0.6, 0.8)
EULER_GAMMA = 0.5772156649

# Full available history. SPY listed 1993-01; Yahoo ^VIX backfills the *new*
# methodology to 1990. The joint sample therefore begins ~1993, and with a 3y
# burn-in the strategy track record starts ~1996.
#
# CAVEAT (data quality): the modern VIX methodology only became official in 2003;
# Yahoo's pre-2003 ^VIX is a back-computed series, and the original real-time
# index of that era (now ^VXO) used a different (8% OTM, Black-Scholes) formula.
# Early-90s equity market structure (decimalization came in 2001, pre-electronic
# options) also differs from today. We therefore explicitly re-test the headline
# conclusion on post-2000 data only and flag if it depends on pre-2000 history.
FULL_START = "1993-01-01"
PRE2000_FLAG_DATE = "2000-01-01"
HOLDOUT_SPLIT = "2016-01-01"   # select threshold on <split, test on >=split

# Per-crisis windows (the centerpiece). (name, start, end, is_oos_of_2015_split)
CRISES: Tuple[Tuple[str, str, str], ...] = (
    ("GFC 2008-09",      "2008-09-01", "2009-03-31"),
    ("Flash crash",      "2010-05-01", "2010-05-31"),
    ("US downgrade",     "2011-08-01", "2011-08-31"),
    ("Aug 2015",         "2015-08-01", "2015-08-31"),
    ("Volmageddon",      "2018-02-01", "2018-02-28"),
    ("Q4 2018",          "2018-10-01", "2018-12-31"),
    ("COVID",            "2020-02-01", "2020-04-30"),
    ("2022 full year",   "2022-01-01", "2022-12-31"),
)


# ===========================================================================
# Full-history data loader (SPY + ^VIX from 1993)
# ===========================================================================

def _load_vix_full(start: str) -> pd.Series:
    import yfinance as yf

    raw = yf.download("^VIX", start=start, interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no ^VIX data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw["Close"].copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = "timestamp"
    return close.rename("vix")


def load_full_history(*, start: str = FULL_START, refresh: bool = False) -> pd.DataFrame:
    """SPY daily OHLCV (+ YZ/GK features) joined with ^VIX, from `start` to now.

    Returns a date-indexed frame carrying compute_features() columns (log_return,
    yz_var/yz_vol, gk_var/gk_vol, cc_vol) plus vix and vix_dec. Cached to parquet;
    independent of the 10-year cache the other scripts use.
    """
    import yfinance as yf

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{SYMBOL}_VIX_1d_full.parquet"
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        print(f"loaded {len(df):,} cached full-history SPY+VIX rows from {cache.name}")
        return df

    raw = yf.download(SYMBOL, start=start, interval="1d",
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no SPY data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "timestamp"
    raw = raw[["open", "high", "low", "close", "volume"]]

    feat = compute_features(raw, TIMEFRAMES["1d"])
    vix = _load_vix_full(start)
    df = feat.join(vix, how="inner")
    df = df.dropna(subset=["vix", "yz_var", "gk_var"])
    df["vix_dec"] = df["vix"] / 100.0
    df.to_parquet(cache)
    print(f"built {len(df):,} full-history SPY+VIX rows -> cached to {cache.name}")
    return df


# ===========================================================================
# PART 1: causal walk-forward HMM filter
# ===========================================================================

def _fit_hmm_model(z: np.ndarray, *, n_seeds: int = N_SEEDS, n_iter: int = N_ITER):
    """Fit a 3-state full-cov GaussianHMM, best of n_seeds restarts (same protocol)."""
    from hmmlearn.hmm import GaussianHMM

    best, best_ll = None, -np.inf
    for seed in range(n_seeds):
        m = GaussianHMM(n_components=3, covariance_type="full",
                        n_iter=n_iter, random_state=seed, tol=1e-4)
        try:
            m.fit(z)
            ll = m.score(z)
        except Exception:
            continue
        if ll > best_ll:
            best_ll, best = ll, m
    return best


def _forward_filter(z: np.ndarray, model) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled forward algorithm in log-space -> FILTERED probs P(state_t | obs_1..t).

    Returns (filt[n,k], order) where order sorts states ascending by mean log-vol
    (feature 1), so order[-1] is the high-vol state column.
    """
    k = model.n_components
    n = z.shape[0]
    emis = np.empty((n, k))
    for j in range(k):
        emis[:, j] = multivariate_normal.logpdf(
            z, mean=model.means_[j], cov=model.covars_[j], allow_singular=True)
    log_trans = np.log(np.clip(model.transmat_, 1e-12, None))
    log_start = np.log(np.clip(model.startprob_, 1e-12, None))

    filt = np.empty((n, k))
    la = log_start + emis[0]
    la -= logsumexp(la)
    filt[0] = np.exp(la)
    log_prev = la
    for t in range(1, n):
        pred = logsumexp(log_prev[:, None] + log_trans, axis=0)  # sum over i -> j
        la = pred + emis[t]
        la -= logsumexp(la)
        filt[t] = np.exp(la)
        log_prev = la

    order = np.argsort(model.means_[:, 1])  # ascending mean log-vol
    return filt, order


@dataclass
class CausalFilter:
    p_high: pd.Series       # filtered P(high-vol regime), OOS dates only
    hard: pd.Series         # filtered argmax regime (0..2, mapped by vol)
    n_refits: int


def walk_forward_filter(feat: pd.DataFrame, *, refit_every: int = REFIT_EVERY,
                        n_seeds: int = N_SEEDS, n_iter: int = N_ITER,
                        burn_in_years: int = BURN_IN_YEARS,
                        cache: bool = True) -> CausalFilter:
    """Refit the HMM on an expanding window; emit causal FILTERED regime probs.

    For each refit segment the model is fixed; the forward filter is re-run over
    the full history with that model (re-decoding the past under current params)
    and read off only within the segment — so every value uses data <= its date.

    Deterministic given the fixed restart seeds, so the result is cached to parquet
    keyed by (cadence, sample span, n rows) to make reruns / sensitivity cheap.
    """
    cols = ["log_return", "yz_vol"]
    sub = feat.dropna(subset=cols).copy()
    cache_path = None
    if cache:
        tag = f"{refit_every}d_{len(sub)}_{sub.index[0].date()}_{sub.index[-1].date()}"
        cache_path = DATA_DIR / f"causal_filter_{tag}.parquet"
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            n_refits = int(cached.attrs.get("n_refits", 0)) if cached.attrs else 0
            return CausalFilter(p_high=cached["p_high"],
                                hard=cached["hard"].astype(int),
                                n_refits=n_refits or len(cached) // max(refit_every, 1))
    obs = np.column_stack([sub["log_return"].to_numpy(), np.log(sub["yz_vol"].to_numpy())])
    idx = sub.index
    n = len(sub)

    start_dt = idx[0] + pd.DateOffset(years=burn_in_years)
    burn = int(np.searchsorted(idx.values, np.datetime64(start_dt)))
    refit_pts = list(range(burn, n, refit_every))

    p_high = np.full(n, np.nan)
    hard = np.full(n, np.nan)
    n_refits = 0
    for si, r in enumerate(refit_pts):
        seg_end = refit_pts[si + 1] if si + 1 < len(refit_pts) else n
        train = obs[:r]
        mu = train.mean(axis=0)
        sd = train.std(axis=0)
        sd[sd == 0] = 1.0
        model = _fit_hmm_model((train - mu) / sd, n_seeds=n_seeds, n_iter=n_iter)
        if model is None:
            continue
        n_refits += 1
        filt, order = _forward_filter((obs[:seg_end] - mu) / sd, model)
        inv = np.empty(model.n_components, dtype=int)
        inv[order] = np.arange(model.n_components)
        hi_col = order[-1]
        for t in range(r, seg_end):
            p_high[t] = filt[t, hi_col]
            hard[t] = inv[int(np.argmax(filt[t]))]

    out = pd.DataFrame({"p_high": p_high, "hard": hard}, index=idx).dropna()
    if cache_path is not None:
        to_save = out.copy()
        to_save.attrs["n_refits"] = n_refits
        to_save.to_parquet(cache_path)
    return CausalFilter(p_high=out["p_high"], hard=out["hard"].astype(int), n_refits=n_refits)


def filter_vs_viterbi(causal: CausalFilter, viterbi: pd.Series,
                      *, search_window: int = REFIT_EVERY) -> Dict[str, float]:
    """Agreement rate + average entry lag (days) of the causal filter behind Viterbi."""
    vit = viterbi.reindex(causal.hard.index).dropna().astype(int)
    common = causal.hard.index.intersection(vit.index)
    ch = causal.hard.reindex(common)
    vt = vit.reindex(common)
    agreement = float((ch == vt).mean())

    vit_high = (vt == HIGH_VOL_REGIME)
    entries = vit_high & ~vit_high.shift(1, fill_value=False)
    causal_high = (ch == HIGH_VOL_REGIME).to_numpy()
    pos = {d: i for i, d in enumerate(common)}
    lags: List[int] = []
    found = 0
    entry_dates = list(common[entries.to_numpy()])
    for d in entry_dates:
        i = pos[d]
        hit = None
        for j in range(i, min(i + search_window, len(common))):
            if causal_high[j]:
                hit = j - i
                break
        if hit is not None:
            lags.append(hit)
            found += 1
    return {
        "agreement": agreement,
        "n_entries": len(entry_dates),
        "n_matched": found,
        "avg_lag_days": float(np.mean(lags)) if lags else float("nan"),
        "median_lag_days": float(np.median(lags)) if lags else float("nan"),
    }


# ===========================================================================
# PART 2: regime-gated short-vol proxy with regime-dependent friction
# ===========================================================================

@dataclass
class GateResult:
    label: str
    threshold: Optional[float]
    gross: pd.Series          # gated P&L, no friction
    net: pd.Series            # gated P&L, friction-adjusted
    want_in: pd.Series
    sharpe_costless: float
    sharpe_friction: float
    max_drawdown: float
    pct_in_market: float
    covid_pnl: float
    bear_pnl: float
    n_trips: int = 0              # gate-caused position flips (excl. initial entry)
    whipsaw_friction: float = 0.0  # friction paid at gate flips (false-alarm cost)
    roll_friction: float = 0.0     # friction paid at scheduled monthly rolls


def simulate_gate(raw_pnl: pd.Series, iv_entry: pd.Series, want_in: pd.Series,
                  regime_high: pd.Series, base_rt: float, *,
                  label: str, threshold: Optional[float]) -> GateResult:
    """Day-by-day gated short-vol P&L with regime-dependent transaction friction.

    - Hold the position on days where want_in is True; flat otherwise.
    - One-way friction = 0.5 * base_rt * mult * IV_entry * ROLL_DAYS, where mult=3
      when that day's filtered regime is high-vol (spreads blow out), else 1.
    - Charge one-way friction on EVERY entry and EVERY exit (whipsaw is real), and
      a full round trip (2 one-way) on each scheduled monthly roll while staying in.
    """
    idx = raw_pnl.index
    wi = want_in.reindex(idx).fillna(False).to_numpy().astype(bool)
    rp = raw_pnl.to_numpy()
    iv = iv_entry.reindex(idx).to_numpy()
    hi = regime_high.reindex(idx).fillna(False).to_numpy().astype(bool)
    months = idx.to_period("M")

    gross = np.zeros(len(idx))
    fric = np.zeros(len(idx))
    prev_in = False
    last_roll_month = None
    n_trips = 0
    whipsaw = 0.0
    roll = 0.0
    for t in range(len(idx)):
        in_now = bool(wi[t])
        if in_now:
            gross[t] = rp[t]
        mult = 3.0 if hi[t] else 1.0
        one_way = 0.5 * base_rt * mult * float(iv[t]) * ROLL_DAYS
        if t == 0:
            if in_now:                          # initial entry (not a gate trip)
                fric[t] += one_way
                last_roll_month = months[t]
            prev_in = in_now
            continue
        if in_now != prev_in:
            fric[t] += one_way                  # gate-caused entry or exit
            whipsaw += one_way
            n_trips += 1
            if in_now:
                last_roll_month = months[t]     # entry resets the roll clock
        elif in_now and months[t] != last_roll_month:
            fric[t] += 2.0 * one_way            # scheduled monthly roll (round trip)
            roll += 2.0 * one_way
            last_roll_month = months[t]
        prev_in = in_now

    gross_s = pd.Series(gross, index=idx)
    net_s = pd.Series(gross - fric, index=idx)
    return GateResult(
        label=label, threshold=threshold,
        gross=gross_s, net=net_s, want_in=pd.Series(wi, index=idx),
        sharpe_costless=sharpe(gross_s), sharpe_friction=sharpe(net_s),
        max_drawdown=max_drawdown_additive(net_s),
        pct_in_market=float(wi.mean() * 100.0),
        covid_pnl=float(net_s.loc["2020-03-01":"2020-03-31"].sum()),
        bear_pnl=float(net_s.loc["2022-01-01":"2022-12-31"].sum()),
        n_trips=n_trips, whipsaw_friction=whipsaw, roll_friction=roll,
    )


def iv_entry_series(df: pd.DataFrame) -> pd.Series:
    """Monthly-locked implied variance (VIX^2), the friction notional base."""
    implied_var = df["vix_dec"] ** 2
    return implied_var.groupby(df.index.to_period("M")).transform("first")


def warning_lag_days(want_in: pd.Series, start: str, end: str,
                     lookback: int = 10) -> Optional[int]:
    """Trading days from window start to the gate's first flat day (filter first
    crossing above threshold). Negative => the filter warned BEFORE the window
    began; None => the gate never went flat inside the window.
    """
    idx = want_in.index
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    start_pos = int(idx.searchsorted(s))
    if start_pos >= len(idx):
        return None
    lo = max(0, start_pos - lookback)
    flat = (~want_in.astype(bool)).to_numpy()
    hi = int(idx.searchsorted(e, side="right"))
    for j in range(lo, min(hi, len(idx))):
        if flat[j]:
            return j - start_pos
    return None


def count_flat_excursions(want_in: pd.Series) -> int:
    """Number of distinct flat episodes (separate filter excursions above threshold)."""
    flat = (~want_in.astype(bool)).to_numpy()
    return int(np.sum(flat & ~np.r_[False, flat[:-1]]))


def _crisis_is_oos(name: str) -> bool:
    """True if the crisis window starts on/after the holdout split (genuine OOS)."""
    for cname, start, _ in CRISES:
        if cname == name:
            return pd.Timestamp(start) >= pd.Timestamp(HOLDOUT_SPLIT)
    return False


def crisis_table(always: GateResult, gated: GateResult) -> List[dict]:
    """Per-crisis: always-on P&L, gated P&L, % days gate flat, warning lag (days)."""
    rows: List[dict] = []
    for name, start, end in CRISES:
        a = float(always.net.loc[start:end].sum())
        g = float(gated.net.loc[start:end].sum())
        wi = gated.want_in.loc[start:end]
        pct_flat = float((~wi.astype(bool)).mean() * 100.0) if len(wi) else float("nan")
        lag = warning_lag_days(gated.want_in, start, end)
        rows.append({"crisis": name, "always": a, "gated": g,
                     "pct_flat": pct_flat, "lag": lag, "n": len(wi)})
    return rows


def slice_perf(net: pd.Series, want_in: pd.Series, start=None, end=None) -> dict:
    """Sharpe / maxDD / %in-market of a (date-sliced) gated net P&L series."""
    n = net.loc[start:end]
    wi = want_in.loc[start:end]
    return {
        "sharpe": sharpe(n), "maxdd": max_drawdown_additive(n),
        "pct_in": float(wi.astype(bool).mean() * 100.0) if len(wi) else float("nan"),
        "n": len(n), "total_pnl": float(n.sum()),
    }


# ===========================================================================
# PART 3: honesty checks
# ===========================================================================

def deflated_sharpe(best_net: pd.Series, all_perperiod_sr: List[float],
                    n_trials: int) -> Dict[str, float]:
    """Bailey & Lopez de Prado deflated Sharpe: P(true SR>0) after N trials.

    Works in per-period (daily) Sharpe units; accounts for non-normal returns
    (skew, kurtosis) and the inflation of the best of N candidate Sharpes.
    """
    r = best_net.dropna().to_numpy()
    T = len(r)
    sr = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # non-excess (normal=3)

    var_sr = float(np.var(all_perperiod_sr, ddof=1)) if len(all_perperiod_sr) > 1 else 0.0
    sr_std = math.sqrt(var_sr)
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr0 = sr_std * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)  # expected max under null

    denom = math.sqrt(max(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr, 1e-9))
    dsr = float(norm.cdf((sr - sr0) * math.sqrt(max(T - 1, 1)) / denom))
    ann = math.sqrt(TRADING_DAYS_PER_YEAR)
    return {
        "sr_ann": sr * ann, "sr0_ann_hurdle": sr0 * ann,
        "deflated_sharpe_prob": dsr, "skew": g3, "kurtosis": g4, "T": T,
    }


def fig_cumulative(always: GateResult, best: GateResult) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.plot(always.net.index, always.net.cumsum().to_numpy(),
            label="always-on (net)", color="tab:gray", lw=1.3)
    ax.plot(best.net.index, best.net.cumsum().to_numpy(),
            label=f"gated thr={best.threshold} (net)", color="tab:blue", lw=1.4)
    # Shade flat (out-of-market) stretches of the gated strategy.
    wi = best.want_in.to_numpy().astype(bool)
    idx = best.want_in.index
    t = 0
    labeled = False
    while t < len(idx):
        if not wi[t]:
            s = t
            while t < len(idx) and not wi[t]:
                t += 1
            ax.axvspan(idx[s], idx[t - 1], color="tab:red", alpha=0.12,
                       label="_g" if labeled else "gate flat")
            labeled = True
        else:
            t += 1
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_title("Short-vol cumulative P&L (net of regime-dependent friction): always-on vs gated")
    ax.set_xlabel("date"); ax.set_ylabel("cumulative P&L (variance units)")
    ax.legend(loc="upper left")
    path = OUTPUT_DIR / "regime_gated_cumulative.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


# ===========================================================================
# main
# ===========================================================================

def _build_inputs(full: bool = True):
    tf = TIMEFRAMES["1d"]
    if full:
        df = load_full_history()
        feat = df  # already carries compute_features() columns
    else:
        df = load_spy_vix()
        feat = compute_features(load_data(tf), tf)
    feat = feat.dropna(subset=["log_return", "yz_vol"])
    raw_pnl = short_vol_proxy(df, "yz_var")
    iv_entry = iv_entry_series(df)
    return df, tf, feat, raw_pnl, iv_entry


def _sweep(raw_pnl, iv_entry, causal: CausalFilter, base_rt: float) -> Dict:
    idx = causal.p_high.index
    rp = raw_pnl.reindex(idx).dropna()
    idx = rp.index
    regime_high = (causal.hard.reindex(idx) == HIGH_VOL_REGIME)
    ivx = iv_entry.reindex(idx)

    results: Dict[str, GateResult] = {}
    always_want = pd.Series(True, index=idx)
    results["always_on"] = simulate_gate(rp, ivx, always_want, regime_high, base_rt,
                                         label="always_on", threshold=None)
    p_high = causal.p_high.reindex(idx)
    for thr in THRESHOLDS:
        # Lag by one day: the position held on day t is set at the prior close,
        # so it may only use the filter conditioned on data through t-1. Using
        # p_high[t] would let the gate dodge day-t's loss with day-t's own vol.
        want = (p_high < thr).shift(1, fill_value=False)
        results[f"thr_{thr}"] = simulate_gate(rp, ivx, want, regime_high, base_rt,
                                              label=f"thr_{thr}", threshold=thr)
    return results


def main() -> None:
    print("=" * 78)
    print("REGIME-GATED SHORT-VOL — does the edge survive CAUSAL regime decoding?")
    print("=" * 78)

    df, tf, feat, raw_pnl, iv_entry = _build_inputs()
    print(f"date range: {df.index.min().date()} -> {df.index.max().date()}  ({len(df):,} days)")
    base_rt = _assumed_friction()["iron_condor"].cost_pct_of_credit / 100.0
    print(f"base round-trip friction = {base_rt*100:.1f}% of credit (FrictionConfig), 3x in high-vol")

    # --- PART 1 ---
    print("\n--- PART 1: causal regime filter (forward algorithm) vs in-sample Viterbi ---")
    causal = walk_forward_filter(feat, refit_every=REFIT_EVERY)
    print(f"  walk-forward HMM: {causal.n_refits} refits (every {REFIT_EVERY}d, expanding, "
          f"{BURN_IN_YEARS}y burn-in); causal days = {len(causal.p_high):,}")
    viterbi = fit_hmm(feat, tf).regimes
    diag = filter_vs_viterbi(causal, viterbi)
    print(f"  agreement (causal hard vs Viterbi)      = {diag['agreement']*100:.1f}%")
    print(f"  high-vol entries (Viterbi)              = {diag['n_entries']} "
          f"(causal matched {diag['n_matched']})")
    print(f"  avg lag, Viterbi->causal high-vol entry = {diag['avg_lag_days']:.1f} days "
          f"(median {diag['median_lag_days']:.0f}) — the cost of causality")

    # --- PART 2 ---
    print("\n--- PART 2: regime-gated short-vol proxy (threshold sweep) ---")
    results = _sweep(raw_pnl, iv_entry, causal, base_rt)
    always = results["always_on"]
    print(f"  {'strategy':12} {'Sharpe_cl':>10} {'Sharpe_fr':>10} {'maxDD':>9} "
          f"{'%inmkt':>8} {'2020-03':>10} {'2022':>9}")
    for key in ["always_on"] + [f"thr_{t}" for t in THRESHOLDS]:
        r = results[key]
        print(f"  {r.label:12} {r.sharpe_costless:>10.2f} {r.sharpe_friction:>10.2f} "
              f"{r.max_drawdown:>9.3f} {r.pct_in_market:>8.1f} {r.covid_pnl:>+10.3f} {r.bear_pnl:>+9.3f}")

    # Best gated threshold by friction-adjusted Sharpe (full sample).
    thr_keys = [f"thr_{t}" for t in THRESHOLDS]
    best_key = max(thr_keys, key=lambda k: results[k].sharpe_friction)
    best = results[best_key]
    print(f"  best gated threshold (friction-adj Sharpe, full sample) = {best.threshold} "
          f"(Sharpe {best.sharpe_friction:.2f})")

    # --- Gate cost vs crash savings (full sample) ---
    print("\n--- gate cost of false alarms vs crash savings (full sample, best threshold) ---")
    print(f"  gate trips (position flips)             = {best.n_trips}")
    print(f"  whipsaw friction paid (gate flips)      = {best.whipsaw_friction:.3f} variance units")
    print(f"  scheduled-roll friction                 = {best.roll_friction:.3f}")
    print(f"  always-on total drawdown               = {always.max_drawdown:.2f} "
          f"(short-vol crashes the gate is meant to dodge)")

    # --- True holdout: choose threshold on <2016, test >=2016 ---
    print("\n--- holdout: threshold chosen on 1996-2015 ONLY, tested out-of-sample 2016-2026 ---")
    insample = {k: slice_perf(results[k].net, results[k].want_in, end=HOLDOUT_SPLIT)
                for k in thr_keys}
    hold_key = max(thr_keys, key=lambda k: insample[k]["sharpe"])
    hold = results[hold_key]
    oos = slice_perf(hold.net, hold.want_in, start=HOLDOUT_SPLIT)
    ins = insample[hold_key]
    print(f"  in-sample (<{HOLDOUT_SPLIT[:4]}) best threshold = {hold.threshold} "
          f"(Sharpe_fr {ins['sharpe']:.2f}, maxDD {ins['maxdd']:.3f}, {ins['n']:,} days)")
    print(f"  OUT-OF-SAMPLE (>={HOLDOUT_SPLIT[:4]}): Sharpe_fr {oos['sharpe']:.2f}, "
          f"maxDD {oos['maxdd']:.3f}, %in-mkt {oos['pct_in']:.1f}, P&L {oos['total_pnl']:+.2f} "
          f"({oos['n']:,} days)")

    # --- PER-CRISIS TABLE (centerpiece). Uses the holdout-selected threshold, so
    #     the post-2015 crises (Volmageddon, Q4'18, COVID, 2022) are genuine OOS. ---
    print(f"\n--- PER-CRISIS TABLE (gated threshold = {hold.threshold}, "
          f"selected on 1996-2015; crises after 2015 are OUT-OF-SAMPLE) ---")
    print(f"  {'crisis':16} {'always P&L':>11} {'gated P&L':>10} {'%flat':>7} "
          f"{'warn lag':>9}  oos?")
    for row in crisis_table(always, hold):
        lag = "never" if row["lag"] is None else f"{row['lag']:+d}d"
        oos_flag = "OOS" if _crisis_is_oos(row["crisis"]) else "sel"
        print(f"  {row['crisis']:16} {row['always']:>+11.3f} {row['gated']:>+10.3f} "
              f"{row['pct_flat']:>6.0f}% {lag:>9}  {oos_flag}")
    print("  (warn lag: trading days from window start to the gate's first flat day; "
          "negative = filter warned before the window began.)")

    # --- PART 3 ---
    print("\n--- PART 3: honesty checks ---")
    perperiod = [float(results[k].net.mean() / results[k].net.std(ddof=1)) for k in thr_keys]
    dsr = deflated_sharpe(best.net, perperiod, n_trials=len(THRESHOLDS))
    n_episodes = count_flat_excursions(best.want_in)
    print(f"  best threshold annualized Sharpe        = {dsr['sr_ann']:.2f}")
    print(f"  expected max Sharpe under null (4 trials)= {dsr['sr0_ann_hurdle']:.2f} (annualized hurdle)")
    print(f"  deflated Sharpe P(true SR>0)            = {dsr['deflated_sharpe_prob']:.3f} "
          f"(skew={dsr['skew']:+.2f}, kurt={dsr['kurtosis']:.1f}, T={dsr['T']:,})")
    print(f"  distinct high-vol episodes tested       = {n_episodes} separate filter "
          f"excursions above threshold (the effective sample of tail tests)")

    print("  refit-cadence sensitivity (best threshold):")
    sens: Dict[int, Tuple[float, float]] = {}
    for cad in (21, REFIT_EVERY, 126):
        cf = causal if cad == REFIT_EVERY else walk_forward_filter(feat, refit_every=cad)
        idx = cf.p_high.index
        rp = raw_pnl.reindex(idx).dropna(); idx = rp.index
        rh = (cf.hard.reindex(idx) == HIGH_VOL_REGIME)
        want = (cf.p_high.reindex(idx) < best.threshold).shift(1, fill_value=False)
        g = simulate_gate(rp, iv_entry.reindex(idx), want, rh, base_rt,
                          label=f"thr_{best.threshold}@{cad}", threshold=best.threshold)
        sens[cad] = (g.sharpe_friction, g.max_drawdown)
        print(f"    refit every {cad:3d}d: Sharpe_fr={g.sharpe_friction:+.2f}  maxDD={g.max_drawdown:.3f}")

    # Pre-2000 data-quality dependence (old VIX methodology / early-90s structure).
    post2000 = slice_perf(best.net, best.want_in, start=PRE2000_FLAG_DATE)
    pre2000_days = int((best.net.index < pd.Timestamp(PRE2000_FLAG_DATE)).sum())
    print(f"  pre-2000 dependence: full Sharpe_fr {best.sharpe_friction:+.2f} (maxDD {best.max_drawdown:.3f}) "
          f"vs post-2000-only Sharpe_fr {post2000['sharpe']:+.2f} (maxDD {post2000['maxdd']:.3f}); "
          f"{pre2000_days:,} pre-2000 days.")
    pre2000_flip = (best.sharpe_friction > 1.0) != (post2000["sharpe"] > 1.0)
    if pre2000_flip:
        print("    !! conclusion DEPENDS on pre-2000 data (old VIX methodology) — treat with extra caution.")
    else:
        print("    conclusion does NOT hinge on pre-2000 data (verdict unchanged post-2000).")

    fig_name = fig_cumulative(always, best)
    print(f"  figure saved: {fig_name}")

    # --- PART 4: verdict ---
    # Viterbi-gated (look-ahead) version to price the causal downgrade.
    vidx = best.net.index
    vit_aligned = viterbi.reindex(vidx)
    # Same one-day execution lag as the causal gate, so the comparison isolates
    # decoding quality (Viterbi still "cheats" by using the full-sample label).
    vit_want = (vit_aligned != HIGH_VOL_REGIME).shift(1, fill_value=True)
    vit_high = (vit_aligned == HIGH_VOL_REGIME)
    rp_v = raw_pnl.reindex(vidx)
    vit_gated = simulate_gate(rp_v, iv_entry.reindex(vidx), vit_want, vit_high, base_rt,
                              label="viterbi_gated", threshold=None)
    downgrade = vit_gated.sharpe_friction - best.sharpe_friction

    print("\n--- PART 4: VERDICT ---")
    _verdict(causal, diag, results, best, vit_gated, downgrade, dsr, sens,
             oos=oos, n_episodes=n_episodes, post2000=post2000, pre2000_flip=pre2000_flip)


def _verdict(causal, diag, results, best, vit_gated, downgrade, dsr, sens,
             *, oos, n_episodes, post2000, pre2000_flip) -> None:
    always = results["always_on"]
    survives_dsr = dsr["deflated_sharpe_prob"] > 0.95 and dsr["sr_ann"] > dsr["sr0_ann_hurdle"]
    sharpe_ok = best.sharpe_friction > 1.0
    dd_ok = best.max_drawdown < 0.5 * always.max_drawdown
    sens_stable = all(s[0] > 1.0 for s in sens.values()) == sharpe_ok
    covid_share = abs(always.covid_pnl) / always.max_drawdown if always.max_drawdown else float("nan")

    print(f"  Q1. Does gating survive CAUSAL decoding? "
          f"{'YES' if (sharpe_ok and dd_ok) else 'NO'} — best causal gated (thr={best.threshold}) "
          f"friction-adjusted Sharpe {best.sharpe_friction:.2f} vs always-on {always.sharpe_friction:.2f}; "
          f"maxDD {best.max_drawdown:.3f} vs always-on {always.max_drawdown:.3f} "
          f"(half-of-always-on bar = {0.5*always.max_drawdown:.3f}).")
    print(f"  Q2. Viterbi->filtered downgrade cost = {downgrade:+.2f} Sharpe "
          f"(look-ahead Viterbi-gated {vit_gated.sharpe_friction:.2f} -> causal {best.sharpe_friction:.2f}); "
          f"plus a {diag['avg_lag_days']:.1f}-day average lag entering high-vol.")
    print(f"  Q3. Friction-adjusted gated Sharpe > 1 AND maxDD < half always-on? "
          f"Sharpe>{1.0}: {'yes' if sharpe_ok else 'NO'}; "
          f"maxDD<half: {'yes' if dd_ok else 'NO'}.")
    if not (sharpe_ok and dd_ok):
        print("  => The regime-gated edge does NOT survive: once the regime is decoded causally "
              "(with realistic lag) and friction blows out 3x on the high-vol exits the gate forces, "
              "the gated strategy fails the Sharpe>1 / drawdown-halving bar. The Viterbi result was "
              "a look-ahead artifact.")
    else:
        print("  => The regime-gated edge survives causal decoding on these bars. Treat with caution: "
              f"deflated Sharpe P(SR>0)={dsr['deflated_sharpe_prob']:.2f} after 4 trials, and conclusions "
              f"are {'stable' if sens_stable else 'NOT stable'} across refit cadences.")
    oos_ok = oos["sharpe"] > 1.0
    print(f"  Q4. True holdout (threshold picked on 1996-2015, tested 2016-2026): "
          f"OOS Sharpe_fr {oos['sharpe']:.2f} {'(>1, holds)' if oos_ok else '(<=1, fails)'}; "
          f"tested across {n_episodes} distinct high-vol excursions over the full sample.")
    print(f"  Q5. Robustness: post-2000-only Sharpe_fr {post2000['sharpe']:.2f}; conclusion "
          f"{'DEPENDS on' if pre2000_flip else 'does NOT depend on'} pre-2000 (old-methodology VIX) data. "
          f"COVID alone is ~{covid_share*100:.0f}% of always-on drawdown, so a chunk of the edge is still "
          f"a few large episodes the deflated Sharpe does not deflate for.")
    print("\n  Caveat: in-sample HMM (Viterbi) is look-ahead; the causal filter is the honest signal. "
          "Friction is the FrictionConfig assumption (NBBO not entitled). Short-vol proxy is a costless-"
          "upper-bound variance position, not a tradable variance swap. Pre-2003 ^VIX is back-computed "
          "(modern methodology) and early-90s market structure differs.")


if __name__ == "__main__":
    from research_utils import load_dotenv_if_present

    load_dotenv_if_present()
    main()
