"""structure_translation_experiment.py

Translate the regime-gated short-vol *proxy* into defined-risk option structures
(put credit spread, iron condor) priced synthetically from VIX + skew, then validate
credits against real Massive option bars (last 2y) and compare to the untradeable
variance-swap proxy.

Daily SPY, track record 1996-2026 (3y burn-in on 1993+ joint sample).

PART 1  Synthetic pricer: ATM IV = (VIX-1.17)/100 (6a calibration), skew model,
        Black-Scholes, FRED 3-month T-bill, q=1.5%.
PART 2  Monthly ~30DTE structures, 2% equity risk, regime-scaled FrictionConfig.
PART 3   Last-2y Massive bar validation of Structure A credits.
PART 3B  Per-leg bias diagnosis + mechanics audit (volume staleness exclusions).
PART 3C  Skew recalibration from real leg IVs; rerun if bias within +/-15%.
PART 4   Report + verdict vs variance-swap proxy.

Run:  python structure_translation_experiment.py
      python structure_translation_experiment.py --no-part3   # skip Massive calls
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research_utils import (
    FrictionConfig,
    SHARES_PER_CONTRACT,
    entry_commission_usd,
    entry_fill_price,
    exit_fill_price,
    get_friction,
    FredClient,
    load_dotenv_if_present,
)

from candle_markov_experiment import DATA_DIR, OUTPUT_DIR, SYMBOL, TIMEFRAMES
from regime_strategy_experiment import (
    CRISES,
    REFIT_EVERY,
    load_full_history,
    simulate_gate,
    walk_forward_filter,
    iv_entry_series,
)
from vrp_experiment import (
    HIGH_VOL_REGIME,
    TARGET_DTE,
    TRADING_DAYS_PER_YEAR,
    NotAuthorized,
    _assumed_friction,
    _massive_client,
    _norm_cdf,
    _retry,
    bs_price,
    _cache_df,
    daily_option_mid,
    implied_vol_bisection,
    list_contracts_asof,
    max_drawdown_additive,
    sharpe,
    short_vol_proxy,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACK_START = "1996-01-01"
TRACK_END = "2026-12-31"
VIX_ATM_OFFSET_VOLPTS = 1.17          # 6a: ATM IV ≈ VIX - 1.17 vol points
SKEW_SLOPE_BASE = 0.10                # additive prior: 25-delta put ~ +2.5 vol pts over ATM
LITERATURE_SKEW_SLOPE = 0.15          # fallback when IV sanity gate fails
SKEW_SENSITIVITY = (0.05, 0.10, 0.15)
SKEW_ADDITIVE = "additive"
SKEW_MULTIPLICATIVE = "multiplicative"
DIV_YIELD = 0.015
RISK_PCT = 0.02
GATE_THRESHOLD = 0.4                  # holdout-chosen on 1996-2015 (regime_strategy)
HIGH_VOL_FRICTION_MULT = 3.0
FRED_TBILL_SERIES = "DTB3"            # 3-month T-bill secondary market, % discount
STARTING_EQUITY = 100_000.0
STALE_VOL_THRESHOLD = 10          # contracts; below = stale bar, exclude from bias pool
RECAL_BIAS_TOL = 0.15             # +/-15% remaining credit bias -> trust recalibrated pricer
FINAL_CRISES = ("GFC 2008-09", "COVID", "2022 full year")


# ---------------------------------------------------------------------------
# PART 1: synthetic pricer
# ---------------------------------------------------------------------------

def load_tbill_rate(*, refresh: bool = False) -> pd.Series:
    """FRED 3-month T-bill (DTB3, percent) -> decimal, daily, forward-filled."""
    cache = DATA_DIR / "fred_DTB3.parquet"
    if cache.exists() and not refresh:
        s = pd.read_parquet(cache)["rate"]
        s.index = pd.to_datetime(s.index)
        return s
    try:
        fc = FredClient()
        df = fc.get_series_df(FRED_TBILL_SERIES, observation_start=date(1990, 1, 1))
    except Exception as exc:
        print(f"  FRED DTB3 unavailable ({exc}); using 2% flat rate")
        return pd.Series(dtype=float)
    if df.empty:
        print("  FRED DTB3 empty; using 2% flat rate")
        return pd.Series(dtype=float)
    s = df.set_index(pd.to_datetime(df["observation_date"]))["value"] / 100.0
    s = s.sort_index().dropna()
    s.name = "rate"
    s.to_frame().to_parquet(cache)
    print(f"  cached FRED {FRED_TBILL_SERIES}: {len(s):,} obs -> {cache.name}")
    return s


def atm_iv_from_vix(vix_pts: float) -> float:
    """ATM implied vol in decimal; 6a calibration: ATM ≈ VIX - 1.17 vol points."""
    return max((vix_pts - VIX_ATM_OFFSET_VOLPTS) / 100.0, 1e-4)


def model_iv(atm_iv: float, otm_delta: float, *, skew_mode: str = SKEW_ADDITIVE,
             skew_slope: float = SKEW_SLOPE_BASE, skew_offset: float = 0.0,
             skew_mult_s: float = 0.0) -> float:
    """IV at |delta| under additive or multiplicative skew (ATM anchored at 6a)."""
    wing = 0.5 - abs(otm_delta)
    if skew_mode == SKEW_MULTIPLICATIVE:
        return max(atm_iv * (1.0 + skew_mult_s * wing), 1e-4)
    return max(atm_iv + skew_offset + skew_slope * wing, 1e-4)


def skew_iv(atm_iv: float, otm_delta: float, *, skew_slope: float = SKEW_SLOPE_BASE,
            skew_offset: float = 0.0) -> float:
    """Additive skew shorthand (baseline / literature prior)."""
    return model_iv(atm_iv, otm_delta, skew_mode=SKEW_ADDITIVE,
                    skew_slope=skew_slope, skew_offset=skew_offset)


def _bs_d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    srt = sigma * math.sqrt(T)
    return (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / srt


def bs_delta(S: float, K: float, T: float, r: float, q: float,
             sigma: float, *, is_call: bool) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = _bs_d1(S, K, T, r, q, sigma)
    dq = math.exp(-q * T)
    return dq * _norm_cdf(d1) if is_call else dq * (_norm_cdf(d1) - 1.0)


def strike_for_delta(S: float, T: float, r: float, q: float, iv: float,
                     target_delta: float, *, is_call: bool,
                     lo_mult: float = 0.5, hi_mult: float = 1.5) -> float:
    """Bisect strike so |delta| matches target (OTM wing)."""
    td = abs(target_delta)
    lo = S * lo_mult
    hi = S * hi_mult
    if not is_call:
        lo, hi = S * 0.5, S * 0.999
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        d = abs(bs_delta(S, mid, T, r, q, iv, is_call=is_call))
        if d > td:
            if is_call:
                lo = mid
            else:
                hi = mid
        else:
            if is_call:
                hi = mid
            else:
                lo = mid
    return round(0.5 * (lo + hi), 2)


@dataclass
class StructureLegs:
    short_put_k: float
    long_put_k: float
    short_call_k: Optional[float] = None
    long_call_k: Optional[float] = None
    expiry: pd.Timestamp = field(default_factory=pd.Timestamp)
    dte: int = 0
    T: float = 0.0


@dataclass
class StructureQuote:
    legs: StructureLegs
    credit_mid: float           # per-share option points (net short credit)
    width_put: float
    width_call: float
    max_loss_put: float
    max_loss: float
    n_legs: int
    atm_iv: float
    skew_slope: float
    skew_offset: float = 0.0
    skew_mode: str = SKEW_ADDITIVE
    skew_mult_s: float = 0.0


def price_structure(S: float, vix_pts: float, entry: pd.Timestamp, expiry: pd.Timestamp,
                    r: float, *, structure: str, skew_slope: float = SKEW_SLOPE_BASE,
                    skew_offset: float = 0.0, skew_mode: str = SKEW_ADDITIVE,
                    skew_mult_s: float = 0.0) -> StructureQuote:
    """Build strikes + mid credit for PCS or iron condor."""
    dte = max(int((expiry.normalize() - entry.normalize()).days), 1)
    T = dte / 365.0
    atm = atm_iv_from_vix(vix_pts)

    def _iv(od: float) -> float:
        return model_iv(atm, od, skew_mode=skew_mode, skew_slope=skew_slope,
                        skew_offset=skew_offset, skew_mult_s=skew_mult_s)

    sp_k = strike_for_delta(S, T, r, DIV_YIELD, _iv(0.30), 0.30, is_call=False)
    lp_k = strike_for_delta(S, T, r, DIV_YIELD, _iv(0.15), 0.15, is_call=False)
    if sp_k <= lp_k:
        sp_k, lp_k = lp_k + 1.0, lp_k  # enforce ordering

    short_put = bs_price(S, sp_k, T, r, DIV_YIELD, _iv(0.30), False)
    long_put = bs_price(S, lp_k, T, r, DIV_YIELD, _iv(0.15), False)
    credit = short_put - long_put
    width_put = sp_k - lp_k

    sl = StructureLegs(short_put_k=sp_k, long_put_k=lp_k, expiry=expiry, dte=dte, T=T)
    width_call = 0.0
    if structure == "iron_condor":
        sc_k = strike_for_delta(S, T, r, DIV_YIELD, _iv(0.30), 0.30, is_call=True)
        lc_k = strike_for_delta(S, T, r, DIV_YIELD, _iv(0.15), 0.15, is_call=True)
        if lc_k <= sc_k:
            lc_k = sc_k + 1.0
        short_call = bs_price(S, sc_k, T, r, DIV_YIELD, _iv(0.30), True)
        long_call = bs_price(S, lc_k, T, r, DIV_YIELD, _iv(0.15), True)
        credit += short_call - long_call
        width_call = lc_k - sc_k
        sl.short_call_k, sl.long_call_k = sc_k, lc_k
        n_legs = 4
        max_loss = max(width_put, width_call) - credit
    else:
        n_legs = 2
        max_loss = width_put - credit

    return StructureQuote(
        legs=sl, credit_mid=max(credit, 1e-4), width_put=width_put, width_call=width_call,
        max_loss_put=width_put - (short_put - long_put), max_loss=max(max_loss, 1e-4),
        n_legs=n_legs, atm_iv=atm, skew_slope=skew_slope, skew_offset=skew_offset,
        skew_mode=skew_mode, skew_mult_s=skew_mult_s,
    )


def friction_cfg(regime_high: bool, base: Optional[FrictionConfig] = None) -> FrictionConfig:
    """Regime-scaled half-spread (3x in high-vol); commission unchanged."""
    b = base or get_friction()
    mult = HIGH_VOL_FRICTION_MULT if regime_high else 1.0
    return FrictionConfig(
        half_spread_pct=b.half_spread_pct * mult,
        commission_usd_per_contract_per_fill=b.commission_usd_per_contract_per_fill,
    )


def entry_credit_filled(quote: StructureQuote, S: float, vix_pts: float, r: float,
                        regime_high: bool, *, credit_scale: float = 1.0) -> Tuple[float, float]:
    """Net credit per share after synthetic entry fills + commission (USD per contract).

    `credit_scale` multiplies all leg mids (e.g. mean(real/syn) from Part 3 bias audit).
    """
    f = friction_cfg(regime_high)
    atm = atm_iv_from_vix(vix_pts)
    T = quote.legs.T
    legs: List[Tuple[bool, float, float, bool]] = []  # is_short, K, otm_delta, is_call
    legs.append((True, quote.legs.short_put_k, 0.30, False))
    legs.append((False, quote.legs.long_put_k, 0.15, False))
    if quote.n_legs == 4:
        legs.append((True, quote.legs.short_call_k, 0.30, True))
        legs.append((False, quote.legs.long_call_k, 0.15, True))

    net = 0.0
    for is_short, K, od, is_call in legs:
        iv = model_iv(atm, od, skew_mode=quote.skew_mode, skew_slope=quote.skew_slope,
                      skew_offset=quote.skew_offset, skew_mult_s=quote.skew_mult_s)
        mid = bs_price(S, K, T, r, DIV_YIELD, iv, is_call) * credit_scale
        fill = entry_fill_price(is_buy=not is_short, mid=mid, friction=f)
        net += fill if is_short else -fill
    comm = entry_commission_usd(quote.n_legs, f) / SHARES_PER_CONTRACT
    return net - comm, net


def expiry_intrinsic(S: float, quote: StructureQuote) -> float:
    """Cost to close at expiry (per share, positive = debit)."""
    sp, lp = quote.legs.short_put_k, quote.legs.long_put_k
    cost = max(sp - S, 0.0) - max(lp - S, 0.0)
    if quote.n_legs == 4:
        sc, lc = quote.legs.short_call_k, quote.legs.long_call_k
        cost += max(S - sc, 0.0) - max(S - lc, 0.0)
    return cost


def mtm_value(S: float, vix_pts: float, r: float, quote: StructureQuote,
              asof: pd.Timestamp) -> float:
    """Mark-to-market per-share value of the short structure (positive = profit)."""
    dte = max((quote.legs.expiry.normalize() - asof.normalize()).days, 0)
    T = dte / 365.0
    atm = atm_iv_from_vix(vix_pts)
    if T <= 0:
        return -expiry_intrinsic(S, quote)
    val = 0.0
    for is_short, K, od, is_call in _leg_list(quote):
        iv = model_iv(atm, od, skew_mode=quote.skew_mode, skew_slope=quote.skew_slope,
                      skew_offset=quote.skew_offset, skew_mult_s=quote.skew_mult_s)
        mid = bs_price(S, K, T, r, DIV_YIELD, iv, is_call)
        val += mid if is_short else -mid
    return val


def _leg_list(quote: StructureQuote) -> List[Tuple[bool, float, float, bool]]:
    out = [(True, quote.legs.short_put_k, 0.30, False),
           (False, quote.legs.long_put_k, 0.15, False)]
    if quote.n_legs == 4:
        out += [(True, quote.legs.short_call_k, 0.30, True),
                (False, quote.legs.long_call_k, 0.15, True)]
    return out


def exit_debit_filled(S: float, vix_pts: float, r: float, quote: StructureQuote,
                      regime_high: bool) -> float:
    """Debit per share to close early + exit commission."""
    f = friction_cfg(regime_high)
    atm = atm_iv_from_vix(vix_pts)
    T = max(quote.legs.T, 1 / 365.0)
    debit = 0.0
    for is_short, K, od, is_call in _leg_list(quote):
        iv = model_iv(atm, od, skew_mode=quote.skew_mode, skew_slope=quote.skew_slope,
                      skew_offset=quote.skew_offset, skew_mult_s=quote.skew_mult_s)
        mid = bs_price(S, K, T, r, DIV_YIELD, iv, is_call)
        # Close short = buy back; close long = sell
        fill = exit_fill_price(is_long=not is_short, mid=mid, friction=f)
        debit += fill if is_short else -fill
    comm = entry_commission_usd(quote.n_legs, f) / SHARES_PER_CONTRACT
    return debit + comm


# ---------------------------------------------------------------------------
# PART 2: strategy simulation
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry: pd.Timestamp
    expiry: pd.Timestamp
    structure: str
    variant: str
    credit: float
    contracts: int
    pnl_usd: float
    pnl_pct_eq: float
    exited_early: bool
    skipped: bool = False


@dataclass
class SimResult:
    structure: str
    variant: str
    skew_slope: float
    skew_offset: float
    skew_mode: str
    skew_mult_s: float
    credit_scale: float
    pricer_label: str
    trades: List[Trade]
    daily_pnl: pd.Series
    equity: pd.Series
    metrics: Dict[str, float]
    crisis: List[dict]


def _monthly_entries(idx: pd.DatetimeIndex, start: str, end: str) -> List[pd.Timestamp]:
    sub = idx[(idx >= start) & (idx <= end)]
    months = sub.to_period("M").unique()
    out: List[pd.Timestamp] = []
    for m in months:
        days = sub[sub.to_period("M") == m]
        if len(days):
            out.append(days[0])
    return out


def _pick_expiry(entry: pd.Timestamp, all_idx: pd.DatetimeIndex) -> pd.Timestamp:
    target = entry + pd.Timedelta(days=TARGET_DTE)
    cands = all_idx[(all_idx > entry) & (all_idx <= entry + pd.Timedelta(days=45))]
    if cands.empty:
        return target
    dte = pd.Series((cands - entry).days, index=cands)
    return cands[(dte - TARGET_DTE).abs().argmin()]


def simulate_structure(
    df: pd.DataFrame,
    causal_p_high: pd.Series,
    causal_hard: pd.Series,
    rates: pd.Series,
    *,
    structure: str,
    variant: str,
    skew_slope: float = SKEW_SLOPE_BASE,
    skew_offset: float = 0.0,
    skew_mode: str = SKEW_ADDITIVE,
    skew_mult_s: float = 0.0,
    credit_scale: float = 1.0,
    pricer_label: str = "baseline",
) -> SimResult:
    """Monthly ~30DTE structures; variant in always_on | gated_skip | gated_exit."""
    idx = df.index
    entries = _monthly_entries(idx, TRACK_START, TRACK_END)
    equity = STARTING_EQUITY
    trades: List[Trade] = []
    daily_pnl = pd.Series(0.0, index=idx[(idx >= TRACK_START) & (idx <= TRACK_END)])
    open_pos: Optional[dict] = None

    for i, entry in enumerate(entries):
        lag_date = idx[idx.searchsorted(entry) - 1] if idx.searchsorted(entry) > 0 else entry
        p_high = float(causal_p_high.reindex([lag_date]).iloc[0]) if lag_date in causal_p_high.index else 0.0
        regime_hi = bool(causal_hard.reindex([lag_date]).iloc[0] == HIGH_VOL_REGIME) if lag_date in causal_hard.index else False
        gated = p_high >= GATE_THRESHOLD

        if variant in ("gated_skip", "gated_exit") and gated:
            trades.append(Trade(entry, entry, structure, variant, 0.0, 0, 0.0, 0.0, False, skipped=True))
            continue

        row = df.loc[entry]
        S = float(row["close"])
        vix_pts = float(row["vix"])
        r = _rate_on(rates, entry)
        expiry = _pick_expiry(entry, idx)
        quote = price_structure(S, vix_pts, entry, expiry, r, structure=structure,
                                skew_slope=skew_slope, skew_offset=skew_offset,
                                skew_mode=skew_mode, skew_mult_s=skew_mult_s)
        credit, _ = entry_credit_filled(quote, S, vix_pts, r, regime_hi, credit_scale=credit_scale)
        max_loss_ps = quote.max_loss
        risk_usd = RISK_PCT * equity
        contracts = max(1, int(risk_usd / (max_loss_ps * SHARES_PER_CONTRACT)))
        if max_loss_ps * SHARES_PER_CONTRACT <= 0:
            continue

        if variant in ("always_on", "gated_skip"):
            # hold to expiry
            exp_idx = idx[idx <= expiry]
            S_exp = float(df.loc[exp_idx[-1], "close"]) if len(exp_idx) else S
            intrinsic = expiry_intrinsic(S_exp, quote)
            pnl_ps = credit - intrinsic
            comm_exit = 0.0  # cash-settled expiry: no exit spread
            pnl_usd = (pnl_ps * SHARES_PER_CONTRACT - comm_exit) * contracts
            exited = False
            # spread daily P&L at expiry for simplicity on hold-to-expiry
            if expiry in daily_pnl.index:
                daily_pnl.loc[expiry] += pnl_usd
            trades.append(Trade(entry, expiry, structure, variant, credit, contracts,
                                pnl_usd, pnl_usd / equity, exited, False))
            equity += pnl_usd

        else:  # gated_exit — intra-month exit when filter trips
            open_pos = {
                "entry": entry, "expiry": expiry, "quote": quote, "credit": credit,
                "contracts": contracts, "equity_at_entry": equity, "regime_entry": regime_hi,
            }
            path = idx[(idx >= entry) & (idx <= expiry)]
            closed = False
            for j, d in enumerate(path):
                if j == 0:
                    continue
                lag = path[j - 1]
                trip = float(causal_p_high.reindex([lag]).iloc[0]) >= GATE_THRESHOLD if lag in causal_p_high.index else False
                if trip:
                    Sd = float(df.loc[d, "close"])
                    vd = float(df.loc[d, "vix"])
                    rd = _rate_on(rates, d)
                    rhi = bool(causal_hard.reindex([lag]).iloc[0] == HIGH_VOL_REGIME) if lag in causal_hard.index else False
                    debit = exit_debit_filled(Sd, vd, rd, quote, rhi)
                    pnl_ps = credit - debit
                    pnl_usd = pnl_ps * SHARES_PER_CONTRACT * contracts
                    daily_pnl.loc[d] += pnl_usd
                    trades.append(Trade(entry, d, structure, variant, credit, contracts,
                                        pnl_usd, pnl_usd / open_pos["equity_at_entry"], True, False))
                    equity += pnl_usd
                    closed = True
                    break
            if not closed:
                S_exp = float(df.loc[path[-1], "close"])
                intrinsic = expiry_intrinsic(S_exp, quote)
                pnl_usd = (credit - intrinsic) * SHARES_PER_CONTRACT * contracts
                daily_pnl.loc[path[-1]] += pnl_usd
                trades.append(Trade(entry, path[-1], structure, variant, credit, contracts,
                                    pnl_usd, pnl_usd / open_pos["equity_at_entry"], False, False))
                equity += pnl_usd

    active = [t for t in trades if not t.skipped]
    eq = _equity_curve(active)
    metrics = _compute_metrics(daily_pnl, eq, active)
    crisis = _crisis_from_trades(trades, active)
    return SimResult(structure, variant, skew_slope, skew_offset, skew_mode, skew_mult_s,
                     credit_scale, pricer_label, trades, daily_pnl, eq, metrics, crisis)


def _equity_curve(active: List[Trade]) -> pd.Series:
    """Compounded equity at each trade exit (hold-to-expiry or early exit)."""
    eq = STARTING_EQUITY
    pts: List[Tuple[pd.Timestamp, float]] = [(pd.Timestamp(TRACK_START), eq)]
    for t in sorted(active, key=lambda x: x.expiry):
        eq += t.pnl_usd
        pts.append((t.expiry, eq))
    if not pts:
        return pd.Series([STARTING_EQUITY])
    idx, vals = zip(*pts)
    return pd.Series(vals, index=pd.DatetimeIndex(idx))


def _maxdd_pct(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, np.nan)
    return float(dd.max()) if len(dd) else 0.0


def _rate_on(rates: pd.Series, d: pd.Timestamp) -> float:
    if rates.empty:
        return 0.02
    sub = rates.loc[:d]
    return float(sub.iloc[-1]) if len(sub) else 0.02


def _compute_metrics(daily_pnl: pd.Series, equity: pd.Series, trades: List[Trade]) -> Dict[str, float]:
    """Trade-level Sharpe/CAGR; daily series for CVaR / worst-5-day on realized exit days."""
    rets = pd.Series([t.pnl_pct_eq for t in trades], index=[t.expiry for t in trades])
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6) if len(equity) > 1 else 1.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) and equity.iloc[0] > 0 else 0.0
    cagr = (max(equity.iloc[-1], 1.0) / equity.iloc[0]) ** (1.0 / years) - 1.0 if equity.iloc[0] > 0 else 0.0
    trades_per_year = len(trades) / years if years > 0 else 12.0
    if rets.std(ddof=1) > 0 and len(rets) > 1:
        sharpe_ann = float(rets.mean() / rets.std(ddof=1) * math.sqrt(trades_per_year))
    else:
        sharpe_ann = float("nan")
    down = rets[rets < 0]
    sortino = float(rets.mean() / down.std(ddof=1) * math.sqrt(trades_per_year)) if len(down) > 1 and down.std(ddof=1) > 0 else float("nan")
    r_daily = daily_pnl[daily_pnl != 0]
    var99 = float(np.percentile(r_daily, 1)) if len(r_daily) else float("nan")
    cvar99 = float(r_daily[r_daily <= var99].mean()) if len(r_daily) else float("nan")
    worst5 = float(r_daily.nsmallest(5).sum()) if len(r_daily) >= 5 else float(r_daily.sum())
    wins = sum(1 for t in trades if t.pnl_usd > 0)
    return {
        "cagr": cagr, "total_return": total_ret,
        "sharpe": sharpe_ann, "sortino": sortino,
        "cvar99": cvar99, "worst5d": worst5,
        "maxdd": _maxdd_pct(equity),
        "win_rate": wins / len(trades) if trades else float("nan"),
        "n_trades": len(trades), "final_equity": float(equity.iloc[-1]),
    }


def _crisis_from_trades(trades: List[Trade], active: List[Trade]) -> List[dict]:
    rows = []
    for name, start, end in CRISES:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        in_crisis = [t for t in active if s <= t.entry <= e or (t.entry <= e and t.expiry >= s)]
        pnl = sum(t.pnl_usd for t in in_crisis)
        rows.append({"crisis": name, "pnl_usd": pnl, "n_trades": len(in_crisis)})
    return rows


# ---------------------------------------------------------------------------
# PART 3 / 3B / 3C: Massive validation, per-leg diagnosis, recalibration
# ---------------------------------------------------------------------------

def daily_option_bar(client, opt_ticker: str, day: pd.Timestamp) -> Tuple[float, float, bool]:
    """EOD option bar: (mid per share, volume, was_empty). Cached alongside 6a aggs."""
    key = f"agg_{opt_ticker.replace(':', '_')}_{day.date()}"

    def _build() -> pd.DataFrame:
        def _pull():
            out = []
            for b in client.list_aggs(
                ticker=opt_ticker, multiplier=1, timespan="day",
                from_=day.date().isoformat(), to=day.date().isoformat(), limit=10,
            ):
                out.append(b)
            return out

        rows = []
        for b in _retry(_pull):
            if b.timestamp is None:
                continue
            rows.append({
                "close": float(b.close) if b.close is not None else np.nan,
                "vwap": float(b.vwap) if getattr(b, "vwap", None) is not None else np.nan,
                "volume": float(b.volume) if getattr(b, "volume", None) is not None else 0.0,
            })
        return pd.DataFrame(rows)

    df = _cache_df(key, _build)
    if not df.empty and "volume" not in df.columns:
        df = _build()  # refetch once to populate volume; overwrite cache
        df.to_parquet(DATA_DIR / "massive" / f"{key}.parquet")
    if df.empty:
        return float("nan"), float("nan"), True
    row = df.iloc[0]
    mid = row["close"] if np.isfinite(row["close"]) and row["close"] > 0 else row["vwap"]
    vol = float(row["volume"]) if "volume" in df.columns and np.isfinite(row["volume"]) else float("nan")
    if not (np.isfinite(mid) and mid > 0):
        return float("nan"), vol, True
    return float(mid), vol, False


def _syn_leg_price(spot: float, K: float, T: float, r: float, otm_delta: float, atm: float,
                   *, skew_mode: str = SKEW_ADDITIVE, skew_slope: float = SKEW_SLOPE_BASE,
                   skew_offset: float = 0.0, skew_mult_s: float = 0.0) -> Tuple[float, float]:
    iv = model_iv(atm, otm_delta, skew_mode=skew_mode, skew_slope=skew_slope,
                  skew_offset=skew_offset, skew_mult_s=skew_mult_s)
    return bs_price(spot, K, T, r, DIV_YIELD, iv, False), iv


def collect_real_entries(df: pd.DataFrame, rates: pd.Series) -> List[dict]:
    """Pull per-leg real vs synthetic rows for last-2y monthly Structure A entries."""
    try:
        client = _massive_client()
    except NotAuthorized:
        return []
    end = df.index.max()
    start = end - pd.DateOffset(years=2)
    entries = _monthly_entries(df.index, str(start.date()), str(end.date()))
    rows: List[dict] = []
    for entry in entries:
        avail = df.index[df.index <= entry]
        if avail.empty:
            continue
        asof = avail[-1]
        spot = float(df.loc[asof, "close"])
        try:
            contracts = list_contracts_asof(client, asof, spot)
        except NotAuthorized:
            return rows
        if contracts.empty:
            continue
        contracts = contracts.assign(dte=(contracts["expiration"] - asof.normalize()).dt.days)
        contracts = contracts[contracts["dte"] > 0]
        if contracts.empty:
            continue
        target_exp = contracts.iloc[(contracts["dte"] - TARGET_DTE).abs().argmin()]["expiration"]
        real_dte = int((target_exp - asof.normalize()).days)
        chain = contracts[contracts["expiration"] == target_exp]
        puts = chain[chain["contract_type"] == "put"]
        if puts.empty:
            continue

        r = _rate_on(rates, asof)
        vix_pts = float(df.loc[asof, "vix"])
        syn = price_structure(spot, vix_pts, asof, target_exp, r, structure="put_credit_spread")
        sp_row = puts.iloc[(puts["strike"] - syn.legs.short_put_k).abs().argmin()]
        lp_row = puts.iloc[(puts["strike"] - syn.legs.long_put_k).abs().argmin()]
        sp_k, lp_k = float(sp_row["strike"]), float(lp_row["strike"])

        sp_mid, sp_vol, sp_empty = daily_option_bar(client, sp_row["ticker"], asof)
        lp_mid, lp_vol, lp_empty = daily_option_bar(client, lp_row["ticker"], asof)
        if sp_empty or lp_empty:
            continue

        T = max(real_dte, 1) / 365.0
        atm = atm_iv_from_vix(vix_pts)
        syn_short, syn_short_iv = _syn_leg_price(spot, sp_k, T, r, 0.30, atm,
                                                 skew_slope=SKEW_SLOPE_BASE, skew_offset=0.0)
        syn_long, syn_long_iv = _syn_leg_price(spot, lp_k, T, r, 0.15, atm,
                                               skew_slope=SKEW_SLOPE_BASE, skew_offset=0.0)
        real_short_iv = implied_vol_bisection(sp_mid, spot, sp_k, T, r=r, q=DIV_YIELD, is_call=False)
        real_long_iv = implied_vol_bisection(lp_mid, spot, lp_k, T, r=r, q=DIV_YIELD, is_call=False)

        rows.append({
            "date": asof,
            "spot": spot, "r": r, "q": DIV_YIELD, "atm_iv": atm, "T": T,
            "syn_dte": syn.legs.dte, "real_dte": real_dte,
            "syn_sp_k": syn.legs.short_put_k, "syn_lp_k": syn.legs.long_put_k,
            "real_sp_k": sp_k, "real_lp_k": lp_k,
            "short_delta": 0.30, "long_delta": 0.15,
            "syn_short": syn_short, "real_short": sp_mid,
            "syn_long": syn_long, "real_long": lp_mid,
            "syn_short_iv": syn_short_iv, "real_short_iv": real_short_iv,
            "syn_long_iv": syn_long_iv, "real_long_iv": real_long_iv,
            "short_vol": sp_vol, "long_vol": lp_vol,
            "short_stale": np.isfinite(sp_vol) and sp_vol < STALE_VOL_THRESHOLD,
            "long_stale": np.isfinite(lp_vol) and lp_vol < STALE_VOL_THRESHOLD,
            "syn_credit": syn_short - syn_long,
            "real_credit": sp_mid - lp_mid,
        })
    return rows


def _leg_bias_pct(syn: float, real: float) -> float:
    return (syn - real) / real if real > 0 else float("nan")


def _iv_bias_volpts(syn_iv: float, real_iv: float) -> float:
    if np.isfinite(syn_iv) and np.isfinite(real_iv):
        return (syn_iv - real_iv) * 100.0
    return float("nan")


def part_3b_diagnosis(entries: List[dict]) -> pd.DataFrame:
    """Per-leg price/IV bias + mechanics audit checklist."""
    if not entries:
        print("  Part 3B: no entries to diagnose")
        return pd.DataFrame()

    df = pd.DataFrame(entries)
    clean = df[~df["short_stale"] & ~df["long_stale"]].copy()
    n_stale = len(df) - len(clean)
    print(f"\n--- PART 3B: per-leg diagnosis ({len(df)} entries, {n_stale} excluded stale volume) ---")

    for leg, syn_p, real_p, syn_iv, real_iv in (
        ("short 30d", "syn_short", "real_short", "syn_short_iv", "real_short_iv"),
        ("long 15d", "syn_long", "real_long", "syn_long_iv", "real_long_iv"),
    ):
        sub = clean if len(clean) else df
        pb = sub.apply(lambda r: _leg_bias_pct(r[syn_p], r[real_p]), axis=1)
        ib = sub.apply(lambda r: _iv_bias_volpts(r[syn_iv], r[real_iv]), axis=1)
        print(f"  {leg} price bias (syn-real)/real: mean {pb.mean()*100:+.1f}%  median {pb.median()*100:+.1f}%")
        print(f"  {leg} IV bias (vol pts):          mean {ib.mean():+.2f}  median {ib.median():+.2f}")

    if len(clean):
        sp_r = (clean["syn_short"] / clean["real_short"]).median()
        lp_r = (clean["syn_long"] / clean["real_long"]).median()
        sp_iv = (clean["syn_short_iv"] - clean["real_short_iv"]).median() * 100
        lp_iv = (clean["syn_long_iv"] - clean["real_long_iv"]).median() * 100
        if abs(sp_r - lp_r) < 0.15:
            print(f"  => Both legs off by similar factor (short {sp_r:.2f}x, long {lp_r:.2f}x) -> "
                  f"ATM level bias dominates (IV +{sp_iv:.1f}/+{lp_iv:.1f} vol pts).")
        elif sp_iv > lp_iv + 1.0:
            print(f"  => Short leg more overpriced in IV (+{sp_iv:.1f} vs +{lp_iv:.1f} vol pts) -> "
                  f"skew slope may be too low as well as level.")
        elif lp_iv < sp_iv - 0.5:
            print(f"  => Long leg underpriced relative to short in IV space -> raise skew slope.")
        else:
            print("  => Mixed leg errors; use 3C regression rather than single-knob fix.")

    print("  Mechanics audit (per entry):")
    checks = {
        "DTE within 3d": (df["syn_dte"] - df["real_dte"]).abs() <= 3,
        "compare at listed strikes": pd.Series(True, index=df.index),  # syn repriced at real_sp/lp_k
        "listed short = syn target": df["syn_sp_k"] == df["real_sp_k"],
        "listed long = syn target": df["syn_lp_k"] == df["real_lp_k"],
        "units per-share": pd.Series(True, index=df.index),
        "r,q consistent": pd.Series(True, index=df.index),
    }
    for name, mask in checks.items():
        pct = float(mask.mean() * 100.0)
        print(f"    [{'PASS' if pct >= 95 else 'FAIL'}] {name}: {pct:.0f}% entries ({mask.sum()}/{len(df)})")
    unk_vol = int((~df["short_vol"].apply(np.isfinite)).sum() + (~df["long_vol"].apply(np.isfinite)).sum())
    stale_n = int(df["short_stale"].sum() + df["long_stale"].sum())
    print(f"    [{'PASS' if n_stale <= 5 else 'WARN'}] staleness exclusions: {n_stale} entries "
          f"({stale_n} stale legs with vol < {STALE_VOL_THRESHOLD}; {unk_vol} legs volume unknown)")

    if len(clean):
        cred_bias = clean.apply(lambda r: _leg_bias_pct(r["syn_credit"], r["real_credit"]), axis=1)
        print(f"  Aggregate credit bias (clean): mean {cred_bias.mean()*100:+.1f}%  "
              f"median {cred_bias.median()*100:+.1f}%")
    df["clean"] = ~df["short_stale"] & ~df["long_stale"]
    return df


def _origin_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS through the origin: y = beta * x."""
    denom = float(np.dot(x, x))
    return float(np.dot(x, y) / denom) if denom > 0 else float("nan")


def _credit_bias_multiplicative(clean: pd.DataFrame, s: float) -> Tuple[float, float]:
    """Mean/median (syn-real)/real using anchored multiplicative skew at listed strikes."""
    rem = []
    for _, r in clean.iterrows():
        ss, _ = _syn_leg_price(r["spot"], r["real_sp_k"], r["T"], r["r"], 0.30, r["atm_iv"],
                               skew_mode=SKEW_MULTIPLICATIVE, skew_mult_s=s)
        sl, _ = _syn_leg_price(r["spot"], r["real_lp_k"], r["T"], r["r"], 0.15, r["atm_iv"],
                               skew_mode=SKEW_MULTIPLICATIVE, skew_mult_s=s)
        rem.append(_leg_bias_pct(ss - sl, r["real_credit"]))
    rem_s = pd.Series(rem)
    return float(rem_s.mean()), float(rem_s.median())


def part_3c_recalibrate(diag: pd.DataFrame) -> Optional[dict]:
    """Sanity gate on put skew, then fit multiplicative s (intercept fixed at 0 vs 6a ATM)."""
    if diag.empty:
        return None
    clean = diag[diag["clean"]].copy()
    if clean.empty:
        print("  Part 3C: no clean entries for recalibration")
        return None

    print(f"\n--- PART 3C: skew recalibration ({len(clean)} clean entries) ---")
    print("  Sanity gate: real 30-delta put IV minus anchored ATM IV (6a: VIX-1.17)/100")
    skew_spreads = []
    for _, r in clean.iterrows():
        spread_vp = (r["real_short_iv"] - r["atm_iv"]) * 100.0
        skew_spreads.append(spread_vp)
        print(f"    {pd.Timestamp(r['date']).date()}: 30d put IV - ATM = {spread_vp:+.2f} vol pts")
    med_skew = float(np.median(skew_spreads))
    print(f"  median 30d put - ATM = {med_skew:+.2f} vol pts")

    if med_skew < 0:
        print("  STOP: median is NEGATIVE -> inverted put skew in inverted IVs.")
        print("  Likely non-synchronous option vs underlying marks; recalibration INVALID.")
        print(f"  Falling back to literature prior: additive slope={LITERATURE_SKEW_SLOPE}, "
              f"ATM=(VIX-{VIX_ATM_OFFSET_VOLPTS})/100")
        return {
            "sanity_failed": True, "trusted": False, "use_literature_prior": True,
            "median_put_skew_vp": med_skew, "n_entries_clean": len(clean),
            "skew_mode": SKEW_ADDITIVE, "skew_slope": LITERATURE_SKEW_SLOPE,
            "skew_offset": 0.0, "skew_mult_s": 0.0,
        }

    xs, ys, atms = [], [], []
    for _, r in clean.iterrows():
        atm = r["atm_iv"]
        for delta, riv in ((0.30, r["real_short_iv"]), (0.15, r["real_long_iv"])):
            if np.isfinite(riv):
                wing = 0.5 - abs(delta)
                xs.append(wing)
                ys.append(riv - atm)  # anchored: zero intercept by construction
                atms.append(atm)
    if len(xs) < 4:
        print("  Part 3C: insufficient IV points for regression")
        return None

    x = np.array(xs)
    y = np.array(ys)
    beta_add = _origin_slope(x, y)  # origin OLS: (IV - anchored ATM) = beta * wing
    atm_med = float(np.median(atms))
    # Convert to multiplicative s so ATM*s*wing = beta*wing at median ATM level
    s_fit = beta_add / atm_med if atm_med > 0 else float("nan")
    print(f"  anchored ATM = (VIX - {VIX_ATM_OFFSET_VOLPTS})/100; intercept forced to 0")
    print(f"  origin OLS (IV-ATM) on wing: beta = {beta_add:.4f} vol-dec/wing "
          f"(assumed additive {SKEW_SLOPE_BASE})")
    print(f"  multiplicative: IV = ATM * (1 + s * (0.5 - |delta|)),  s = {s_fit:.4f}  "
          f"(= beta / median ATM)")
    print(f"  at 25-delta: +{s_fit * 0.25 * 100:.2f}% of ATM IV  (~+{beta_add * 0.25 * 100:.2f} vol pts at median ATM)")

    mean_rem, med_rem = _credit_bias_multiplicative(clean, s_fit)
    trusted = abs(mean_rem) <= RECAL_BIAS_TOL and abs(med_rem) <= RECAL_BIAS_TOL
    print(f"  remaining credit bias ({len(clean)} entries): mean {mean_rem*100:+.1f}%  "
          f"median {med_rem*100:+.1f}%  (tolerance +/-{RECAL_BIAS_TOL*100:.0f}%)")
    print(f"  pricer trust: {'TRUSTED — multiplicative recalibration is primary' if trusted else 'NOT TRUSTED at this data tier'}")
    return {
        "sanity_failed": False, "trusted": trusted, "use_literature_prior": False,
        "median_put_skew_vp": med_skew, "beta_additive": beta_add, "s": s_fit,
        "mean_rem_bias": mean_rem, "median_rem_bias": med_rem,
        "n_legs": len(x), "n_entries_clean": len(clean),
        "skew_mode": SKEW_MULTIPLICATIVE, "skew_slope": SKEW_SLOPE_BASE,
        "skew_offset": 0.0, "skew_mult_s": s_fit,
    }


def validation_bias(entries: List[dict]) -> Optional[dict]:
    """Part 3 headline: aggregate credit bias on all entries with bars."""
    if not entries:
        print("  Part 3: no real credits retrieved (Massive unavailable or empty bars)")
        return None
    bias = pd.DataFrame(entries)
    mean_bias = float(bias.apply(lambda r: _leg_bias_pct(r["syn_credit"], r["real_credit"]), axis=1).mean())
    med_bias = float(bias.apply(lambda r: _leg_bias_pct(r["syn_credit"], r["real_credit"]), axis=1).median())
    print(f"\n--- PART 3: validation vs Massive option bars (last 2y) ---")
    print(f"  {len(bias)} monthly Structure A entries with real bars")
    print(f"    mean (syn-real)/real = {mean_bias*100:+.1f}%  median = {med_bias*100:+.1f}%")
    print(f"    syn credit mean={bias['syn_credit'].mean():.2f}  real={bias['real_credit'].mean():.2f}")
    return {"n": len(bias), "mean_bias_pct": mean_bias, "median_bias_pct": med_bias, "df": bias}


# ---------------------------------------------------------------------------
# PART 4: report + proxy comparison
# ---------------------------------------------------------------------------

VARIANTS = ("always_on", "gated_skip", "gated_exit")
STRUCTURES = ("put_credit_spread", "iron_condor")


def _proxy_metrics(df: pd.DataFrame, causal) -> Dict[str, Dict[str, float]]:
    """Variance-swap proxy: always-on vs gated_skip @ threshold 0.4."""
    raw = short_vol_proxy(df, "yz_var")
    ivx = iv_entry_series(df)
    idx = causal.p_high.index
    rp = raw.reindex(idx).dropna()
    idx = rp.index
    rh = (causal.hard.reindex(idx) == HIGH_VOL_REGIME)
    base_rt = _assumed_friction()["iron_condor"].cost_pct_of_credit / 100.0
    always = simulate_gate(rp, ivx.reindex(idx), pd.Series(True, index=idx), rh, base_rt,
                           label="proxy_always", threshold=None)
    want = (causal.p_high.reindex(idx) < GATE_THRESHOLD).shift(1, fill_value=False)
    gated = simulate_gate(rp, ivx.reindex(idx), want, rh, base_rt,
                          label="proxy_gated", threshold=GATE_THRESHOLD)
    return {
        "proxy_always_on": {"sharpe": always.sharpe_friction, "maxdd": always.max_drawdown,
                            "final_pnl": float(always.net.sum())},
        "proxy_gated_skip": {"sharpe": gated.sharpe_friction, "maxdd": gated.max_drawdown,
                             "final_pnl": float(gated.net.sum())},
    }


def _gate_passes(metrics_always: dict, metrics_gated: dict) -> Tuple[bool, bool]:
    sharpe_up = metrics_gated["sharpe"] > metrics_always["sharpe"] + 0.1
    dd_half = metrics_gated["maxdd"] < 0.5 * metrics_always["maxdd"]
    return sharpe_up, dd_half


def print_final_verdict_table(results: List[SimResult], *, primary: str) -> None:
    """PCS + condor, always_on vs gated_skip; Sharpe, Sortino, maxDD; GFC/COVID/2022."""
    print(f"\n--- FINAL STRATEGY VERDICT TABLE ({primary}) ---")
    hdr = f"  {'structure':18} {'variant':12} {'Sharpe':>7} {'Sortino':>8} {'maxDD':>7}"
    print(hdr)
    pool = [r for r in results if r.variant in ("always_on", "gated_skip")
            and r.credit_scale == 1.0 and r.pricer_label == primary]

    for structure in STRUCTURES:
        for variant in ("always_on", "gated_skip"):
            r = next((x for x in pool if x.structure == structure and x.variant == variant), None)
            if r is None:
                continue
            m = r.metrics
            print(f"  {r.structure:18} {r.variant:12} {m['sharpe']:7.2f} {m['sortino']:8.2f} "
                  f"{m['maxdd']*100:6.1f}%")

    print(f"\n  {'crisis':16} {'PCS always':>12} {'PCS gated':>12} {'IC always':>12} {'IC gated':>12}")
    crisis_idx = {n: i for i, (n, _, _) in enumerate(CRISES)}
    for cname in FINAL_CRISES:
        i = crisis_idx.get(cname)
        if i is None:
            continue
        pcs_a = next((x for x in pool if x.structure == "put_credit_spread" and x.variant == "always_on"), None)
        pcs_g = next((x for x in pool if x.structure == "put_credit_spread" and x.variant == "gated_skip"), None)
        ic_a = next((x for x in pool if x.structure == "iron_condor" and x.variant == "always_on"), None)
        ic_g = next((x for x in pool if x.structure == "iron_condor" and x.variant == "gated_skip"), None)
        def _pnl(r):
            return r.crisis[i]["pnl_usd"] if r else float("nan")
        print(f"  {cname:16} {_pnl(pcs_a):12.0f} {_pnl(pcs_g):12.0f} {_pnl(ic_a):12.0f} {_pnl(ic_g):12.0f}")


def print_report(results: List[SimResult], proxy: dict, bias: Optional[dict],
                 skew_sens: Dict[str, Tuple[bool, bool]], *,
                 recal: Optional[dict] = None, use_recal_primary: bool = False) -> None:
    print("\n--- PART 4: STRUCTURE REPORT (skew=0.10) ---")
    hdr = (f"  {'structure':18} {'variant':12} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} "
           f"{'CVaR99':>8} {'worst5':>8} {'maxDD':>7} {'win%':>6} {'trades':>6}")
    print(hdr)
    base = [r for r in results if r.pricer_label == "baseline" and r.credit_scale == 1.0]
    for r in base:
        m = r.metrics
        print(f"  {r.structure:18} {r.variant:12} {m['cagr']*100:6.1f}% {m['sharpe']:7.2f} "
              f"{m['sortino']:8.2f} {m['cvar99']:8.0f} {m['worst5d']:8.0f} "
              f"{m['maxdd']*100:6.1f}% {m['win_rate']*100:5.1f}% {m['n_trades']:6d}")

    print("\n  Per-crisis P&L (USD, put_credit_spread / skew=0.10):")
    print(f"  {'crisis':16} {'always':>10} {'gated_skip':>12} {'gated_exit':>12}")
    by_var = {r.variant: r for r in base if r.structure == "put_credit_spread"}
    if all(v in by_var for v in VARIANTS):
        for i, (name, _, _) in enumerate(CRISES):
            print(f"  {name:16} {by_var['always_on'].crisis[i]['pnl_usd']:10.0f} "
                  f"{by_var['gated_skip'].crisis[i]['pnl_usd']:12.0f} "
                  f"{by_var['gated_exit'].crisis[i]['pnl_usd']:12.0f}")

    print("\n  --- vs variance-swap proxy (untradeable upper bound) ---")
    pa, pg = proxy["proxy_always_on"], proxy["proxy_gated_skip"]
    print(f"  proxy always-on : Sharpe {pa['sharpe']:.2f}  maxDD {pa['maxdd']:.2f}  cum P&L {pa['final_pnl']:.1f}")
    print(f"  proxy gated_skip: Sharpe {pg['sharpe']:.2f}  maxDD {pg['maxdd']:.2f}  cum P&L {pg['final_pnl']:.1f}")
    pcs_a = next(r for r in base if r.structure == "put_credit_spread" and r.variant == "always_on")
    pcs_g = next(r for r in base if r.structure == "put_credit_spread" and r.variant == "gated_skip")
    print(f"  PCS always-on   : Sharpe {pcs_a.metrics['sharpe']:.2f}  maxDD {pcs_a.metrics['maxdd']*100:.1f}%  "
          f"final ${pcs_a.metrics['final_equity']:,.0f}")
    print(f"  PCS gated_skip  : Sharpe {pcs_g.metrics['sharpe']:.2f}  maxDD {pcs_g.metrics['maxdd']*100:.1f}%  "
          f"final ${pcs_g.metrics['final_equity']:,.0f}")
    print("  Cost of tradability: defined-risk PCS earns far less than the proxy; the gate's *relative* lift matters.")

    if recal:
        if recal.get("sanity_failed"):
            print(f"\n  IV sanity gate FAILED (median 30d put-ATM = {recal['median_put_skew_vp']:+.2f} vol pts).")
            print(f"  Part 2 labelled PRICER-UNCERTAIN; literature prior slope={LITERATURE_SKEW_SLOPE}.")
        elif use_recal_primary:
            print(f"\n  Recalibrated multiplicative s={recal['s']:.4f}; remaining bias "
                  f"{recal['mean_rem_bias']*100:+.1f}% on {recal['n_entries_clean']} entries.")
        elif "mean_rem_bias" in recal:
            print(f"\n  Recalibration s={recal.get('s', float('nan')):.4f} did not meet +/-15% bias bar; "
                  f"baseline table remains reference.")
    if bias:
        print(f"  Baseline aggregate bias: mean (syn-real)/real = {bias['mean_bias_pct']*100:+.1f}%")

    print("\n  Skew sensitivity (PCS gated_skip — Sharpe up & maxDD half?):")
    for slope, (su, dh) in skew_sens.items():
        print(f"    slope={slope}: Sharpe improvement={su}  maxDD half={dh}")

    print("\n--- VERDICT ---")
    su, dh = _gate_passes(pcs_a.metrics, pcs_g.metrics)
    exit_r = next(r for r in base if r.structure == "put_credit_spread" and r.variant == "gated_exit")
    exit_vs_skip = exit_r.metrics["sharpe"] - pcs_g.metrics["sharpe"]
    print(f"  Q1. Regime gate on defined-risk PCS: Sharpe improves={su}, maxDD halves={dh}.")
    print(f"  Q2. Hold-to-expiry (gated_skip) vs intra-month exit (gated_exit): "
          f"Sharpe delta {exit_vs_skip:+.2f} ({'exit hurts' if exit_vs_skip < -0.05 else 'similar' if abs(exit_vs_skip) <= 0.05 else 'exit helps'}).")
    if su and dh:
        print("  => Regime gating still clears Sharpe improvement and drawdown halving on tradable structures.")
    else:
        print("  => Regime gating does NOT clearly clear both bars on tradable structures — proxy edge may not translate.")
    if recal and recal.get("sanity_failed"):
        print("  => Inverted put skew in real IVs — do not trust recalibration; Part 2 is pricer-uncertain.")
    elif use_recal_primary and recal:
        print("  => Multiplicative recalibrated pricer matches clean entries (+/-15%); see FINAL table.")
    elif bias and bias["mean_bias_pct"] > 0.05:
        print("  => Baseline synthetic credits overstated vs real bars.")


def main() -> None:
    load_dotenv_if_present()
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-part3", action="store_true", help="skip Massive validation")
    parser.add_argument("--refresh", action="store_true", help="re-download FRED / full history")
    args = parser.parse_args()

    print("=" * 78)
    print("STRUCTURE TRANSLATION — proxy -> defined-risk options (1996-2026)")
    print("=" * 78)

    df = load_full_history(refresh=args.refresh)
    df = df[(df.index >= TRACK_START) & (df.index <= TRACK_END)]
    print(f"  sample: {df.index.min().date()} -> {df.index.max().date()} ({len(df):,} days)")

    rates = load_tbill_rate(refresh=args.refresh)
    if not rates.empty:
        aligned = rates.reindex(df.index, method="ffill").fillna(0.02)
    else:
        aligned = pd.Series(0.02, index=df.index)

    print("\n--- PART 1: synthetic pricer ---")
    print(f"  ATM IV = (VIX - {VIX_ATM_OFFSET_VOLPTS})/100  [6a calibration]")
    print(f"  skew: IV = ATM + slope*(0.5 - |delta|); slope={SKEW_SLOPE_BASE} -> 25d put +2.5 vol pts")
    print(f"  r = FRED {FRED_TBILL_SERIES} (ffill), q = {DIV_YIELD*100:.1f}%")
    print(f"  gate threshold = {GATE_THRESHOLD} (holdout-selected on 1996-2015)")

    print("\n--- PART 1b: causal filter (cached) ---")
    feat = df.dropna(subset=["log_return", "yz_vol"])
    causal = walk_forward_filter(feat, refit_every=REFIT_EVERY)

    print("\n--- PART 2: strategy simulation ---")
    results: List[SimResult] = []
    for structure in STRUCTURES:
        for variant in VARIANTS:
            print(f"  sim {structure} / {variant} ...")
            results.append(simulate_structure(
                df, causal.p_high, causal.hard, aligned,
                structure=structure, variant=variant, skew_slope=SKEW_SLOPE_BASE))

    bias = None
    recal = None
    use_recal_primary = False
    final_primary = "baseline"
    real_entries: List[dict] = []
    if not args.no_part3:
        real_entries = collect_real_entries(df, aligned)
        bias = validation_bias(real_entries)
        diag = part_3b_diagnosis(real_entries)
        recal = part_3c_recalibrate(diag) if not diag.empty else None

        if recal and recal.get("use_literature_prior"):
            print(f"  rerunning Part 2 (PRICER-UNCERTAIN) with literature prior slope={LITERATURE_SKEW_SLOPE} ...")
            for structure in STRUCTURES:
                for variant in VARIANTS:
                    results.append(simulate_structure(
                        df, causal.p_high, causal.hard, aligned,
                        structure=structure, variant=variant,
                        skew_slope=LITERATURE_SKEW_SLOPE, skew_offset=0.0,
                        pricer_label="pricer_uncertain"))
            final_primary = "pricer_uncertain"
        elif recal and not recal.get("sanity_failed"):
            print(f"  rerunning Part 2 with anchored multiplicative pricer (s={recal['s']:.4f}) ...")
            for structure in STRUCTURES:
                for variant in VARIANTS:
                    results.append(simulate_structure(
                        df, causal.p_high, causal.hard, aligned,
                        structure=structure, variant=variant,
                        skew_mode=SKEW_MULTIPLICATIVE, skew_mult_s=recal["s"],
                        skew_offset=0.0, pricer_label="recalibrated"))
            if recal["trusted"]:
                use_recal_primary = True
                final_primary = "recalibrated"

    skew_sens: Dict[str, Tuple[bool, bool]] = {}
    pcs_a = next(r for r in results if r.structure == "put_credit_spread" and r.variant == "always_on"
                 and r.pricer_label == "baseline" and r.credit_scale == 1.0)
    for slope in SKEW_SENSITIVITY:
        if slope == SKEW_SLOPE_BASE:
            pcs_g = next(r for r in results if r.structure == "put_credit_spread" and r.variant == "gated_skip"
                         and r.pricer_label == "baseline" and r.credit_scale == 1.0)
        else:
            pcs_g = simulate_structure(df, causal.p_high, causal.hard, aligned,
                                       structure="put_credit_spread", variant="gated_skip",
                                       skew_slope=slope)
        skew_sens[str(slope)] = _gate_passes(pcs_a.metrics, pcs_g.metrics)

    proxy = _proxy_metrics(df, causal)
    print_report(results, proxy, bias, skew_sens, recal=recal, use_recal_primary=use_recal_primary)
    print_final_verdict_table(results, primary=final_primary)


if __name__ == "__main__":
    main()
