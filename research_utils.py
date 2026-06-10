"""Shared utilities for the SPY volatility research scripts.

Vendored from a private trading project so this repo runs standalone:
friction assumptions, optional .env loading, and a minimal FRED client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parent

SHARES_PER_CONTRACT = 100


@dataclass(frozen=True)
class FrictionConfig:
    """Train and live share this object; defaults match retail Alpaca-style assumptions."""

    half_spread_pct: float = 0.05
    commission_usd_per_contract_per_fill: float = 0.65

    @classmethod
    def frictionless(cls) -> FrictionConfig:
        return cls(half_spread_pct=0.0, commission_usd_per_contract_per_fill=0.0)

    @property
    def is_frictionless(self) -> bool:
        return self.half_spread_pct == 0.0 and self.commission_usd_per_contract_per_fill == 0.0


def bid_from_mid(mid: float, friction: FrictionConfig) -> float:
    if mid <= 0:
        return 0.0
    if friction.half_spread_pct <= 0:
        return float(mid)
    return max(float(mid) * (1.0 - friction.half_spread_pct), 1e-8)


def ask_from_mid(mid: float, friction: FrictionConfig) -> float:
    if mid <= 0:
        return 0.0
    if friction.half_spread_pct <= 0:
        return float(mid)
    return max(float(mid) * (1.0 + friction.half_spread_pct), 1e-8)


def entry_fill_price(*, is_buy: bool, mid: float, friction: FrictionConfig) -> float:
    return ask_from_mid(mid, friction) if is_buy else bid_from_mid(mid, friction)


def exit_fill_price(*, is_long: bool, mid: float, friction: FrictionConfig) -> float:
    return bid_from_mid(mid, friction) if is_long else ask_from_mid(mid, friction)


def entry_commission_usd(n_legs: int, friction: FrictionConfig) -> float:
    return n_legs * friction.commission_usd_per_contract_per_fill


@lru_cache(maxsize=1)
def get_friction() -> FrictionConfig:
    """Read optional FRICTION_* env overrides; otherwise use module defaults."""
    hs = float(os.environ.get("FRICTION_HALF_SPREAD_PCT", "0.05"))
    comm = float(os.environ.get("FRICTION_COMMISSION_USD", "0.65"))
    return FrictionConfig(half_spread_pct=hs, commission_usd_per_contract_per_fill=comm)


def load_dotenv_if_present() -> None:
    """Populate os.environ from ./.env without overwriting existing keys."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


class FredClient:
    """Minimal read-only FRED observations client (DTB3 and other series)."""

    BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY", "")

    @retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.get(self.BASE, params=params, timeout=60)
        r.raise_for_status()
        return r.json()

    def get_series_df(
        self,
        series_id: str,
        observation_start: Optional[date] = None,
        observation_end: Optional[date] = None,
    ) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("FRED_API_KEY is not set")
        params: Dict[str, Any] = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if observation_start is not None:
            params["observation_start"] = observation_start.isoformat()
        if observation_end is not None:
            params["observation_end"] = observation_end.isoformat()
        rows = self._get(params).get("observations") or []
        if not rows:
            return pd.DataFrame(columns=["observation_date", "series_id", "value"])
        out: List[Dict[str, Any]] = []
        for r in rows:
            raw = r.get("value")
            try:
                v: Optional[float] = float(raw) if raw not in (None, "", ".") else None
            except (TypeError, ValueError):
                v = None
            try:
                d = datetime.strptime(r["date"], "%Y-%m-%d").date()
            except Exception:
                continue
            out.append({"observation_date": d, "series_id": series_id, "value": v})
        df = pd.DataFrame(out)
        if df.empty:
            return df
        df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
        df["series_id"] = df["series_id"].astype("string")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df
