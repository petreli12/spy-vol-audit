# spy-vol-audit

A 30-year statistical audit of candlestick patterns, volatility regimes, and the variance risk premium on SPY.

I built four reproducible scripts to ask whether candlestick folklore predicts the next bar, whether a causal volatility-regime filter can dodge short-vol tail risk, and whether any of it survives honest measurement. Candlestick direction signals fail after costs. A three-state HMM on volatility is real and persistent. The variance risk premium exists at roughly +3.8 annualized vol points once I fixed a Garman-Klass estimator bug that had inflated Sharpe from 6.15 to a still-positive 1.11. Causal regime gating on a stylized short-vol proxy holds up on a 2016-2026 holdout (friction-adjusted Sharpe 2.76), and the relative edge survives translation into defined-risk spreads, but its absolute size depends on whether synthetic credits match real quoted marks; historical bar validation at this data tier showed ~80% overstatement and a failed IV sanity gate. Everything here is simulated research output, not a live trading record.

**[Read the full writeup](WRITEUP.md)**

## Headline results

### Cross-timeframe candle results

| Timeframe | n_obs | χ² p-value | Strongest dev (pp) | Gross edge (bp) | Net @ 2 bp (bp) | HMM vol spread |
|-----------|------:|-----------:|-------------------:|----------------:|----------------:|---------------:|
| 1d        | 2,494 | 0.7915     | +5.10              | 7.43            | +5.43           | 3.47×          |
| 1h        | 7,219 | 0.1856     | −3.81              | 1.51            | −0.49           | 2.26×          |
| 15m       | 36,187| 0.0033     | +2.70              | 0.56            | −1.44           | 2.08×          |

### GK vs Yang-Zhang (short-vol proxy)

| Metric | GK (old) | Yang-Zhang (new) |
|--------|----------:|-----------------:|
| Mean VRP (vol points) | +7.65 | +3.81 |
| VRP % days positive | 99.9% | 88.9% |
| Proxy Sharpe (costless) | 6.15 | 1.11 |
| Proxy Sharpe (8% friction) | 4.90 | 0.65 |
| 2020-03 P&L (costless) | −5.00 | −14.44 |

### Per-crisis gate table (threshold 0.4, selected pre-2016)

| Crisis | Always-on P&L | Gated P&L | % flat | Warn lag | OOS? |
|--------|--------------:|----------:|-------:|---------:|:----:|
| GFC 2008-09 | −21.00 | −0.52 | 96% | +6d | sel |
| COVID (Feb-Apr 2020) | −14.92 | −0.76 | 73% | +16d | OOS |
| 2022 full year | +0.29 | −1.41 | 58% | +20d | OOS |

## Figures

![Variance risk premium time series](output/vrp_timeseries.png)

![Cumulative P&L: always-on vs causally gated short-vol proxy](output/regime_gated_cumulative.png)

More figures and tables: **[WRITEUP.md](WRITEUP.md)**.

## Related work

This audit sits in a line of work on variance risk premia and volatility timing: Marshall, Young, and Rose (2006) on candlestick efficacy; Caginalp and Laurent (1998) on predictive candle patterns; Duvinage, Mazza, and Petitjean (2013) on intraday pattern profitability; Carr and Wu (2009) on variance swap rates; Corsi (2009) HAR-RV forecasting; Moreira and Muir (2017) on volatility-managed portfolios; Cederburg, O'Doherty, Wang, and Yan (2020) on the real-time out-of-sample performance of volatility-managed portfolios; and López de Prado's *Advances in Financial Machine Learning* on leakage and multiple-testing discipline. I treat published VRP magnitudes (+2 to +4 vol points on SPY) as the sanity check that exposed my Garman-Klass bug.

## Quickstart

```bash
git clone https://github.com/petreli12/spy-vol-audit.git
cd spy-vol-audit
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional; daily yfinance paths need no keys
```

Run the four scripts (daily yfinance path, no API keys required):

```bash
python candle_markov_experiment.py --timeframes 1d      # ~30 s cold; ~5 s cached
python vrp_experiment.py --no-part6                     # ~30 s cold
python regime_strategy_experiment.py                      # ~5-15 min cold HMM; ~5 s cached
python structure_translation_experiment.py --no-part3     # ~60 s cold
```

Figures land in `output/`. Parquet caches land in `data/` (gitignored).

## Data requirements

| Data | Source | API key? |
|------|--------|----------|
| SPY daily, ^VIX | yfinance | No |
| SPY 1h / 15m | Alpaca IEX, Massive fallback | Optional (`ALPACA_*`, `MASSIVE_API_KEY`) |
| FRED DTB3 (risk-free rate) | FRED API | Optional (`FRED_API_KEY`; 2% flat fallback) |
| Option bar validation | Massive aggregates | Optional (`MASSIVE_API_KEY`; skip with `--no-part3` / `--no-part6`) |

## Repository layout

| Path | Role |
|------|------|
| `candle_markov_experiment.py` | Candle states, chi-square tests, HMM regimes |
| `vrp_experiment.py` | HAR-RV forecast, VRP series, short-vol proxy |
| `regime_strategy_experiment.py` | Causal walk-forward gate, threshold sweep |
| `structure_translation_experiment.py` | Synthetic pricer, PCS translation, Massive validation |
| `research_utils.py` | Friction config, `.env` loader, minimal FRED client |
| `WRITEUP.md` | Full narrative audit |
| `output/` | Generated figures (PNG); referenced by README and writeup |
| `requirements.txt` | Pinned Python dependencies |
| `.env.example` | Optional API key template (no keys required for daily yfinance path) |
| `LICENSE` | MIT license |

## Disclaimer

This repository is research software for education and replication. It is not investment advice, not a trading system, and not a live performance record.

## License

MIT License. See [LICENSE](LICENSE).
