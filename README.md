# spy-vol-audit

A 30-year statistical audit of candlestick patterns, volatility regimes, and the variance risk premium on SPY.

I built four reproducible scripts to ask whether candlestick folklore predicts the next bar, whether a causal volatility-regime filter can dodge short-vol tail risk, and whether any of it survives honest measurement. Candlestick direction signals fail after costs. A three-state HMM on volatility is real and persistent. The variance risk premium exists at roughly +3.8 annualized vol points once I fixed a Garman-Klass estimator bug that had inflated Sharpe from 6.15 to a still-positive 1.11. Causal regime gating on a stylized short-vol proxy holds up on a 2016-2026 holdout (friction-adjusted Sharpe 2.76), and the relative edge survives translation into defined-risk spreads, but its absolute size depends on whether synthetic credits match real quoted marks; historical bar validation at this data tier showed ~80% overstatement and a failed IV sanity gate. Everything here is simulated research output, not a live trading record.

**[Read the full writeup](WRITEUP.md)** and the **[Medium article](https://medium.com/@olayemioladapo1/i-audited-30-years-of-spy-candlesticks-and-the-variance-risk-premium-9f0bb733965e)** for more information.


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

This audit sits in a line of work on variance risk premia, volatility timing, and technical-pattern testing: Marshall, Young, and Rose (2006) and Marshall, Young, and Cahan (2008) on candlestick efficacy; Caginalp and Laurent (1998) on predictive price patterns; Duvinage, Mazza, and Petitjean (2013) on intraday candlestick profitability; Carr and Wu (2009) on variance risk premiums; Corsi (2009) on HAR-RV forecasting; Moreira and Muir (2017) on volatility-managed portfolios; Cederburg, O'Doherty, Wang, and Yan (2020) on real-time out-of-sample performance of volatility-managed portfolios; and López de Prado (2018) on backtest overfitting and multiple-testing discipline. Estimators and inference follow Garman and Klass (1980), Yang and Zhang (2000), Hamilton (1989), Newey and West (1987), Wilson (1927), and Bailey and López de Prado (2014). I treat published SPY VRP magnitudes (+2 to +4 vol points) as the sanity check that exposed my Garman-Klass bug. Full citations are in [References](#references) below and in [WRITEUP.md](WRITEUP.md#references).

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

## References

**Related work**

- Caginalp, G., & Laurent, H. (1998). The predictive power of price patterns. 
  *Applied Mathematical Finance*, 5(3-4), 181-205. https://doi.org/10.1080/135048698334637
- Carr, P., & Wu, L. (2009). Variance risk premiums. *Review of Financial 
  Studies*, 22(3), 1311-1341. https://doi.org/10.1093/rfs/hhn062
- Cederburg, S., O'Doherty, M. S., Wang, F., & Yan, X. S. (2020). On the 
  performance of volatility-managed portfolios. *Journal of Financial 
  Economics*, 138(1), 95-117. https://doi.org/10.1016/j.jfineco.2020.04.015
- Corsi, F. (2009). A simple approximate long-memory model of realized 
  volatility. *Journal of Financial Econometrics*, 7(2), 174-196. https://doi.org/10.1093/jjfinec/nbp001
- Duvinage, M., Mazza, P., & Petitjean, M. (2013). The intra-day performance 
  of market timing strategies and trading systems based on Japanese 
  candlesticks. *Quantitative Finance*, 13(7), 1059-1070. https://doi.org/10.1080/14697688.2013.768774
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. 
  Hoboken, NJ: Wiley. https://doi.org/10.1002/9781119482086
- Marshall, B. R., Young, M. R., & Rose, L. C. (2006). Candlestick technical 
  trading strategies: Can they create value for investors? *Journal of 
  Banking & Finance*, 30(8), 2303-2323. https://doi.org/10.1016/j.jbankfin.2005.08.001
- Marshall, B. R., Young, M. R., & Cahan, R. (2008). Are candlestick 
  technical trading strategies profitable in the Japanese equity market? 
  *Review of Quantitative Finance and Accounting*, 31(2), 191-207. https://doi.org/10.1007/s11156-007-0068-1
- Moreira, A., & Muir, T. (2017). Volatility-managed portfolios. *Journal 
  of Finance*, 72(4), 1611-1644. https://doi.org/10.1111/jofi.12518

**Methods**

- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: 
  Correcting for selection bias, backtest overfitting, and non-normality. 
  *Journal of Portfolio Management*, 40(5), 94-107. https://doi.org/10.3905/jpm.2014.40.5.094
- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price 
  volatilities from historical data. *Journal of Business*, 53(1), 67-78. https://doi.org/10.1086/296072
- Hamilton, J. D. (1989). A new approach to the economic analysis of 
  nonstationary time series and the business cycle. *Econometrica*, 57(2), 
  357-384. https://doi.org/10.2307/1912559
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, 
  heteroskedasticity and autocorrelation consistent covariance matrix. 
  *Econometrica*, 55(3), 703-708. https://doi.org/10.2307/1913610
- Rogers, L. C. G., & Satchell, S. E. (1991). Estimating variance from high, 
  low and closing prices. *Annals of Applied Probability*, 1(4), 504-512. https://doi.org/10.1214/aoap/1177007967
- Wilson, E. B. (1927). Probable inference, the law of succession, and 
  statistical inference. *Journal of the American Statistical Association*, 
  22(158), 209-212. https://doi.org/10.2307/2276774
- Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation 
  based on high, low, open, and close prices. *Journal of Business*, 73(3), 
  477-491. https://doi.org/10.1086/209655

## Disclaimer

This repository is research software for education and replication. It is not investment advice, not a trading system, and not a live performance record.

## License

MIT License. See [LICENSE](LICENSE).
