# Backtesting Module Design

## Overview

Integrate Lumibot backtesting framework with the existing ORB trading bot to validate strategy performance against historical data from Alpaca.

## Decisions

- **Framework:** Lumibot (native Alpaca integration, same code for backtest/live)
- **Data source:** Alpaca API (1-minute bars)
- **Default period:** Last 30 days (customizable via CLI)
- **Metrics:** Complete set with interactive HTML charts

## Architecture

```
backtesting/
├── __init__.py
├── orb_backtest.py    # ORB strategy adapted to Lumibot
├── data_loader.py     # Load/cache data from Alpaca
├── run_backtest.py    # CLI entrypoint
└── reports.py         # Additional metrics and reporting
```

## Data Flow

```
CLI (run_backtest.py)
        ↓
data_loader.py → Alpaca API → Local cache (parquet)
        ↓
orb_backtest.py (Lumibot Strategy)
        ↓
Lumibot backtest engine
        ↓
reports.py → Console output + HTML report
```

## Strategy Mapping

| Current Code | Lumibot Equivalent |
|--------------|-------------------|
| `strategy/orb.py` → `calculate_opening_range()` | `on_trading_iteration()` (09:30-09:45) |
| `strategy/orb.py` → `check_breakout()` | `on_trading_iteration()` (post 09:45) |
| `strategy/orb.py` → `_check_long_conditions()` | Reused directly |
| `strategy/orb.py` → `_check_short_conditions()` | Reused directly |
| `data/indicators.py` | Imported and reused |
| `config/settings.py` | Default parameters source |

## Metrics

| Metric | Description |
|--------|-------------|
| Total Return | Overall % return |
| Win Rate | % winning trades |
| Profit Factor | Gross profit / Gross loss |
| Sharpe Ratio | Risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Avg Win / Avg Loss | Average gain vs loss per trade |
| Expectancy | Expected return per trade |
| Total Trades | Number of executed trades |

## CLI Interface

```bash
# Basic (last 30 days, scanner symbols)
python -m backtesting.run_backtest

# Custom symbols
python -m backtesting.run_backtest --symbols AAPL,TSLA,NVDA

# Custom date range
python -m backtesting.run_backtest --start 2024-12-01 --end 2024-12-31

# Parameter tuning
python -m backtesting.run_backtest --orb-period 10 --rsi-overbought 75
```

## Output

```
$ python -m backtesting.run_backtest --days 30

ORB Strategy Backtest Results (30 days)
==========================================
Total Return:     +8.42%
Win Rate:         62.5% (15/24 trades)
Profit Factor:    1.87
Sharpe Ratio:     1.24
Max Drawdown:     -3.21%
Avg Win:          +1.12%
Avg Loss:         -0.68%
Expectancy:       +0.35% per trade

Report saved: reports/backtest_2025-01-21.html
```

## Dependencies

```
lumibot>=3.0.0
```

## Limitations

- No historical sentiment data (Finnhub free tier limitation)
- Backtest assumes sentiment = 0 (neutral) for all signals
- Pre-market gap data may be limited for older dates

## Files to Create

1. `backtesting/__init__.py`
2. `backtesting/data_loader.py`
3. `backtesting/orb_backtest.py`
4. `backtesting/run_backtest.py`
5. `backtesting/reports.py`
6. Update `requirements.txt` with lumibot
