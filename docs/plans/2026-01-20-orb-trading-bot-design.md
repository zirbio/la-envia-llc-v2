# Alpaca ORB Trading Bot - Design Document

**Date:** 2026-01-20
**Status:** Approved
**Author:** Silvio Requena + Claude

---

## Overview

Semi-autonomous day trading bot using Opening Range Breakout (ORB) strategy with VWAP and Volume confirmation. Sends trade signals via Telegram for manual confirmation before execution.

---

## Specifications

| Parameter | Value |
|-----------|-------|
| **Market** | US Stocks |
| **Strategy** | ORB + VWAP + Volume |
| **Capital** | $10,000 - $25,000 |
| **Risk per trade** | 2% max |
| **Trades per day** | 1-3 max |
| **Trading hours** | 9:30 - 11:30 AM EST |
| **Mode** | Semi-autonomous (Telegram confirmation) |
| **Broker** | Alpaca (Paper Trading) |

---

## Strategy Rules

### Pre-Market Scanner (6:00 - 9:25 AM EST)

Criteria for watchlist:
- Gap > 2% (up or down)
- Pre-market volume > 500K shares
- Price between $10 - $500
- Average daily volume > 1M (last 20 days)
- No earnings that day

Select TOP 5-10 candidates.

### Opening Range (9:30 - 9:45 AM EST)

For each ticker in watchlist:
- Record HIGH of first 15 minutes
- Record LOW of first 15 minutes
- Calculate range = HIGH - LOW

### Entry Signals (9:45 - 11:30 AM EST)

**LONG if:**
- Price breaks ABOVE ORB High
- Price is ABOVE VWAP
- Candle volume > 1.5x average
- RSI < 70 (not overbought)

**SHORT if:**
- Price breaks BELOW ORB Low
- Price is BELOW VWAP
- Candle volume > 1.5x average
- RSI > 30 (not oversold)

### Risk Management

- **Stop Loss:** Opposite side of ORB (if LONG → stop at ORB Low)
- **Take Profit:** 2:1 ratio (risk $200 → target $400)
- **Position Size:** (Capital × 2%) / Distance to Stop
- **Session Close:** Force close all positions at 11:30 AM EST

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                ALPACA TRADING BOT (Semi-Auto)               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  PRE-MARKET  │───▶│   SCANNER    │───▶│  WATCHLIST   │  │
│  │   6:00 AM    │    │  Gap + Vol   │    │  Top 5-10    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                              │                              │
│                              ▼                              │
│                     ┌──────────────┐                        │
│                     │  TELEGRAM    │  ◀── "Watchlist ready" │
│                     │   Alert      │                        │
│                     └──────────────┘                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  MARKET OPEN │───▶│   SIGNAL     │───▶│  TELEGRAM    │  │
│  │   9:30 AM    │    │  ORB+VWAP    │    │  + Details   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                  │          │
│                                                  ▼          │
│                                          ┌──────────────┐   │
│                                          │  USER DECISION│  │
│                                          │  YES / NO     │  │
│                                          └──────────────┘   │
│                                                  │          │
│                              ┌───────────────────┘          │
│                              ▼                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   EXECUTE    │───▶│   MONITOR    │───▶│ AUTO CLOSE   │  │
│  │   If YES     │    │  Stop/Target │    │  11:30 AM    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
alpaca-trading-bot/
├── config/
│   ├── settings.py          # API keys, capital, risk params
│   └── telegram_config.py   # Bot token, chat ID
│
├── scanner/
│   ├── premarket.py         # Scan gaps + volume
│   └── watchlist.py         # Manage candidate list
│
├── strategy/
│   ├── orb.py               # Calculate Opening Range
│   ├── vwap.py              # Calculate VWAP real-time
│   ├── signals.py           # Generate entry signals
│   └── risk_manager.py      # Position sizing, stop/target
│
├── execution/
│   ├── orders.py            # Send orders to Alpaca
│   └── positions.py         # Monitor open positions
│
├── notifications/
│   └── telegram_bot.py      # Alerts + wait for confirmation
│
├── data/
│   ├── market_data.py       # Get data from Alpaca
│   └── indicators.py        # RSI, relative volume
│
├── main.py                  # Main loop
├── scheduler.py             # Schedule tasks by time
└── requirements.txt         # Dependencies
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Broker API | alpaca-py (official SDK) |
| Market Data | Alpaca Market Data API |
| Notifications | python-telegram-bot |
| Scheduler | APScheduler |
| Logs | loguru |
| Data Analysis | pandas |

---

## Telegram Bot Commands

| Command | Action |
|---------|--------|
| `/status` | View open positions |
| `/watchlist` | View today's watchlist |
| `/stop` | Pause the bot |
| `/start` | Resume the bot |
| `/close` | Close all positions |
| `/stats` | Weekly statistics |

---

## Daily Execution Flow

1. **06:00 AM** - Bot starts, connects to Alpaca, begins pre-market scan
2. **09:25 AM** - Telegram alert with watchlist
3. **09:30 AM** - Market opens, start calculating Opening Range
4. **09:45 AM** - ORB defined, start monitoring for breakouts
5. **09:45-11:30 AM** - Signal detection, Telegram alerts, wait for confirmation
6. **On confirmation** - Execute order, monitor position
7. **11:30 AM** - Force close all positions, send daily summary

---

## Implementation Phases

### Phase 1: Foundation
- [ ] Project setup and dependencies
- [ ] Alpaca API connection
- [ ] Basic market data retrieval

### Phase 2: Scanner
- [ ] Pre-market gap scanner
- [ ] Volume filter
- [ ] Watchlist management

### Phase 3: Strategy
- [ ] Opening Range calculation
- [ ] VWAP calculation
- [ ] Signal generation logic

### Phase 4: Execution
- [ ] Order placement
- [ ] Position monitoring
- [ ] Stop/Target management

### Phase 5: Notifications
- [ ] Telegram bot setup
- [ ] Alert formatting
- [ ] Confirmation handling

### Phase 6: Integration
- [ ] Scheduler setup
- [ ] Main loop
- [ ] Error handling and logging

### Phase 7: Testing
- [ ] Backtesting with historical data
- [ ] Paper trading validation
- [ ] Performance optimization

---

## Success Metrics

- **Win Rate Target:** > 55%
- **Profit Factor Target:** > 1.5
- **Max Drawdown:** < 10% of capital
- **Sharpe Ratio:** > 1.0

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| API downtime | Retry logic, graceful degradation |
| Slippage | Use limit orders when possible |
| Overtrading | Hard limit of 3 trades/day |
| False breakouts | VWAP + Volume confirmation |
| Overnight positions | Force close at 11:30 AM |

---

## Next Steps

1. Configure Telegram bot
2. Implement code module by module
3. Backtest strategy with historical data
4. Run in paper trading mode
5. Optimize parameters based on results
