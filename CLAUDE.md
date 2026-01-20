# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Alpaca ORB Trading Bot - A semi-autonomous day trading system implementing Opening Range Breakout (ORB) strategy for US stocks. The bot scans pre-market gappers, calculates opening ranges, detects breakout signals with multi-indicator confirmation (VWAP/Volume/RSI/MACD/Sentiment), and sends alerts via Telegram for manual confirmation before execution.

**Trading Mode:** Paper trading (default), US Eastern Time market hours

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the trading bot
python main.py

# Run tests
python test_live.py       # Full live scanner test (async)
python test_signal.py     # Signal generation validation
python test_alpaca.py     # API connectivity test
python test_telegram.py   # Telegram bot test
```

## Architecture

```
config/settings.py      → Centralized config (dataclasses for Alpaca, Telegram, Trading params)
data/market_data.py     → Alpaca API client (bars, quotes, volume data)
data/indicators.py      → Technical indicators (RSI, VWAP, MACD, Stochastic, Bollinger)
data/sentiment.py       → Finnhub API sentiment + keyword fallback with caching
scanner/premarket.py    → Pre-market gap scanning with composite scoring
strategy/orb.py         → ORB calculation, signal generation, Kelly Criterion sizing
execution/orders.py     → Bracket order execution, position management
notifications/telegram_bot.py → Telegram commands and signal alerts
main.py                 → TradingBot orchestrator with APScheduler
```

### Daily Execution Flow

1. **09:25** - Pre-market scan (gap + sentiment analysis)
2. **09:45** - ORB calculation for watchlist (first 15 min high/low)
3. **09:46** - Breakout monitoring starts (10-sec intervals)
4. **09:46-16:00** - Signal detection → Telegram alert → User confirm → Execute
5. **16:00** - Force close positions, daily summary (market close)

### Signal Generation

**LONG:** Price > ORB High + above VWAP + volume spike (1.5x) + RSI < 70 + MACD bullish + sentiment >= -0.3

**SHORT:** Price < ORB Low + below VWAP + volume spike (1.5x) + RSI > 30 + MACD bearish + sentiment <= 0.3

### Key Singletons

Modules expose pre-instantiated singletons for import:
- `from data.market_data import market_data`
- `from scanner.premarket import premarket_scanner, SCAN_UNIVERSE`
- `from strategy.orb import orb_strategy`
- `from execution.orders import order_executor`

## Configuration

Copy `.env.example` to `.env` and set:
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_PAPER=true`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `FINNHUB_API_KEY`, `SENTIMENT_ENABLED=true`

## Key Parameters (config/settings.py)

- Capital: $25,000 | Risk per trade: 2% | Max daily trades: 3
- ORB period: 15 minutes | Min gap: 2% | Price range: $10-$500
- Min premarket volume: 500K | Min avg daily volume: 1M
- RSI overbought/oversold: 70/30 | Take profit ratio: 2:1

## Important Constraints

- **No artificial data** - All market data from Alpaca API, sentiment from Finnhub
- **Fully async** - Main bot, Telegram, and all API calls use asyncio/aiohttp
- **Rate limits** - 500ms delay between Finnhub calls (60/min), 10-sec monitoring interval
- **Sentiment cache** - 30-minute TTL to reduce API calls

## Logging

- **stdout:** INFO level (real-time)
- **File:** `logs/trading_YYYY-MM-DD.log` (DEBUG level, 30-day retention)
