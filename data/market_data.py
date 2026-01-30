"""
Market data retrieval from Alpaca API

This module uses REST API polling for market data. For lower latency in production,
consider migrating to WebSocket streaming:

WebSocket Streaming (Future Enhancement):
- URL: wss://stream.data.alpaca.markets/v2/{feed}
- Feeds: iex (free), sip (paid, real-time)
- Subscription types: trades, quotes, bars
- Benefits: ~10-100ms latency vs 2-10s polling
- Limit: 1 connection per subscription type

Example WebSocket setup (for future implementation):
```python
from alpaca.data.live import StockDataStream

stream = StockDataStream(api_key, secret_key)

@stream.on_bar(symbols=['AAPL'])
async def on_bar(bar):
    # Handle real-time bar data
    pass

stream.run()
```

Current implementation uses REST polling which is suitable for:
- Paper trading with lower frequency requirements
- Development and testing
- Strategies with 2-10 second decision windows
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import pytz
from alpaca.data import StockHistoricalDataClient, StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import MostActivesRequest, MarketMoversRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from loguru import logger

from config.settings import settings
from utils.rate_limit import exponential_backoff_with_jitter

# US Eastern timezone for market hours calculations
EST = pytz.timezone('US/Eastern')

# Rate limiting constants for API calls
# Alpaca limits: ~200 req/min for data API, ~200 req/min for trading API
BATCH_DELAY_SECONDS = 0.3  # 300ms delay between batches to respect API limits
MAX_RETRIES = 3  # Maximum retry attempts for failed requests


class MarketDataClient:
    """
    Client for retrieving market data from Alpaca.

    IMPORTANT: This client uses SYNCHRONOUS methods (Alpaca SDK is sync).
    When calling from async contexts, use asyncio.to_thread():
        result = await asyncio.to_thread(market_data.get_bars, symbol)

    Features:
    - REST API polling for bars, quotes, and asset data
    - Batch requests for efficient multi-symbol queries
    - Automatic retry with exponential backoff for transient failures
    - Static rate limiting via BATCH_DELAY_SECONDS between requests

    Rate Limits (Alpaca):
    - Data API: ~200 requests/minute (free tier)
    - SDK doesn't expose rate limit headers, so we use static delays

    For production use with lower latency requirements, consider:
    - WebSocket streaming (see module docstring)
    - Alpaca paid tiers for higher rate limits
    """

    def __init__(self):
        self.data_client = StockHistoricalDataClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key
        )
        self.trading_client = TradingClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key,
            paper=settings.alpaca.paper
        )
        # Screener client for most actives and market movers
        self.screener_client = ScreenerClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key
        )
        # Phase 2: Cache for intraday volume profiles by symbol
        self.volume_profiles: dict[str, dict[int, float]] = {}
        # Note: Alpaca SDK doesn't expose rate limit headers, so we rely on
        # static BATCH_DELAY_SECONDS and retry backoff for rate limit handling

    def _retry_with_backoff(
        self,
        func,
        *args,
        max_retries: int = MAX_RETRIES,
        **kwargs
    ):
        """
        Execute a function with retry and exponential backoff.

        Uses jitter to prevent thundering herd when multiple requests retry.

        IMPORTANT: This is a SYNCHRONOUS method that uses time.sleep().
        The Alpaca SDK is synchronous, so this method blocks during retries.
        When calling from async contexts, wrap with asyncio.to_thread():
            result = await asyncio.to_thread(self._retry_with_backoff, func, *args)

        Args:
            func: Function to execute (must be synchronous)
            *args: Positional arguments for func
            max_retries: Maximum retry attempts
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            Last exception if all retries exhausted
        """
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                # Don't retry on final attempt
                if attempt >= max_retries:
                    logger.error(
                        f"API call failed after {max_retries + 1} attempts: {e}"
                    )
                    raise

                # Check if error is retryable
                error_str = str(e).lower()
                retryable = any(pattern in error_str for pattern in [
                    'timeout', 'connection', 'rate', 'throttl',
                    'temporary', 'unavailable', '429', '500', '502', '503', '504'
                ])

                if not retryable:
                    logger.error(f"Non-retryable error: {e}")
                    raise

                # Calculate backoff with jitter
                delay = exponential_backoff_with_jitter(
                    attempt,
                    base_delay=1.0,
                    max_delay=30.0,
                    jitter_factor=0.5
                )

                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

                time.sleep(delay)

        # Should never reach here - loop either returns or raises
        raise RuntimeError("Unexpected state in _retry_with_backoff")

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.Minute,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get historical bars for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            timeframe: Bar timeframe (Minute, Hour, Day)
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars

        Returns:
            DataFrame with OHLCV data
        """
        try:
            if start is None:
                start = datetime.now(EST) - timedelta(days=5)

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                limit=limit
            )

            bars = self.data_client.get_stock_bars(request)

            if symbol not in bars.data or not bars.data[symbol]:
                logger.warning(f"No bars data for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
                'vwap': float(bar.vwap) if bar.vwap else None
            } for bar in bars.data[symbol]])

            df.set_index('timestamp', inplace=True)
            return df

        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        """
        Get latest quote for a symbol with freshness validation

        Args:
            symbol: Stock symbol

        Returns:
            Dict with bid, ask, mid price, timestamp, and age_seconds
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)

            if symbol not in quotes:
                return None

            quote = quotes[symbol]
            quote_time = quote.timestamp
            now = datetime.now(timezone.utc)
            age_seconds = (now - quote_time).total_seconds()

            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            mid = (bid + ask) / 2

            # Log quote details for debugging
            logger.debug(
                f"{symbol} quote: bid=${bid:.2f}, ask=${ask:.2f}, "
                f"mid=${mid:.2f}, age={age_seconds:.1f}s"
            )

            # Warn if quote is stale (>60 seconds old)
            if age_seconds > 60:
                logger.warning(
                    f"{symbol}: Quote is {age_seconds:.0f}s old, may be stale"
                )

            return {
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'bid_size': int(quote.bid_size),
                'ask_size': int(quote.ask_size),
                'timestamp': quote_time,
                'age_seconds': age_seconds
            }

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    def get_tradable_assets(self) -> list[str]:
        """
        Get list of tradable US stock symbols

        Returns:
            List of tradable symbols
        """
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE
            )
            assets = self.trading_client.get_all_assets(request)

            tradable = [
                asset.symbol for asset in assets
                if asset.tradable and asset.fractionable
            ]

            logger.info(f"Found {len(tradable)} tradable assets")
            return tradable

        except Exception as e:
            logger.error(f"Error getting tradable assets: {e}")
            return []

    def get_daily_bars(self, symbol: str, days: int = 20) -> pd.DataFrame:
        """
        Get daily bars for calculating average volume

        Args:
            symbol: Stock symbol
            days: Number of days to retrieve

        Returns:
            DataFrame with daily OHLCV
        """
        start = datetime.now(EST) - timedelta(days=days + 5)
        return self.get_bars(
            symbol=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            limit=days
        )

    def get_premarket_data(self, symbol: str) -> Optional[dict]:
        """
        Get pre-market data for a symbol including High/Low levels

        Args:
            symbol: Stock symbol

        Returns:
            Dict with pre-market price change, volume, and High/Low levels
        """
        try:
            # Get previous day close
            daily = self.get_daily_bars(symbol, days=2)
            if daily.empty or len(daily) < 1:
                return None

            prev_close = daily['close'].iloc[-1]

            # Get latest quote (includes pre-market)
            quote = self.get_latest_quote(symbol)
            if not quote:
                return None

            current_price = quote['mid']
            gap_percent = ((current_price - prev_close) / prev_close) * 100

            # Get today's pre-market bars for volume and High/Low
            today = datetime.now(EST).replace(hour=4, minute=0, second=0, microsecond=0)
            premarket_bars = self.get_bars(
                symbol=symbol,
                timeframe=TimeFrame.Minute,
                start=today,
                limit=330  # 4:00 AM to 9:30 AM = 5.5 hours = 330 minutes
            )

            premarket_volume = 0
            premarket_high = current_price
            premarket_low = current_price

            if not premarket_bars.empty:
                premarket_volume = int(premarket_bars['volume'].sum())
                premarket_high = float(premarket_bars['high'].max())
                premarket_low = float(premarket_bars['low'].min())

            return {
                'symbol': symbol,
                'prev_close': prev_close,
                'current_price': current_price,
                'gap_percent': gap_percent,
                'premarket_volume': premarket_volume,
                'premarket_high': premarket_high,
                'premarket_low': premarket_low
            }

        except Exception as e:
            logger.error(f"Error getting pre-market data for {symbol}: {e}")
            return None

    def get_previous_close(self, symbol: str) -> Optional[float]:
        """
        Get previous day's closing price

        Args:
            symbol: Stock symbol

        Returns:
            Previous close price or None
        """
        try:
            daily = self.get_daily_bars(symbol, days=2)
            if daily.empty or len(daily) < 1:
                return None
            return float(daily['close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error getting previous close for {symbol}: {e}")
            return None

    def get_open_price(self, symbol: str) -> Optional[float]:
        """
        Get today's opening price (9:30 AM bar)

        Args:
            symbol: Stock symbol

        Returns:
            Today's open price or None
        """
        try:
            today = datetime.now(EST).replace(hour=9, minute=30, second=0, microsecond=0)
            bars = self.get_bars(
                symbol=symbol,
                timeframe=TimeFrame.Minute,
                start=today,
                limit=5
            )
            if bars.empty:
                return None
            return float(bars['open'].iloc[0])
        except Exception as e:
            logger.error(f"Error getting open price for {symbol}: {e}")
            return None

    def get_avg_daily_volume(self, symbol: str, days: int = 20) -> int:
        """
        Calculate average daily volume

        Args:
            symbol: Stock symbol
            days: Number of days for average

        Returns:
            Average daily volume
        """
        daily = self.get_daily_bars(symbol, days)
        if daily.empty:
            return 0
        return int(daily['volume'].mean())

    def build_intraday_volume_profile(self, symbol: str, lookback_days: int = 20) -> dict[int, float]:
        """
        Phase 2: Build intraday volume profile for time-adjusted RVOL.

        Calculates average volume by minute-of-day over last N trading days.

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back

        Returns:
            Dict mapping minute_index (0=9:30, 1=9:31, etc.) to average volume
        """
        try:
            # Get minute bars for the last lookback_days
            start = datetime.now(EST) - timedelta(days=lookback_days + 7)  # Extra buffer for weekends

            bars = self.get_bars(
                symbol=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                limit=lookback_days * 390  # ~390 minutes per trading day
            )

            if bars.empty:
                logger.warning(f"No bars data for {symbol} volume profile")
                return {}

            # Group bars by minute-of-day (0-389 where 0 = 9:30 AM)
            minute_volumes: dict[int, list[int]] = {}

            for idx, row in bars.iterrows():
                # Get timestamp and calculate minute index from market open (9:30)
                ts = idx
                if hasattr(ts, 'hour') and hasattr(ts, 'minute'):
                    # Calculate minutes since 9:30 AM
                    market_open_minutes = 9 * 60 + 30  # 9:30 AM in minutes
                    bar_minutes = ts.hour * 60 + ts.minute
                    minute_idx = bar_minutes - market_open_minutes

                    # Only include regular trading hours (0-389)
                    if 0 <= minute_idx < 390:
                        if minute_idx not in minute_volumes:
                            minute_volumes[minute_idx] = []
                        minute_volumes[minute_idx].append(int(row['volume']))

            # Calculate average volume for each minute
            profile = {}
            for minute_idx, volumes in minute_volumes.items():
                if volumes:
                    profile[minute_idx] = sum(volumes) / len(volumes)

            if profile:
                logger.debug(f"{symbol}: Built volume profile with {len(profile)} minutes")

            return profile

        except Exception as e:
            logger.error(f"Error building volume profile for {symbol}: {e}")
            return {}

    def cache_volume_profile(self, symbol: str, lookback_days: int = 20) -> None:
        """
        Phase 2: Build and cache volume profile for a symbol.

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back
        """
        profile = self.build_intraday_volume_profile(symbol, lookback_days)
        if profile:
            self.volume_profiles[symbol] = profile
            logger.info(f"Cached volume profile for {symbol}")

    def calculate_time_adjusted_rvol(
        self,
        symbol: str,
        current_minute: int,
        cumulative_volume: int
    ) -> float:
        """
        Phase 2: Calculate time-adjusted relative volume.

        RVOL = cumulative_volume_today / expected_volume_at_this_time

        Args:
            symbol: Stock symbol
            current_minute: Minutes since 9:30 (e.g., 16 = 9:46 AM)
            cumulative_volume: Total volume traded today up to current_minute

        Returns:
            Time-adjusted relative volume ratio (1.0 = average, 2.0 = 2x typical)
        """
        profile = self.volume_profiles.get(symbol)

        if not profile:
            # Fallback to simple RVOL if no profile available
            logger.debug(f"{symbol}: No volume profile, using fallback RVOL")
            return 1.0

        # Calculate expected cumulative volume up to current_minute
        expected = sum(
            profile.get(i, 0)
            for i in range(current_minute + 1)
        )

        if expected <= 0:
            return 1.0

        rvol = cumulative_volume / expected

        logger.debug(
            f"{symbol}: Time-adjusted RVOL at minute {current_minute}: "
            f"{rvol:.2f}x (cumul={cumulative_volume:,}, expected={expected:,.0f})"
        )

        return rvol

    def get_cumulative_volume_today(self, symbol: str) -> int:
        """
        Phase 2: Get cumulative volume for today's session.

        Args:
            symbol: Stock symbol

        Returns:
            Total volume traded today from 9:30 to now
        """
        try:
            # Get today's bars from market open
            today = datetime.now(EST).replace(hour=9, minute=30, second=0, microsecond=0)

            bars = self.get_bars(
                symbol=symbol,
                timeframe=TimeFrame.Minute,
                start=today,
                limit=390
            )

            if bars.empty:
                return 0

            return int(bars['volume'].sum())

        except Exception as e:
            logger.error(f"Error getting cumulative volume for {symbol}: {e}")
            return 0

    def get_current_minute_index(self) -> int:
        """
        Phase 2: Get current minute index since market open (9:30 AM).

        Returns:
            Minute index (0 = 9:30, 16 = 9:46, etc.)
        """
        now = datetime.now(EST)
        market_open_minutes = 9 * 60 + 30
        current_minutes = now.hour * 60 + now.minute
        return max(0, current_minutes - market_open_minutes)

    def get_5min_bars(self, symbol: str, limit: int = 50) -> pd.DataFrame:
        """
        Get 5-minute bars for trailing stop EMA calculation

        Args:
            symbol: Stock symbol
            limit: Maximum number of bars to retrieve

        Returns:
            DataFrame with OHLCV data in 5-minute intervals
        """
        return self.get_bars(
            symbol=symbol,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            limit=limit
        )

    def get_latest_quotes_batch(self, symbols: list[str]) -> dict[str, dict]:
        """
        Get latest quotes for multiple symbols in a single API call.

        More efficient than calling get_latest_quote() for each symbol
        when checking multiple symbols in the watchlist.

        Uses retry with exponential backoff for transient failures.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to quote data (bid, ask, mid, age_seconds, etc.)
        """
        if not symbols:
            return {}

        def _fetch_quotes():
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            return self.data_client.get_stock_latest_quote(request)

        try:
            # Use retry with backoff for the API call
            quotes = self._retry_with_backoff(_fetch_quotes)

            now = datetime.now(timezone.utc)
            result = {}
            stale_count = 0

            for symbol in symbols:
                if symbol in quotes:
                    quote = quotes[symbol]
                    bid = float(quote.bid_price) if quote.bid_price else 0.0
                    ask = float(quote.ask_price) if quote.ask_price else 0.0
                    # Avoid division by zero during market halts or for illiquid stocks
                    mid = (bid + ask) / 2 if (bid + ask) > 0 else 0.0

                    quote_time = quote.timestamp
                    age_seconds = (now - quote_time).total_seconds() if quote_time else 0.0

                    if age_seconds > 60:
                        stale_count += 1

                    result[symbol] = {
                        'symbol': symbol,
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'bid_size': int(quote.bid_size) if quote.bid_size else 0,
                        'ask_size': int(quote.ask_size) if quote.ask_size else 0,
                        'timestamp': quote_time,
                        'age_seconds': age_seconds
                    }

            if stale_count > 0:
                logger.warning(f"Batch quotes: {stale_count}/{len(result)} quotes are >60s old")

            logger.debug(f"Fetched {len(result)}/{len(symbols)} quotes in batch")
            return result

        except Exception as e:
            logger.error(f"Error getting batch quotes (after retries): {e}")
            return {}

    def get_bars_batch(
        self,
        symbols: list[str],
        timeframe: TimeFrame = TimeFrame.Minute,
        limit: int = 50
    ) -> dict[str, pd.DataFrame]:
        """
        Get bars for multiple symbols in a single API call.

        More efficient than calling get_bars() for each symbol
        when building indicators for multiple symbols.

        Uses retry with exponential backoff for transient failures.

        Args:
            symbols: List of stock symbols
            timeframe: Bar timeframe
            limit: Maximum number of bars per symbol

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        if not symbols:
            return {}

        def _fetch_bars():
            start = datetime.now(EST) - timedelta(days=5)
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start,
                limit=limit
            )
            return self.data_client.get_stock_bars(request)

        try:
            # Use retry with backoff for the API call
            bars = self._retry_with_backoff(_fetch_bars)

            result = {}
            for symbol in symbols:
                if symbol in bars.data and bars.data[symbol]:
                    df = pd.DataFrame([{
                        'timestamp': bar.timestamp,
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume),
                        'vwap': float(bar.vwap) if bar.vwap else None
                    } for bar in bars.data[symbol]])

                    df.set_index('timestamp', inplace=True)
                    result[symbol] = df

            logger.debug(f"Fetched bars for {len(result)}/{len(symbols)} symbols in batch")
            return result

        except Exception as e:
            logger.error(f"Error getting batch bars (after retries): {e}")
            return {}

    # ========== Dynamic Universe Methods ==========

    def get_tradeable_universe(self) -> list[str]:
        """
        Get all tradeable US equity symbols from Alpaca (Tier 1 filter).

        Filters for:
        - Active status
        - Tradeable
        - Shortable (optional via config)
        - No special characters (., /)
        - Symbol length <= max_symbol_length

        Returns:
            List of tradeable symbols (~5000)
        """
        try:
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE
            )
            assets = self.trading_client.get_all_assets(request)

            universe_config = settings.universe
            max_len = universe_config.max_symbol_length
            require_shortable = universe_config.require_shortable

            tradeable = []
            for asset in assets:
                # Basic tradability check
                if not asset.tradable:
                    continue

                # Shortable check (optional)
                if require_shortable and not asset.shortable:
                    continue

                # Exclude symbols with special characters (warrants, preferred, etc.)
                symbol = asset.symbol
                if any(c in symbol for c in ['.', '/', '-', '+']):
                    continue

                # Exclude long symbols (usually warrants, units, etc.)
                if len(symbol) > max_len:
                    continue

                tradeable.append(symbol)

            logger.info(f"Tier 1 - Found {len(tradeable)} tradeable assets from {len(assets)} total")
            return tradeable

        except Exception as e:
            logger.error(f"Error getting tradeable universe: {e}")
            return []

    def prefilter_by_price_batch(
        self,
        symbols: list[str],
        min_price: float,
        max_price: float,
        batch_size: int = 500
    ) -> list[str]:
        """
        Filter symbols by current price using batch quotes (Tier 2 filter).

        Uses batch size of 500 because quote requests are lightweight (single API call
        returns current bid/ask for all symbols). Alpaca handles up to 1000 symbols
        per quote request efficiently.

        Args:
            symbols: List of symbols to filter
            min_price: Minimum price (e.g., $10)
            max_price: Maximum price (e.g., $500)
            batch_size: Symbols per API call (default 500, max recommended 1000)

        Returns:
            List of symbols within price range
        """
        if not symbols:
            return []

        filtered = []
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        start_time = datetime.now(EST)

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1

            # Rate limiting: add delay between batches to respect API limits
            if batch_num > 1:
                time.sleep(BATCH_DELAY_SECONDS)

            try:
                quotes = self.get_latest_quotes_batch(batch)

                for symbol, quote in quotes.items():
                    price = quote.get('mid', 0)
                    if price and min_price <= price <= max_price:
                        filtered.append(symbol)

                # Progress logging every 5 batches or at end
                if batch_num % 5 == 0 or batch_num == total_batches:
                    elapsed = (datetime.now(EST) - start_time).total_seconds()
                    logger.debug(
                        f"Tier 2 - Price filter batch {batch_num}/{total_batches}: "
                        f"{len(filtered)} symbols in range ({elapsed:.1f}s elapsed)"
                    )

            except Exception as e:
                logger.warning(f"Error in price filter batch {batch_num}: {e}")
                continue

        logger.info(
            f"Tier 2 - Price filter complete: {len(filtered)}/{len(symbols)} symbols "
            f"in ${min_price}-${max_price} range"
        )
        return filtered

    def prefilter_by_avg_volume_batch(
        self,
        symbols: list[str],
        min_avg_volume: int,
        batch_size: int = 100,
        lookback_days: int = 20
    ) -> list[str]:
        """
        Filter symbols by average daily volume using batch bars (Tier 3 filter).

        Uses smaller batch size of 100 because daily bars requests are heavier:
        - Each symbol returns 20 days of OHLCV data
        - Response payload is ~20x larger than quotes
        - Alpaca recommends smaller batches for historical data

        Uses retry with exponential backoff and adaptive delays based on failures.

        Args:
            symbols: List of symbols to filter
            min_avg_volume: Minimum average daily volume (e.g., 1,000,000)
            batch_size: Symbols per API call (default 100, max recommended 200)
            lookback_days: Days to average volume over

        Returns:
            List of symbols meeting volume threshold
        """
        if not symbols:
            return []

        filtered = []
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        start_time = datetime.now(EST)
        consecutive_batch_failures = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1

            # Adaptive rate limiting: increase delay after failures
            if batch_num > 1:
                base_delay = BATCH_DELAY_SECONDS * 2  # 600ms for heavier requests
                # Add extra delay if we've had recent failures
                adaptive_delay = base_delay * (1 + consecutive_batch_failures * 0.5)
                adaptive_delay = min(adaptive_delay, 5.0)  # Cap at 5 seconds
                time.sleep(adaptive_delay)

            def _fetch_volume_bars():
                bar_start = datetime.now(EST) - timedelta(days=lookback_days + 7)
                request = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=bar_start,
                    limit=lookback_days
                )
                return self.data_client.get_stock_bars(request)

            try:
                # Use retry with backoff for the API call
                bars = self._retry_with_backoff(_fetch_volume_bars)
                consecutive_batch_failures = 0  # Reset on success

                for symbol in batch:
                    if symbol in bars.data and bars.data[symbol]:
                        volumes = [int(bar.volume) for bar in bars.data[symbol]]
                        if volumes:
                            avg_vol = sum(volumes) / len(volumes)
                            if avg_vol >= min_avg_volume:
                                filtered.append(symbol)

                # Progress logging every 10 batches or at end
                if batch_num % 10 == 0 or batch_num == total_batches:
                    elapsed = (datetime.now(EST) - start_time).total_seconds()
                    pct_complete = (batch_num / total_batches) * 100
                    logger.info(
                        f"Tier 3 - Volume filter: {batch_num}/{total_batches} batches "
                        f"({pct_complete:.0f}%) - {len(filtered)} high-volume symbols ({elapsed:.0f}s)"
                    )

            except Exception as e:
                consecutive_batch_failures += 1
                logger.warning(
                    f"Error in volume filter batch {batch_num} (failures: {consecutive_batch_failures}): {e}"
                )
                # If too many consecutive failures, log a more severe warning
                if consecutive_batch_failures >= 3:
                    logger.error(
                        f"Multiple consecutive batch failures ({consecutive_batch_failures}). "
                        f"Consider checking API connectivity or rate limits."
                    )
                continue

        logger.info(
            f"Tier 3 - Volume filter complete: {len(filtered)}/{len(symbols)} symbols "
            f"with avg volume >= {min_avg_volume:,}"
        )
        return filtered

    # ========== Screener Methods ==========

    def get_most_active_stocks(self, top: int = 20) -> list[dict]:
        """
        Get most active stocks by volume using Alpaca's Screener API.

        This is an alternative/supplement to the dynamic universe building.
        Returns real-time most active stocks based on SIP data.

        Args:
            top: Number of top symbols to return (default 20)

        Returns:
            List of dicts with symbol and volume data, sorted by activity
        """
        try:
            request = MostActivesRequest(top=top)
            response = self._retry_with_backoff(
                self.screener_client.get_most_actives,
                request_params=request
            )

            # Parse response - structure varies by SDK version
            result = []
            if hasattr(response, 'most_actives'):
                for item in response.most_actives:
                    result.append({
                        'symbol': item.symbol,
                        'volume': getattr(item, 'volume', 0),
                        'trade_count': getattr(item, 'trade_count', 0),
                    })
            elif isinstance(response, dict):
                for item in response.get('most_actives', []):
                    result.append({
                        'symbol': item.get('symbol'),
                        'volume': item.get('volume', 0),
                        'trade_count': item.get('trade_count', 0),
                    })

            logger.info(f"Screener: Found {len(result)} most active stocks")
            return result

        except Exception as e:
            logger.warning(f"Error getting most active stocks from screener: {e}")
            return []

    def get_market_movers(self, top: int = 10) -> dict[str, list[dict]]:
        """
        Get top market movers (gainers and losers) using Alpaca's Screener API.

        Returns stocks with the largest price changes based on SIP data.
        Change is calculated from previous close to latest close.

        Args:
            top: Number of top symbols per category (default 10)

        Returns:
            Dict with 'gainers' and 'losers' lists, each containing
            symbol, change_percent, and price data
        """
        try:
            request = MarketMoversRequest(top=top)
            response = self._retry_with_backoff(
                self.screener_client.get_market_movers,
                request_params=request
            )

            result = {'gainers': [], 'losers': []}

            # Parse response - handle both object and dict formats
            if hasattr(response, 'gainers'):
                for item in response.gainers:
                    result['gainers'].append({
                        'symbol': item.symbol,
                        'change_percent': getattr(item, 'percent_change', 0),
                        'price': getattr(item, 'price', 0),
                        'volume': getattr(item, 'volume', 0),
                    })
            elif isinstance(response, dict):
                for item in response.get('gainers', []):
                    result['gainers'].append({
                        'symbol': item.get('symbol'),
                        'change_percent': item.get('percent_change', 0),
                        'price': item.get('price', 0),
                        'volume': item.get('volume', 0),
                    })

            if hasattr(response, 'losers'):
                for item in response.losers:
                    result['losers'].append({
                        'symbol': item.symbol,
                        'change_percent': getattr(item, 'percent_change', 0),
                        'price': getattr(item, 'price', 0),
                        'volume': getattr(item, 'volume', 0),
                    })
            elif isinstance(response, dict):
                for item in response.get('losers', []):
                    result['losers'].append({
                        'symbol': item.get('symbol'),
                        'change_percent': item.get('percent_change', 0),
                        'price': item.get('price', 0),
                        'volume': item.get('volume', 0),
                    })

            logger.info(
                f"Screener: Found {len(result['gainers'])} gainers and "
                f"{len(result['losers'])} losers"
            )
            return result

        except Exception as e:
            logger.warning(f"Error getting market movers from screener: {e}")
            return {'gainers': [], 'losers': []}

    def get_screener_candidates(
        self,
        min_price: float = 10.0,
        max_price: float = 500.0,
        top_actives: int = 50,
        top_movers: int = 20
    ) -> list[str]:
        """
        Get candidate symbols from Alpaca Screener API.

        Combines most active stocks and market movers to find potential
        trading candidates. This can supplement the dynamic universe.

        Args:
            min_price: Minimum price filter
            max_price: Maximum price filter
            top_actives: Number of most active stocks to consider
            top_movers: Number of top movers (gainers/losers) to consider

        Returns:
            List of unique symbols passing basic filters
        """
        candidates = set()

        # Get most active stocks
        actives = self.get_most_active_stocks(top=top_actives)
        for item in actives:
            candidates.add(item['symbol'])

        # Get market movers
        movers = self.get_market_movers(top=top_movers)
        for item in movers.get('gainers', []):
            candidates.add(item['symbol'])
        for item in movers.get('losers', []):
            candidates.add(item['symbol'])

        # Filter by price using batch quotes
        if candidates:
            symbols = list(candidates)
            filtered = self.prefilter_by_price_batch(
                symbols=symbols,
                min_price=min_price,
                max_price=max_price,
                batch_size=100  # Smaller batch for quick check
            )
            logger.info(
                f"Screener candidates: {len(filtered)}/{len(candidates)} passed price filter"
            )
            return filtered

        return []


# Global client instance
market_data = MarketDataClient()
