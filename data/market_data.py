"""
Market data retrieval from Alpaca API
"""
from datetime import datetime, timedelta, timezone
from typing import Optional
import pandas as pd
from alpaca.data import StockHistoricalDataClient, StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from loguru import logger

from config.settings import settings


class MarketDataClient:
    """Client for retrieving market data from Alpaca"""

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
        # Phase 2: Cache for intraday volume profiles by symbol
        self.volume_profiles: dict[str, dict[int, float]] = {}

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
                start = datetime.now() - timedelta(days=5)

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
        start = datetime.now() - timedelta(days=days + 5)
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
            today = datetime.now().replace(hour=4, minute=0, second=0, microsecond=0)
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
            today = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
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
            start = datetime.now() - timedelta(days=lookback_days + 7)  # Extra buffer for weekends

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
            today = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)

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
        now = datetime.now()
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

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to quote data (bid, ask, mid, age_seconds, etc.)
        """
        if not symbols:
            return {}

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_stock_latest_quote(request)

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
            logger.error(f"Error getting batch quotes: {e}")
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

        Args:
            symbols: List of stock symbols
            timeframe: Bar timeframe
            limit: Maximum number of bars per symbol

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        if not symbols:
            return {}

        try:
            start = datetime.now() - timedelta(days=5)

            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start,
                limit=limit
            )

            bars = self.data_client.get_stock_bars(request)

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
            logger.error(f"Error getting batch bars: {e}")
            return {}


# Global client instance
market_data = MarketDataClient()
