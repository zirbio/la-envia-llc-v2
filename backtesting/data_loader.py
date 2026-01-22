"""
Data loader for backtesting - fetches and caches historical data from Alpaca
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
from loguru import logger

from alpaca.data import StockHistoricalDataClient, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError

from config.settings import settings


# Custom exceptions for specific error handling
class DataLoaderError(Exception):
    """Base exception for data loader errors"""
    pass


class NetworkError(DataLoaderError):
    """Network-related errors (retry-able)"""
    pass


class InvalidSymbolError(DataLoaderError):
    """Invalid or unknown symbol (permanent error)"""
    pass


class RateLimitError(DataLoaderError):
    """API rate limit exceeded (retry-able with backoff)"""
    pass


class BacktestDataLoader:
    """Load and cache historical data from Alpaca for backtesting"""

    # Default cache age in days before data is considered stale
    DEFAULT_MAX_CACHE_AGE_DAYS = 7

    def __init__(self, cache_dir: str = "data/cache", max_cache_age_days: int = None):
        """
        Initialize the data loader.

        Args:
            cache_dir: Directory to store cached data files
            max_cache_age_days: Maximum age of cache files before refresh (default: 7 days)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_age_days = max_cache_age_days or self.DEFAULT_MAX_CACHE_AGE_DAYS

        self.client = StockHistoricalDataClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key
        )

    def _get_cache_path(self, symbol: str, start: datetime, end: datetime) -> Path:
        """Generate cache file path for a symbol and date range"""
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return self.cache_dir / f"{symbol}_{start_str}_{end_str}_1min.parquet"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if a cache file exists and is still valid (not too old).

        Args:
            cache_path: Path to the cache file

        Returns:
            True if cache exists and is within max age, False otherwise
        """
        if not cache_path.exists():
            return False

        # Check file age
        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age_days = (datetime.now() - file_mtime).days

        if age_days > self.max_cache_age_days:
            logger.debug(f"Cache file {cache_path.name} is {age_days} days old (max: {self.max_cache_age_days})")
            return False

        return True

    def clear_cache(self, symbol: str = None):
        """
        Clear cached data files.

        Args:
            symbol: If provided, only clear cache for this symbol.
                   If None, clear all cached files.
        """
        if symbol:
            # Clear cache files for specific symbol
            pattern = f"{symbol}_*.parquet"
            files = list(self.cache_dir.glob(pattern))
        else:
            # Clear all cache files
            files = list(self.cache_dir.glob("*.parquet"))

        for file in files:
            try:
                file.unlink()
                logger.info(f"Deleted cache file: {file.name}")
            except OSError as e:
                logger.warning(f"Failed to delete cache file {file.name}: {e}")

        logger.info(f"Cleared {len(files)} cache file(s)")
        return len(files)

    def load_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Load historical 1-minute bars for a symbol

        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with OHLCV data

        Raises:
            InvalidSymbolError: If the symbol is not found or invalid
            NetworkError: If there's a network connectivity issue
            RateLimitError: If API rate limit is exceeded
        """
        cache_path = self._get_cache_path(symbol, start, end)

        # Try to load from cache if valid (exists and not too old)
        if use_cache and self._is_cache_valid(cache_path):
            logger.info(f"Loading {symbol} from cache: {cache_path}")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_path}: {e}. Fetching fresh data.")

        # Fetch from Alpaca
        logger.info(f"Fetching {symbol} from Alpaca: {start} to {end}")

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end
            )

            bars = self.client.get_stock_bars(request)

            if symbol not in bars.data or not bars.data[symbol]:
                logger.warning(f"No data found for {symbol}")
                raise InvalidSymbolError(f"No data available for symbol: {symbol}")

            df = pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume),
                'vwap': float(bar.vwap) if bar.vwap else None
            } for bar in bars.data[symbol]])

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()

            # Cache the data
            if use_cache:
                df.to_parquet(cache_path)
                logger.info(f"Cached {symbol} data to {cache_path}")

            return df

        except InvalidSymbolError:
            # Re-raise our custom exceptions
            raise
        except APIError as e:
            error_str = str(e).lower()
            if "not found" in error_str or "invalid" in error_str or "unknown" in error_str:
                logger.error(f"Invalid symbol {symbol}: {e}")
                raise InvalidSymbolError(f"Invalid or unknown symbol: {symbol}") from e
            elif "rate limit" in error_str or "too many" in error_str:
                logger.warning(f"Rate limit hit for {symbol}: {e}")
                raise RateLimitError(f"API rate limit exceeded: {e}") from e
            else:
                logger.error(f"API error fetching data for {symbol}: {e}")
                raise NetworkError(f"API error: {e}") from e
        except ConnectionError as e:
            logger.error(f"Network error fetching data for {symbol}: {e}")
            raise NetworkError(f"Network connection error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {symbol}: {e}")
            raise DataLoaderError(f"Failed to load data for {symbol}: {e}") from e

    def load_multiple(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        use_cache: bool = True,
        max_workers: int = 5
    ) -> dict[str, pd.DataFrame]:
        """
        Load historical data for multiple symbols (in parallel for faster loading)

        Args:
            symbols: List of stock symbols
            start: Start datetime
            end: End datetime
            use_cache: Whether to use cached data
            max_workers: Maximum number of parallel workers (default: 5)

        Returns:
            Dict mapping symbol to DataFrame
        """
        data = {}
        errors = {}

        def load_symbol(symbol: str) -> tuple[str, pd.DataFrame, Exception]:
            """Load data for a single symbol, returning (symbol, df, error)"""
            try:
                df = self.load_data(symbol, start, end, use_cache)
                return (symbol, df, None)
            except InvalidSymbolError as e:
                # Permanent error - don't retry
                return (symbol, pd.DataFrame(), e)
            except (NetworkError, RateLimitError) as e:
                # Transient error - could retry but we'll log for now
                return (symbol, pd.DataFrame(), e)
            except Exception as e:
                return (symbol, pd.DataFrame(), e)

        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_symbol, symbol): symbol for symbol in symbols}

            for future in as_completed(futures):
                symbol, df, error = future.result()

                if error:
                    error_type = type(error).__name__
                    logger.warning(f"Skipping {symbol} - {error_type}: {error}")
                    errors[symbol] = error
                elif df.empty:
                    logger.warning(f"Skipping {symbol} - no data available")
                else:
                    data[symbol] = df
                    logger.debug(f"Loaded {len(df)} bars for {symbol}")

        if errors:
            logger.info(f"Loaded {len(data)}/{len(symbols)} symbols ({len(errors)} errors)")

        return data

    def get_trading_days(self, start: datetime, end: datetime) -> list[datetime]:
        """
        Get list of trading days in date range

        Args:
            start: Start datetime
            end: End datetime

        Returns:
            List of trading day dates
        """
        # Use SPY as a proxy for trading days
        df = self.load_data("SPY", start, end)
        if df.empty:
            return []

        # Get unique dates
        dates = df.index.normalize().unique().tolist()
        return sorted(dates)


# Global instance
backtest_data_loader = BacktestDataLoader()
