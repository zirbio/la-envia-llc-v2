"""
Market data retrieval from Alpaca API
"""
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from alpaca.data import StockHistoricalDataClient, StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
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
        Get latest quote for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dict with bid, ask, and mid price
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)

            if symbol not in quotes:
                return None

            quote = quotes[symbol]
            return {
                'symbol': symbol,
                'bid': float(quote.bid_price),
                'ask': float(quote.ask_price),
                'mid': (float(quote.bid_price) + float(quote.ask_price)) / 2,
                'bid_size': int(quote.bid_size),
                'ask_size': int(quote.ask_size),
                'timestamp': quote.timestamp
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
        Get pre-market data for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dict with pre-market price change and volume
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

            # Get today's pre-market bars for volume
            today = datetime.now().replace(hour=4, minute=0, second=0, microsecond=0)
            premarket_bars = self.get_bars(
                symbol=symbol,
                timeframe=TimeFrame.Minute,
                start=today,
                limit=300
            )

            premarket_volume = int(premarket_bars['volume'].sum()) if not premarket_bars.empty else 0

            return {
                'symbol': symbol,
                'prev_close': prev_close,
                'current_price': current_price,
                'gap_percent': gap_percent,
                'premarket_volume': premarket_volume
            }

        except Exception as e:
            logger.error(f"Error getting pre-market data for {symbol}: {e}")
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


# Global client instance
market_data = MarketDataClient()
