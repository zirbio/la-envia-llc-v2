"""
Technical indicators for trading signals
"""
import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)

    Args:
        prices: Series of closing prices
        period: RSI period (default 14)

    Returns:
        Series with RSI values
    """
    delta = prices.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP)

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns

    Returns:
        Series with VWAP values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_tp_volume = (typical_price * df['volume']).cumsum()
    cumulative_volume = df['volume'].cumsum()

    vwap = cumulative_tp_volume / cumulative_volume
    return vwap


def calculate_relative_volume(
    current_volume: int,
    avg_volume: int
) -> float:
    """
    Calculate relative volume compared to average

    Args:
        current_volume: Current bar's volume
        avg_volume: Average volume

    Returns:
        Relative volume ratio
    """
    if avg_volume == 0:
        return 0.0
    return current_volume / avg_volume


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average

    Args:
        prices: Series of prices
        period: EMA period

    Returns:
        Series with EMA values
    """
    return prices.ewm(span=period, adjust=False).mean()


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average

    Args:
        prices: Series of prices
        period: SMA period

    Returns:
        Series with SMA values
    """
    return prices.rolling(window=period).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period

    Returns:
        Series with ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands

    Args:
        prices: Series of closing prices
        period: SMA period
        std_dev: Standard deviation multiplier

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def detect_volume_spike(
    current_volume: int,
    avg_volume: int,
    threshold: float = 1.5
) -> bool:
    """
    Detect if current volume is a spike

    Args:
        current_volume: Current bar's volume
        avg_volume: Average volume
        threshold: Minimum ratio for spike

    Returns:
        True if volume spike detected
    """
    if avg_volume == 0:
        return False
    return (current_volume / avg_volume) >= threshold


def is_above_vwap(price: float, vwap: float) -> bool:
    """Check if price is above VWAP"""
    return price > vwap


def is_below_vwap(price: float, vwap: float) -> bool:
    """Check if price is below VWAP"""
    return price < vwap


def is_overbought(rsi: float, threshold: int = 70) -> bool:
    """Check if RSI indicates overbought"""
    return rsi >= threshold


def is_oversold(rsi: float, threshold: int = 30) -> bool:
    """Check if RSI indicates oversold"""
    return rsi <= threshold


class IndicatorCalculator:
    """Helper class to calculate all indicators for a DataFrame"""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume'
        """
        self.df = df.copy()

    def add_all_indicators(
        self,
        rsi_period: int = 14,
        ema_periods: list[int] = None,
        bb_period: int = 20
    ) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame

        Args:
            rsi_period: RSI period
            ema_periods: List of EMA periods to calculate
            bb_period: Bollinger Bands period

        Returns:
            DataFrame with indicators added
        """
        if ema_periods is None:
            ema_periods = [9, 20, 50]

        # VWAP
        self.df['vwap'] = calculate_vwap(self.df)

        # RSI
        self.df['rsi'] = calculate_rsi(self.df['close'], rsi_period)

        # EMAs
        for period in ema_periods:
            self.df[f'ema_{period}'] = calculate_ema(self.df['close'], period)

        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(
            self.df['close'], bb_period
        )
        self.df['bb_upper'] = upper
        self.df['bb_middle'] = middle
        self.df['bb_lower'] = lower

        # ATR
        self.df['atr'] = calculate_atr(self.df)

        return self.df

    def get_latest_indicators(self) -> dict:
        """
        Get the most recent indicator values

        Returns:
            Dict with latest indicator values
        """
        if self.df.empty:
            return {}

        latest = self.df.iloc[-1]
        return {
            'close': latest['close'],
            'volume': latest['volume'],
            'vwap': latest.get('vwap'),
            'rsi': latest.get('rsi'),
            'ema_9': latest.get('ema_9'),
            'ema_20': latest.get('ema_20'),
            'atr': latest.get('atr')
        }
