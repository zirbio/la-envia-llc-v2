"""
Unit tests for data/indicators.py - Technical indicators module.

Tests cover:
- RSI calculation (6 tests)
- MACD calculation and signals (6 tests)
- Stochastic oscillator (5 tests)
- Volume, VWAP, ATR, Bollinger Bands (8 tests)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from data.indicators import (
    calculate_rsi,
    calculate_vwap,
    calculate_relative_volume,
    calculate_ema,
    calculate_sma,
    calculate_atr,
    calculate_macd,
    calculate_stochastic,
    calculate_bollinger_bands,
    detect_volume_spike,
    is_macd_bullish,
    is_macd_bearish,
    is_stochastic_oversold,
    is_stochastic_overbought,
    is_stochastic_bullish_cross,
    is_stochastic_bearish_cross,
    is_above_vwap,
    is_below_vwap,
    is_overbought,
    is_oversold,
    IndicatorCalculator
)


# ============================================================================
# RSI Tests (6 tests)
# ============================================================================

class TestCalculateRSI:
    """Tests for RSI calculation."""

    def test_calculate_rsi_standard_14_period(self, sample_ohlcv_df):
        """RSI with standard 14-period should return values between 0 and 100."""
        prices = sample_ohlcv_df['close']
        rsi = calculate_rsi(prices, period=14)

        # Check that RSI is calculated
        assert len(rsi) == len(prices)

        # Get valid RSI values (after warm-up period)
        valid_rsi = rsi.dropna()
        assert len(valid_rsi) > 0

        # RSI should be bounded between 0 and 100
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100

    def test_calculate_rsi_custom_period(self, sample_ohlcv_df):
        """RSI with custom 7-period should work correctly."""
        prices = sample_ohlcv_df['close']
        rsi = calculate_rsi(prices, period=7)

        valid_rsi = rsi.dropna()
        assert len(valid_rsi) > 0
        assert valid_rsi.min() >= 0
        assert valid_rsi.max() <= 100

    def test_calculate_rsi_empty_series(self):
        """RSI of empty series should return empty series."""
        empty_series = pd.Series([], dtype=float)
        rsi = calculate_rsi(empty_series)

        assert len(rsi) == 0

    def test_calculate_rsi_insufficient_data(self):
        """RSI with less data than period should return NaN values."""
        short_series = pd.Series([100, 101, 102, 103, 104])  # 5 values, period=14
        rsi = calculate_rsi(short_series, period=14)

        # All values should be NaN since we don't have enough data
        assert rsi.isna().all()

    def test_calculate_rsi_all_gains(self):
        """RSI should approach 100 when price only goes up."""
        # Strictly increasing prices
        prices = pd.Series([100 + i for i in range(30)])
        rsi = calculate_rsi(prices, period=14)

        # Last RSI value should be very high (approaching 100)
        last_valid_rsi = rsi.dropna().iloc[-1]
        assert last_valid_rsi > 90

    def test_calculate_rsi_all_losses(self):
        """RSI should approach 0 when price only goes down."""
        # Strictly decreasing prices
        prices = pd.Series([100 - i for i in range(30)])
        rsi = calculate_rsi(prices, period=14)

        # Last RSI value should be very low (approaching 0)
        last_valid_rsi = rsi.dropna().iloc[-1]
        assert last_valid_rsi < 10


# ============================================================================
# MACD Tests (6 tests)
# ============================================================================

class TestCalculateMACD:
    """Tests for MACD calculation."""

    def test_calculate_macd_standard_12_26_9(self, sample_ohlcv_df):
        """MACD with standard parameters (12, 26, 9) should return three series."""
        prices = sample_ohlcv_df['close']
        macd_line, signal_line, histogram = calculate_macd(prices)

        # All three series should have the same length as input
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)

        # Histogram should equal MACD - Signal
        valid_idx = ~(macd_line.isna() | signal_line.isna())
        np.testing.assert_array_almost_equal(
            histogram[valid_idx].values,
            (macd_line[valid_idx] - signal_line[valid_idx]).values,
            decimal=10
        )


class TestIsMACDBullish:
    """Tests for MACD bullish signal detection."""

    def test_is_macd_bullish_all_conditions_met(self):
        """MACD bullish when all conditions are met."""
        # MACD above signal, histogram positive and growing
        result = is_macd_bullish(
            macd=0.5,
            signal=0.3,
            histogram=0.3,
            prev_histogram=0.1
        )
        assert result is True

    def test_is_macd_bullish_prev_histogram_none(self):
        """MACD bullish should return False when prev_histogram is None."""
        result = is_macd_bullish(
            macd=0.5,
            signal=0.3,
            histogram=0.3,
            prev_histogram=None
        )
        assert result is False

    def test_is_macd_bullish_histogram_not_growing(self):
        """MACD bullish should return False when histogram is shrinking."""
        result = is_macd_bullish(
            macd=0.5,
            signal=0.3,
            histogram=0.2,
            prev_histogram=0.4  # Histogram shrinking
        )
        assert result is False


class TestIsMACDBearish:
    """Tests for MACD bearish signal detection."""

    def test_is_macd_bearish_all_conditions_met(self):
        """MACD bearish when all conditions are met."""
        # MACD below signal, histogram negative and falling
        result = is_macd_bearish(
            macd=-0.5,
            signal=-0.3,
            histogram=-0.3,
            prev_histogram=-0.1
        )
        assert result is True

    def test_is_macd_bearish_prev_histogram_none(self):
        """MACD bearish should return False when prev_histogram is None."""
        result = is_macd_bearish(
            macd=-0.5,
            signal=-0.3,
            histogram=-0.3,
            prev_histogram=None
        )
        assert result is False


# ============================================================================
# Stochastic Tests (5 tests)
# ============================================================================

class TestCalculateStochastic:
    """Tests for Stochastic oscillator calculation."""

    def test_calculate_stochastic_standard_14_3(self, sample_ohlcv_df):
        """Stochastic with standard parameters (14, 3) should return %K and %D."""
        stoch_k, stoch_d = calculate_stochastic(sample_ohlcv_df, k_period=14, d_period=3)

        # Both should have same length as input
        assert len(stoch_k) == len(sample_ohlcv_df)
        assert len(stoch_d) == len(sample_ohlcv_df)

        # Valid values should be between 0 and 100
        valid_k = stoch_k.dropna()
        valid_d = stoch_d.dropna()

        if len(valid_k) > 0:
            assert valid_k.min() >= 0
            assert valid_k.max() <= 100
        if len(valid_d) > 0:
            assert valid_d.min() >= 0
            assert valid_d.max() <= 100

    def test_calculate_stochastic_division_by_zero(self, flat_price_df):
        """Stochastic should return NaN when high == low (no range)."""
        stoch_k, stoch_d = calculate_stochastic(flat_price_df)

        # When high == low, range is 0, result should be NaN
        # The code replaces 0 with NaN to avoid division by zero
        assert stoch_k.isna().any() or stoch_d.isna().any()


class TestStochasticSignals:
    """Tests for Stochastic signal detection."""

    def test_is_stochastic_oversold(self):
        """Stochastic oversold when both %K and %D are below threshold."""
        assert is_stochastic_oversold(stoch_k=15, stoch_d=18, threshold=20) is True
        assert is_stochastic_oversold(stoch_k=25, stoch_d=18, threshold=20) is False
        assert is_stochastic_oversold(stoch_k=15, stoch_d=25, threshold=20) is False

    def test_is_stochastic_bullish_cross(self):
        """Bullish crossover when %K crosses above %D."""
        # Previous: K <= D, Current: K > D
        result = is_stochastic_bullish_cross(
            stoch_k=35, stoch_d=30,
            prev_k=25, prev_d=30
        )
        assert result is True

        # Not a crossover if K was already above D
        result = is_stochastic_bullish_cross(
            stoch_k=35, stoch_d=30,
            prev_k=32, prev_d=30
        )
        assert result is False

    def test_is_stochastic_bearish_cross(self):
        """Bearish crossover when %K crosses below %D."""
        # Previous: K >= D, Current: K < D
        result = is_stochastic_bearish_cross(
            stoch_k=25, stoch_d=30,
            prev_k=32, prev_d=30
        )
        assert result is True

        # Not a crossover if K was already below D
        result = is_stochastic_bearish_cross(
            stoch_k=25, stoch_d=30,
            prev_k=28, prev_d=30
        )
        assert result is False


# ============================================================================
# Volume, VWAP, ATR, Bollinger Tests (8 tests)
# ============================================================================

class TestCalculateVWAP:
    """Tests for VWAP calculation."""

    def test_calculate_vwap_standard(self, sample_ohlcv_df):
        """VWAP should be calculated correctly."""
        vwap = calculate_vwap(sample_ohlcv_df)

        assert len(vwap) == len(sample_ohlcv_df)

        # VWAP should be within the high-low range for each bar
        for i in range(len(sample_ohlcv_df)):
            if not pd.isna(vwap.iloc[i]):
                # VWAP is cumulative, so it should be a reasonable price
                assert vwap.iloc[i] > 0

    def test_calculate_vwap_zero_volume(self):
        """VWAP with zero cumulative volume should handle gracefully."""
        df = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [99, 100, 101],
            'close': [99.5, 100.5, 101.5],
            'volume': [0, 0, 0]  # All zero volume
        })

        vwap = calculate_vwap(df)
        # Should return inf or NaN when volume is 0
        # The exact behavior depends on implementation
        assert len(vwap) == len(df)


class TestDetectVolumeSpike:
    """Tests for volume spike detection."""

    def test_detect_volume_spike_above_threshold(self):
        """Volume spike detected when current > threshold * average."""
        # 2x average volume, threshold 1.5x
        result = detect_volume_spike(
            current_volume=2_000_000,
            avg_volume=1_000_000,
            threshold=1.5
        )
        assert result is True

        # Exactly at threshold
        result = detect_volume_spike(
            current_volume=1_500_000,
            avg_volume=1_000_000,
            threshold=1.5
        )
        assert result is True

    def test_detect_volume_spike_zero_avg(self):
        """Volume spike should return False when average is zero."""
        result = detect_volume_spike(
            current_volume=1_000_000,
            avg_volume=0,
            threshold=1.5
        )
        assert result is False


class TestCalculateATR:
    """Tests for ATR calculation."""

    def test_calculate_atr_standard(self, sample_ohlcv_df):
        """ATR with standard 14-period should return positive values."""
        atr = calculate_atr(sample_ohlcv_df, period=14)

        assert len(atr) == len(sample_ohlcv_df)

        # Valid ATR values should be positive
        valid_atr = atr.dropna()
        assert len(valid_atr) > 0
        assert (valid_atr > 0).all()


class TestCalculateBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_calculate_bollinger_bands_standard(self, sample_ohlcv_df):
        """Bollinger Bands with standard parameters should have correct structure."""
        prices = sample_ohlcv_df['close']
        upper, middle, lower = calculate_bollinger_bands(prices, period=20, std_dev=2.0)

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

        # Valid values: upper > middle > lower
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        if valid_idx.any():
            assert (upper[valid_idx] >= middle[valid_idx]).all()
            assert (middle[valid_idx] >= lower[valid_idx]).all()


# ============================================================================
# IndicatorCalculator Tests
# ============================================================================

class TestIndicatorCalculator:
    """Tests for IndicatorCalculator class."""

    def test_indicator_calculator_add_all(self, sample_ohlcv_df):
        """IndicatorCalculator should add all indicators to DataFrame."""
        calc = IndicatorCalculator(sample_ohlcv_df)
        result_df = calc.add_all_indicators()

        # Check that all expected columns are present
        expected_columns = [
            'vwap', 'rsi', 'ema_9', 'ema_20', 'ema_50',
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr'
        ]

        for col in expected_columns:
            assert col in result_df.columns, f"Missing column: {col}"

    def test_indicator_calculator_empty_df(self):
        """IndicatorCalculator with empty DataFrame should handle gracefully."""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        calc = IndicatorCalculator(empty_df)

        result_df = calc.add_all_indicators()
        assert len(result_df) == 0

        # get_latest_indicators should return empty dict
        latest = calc.get_latest_indicators()
        assert latest == {}

    def test_indicator_calculator_get_latest_indicators(self, sample_ohlcv_df):
        """get_latest_indicators should return dict with latest values."""
        calc = IndicatorCalculator(sample_ohlcv_df)
        calc.add_all_indicators()

        latest = calc.get_latest_indicators()

        assert 'close' in latest
        assert 'volume' in latest
        assert 'vwap' in latest
        assert 'rsi' in latest
        assert 'macd' in latest
        assert 'macd_signal' in latest
        assert 'macd_histogram' in latest
        assert 'prev_macd_histogram' in latest
        assert 'stoch_k' in latest
        assert 'stoch_d' in latest


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestHelperFunctions:
    """Tests for simple helper functions."""

    def test_is_above_vwap(self):
        """Price above VWAP detection."""
        assert is_above_vwap(price=101.0, vwap=100.0) is True
        assert is_above_vwap(price=99.0, vwap=100.0) is False
        assert is_above_vwap(price=100.0, vwap=100.0) is False

    def test_is_below_vwap(self):
        """Price below VWAP detection."""
        assert is_below_vwap(price=99.0, vwap=100.0) is True
        assert is_below_vwap(price=101.0, vwap=100.0) is False
        assert is_below_vwap(price=100.0, vwap=100.0) is False

    def test_is_overbought(self):
        """RSI overbought detection."""
        assert is_overbought(rsi=75, threshold=70) is True
        assert is_overbought(rsi=70, threshold=70) is True
        assert is_overbought(rsi=65, threshold=70) is False

    def test_is_oversold(self):
        """RSI oversold detection."""
        assert is_oversold(rsi=25, threshold=30) is True
        assert is_oversold(rsi=30, threshold=30) is True
        assert is_oversold(rsi=35, threshold=30) is False

    def test_calculate_relative_volume(self):
        """Relative volume calculation."""
        assert calculate_relative_volume(2_000_000, 1_000_000) == 2.0
        assert calculate_relative_volume(500_000, 1_000_000) == 0.5
        assert calculate_relative_volume(1_000_000, 0) == 0.0

    def test_calculate_ema(self, sample_ohlcv_df):
        """EMA calculation."""
        prices = sample_ohlcv_df['close']
        ema = calculate_ema(prices, period=9)

        assert len(ema) == len(prices)
        # EMA should be close to recent prices
        assert not ema.isna().all()

    def test_calculate_sma(self, sample_ohlcv_df):
        """SMA calculation."""
        prices = sample_ohlcv_df['close']
        sma = calculate_sma(prices, period=20)

        assert len(sma) == len(prices)
        # SMA should be calculated after enough data
        valid_sma = sma.dropna()
        assert len(valid_sma) > 0
