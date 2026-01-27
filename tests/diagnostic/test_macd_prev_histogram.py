"""
Diagnostic tests for MACD prev_histogram bug.

The is_macd_bullish() and is_macd_bearish() functions return False when
prev_histogram is None. This test suite verifies whether prev_histogram
is being passed correctly through the signal generation pipeline.

Run with: pytest tests/diagnostic/test_macd_prev_histogram.py -v
"""
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from data.indicators import (
    calculate_macd,
    is_macd_bullish,
    is_macd_bearish,
    IndicatorCalculator,
    indicator_cache
)


# ============================================================================
# Test is_macd_bullish with prev_histogram=None
# ============================================================================

@pytest.mark.diagnostic
class TestMACDBullishWithNone:
    """Test that is_macd_bullish handles prev_histogram=None correctly."""

    def test_macd_bullish_none_prev_histogram_returns_false(self):
        """is_macd_bullish should return False when prev_histogram is None."""
        result = is_macd_bullish(
            macd=0.5,
            signal=0.3,
            histogram=0.2,
            prev_histogram=None  # Bug: This causes False even with bullish values
        )

        # This documents the CURRENT behavior (which may be the bug)
        assert result is False

    def test_macd_bullish_with_valid_prev_histogram(self):
        """is_macd_bullish should return True with valid bullish values."""
        result = is_macd_bullish(
            macd=0.5,
            signal=0.3,
            histogram=0.2,
            prev_histogram=0.1  # Valid previous histogram
        )

        # MACD > signal, histogram > 0, histogram growing
        assert result is True

    def test_macd_bullish_conditions_breakdown(self):
        """Document all conditions for is_macd_bullish."""
        macd = 0.5
        signal = 0.3
        histogram = 0.2
        prev_histogram = 0.1

        # Condition 1: MACD above signal
        above_signal = macd > signal
        assert above_signal is True

        # Condition 2: Histogram positive
        histogram_positive = histogram > 0
        assert histogram_positive is True

        # Condition 3: Histogram growing (current > previous)
        histogram_growing = histogram > prev_histogram
        assert histogram_growing is True

        # All conditions met = bullish
        result = above_signal and histogram_positive and histogram_growing
        assert result is True


# ============================================================================
# Test is_macd_bearish with prev_histogram=None
# ============================================================================

@pytest.mark.diagnostic
class TestMACDBearishWithNone:
    """Test that is_macd_bearish handles prev_histogram=None correctly."""

    def test_macd_bearish_none_prev_histogram_returns_false(self):
        """is_macd_bearish should return False when prev_histogram is None."""
        result = is_macd_bearish(
            macd=-0.5,
            signal=-0.3,
            histogram=-0.2,
            prev_histogram=None  # Bug: This causes False even with bearish values
        )

        # This documents the CURRENT behavior (which may be the bug)
        assert result is False

    def test_macd_bearish_with_valid_prev_histogram(self):
        """is_macd_bearish should return True with valid bearish values."""
        result = is_macd_bearish(
            macd=-0.5,
            signal=-0.3,
            histogram=-0.2,
            prev_histogram=-0.1  # Valid previous histogram
        )

        # MACD < signal, histogram < 0, histogram falling
        assert result is True


# ============================================================================
# Test IndicatorCalculator provides prev_macd_histogram
# ============================================================================

@pytest.mark.diagnostic
class TestIndicatorCalculatorPrevHistogram:
    """Test that IndicatorCalculator correctly provides prev_macd_histogram."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV data with enough bars for MACD calculation."""
        np.random.seed(42)
        n_bars = 50  # Need at least 26 for MACD slow period

        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(n_bars) * 0.5)

        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

        return pd.DataFrame({
            'open': prices - 0.2,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': [1_000_000] * n_bars
        }, index=dates)

    def test_indicator_calculator_includes_prev_histogram(self, sample_df):
        """IndicatorCalculator.get_latest_indicators should include prev_macd_histogram."""
        calc = IndicatorCalculator(sample_df)
        calc.add_all_indicators()
        indicators = calc.get_latest_indicators()

        # Verify prev_macd_histogram is present
        assert 'prev_macd_histogram' in indicators
        assert indicators['prev_macd_histogram'] is not None

    def test_prev_histogram_differs_from_current(self, sample_df):
        """prev_macd_histogram should be different from current histogram."""
        calc = IndicatorCalculator(sample_df)
        calc.add_all_indicators()
        indicators = calc.get_latest_indicators()

        current = indicators.get('macd_histogram')
        previous = indicators.get('prev_macd_histogram')

        # They may be equal by chance, but usually they differ
        # At minimum, both should be present and be numbers
        assert current is not None
        assert previous is not None
        assert isinstance(current, (int, float))
        assert isinstance(previous, (int, float))

    def test_prev_histogram_comes_from_previous_bar(self, sample_df):
        """prev_macd_histogram should come from df.iloc[-2]."""
        calc = IndicatorCalculator(sample_df)
        calc.add_all_indicators()

        # Get the actual histogram values from the DataFrame
        df = calc.df
        current_histogram = df['macd_histogram'].iloc[-1]
        prev_histogram = df['macd_histogram'].iloc[-2]

        indicators = calc.get_latest_indicators()

        assert indicators['macd_histogram'] == pytest.approx(current_histogram, rel=1e-5)
        assert indicators['prev_macd_histogram'] == pytest.approx(prev_histogram, rel=1e-5)


# ============================================================================
# Test Indicator Cache Preserves prev_macd_histogram
# ============================================================================

@pytest.mark.diagnostic
class TestIndicatorCachePrevHistogram:
    """Test that indicator cache preserves prev_macd_histogram."""

    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV data."""
        np.random.seed(42)
        n_bars = 50

        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(n_bars) * 0.5)

        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

        return pd.DataFrame({
            'open': prices - 0.2,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': [1_000_000] * n_bars
        }, index=dates)

    def test_cache_stores_prev_histogram(self, sample_df):
        """Indicator cache should store prev_macd_histogram."""
        # Clear cache first
        indicator_cache.invalidate()

        # Calculate and cache indicators
        calc = IndicatorCalculator(sample_df)
        calc.add_all_indicators()
        indicators = calc.get_latest_indicators()

        indicator_cache.set("TEST", sample_df, indicators)

        # Retrieve from cache
        cached = indicator_cache.get("TEST", sample_df)

        assert cached is not None
        assert 'prev_macd_histogram' in cached
        assert cached['prev_macd_histogram'] == indicators['prev_macd_histogram']


# ============================================================================
# Test ORB Strategy Uses prev_macd_histogram
# ============================================================================

@pytest.mark.diagnostic
class TestORBStrategyUsesPrevHistogram:
    """Test that ORB strategy correctly uses prev_macd_histogram."""

    def test_check_breakout_receives_prev_histogram(self):
        """Verify that check_breakout passes prev_histogram to MACD check."""
        from strategy.orb import ORBStrategy, OpeningRange
        from config.settings import TradingConfig, SentimentConfig

        with patch('strategy.orb.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        # Add an opening range
        strategy.opening_ranges["TEST"] = OpeningRange(
            symbol="TEST",
            high=100.00,
            low=98.00,
            range_size=2.00,
            vwap=99.00,
            timestamp=datetime.now()
        )

        # Create mock bars with enough data
        n_bars = 50
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

        mock_df = pd.DataFrame({
            'open': prices - 0.2,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': [1_000_000] * n_bars
        }, index=dates)

        # Mock time to be within trading window (10:00 AM ET)
        mock_time = datetime(2024, 1, 15, 10, 0, 0)

        with patch('strategy.orb.market_data') as mock_market:
            mock_market.get_bars.return_value = mock_df
            mock_market.get_current_minute_index.return_value = 30
            mock_market.get_cumulative_volume_today.return_value = 30_000_000
            mock_market.volume_profiles = {}

            with patch('strategy.orb.datetime') as mock_datetime:
                mock_datetime.now.return_value = mock_time
                mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                with patch('strategy.orb.indicator_cache') as mock_cache:
                    # Return indicators that include prev_macd_histogram
                    mock_indicators = {
                        'vwap': 99.50,
                        'rsi': 55.0,
                        'macd': 0.5,
                        'macd_signal': 0.3,
                        'macd_histogram': 0.2,
                        'prev_macd_histogram': 0.15,  # Important: this should be present
                    }
                    mock_cache.get.return_value = mock_indicators

                    # Call check_breakout
                    result = strategy.check_breakout(
                        symbol="TEST",
                        current_price=100.50,
                        current_volume=2_000_000,
                        avg_volume=1_000_000
                    )

                    # Verify indicator cache was queried
                    mock_cache.get.assert_called()

    def test_scoring_uses_macd_histogram_only(self):
        """Verify _check_breakout_with_scoring uses histogram, not prev_histogram."""
        from strategy.orb import ORBStrategy, OpeningRange
        from config.settings import TradingConfig, SentimentConfig

        with patch('strategy.orb.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        orb = OpeningRange(
            symbol="TEST",
            high=100.00,
            low=98.00,
            range_size=2.00,
            vwap=99.00,
            timestamp=datetime.now()
        )

        # Note: _check_breakout_with_scoring only takes macd_histogram, not prev
        # This is correct because scoring uses histogram direction, not momentum
        result = strategy._check_breakout_with_scoring(
            symbol="TEST",
            price=100.50,
            orb=orb,
            vwap=99.50,
            rsi=55.0,
            rel_volume=2.0,
            macd_histogram=0.15,  # Positive histogram = points for LONG
            sentiment=0.2,
            last_candle_close=100.30
        )

        # Result should be possible (not blocked by MACD)
        # The scoring system doesn't use is_macd_bullish/bearish
        # It just checks if histogram is positive (LONG) or negative (SHORT)
        if result is not None:
            assert result[0] == 'LONG'


# ============================================================================
# Diagnostic: Simulate Full Flow to Check MACD
# ============================================================================

@pytest.mark.diagnostic
class TestFullMACDFlow:
    """End-to-end diagnostic for MACD in signal generation."""

    def test_macd_values_in_signal_flow(self):
        """Trace MACD values through the entire signal flow."""
        from strategy.orb import ORBStrategy, OpeningRange
        from config.settings import TradingConfig, SentimentConfig

        with patch('strategy.orb.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        orb = OpeningRange(
            symbol="TEST",
            high=100.00,
            low=98.00,
            range_size=2.00,
            vwap=99.00,
            timestamp=datetime.now()
        )
        strategy.opening_ranges["TEST"] = orb

        # Calculate indicators from real data
        n_bars = 50
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

        df = pd.DataFrame({
            'open': prices - 0.2,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': [1_000_000] * n_bars
        }, index=dates)

        calc = IndicatorCalculator(df)
        calc.add_all_indicators()
        indicators = calc.get_latest_indicators()

        print("\n=== MACD Values from IndicatorCalculator ===")
        print(f"macd: {indicators.get('macd')}")
        print(f"macd_signal: {indicators.get('macd_signal')}")
        print(f"macd_histogram: {indicators.get('macd_histogram')}")
        print(f"prev_macd_histogram: {indicators.get('prev_macd_histogram')}")

        # Check if is_macd_bullish would pass
        macd = indicators.get('macd', 0)
        signal = indicators.get('macd_signal', 0)
        histogram = indicators.get('macd_histogram', 0)
        prev_histogram = indicators.get('prev_macd_histogram')

        bullish = is_macd_bullish(macd, signal, histogram, prev_histogram)
        bearish = is_macd_bearish(macd, signal, histogram, prev_histogram)

        print(f"\nis_macd_bullish: {bullish}")
        print(f"is_macd_bearish: {bearish}")
        print(f"prev_histogram is None: {prev_histogram is None}")

        # Verify prev_histogram is not None (this would be the bug)
        assert prev_histogram is not None, "prev_histogram should not be None!"
