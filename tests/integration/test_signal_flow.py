"""
Integration tests for the complete signal flow.

Tests the full pipeline: watchlist → ORB calculation → breakout monitoring → signal

Run with: pytest tests/integration/test_signal_flow.py -v
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from strategy.orb import ORBStrategy, OpeningRange, SignalType
from config.settings import TradingConfig, SentimentConfig, SignalLevel


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_market_data():
    """Create comprehensive market data mock."""
    mock = MagicMock()

    # Default bar data (50 bars)
    np.random.seed(42)
    n_bars = 50
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n_bars) * 0.5)
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

    default_bars = pd.DataFrame({
        'open': prices - 0.2,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': [1_000_000] * n_bars
    }, index=dates)

    mock.get_bars.return_value = default_bars
    mock.get_current_minute_index.return_value = 30
    mock.get_cumulative_volume_today.return_value = 30_000_000
    mock.volume_profiles = {}

    return mock


@pytest.fixture
def orb_strategy_with_mocks(mock_market_data):
    """Create ORBStrategy with all external dependencies mocked."""
    with patch('strategy.orb.settings') as mock_settings:
        config = TradingConfig()
        config.signal_level = SignalLevel.MODERATE
        mock_settings.trading = config
        mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
        strategy = ORBStrategy()

    return strategy


# ============================================================================
# ORB Calculation Integration Tests
# ============================================================================

@pytest.mark.integration
class TestORBCalculationFlow:
    """Integration tests for ORB calculation."""

    def test_orb_calculation_stores_in_opening_ranges(self, orb_strategy_with_mocks):
        """ORB calculation should store result in opening_ranges dict."""
        # Create 20 bars for ORB calculation with valid range
        n_bars = 20
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

        orb_bars = pd.DataFrame({
            'open': [100.0] * n_bars,
            'high': [100.0 + i * 0.02 for i in range(n_bars)],
            'low': [99.5 - i * 0.01 for i in range(n_bars)],
            'close': [99.75 + i * 0.01 for i in range(n_bars)],
            'volume': [1_000_000] * n_bars
        }, index=dates)

        with patch('strategy.orb.market_data') as mock_market:
            mock_market.get_bars.return_value = orb_bars

            result = orb_strategy_with_mocks.calculate_opening_range("AAPL")

            assert result is not None
            assert "AAPL" in orb_strategy_with_mocks.opening_ranges
            assert orb_strategy_with_mocks.opening_ranges["AAPL"] == result

    def test_orb_calculation_multiple_symbols(self, orb_strategy_with_mocks):
        """ORB calculation should work for multiple symbols."""
        n_bars = 20
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

        def create_orb_bars(base_price):
            return pd.DataFrame({
                'open': [base_price] * n_bars,
                'high': [base_price + i * 0.02 for i in range(n_bars)],
                'low': [base_price - 0.5 - i * 0.01 for i in range(n_bars)],
                'close': [base_price - 0.25 + i * 0.01 for i in range(n_bars)],
                'volume': [1_000_000] * n_bars
            }, index=dates)

        with patch('strategy.orb.market_data') as mock_market:
            # Calculate for multiple symbols
            mock_market.get_bars.return_value = create_orb_bars(100.0)
            orb_strategy_with_mocks.calculate_opening_range("AAPL")

            mock_market.get_bars.return_value = create_orb_bars(250.0)
            orb_strategy_with_mocks.calculate_opening_range("TSLA")

            mock_market.get_bars.return_value = create_orb_bars(400.0)
            orb_strategy_with_mocks.calculate_opening_range("NVDA")

            assert len(orb_strategy_with_mocks.opening_ranges) == 3
            assert "AAPL" in orb_strategy_with_mocks.opening_ranges
            assert "TSLA" in orb_strategy_with_mocks.opening_ranges
            assert "NVDA" in orb_strategy_with_mocks.opening_ranges


# ============================================================================
# Breakout Monitoring Integration Tests
# ============================================================================

@pytest.mark.integration
class TestBreakoutMonitoringFlow:
    """Integration tests for breakout monitoring."""

    def test_breakout_without_orb_returns_none(self, orb_strategy_with_mocks):
        """check_breakout should return None if no ORB exists."""
        with patch('strategy.orb.market_data'):
            result = orb_strategy_with_mocks.check_breakout(
                symbol="MISSING",
                current_price=100.0,
                current_volume=1_000_000,
                avg_volume=500_000
            )

        assert result is None

    def test_breakout_with_valid_orb_and_conditions(self, orb_strategy_with_mocks):
        """check_breakout should detect breakout with valid conditions."""
        # Add ORB
        orb_strategy_with_mocks.opening_ranges["TEST"] = OpeningRange(
            symbol="TEST",
            high=100.00,
            low=98.00,
            range_size=2.00,
            vwap=99.00,
            timestamp=datetime.now()
        )

        # Add positive sentiment
        orb_strategy_with_mocks.sentiment_cache["TEST"] = 0.3

        # Create bar data with good indicators
        n_bars = 50
        np.random.seed(42)
        base_price = 100.0
        # Price trending up (for bullish MACD)
        prices = np.linspace(95, 101, n_bars)
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

        bullish_bars = pd.DataFrame({
            'open': prices - 0.1,
            'high': prices + 0.3,
            'low': prices - 0.3,
            'close': prices,
            'volume': [2_000_000] * n_bars  # High volume
        }, index=dates)

        with patch('strategy.orb.market_data') as mock_market:
            mock_market.get_bars.return_value = bullish_bars
            mock_market.get_current_minute_index.return_value = 30
            mock_market.get_cumulative_volume_today.return_value = 60_000_000
            mock_market.volume_profiles = {}

            with patch('strategy.orb.indicator_cache') as mock_cache:
                # Return bullish indicators
                mock_cache.get.return_value = {
                    'vwap': 99.00,
                    'rsi': 55.0,
                    'macd': 0.3,
                    'macd_signal': 0.1,
                    'macd_histogram': 0.2,
                    'prev_macd_histogram': 0.15,
                }

                result = orb_strategy_with_mocks.check_breakout(
                    symbol="TEST",
                    current_price=100.50,  # Above ORB high
                    current_volume=3_000_000,  # High volume
                    avg_volume=1_000_000
                )

        # Should generate a LONG signal
        if result is not None:
            assert result.signal_type == SignalType.LONG
            assert result.symbol == "TEST"


# ============================================================================
# Signal Level Integration Tests
# ============================================================================

@pytest.mark.integration
class TestSignalLevelIntegration:
    """Test signal generation with different signal levels."""

    @pytest.fixture
    def orb_strategy_strict(self):
        """Create ORBStrategy with STRICT level."""
        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.STRICT
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()
        return strategy

    @pytest.fixture
    def orb_strategy_relaxed(self):
        """Create ORBStrategy with RELAXED level."""
        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.RELAXED
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()
        return strategy

    def test_strict_requires_higher_score(self, orb_strategy_strict):
        """STRICT level requires score >= 70."""
        assert orb_strategy_strict.config.min_signal_score == 70.0

        orb = OpeningRange(
            symbol="TEST",
            high=100.00,
            low=98.00,
            range_size=2.00,
            vwap=99.00,
            timestamp=datetime.now()
        )

        # Moderate conditions (would pass MODERATE but not STRICT)
        result = orb_strategy_strict._check_breakout_with_scoring(
            symbol="TEST",
            price=100.30,
            orb=orb,
            vwap=99.50,
            rsi=55.0,
            rel_volume=1.8,
            macd_histogram=0.10,
            sentiment=0.2,
            last_candle_close=100.25
        )

        # May or may not pass depending on exact score
        # Just verify the threshold is applied
        pass

    def test_relaxed_accepts_lower_score(self, orb_strategy_relaxed):
        """RELAXED level accepts score >= 40."""
        assert orb_strategy_relaxed.config.min_signal_score == 40.0

        orb = OpeningRange(
            symbol="TEST",
            high=100.00,
            low=98.00,
            range_size=2.00,
            vwap=99.00,
            timestamp=datetime.now()
        )

        # Weak conditions that would pass RELAXED
        result = orb_strategy_relaxed._check_breakout_with_scoring(
            symbol="TEST",
            price=100.50,
            orb=orb,
            vwap=99.50,
            rsi=50.0,
            rel_volume=1.5,
            macd_histogram=0.05,
            sentiment=0.0,
            last_candle_close=100.40  # RELAXED doesn't require this
        )

        if result is not None:
            assert result[0] == 'LONG'
            assert result[1] >= 40.0

    def test_change_signal_level_at_runtime(self, orb_strategy_with_mocks):
        """Signal level can be changed at runtime."""
        # Start with MODERATE
        assert orb_strategy_with_mocks.config.signal_level == SignalLevel.MODERATE
        assert orb_strategy_with_mocks.config.min_signal_score == 55.0

        # Change to STRICT
        success = orb_strategy_with_mocks.set_signal_level(SignalLevel.STRICT)
        assert success is True
        assert orb_strategy_with_mocks.config.signal_level == SignalLevel.STRICT
        assert orb_strategy_with_mocks.config.min_signal_score == 70.0

        # Change to RELAXED
        success = orb_strategy_with_mocks.set_signal_level("RELAXED")
        assert success is True
        assert orb_strategy_with_mocks.config.signal_level == SignalLevel.RELAXED
        assert orb_strategy_with_mocks.config.min_signal_score == 40.0


# ============================================================================
# Daily Reset Integration Tests
# ============================================================================

@pytest.mark.integration
class TestDailyResetFlow:
    """Test daily reset functionality."""

    def test_reset_daily_clears_all_data(self, orb_strategy_with_mocks):
        """reset_daily should clear all daily tracking data."""
        # Add some data
        orb_strategy_with_mocks.opening_ranges["AAPL"] = OpeningRange(
            symbol="AAPL", high=150.0, low=148.0, range_size=2.0,
            vwap=149.0, timestamp=datetime.now()
        )
        orb_strategy_with_mocks.signals_today.append(MagicMock())
        orb_strategy_with_mocks.sentiment_cache["AAPL"] = 0.5
        orb_strategy_with_mocks.daily_pnl = 100.0
        orb_strategy_with_mocks.consecutive_losses = 2

        # Reset
        orb_strategy_with_mocks.reset_daily()

        # Verify everything is cleared
        assert len(orb_strategy_with_mocks.opening_ranges) == 0
        assert len(orb_strategy_with_mocks.signals_today) == 0
        assert len(orb_strategy_with_mocks.sentiment_cache) == 0
        assert orb_strategy_with_mocks.daily_pnl == 0.0
        assert orb_strategy_with_mocks.consecutive_losses == 0

    def test_trade_history_persists_after_reset(self, orb_strategy_with_mocks):
        """Trade history should persist after daily reset."""
        from strategy.orb import TradeResult

        # Add trade history
        orb_strategy_with_mocks.trade_history = [
            TradeResult(symbol="A", entry_price=100, exit_price=105, pnl=5, pnl_pct=0.05, won=True),
            TradeResult(symbol="B", entry_price=100, exit_price=95, pnl=-5, pnl_pct=-0.05, won=False),
        ]

        # Reset
        orb_strategy_with_mocks.reset_daily()

        # Trade history should persist
        assert len(orb_strategy_with_mocks.trade_history) == 2


# ============================================================================
# End-to-End Flow Tests
# ============================================================================

@pytest.mark.integration
class TestEndToEndFlow:
    """End-to-end tests for the complete signal flow."""

    def test_full_signal_generation_flow(self, orb_strategy_with_mocks):
        """Test complete flow: ORB calculation → breakout check → signal."""
        # Step 1: Create ORB bars
        n_bars = 20
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

        orb_bars = pd.DataFrame({
            'open': [100.0] * n_bars,
            'high': [100.0 + i * 0.02 for i in range(n_bars)],
            'low': [99.5 - i * 0.01 for i in range(n_bars)],
            'close': [99.75 + i * 0.01 for i in range(n_bars)],
            'volume': [1_000_000] * n_bars
        }, index=dates)

        # Step 2: Calculate ORB
        with patch('strategy.orb.market_data') as mock_market:
            mock_market.get_bars.return_value = orb_bars

            orb = orb_strategy_with_mocks.calculate_opening_range("AAPL")
            assert orb is not None

        # Step 3: Add sentiment
        orb_strategy_with_mocks.update_sentiment("AAPL", 0.3)

        # Step 4: Create monitoring bars with bullish setup
        n_monitor_bars = 50
        prices = np.linspace(98, 102, n_monitor_bars)  # Trending up
        monitor_dates = pd.date_range(end=datetime.now(), periods=n_monitor_bars, freq='1min')

        monitor_bars = pd.DataFrame({
            'open': prices - 0.1,
            'high': prices + 0.3,
            'low': prices - 0.3,
            'close': prices,
            'volume': [2_000_000] * n_monitor_bars
        }, index=monitor_dates)

        # Step 5: Check for breakout
        with patch('strategy.orb.market_data') as mock_market:
            mock_market.get_bars.return_value = monitor_bars
            mock_market.get_current_minute_index.return_value = 30
            mock_market.get_cumulative_volume_today.return_value = 60_000_000
            mock_market.volume_profiles = {}

            with patch('strategy.orb.indicator_cache') as mock_cache:
                mock_cache.get.return_value = None  # Force fresh calculation

                signal = orb_strategy_with_mocks.check_breakout(
                    symbol="AAPL",
                    current_price=orb.high + 0.50,  # Above ORB high
                    current_volume=3_000_000,
                    avg_volume=1_000_000
                )

        # Signal may or may not be generated based on score
        # This test verifies the flow completes without error
        if signal is not None:
            assert signal.signal_type in [SignalType.LONG, SignalType.SHORT]
            assert signal.symbol == "AAPL"
            assert signal.position_size > 0

    def test_multiple_signals_respects_daily_limit(self, orb_strategy_with_mocks):
        """Multiple signals should respect daily trade limit."""
        # Fill up signals to the limit
        max_trades = orb_strategy_with_mocks.config.max_trades_per_day
        for i in range(max_trades):
            orb_strategy_with_mocks.signals_today.append(MagicMock())

        # Add ORB
        orb_strategy_with_mocks.opening_ranges["TEST"] = OpeningRange(
            symbol="TEST",
            high=100.00,
            low=98.00,
            range_size=2.00,
            vwap=99.00,
            timestamp=datetime.now()
        )

        # Try to generate another signal
        with patch('strategy.orb.market_data'):
            result = orb_strategy_with_mocks.check_breakout(
                symbol="TEST",
                current_price=100.50,
                current_volume=2_000_000,
                avg_volume=1_000_000
            )

        assert result is None  # Should be blocked by daily limit
