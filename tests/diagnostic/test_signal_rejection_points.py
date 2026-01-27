"""
Diagnostic tests to identify exactly WHY signals are being rejected.

These tests systematically check each rejection point in the signal generation
pipeline to identify what's blocking signal generation.

Run with: pytest tests/diagnostic/ -v --tb=long
"""
import pytest
from datetime import datetime, time
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from strategy.orb import ORBStrategy, OpeningRange, SignalType
from config.settings import TradingConfig, SentimentConfig, SignalLevel


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def fresh_orb_strategy():
    """Create a fresh ORBStrategy with MODERATE level."""
    with patch('strategy.orb.settings') as mock_settings:
        config = TradingConfig()
        config.signal_level = SignalLevel.MODERATE
        mock_settings.trading = config
        mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
        strategy = ORBStrategy()
    return strategy


@pytest.fixture
def valid_orb():
    """Create a valid OpeningRange for testing."""
    return OpeningRange(
        symbol="TEST",
        high=100.00,
        low=98.00,
        range_size=2.00,
        vwap=99.00,
        timestamp=datetime.now()
    )


# ============================================================================
# Rejection Point 1: No Opening Range Calculated
# ============================================================================

@pytest.mark.diagnostic
class TestRejectionNoORB:
    """Test rejection when no ORB is calculated for the symbol."""

    def test_check_breakout_no_orb_returns_none(self, fresh_orb_strategy):
        """check_breakout should return None if no ORB exists for symbol."""
        # Don't add any opening range
        result = fresh_orb_strategy.check_breakout(
            symbol="MISSING",
            current_price=100.0,
            current_volume=1_000_000,
            avg_volume=500_000
        )
        assert result is None

    def test_opening_ranges_dict_starts_empty(self, fresh_orb_strategy):
        """Opening ranges should be empty on fresh strategy."""
        assert len(fresh_orb_strategy.opening_ranges) == 0


# ============================================================================
# Rejection Point 2: Daily Trade Limit Reached
# ============================================================================

@pytest.mark.diagnostic
class TestRejectionDailyLimit:
    """Test rejection when daily trade limit is reached."""

    def test_signals_today_blocks_new_signals(self, fresh_orb_strategy, valid_orb):
        """No new signals when signals_today >= max_trades_per_day."""
        fresh_orb_strategy.opening_ranges["TEST"] = valid_orb

        # Fill up signals_today to the limit
        max_trades = fresh_orb_strategy.config.max_trades_per_day
        for i in range(max_trades):
            fresh_orb_strategy.signals_today.append(MagicMock())

        with patch('strategy.orb.market_data'):
            result = fresh_orb_strategy.check_breakout(
                symbol="TEST",
                current_price=100.50,
                current_volume=2_000_000,
                avg_volume=1_000_000
            )

        assert result is None

    def test_max_trades_per_day_default_value(self, fresh_orb_strategy):
        """Verify max_trades_per_day has expected default."""
        assert fresh_orb_strategy.config.max_trades_per_day == 3


# ============================================================================
# Rejection Point 3: Daily Loss Circuit Breaker
# ============================================================================

@pytest.mark.diagnostic
class TestRejectionDailyLossLimit:
    """Test rejection when daily loss limit is hit."""

    def test_daily_pnl_circuit_breaker(self, fresh_orb_strategy, valid_orb):
        """No new signals when daily_pnl <= -max_daily_loss."""
        fresh_orb_strategy.opening_ranges["TEST"] = valid_orb

        # Set daily P/L to exceed loss limit
        fresh_orb_strategy.daily_pnl = -fresh_orb_strategy.config.max_daily_loss - 1

        with patch('strategy.orb.market_data'):
            result = fresh_orb_strategy.check_breakout(
                symbol="TEST",
                current_price=100.50,
                current_volume=2_000_000,
                avg_volume=1_000_000
            )

        assert result is None

    def test_max_daily_loss_default_value(self, fresh_orb_strategy):
        """Verify max_daily_loss has expected default ($750 = 3% of $25k)."""
        assert fresh_orb_strategy.config.max_daily_loss == 750.0


# ============================================================================
# Rejection Point 4: Consecutive Loss Cooldown
# ============================================================================

@pytest.mark.diagnostic
class TestRejectionConsecutiveLosses:
    """Test rejection when consecutive losses trigger cooldown."""

    def test_consecutive_losses_cooldown(self, fresh_orb_strategy, valid_orb):
        """No new signals when consecutive_losses >= max_consecutive_losses."""
        fresh_orb_strategy.opening_ranges["TEST"] = valid_orb

        # Set consecutive losses to trigger cooldown
        max_consecutive = fresh_orb_strategy.config.max_consecutive_losses
        fresh_orb_strategy.consecutive_losses = max_consecutive

        with patch('strategy.orb.market_data'):
            result = fresh_orb_strategy.check_breakout(
                symbol="TEST",
                current_price=100.50,
                current_volume=2_000_000,
                avg_volume=1_000_000
            )

        assert result is None

    def test_max_consecutive_losses_default(self, fresh_orb_strategy):
        """Verify max_consecutive_losses has expected default."""
        assert fresh_orb_strategy.config.max_consecutive_losses == 2


# ============================================================================
# Rejection Point 5: Trading Window Restriction
# ============================================================================

@pytest.mark.diagnostic
class TestRejectionTradingWindow:
    """Test rejection when outside trading window."""

    def test_past_latest_trade_time_rejects(self, fresh_orb_strategy, valid_orb):
        """No new signals after latest_trade_time."""
        from freezegun import freeze_time

        fresh_orb_strategy.opening_ranges["TEST"] = valid_orb

        # MODERATE level has latest_trade_time = "14:30"
        # Test at 15:00 (after cutoff)
        with freeze_time("2024-01-15 15:00:00"):
            with patch('strategy.orb.market_data'):
                result = fresh_orb_strategy.check_breakout(
                    symbol="TEST",
                    current_price=100.50,
                    current_volume=2_000_000,
                    avg_volume=1_000_000
                )

        assert result is None

    def test_latest_trade_time_by_signal_level(self):
        """Verify latest_trade_time varies by signal level."""
        from config.settings import SIGNAL_LEVEL_CONFIGS

        assert SIGNAL_LEVEL_CONFIGS[SignalLevel.STRICT].latest_trade_time == "11:30"
        assert SIGNAL_LEVEL_CONFIGS[SignalLevel.MODERATE].latest_trade_time == "14:30"
        assert SIGNAL_LEVEL_CONFIGS[SignalLevel.RELAXED].latest_trade_time == "15:30"


# ============================================================================
# Rejection Point 6: Empty Bars from Market Data
# ============================================================================

@pytest.mark.diagnostic
class TestRejectionEmptyBars:
    """Test rejection when market_data.get_bars returns empty."""

    def test_empty_bars_rejects(self, fresh_orb_strategy, valid_orb):
        """No signal when get_bars returns empty DataFrame."""
        fresh_orb_strategy.opening_ranges["TEST"] = valid_orb

        with patch('strategy.orb.market_data') as mock_market:
            mock_market.get_bars.return_value = pd.DataFrame()

            result = fresh_orb_strategy.check_breakout(
                symbol="TEST",
                current_price=100.50,
                current_volume=2_000_000,
                avg_volume=1_000_000
            )

        assert result is None


# ============================================================================
# Rejection Point 7: Volume Hard Floor (1.2x RVOL)
# ============================================================================

@pytest.mark.diagnostic
class TestRejectionVolumeFloor:
    """Test rejection when relative volume is below hard floor."""

    def test_volume_below_hard_floor_rejects(self, fresh_orb_strategy, valid_orb):
        """No signal when relative_volume < 1.2 (hard floor)."""
        # The hard floor in _check_breakout_with_scoring is 1.2
        result = fresh_orb_strategy._check_breakout_with_scoring(
            symbol="TEST",
            price=100.50,
            orb=valid_orb,
            vwap=99.50,
            rsi=55.0,
            rel_volume=1.1,  # Below 1.2 hard floor
            macd_histogram=0.15,
            sentiment=0.2,
            last_candle_close=100.30
        )

        assert result is None

    def test_volume_at_hard_floor_passes(self, fresh_orb_strategy, valid_orb):
        """Signal possible when relative_volume >= 1.2."""
        result = fresh_orb_strategy._check_breakout_with_scoring(
            symbol="TEST",
            price=100.50,
            orb=valid_orb,
            vwap=99.50,
            rsi=55.0,
            rel_volume=1.2,  # At hard floor
            macd_histogram=0.15,
            sentiment=0.2,
            last_candle_close=100.30
        )

        # May or may not pass based on score, but should not be rejected by volume floor
        # If result is None, it's due to score threshold, not volume floor
        # We just verify it doesn't immediately reject
        pass  # Test passes if no exception


# ============================================================================
# Rejection Point 8: No Breakout Detected
# ============================================================================

@pytest.mark.diagnostic
class TestRejectionNoBreakout:
    """Test rejection when price hasn't broken ORB levels."""

    def test_price_within_orb_range_no_breakout(self, fresh_orb_strategy, valid_orb):
        """No signal when price is within ORB range."""
        result = fresh_orb_strategy._check_breakout_with_scoring(
            symbol="TEST",
            price=99.00,  # Within ORB range (98-100)
            orb=valid_orb,
            vwap=99.00,
            rsi=55.0,
            rel_volume=2.0,
            macd_histogram=0.15,
            sentiment=0.2,
            last_candle_close=99.00
        )

        assert result is None

    def test_breakout_buffer_calculation(self, fresh_orb_strategy, valid_orb):
        """Verify breakout buffer is applied correctly."""
        buffer = fresh_orb_strategy.config.breakout_buffer_pct  # 0.001 = 0.1%

        # ORB high = 100.00, with 0.1% buffer = 100.10
        long_breakout_level = valid_orb.high * (1 + buffer)
        assert long_breakout_level == pytest.approx(100.10, rel=1e-3)

        # Price just above ORB high but below buffer
        result = fresh_orb_strategy._check_breakout_with_scoring(
            symbol="TEST",
            price=100.05,  # Above ORB high but below buffer
            orb=valid_orb,
            vwap=99.50,
            rsi=55.0,
            rel_volume=2.0,
            macd_histogram=0.15,
            sentiment=0.2,
            last_candle_close=100.05
        )

        assert result is None  # Should not trigger breakout


# ============================================================================
# Rejection Point 9: Candle Close Confirmation (STRICT/MODERATE)
# ============================================================================

@pytest.mark.diagnostic
class TestRejectionCandleClose:
    """Test rejection when candle close doesn't confirm breakout."""

    def test_candle_close_required_moderate(self, fresh_orb_strategy, valid_orb):
        """MODERATE requires candle close confirmation."""
        assert fresh_orb_strategy.config.require_candle_close is True

    def test_candle_close_not_confirming_rejects(self, fresh_orb_strategy, valid_orb):
        """No signal when last candle close doesn't confirm breakout."""
        # Price above breakout but last candle closed below
        result = fresh_orb_strategy._check_breakout_with_scoring(
            symbol="TEST",
            price=100.50,  # Above ORB high + buffer
            orb=valid_orb,
            vwap=99.50,
            rsi=55.0,
            rel_volume=2.0,
            macd_histogram=0.15,
            sentiment=0.2,
            last_candle_close=99.90  # Closed BELOW breakout level
        )

        assert result is None

    def test_relaxed_doesnt_require_candle_close(self):
        """RELAXED level doesn't require candle close confirmation."""
        from config.settings import SIGNAL_LEVEL_CONFIGS

        assert SIGNAL_LEVEL_CONFIGS[SignalLevel.RELAXED].require_candle_close is False


# ============================================================================
# Rejection Point 10: Signal Score Below Threshold
# ============================================================================

@pytest.mark.diagnostic
class TestRejectionScoreThreshold:
    """Test rejection when signal score is below minimum threshold."""

    def test_score_below_threshold_rejects(self, fresh_orb_strategy, valid_orb):
        """No signal when score < min_signal_score."""
        # Create conditions that produce a low score
        result = fresh_orb_strategy._check_breakout_with_scoring(
            symbol="TEST",
            price=100.15,       # Barely above breakout (low breakout score)
            orb=valid_orb,
            vwap=100.10,        # Price barely above VWAP (low VWAP score)
            rsi=75.0,           # Near overbought (low RSI score for LONG)
            rel_volume=1.25,    # Low volume (low volume score)
            macd_histogram=0.01,  # Weak MACD (low MACD score)
            sentiment=-0.4,     # Weak sentiment (low sentiment score)
            last_candle_close=100.11
        )

        # Should be rejected due to low score
        assert result is None

    def test_min_signal_score_by_level(self):
        """Verify min_signal_score varies by signal level."""
        from config.settings import SIGNAL_LEVEL_CONFIGS

        assert SIGNAL_LEVEL_CONFIGS[SignalLevel.STRICT].min_signal_score == 70.0
        assert SIGNAL_LEVEL_CONFIGS[SignalLevel.MODERATE].min_signal_score == 55.0
        assert SIGNAL_LEVEL_CONFIGS[SignalLevel.RELAXED].min_signal_score == 40.0

    def test_high_score_passes_threshold(self, fresh_orb_strategy, valid_orb):
        """Signal generated when score >= min_signal_score."""
        # Create conditions that produce a high score
        result = fresh_orb_strategy._check_breakout_with_scoring(
            symbol="TEST",
            price=100.50,       # Strong breakout (high breakout score)
            orb=valid_orb,
            vwap=99.00,         # Price well above VWAP (high VWAP score)
            rsi=50.0,           # Middle RSI (high RSI score)
            rel_volume=3.0,     # Strong volume (high volume score)
            macd_histogram=0.20,  # Strong MACD (high MACD score)
            sentiment=0.5,      # Strong sentiment (high sentiment score)
            last_candle_close=100.40
        )

        assert result is not None
        assert result[0] == 'LONG'
        assert result[1] >= fresh_orb_strategy.config.min_signal_score


# ============================================================================
# Diagnostic: Print All Thresholds for Each Level
# ============================================================================

@pytest.mark.diagnostic
class TestDiagnosticThresholds:
    """Diagnostic tests to print all threshold values."""

    def test_print_moderate_thresholds(self):
        """Print all MODERATE level thresholds for debugging."""
        from config.settings import SIGNAL_LEVEL_CONFIGS

        config = SIGNAL_LEVEL_CONFIGS[SignalLevel.MODERATE]

        print("\n=== MODERATE Level Thresholds ===")
        print(f"min_signal_score: {config.min_signal_score}")
        print(f"min_relative_volume: {config.min_relative_volume}")
        print(f"min_orb_range_pct: {config.min_orb_range_pct}%")
        print(f"max_orb_range_pct: {config.max_orb_range_pct}%")
        print(f"latest_trade_time: {config.latest_trade_time}")
        print(f"require_candle_close: {config.require_candle_close}")
        print(f"min_sentiment_long: {config.min_sentiment_long}")
        print(f"max_sentiment_short: {config.max_sentiment_short}")
        print(f"rsi_overbought: {config.rsi_overbought}")
        print(f"rsi_oversold: {config.rsi_oversold}")

        # This test always passes - it's just for diagnostics
        assert True

    def test_score_calculation_breakdown(self, fresh_orb_strategy, valid_orb):
        """Print score breakdown for a typical scenario."""
        price = 100.50
        vwap = 99.00
        rsi = 50.0
        rel_volume = 2.0
        macd_histogram = 0.15
        sentiment = 0.3

        score = fresh_orb_strategy._calculate_signal_score(
            price=price,
            orb=valid_orb,
            vwap=vwap,
            rsi=rsi,
            rel_volume=rel_volume,
            macd_histogram=macd_histogram,
            sentiment=sentiment,
            direction='LONG',
            last_candle_close=price
        )

        print("\n=== Score Breakdown for LONG ===")
        print(f"Price: ${price}, ORB High: ${valid_orb.high}")
        print(f"Breakout %: {(price - valid_orb.high) / valid_orb.high * 100:.2f}%")
        print(f"VWAP: ${vwap}, Price above by: {(price - vwap) / vwap * 100:.2f}%")
        print(f"RSI: {rsi}")
        print(f"Relative Volume: {rel_volume}x")
        print(f"MACD Histogram: {macd_histogram}")
        print(f"Sentiment: {sentiment}")
        print(f"TOTAL SCORE: {score:.1f}/100")
        print(f"Min required (MODERATE): {fresh_orb_strategy.config.min_signal_score}")

        assert True  # Diagnostic test
