"""
Integration tests for strategy/orb.py - Opening Range Breakout Strategy.

Tests cover:
- Opening Range calculation (4 tests)
- LONG breakout conditions (7 tests)
- SHORT breakout conditions (7 tests)
- Kelly Criterion sizing (4 tests)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from strategy.orb import (
    ORBStrategy,
    OpeningRange,
    TradeSignal,
    SignalType,
    TradeResult
)
from config.settings import TradingConfig, SentimentConfig


# ============================================================================
# Opening Range Calculation Tests (4 tests)
# ============================================================================

class TestCalculateOpeningRange:
    """Tests for opening range calculation."""

    @pytest.fixture
    def orb_strategy(self):
        """Create fresh ORBStrategy instance for each test."""
        with patch('strategy.orb.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()
        return strategy

    @pytest.fixture
    def orb_bars(self):
        """Create sample bars for ORB calculation (20 bars, enough for 15-min ORB).

        ORB range is designed to be ~1% which passes MODERATE filters (0.2%-2.5%).
        """
        n_bars = 20
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='1min')

        return pd.DataFrame({
            'open': [100.0] * n_bars,
            'high': [100.0 + i * 0.02 for i in range(n_bars)],  # Gradually increasing high
            'low': [99.5 - i * 0.01 for i in range(n_bars)],    # Gradually decreasing low
            'close': [99.75 + i * 0.01 for i in range(n_bars)],
            'volume': [1_000_000] * n_bars
        }, index=dates)

    def test_calculate_opening_range_success(self, orb_strategy, orb_bars):
        """Opening range should be calculated from first N minutes."""
        with patch('strategy.orb.market_data') as mock_market_data:
            mock_market_data.get_bars.return_value = orb_bars

            result = orb_strategy.calculate_opening_range("AAPL")

            assert result is not None
            assert isinstance(result, OpeningRange)
            assert result.symbol == "AAPL"
            assert result.high > 0
            assert result.low > 0
            assert result.range_size == result.high - result.low
            assert result.vwap > 0

    def test_calculate_opening_range_insufficient_bars(self, orb_strategy):
        """Opening range should return None if not enough bars."""
        with patch('strategy.orb.market_data') as mock_market_data:
            # Return only 5 bars (need 15 for ORB)
            short_bars = pd.DataFrame({
                'open': [100.0] * 5,
                'high': [101.0] * 5,
                'low': [99.0] * 5,
                'close': [100.5] * 5,
                'volume': [1_000_000] * 5
            })
            mock_market_data.get_bars.return_value = short_bars

            result = orb_strategy.calculate_opening_range("AAPL")

            assert result is None

    def test_calculate_opening_range_empty_bars(self, orb_strategy):
        """Opening range should return None for empty bars."""
        with patch('strategy.orb.market_data') as mock_market_data:
            mock_market_data.get_bars.return_value = pd.DataFrame()

            result = orb_strategy.calculate_opening_range("AAPL")

            assert result is None

    def test_calculate_opening_range_caches_result(self, orb_strategy, orb_bars):
        """Opening range should be cached in strategy."""
        with patch('strategy.orb.market_data') as mock_market_data:
            mock_market_data.get_bars.return_value = orb_bars

            result = orb_strategy.calculate_opening_range("AAPL")

            assert "AAPL" in orb_strategy.opening_ranges
            assert orb_strategy.opening_ranges["AAPL"] == result


# ============================================================================
# LONG Breakout Condition Tests (7 tests)
# ============================================================================

class TestLongBreakoutConditions:
    """Tests for LONG breakout signal conditions."""

    @pytest.fixture
    def orb_strategy_with_range(self):
        """Create ORBStrategy with pre-calculated opening range."""
        with patch('strategy.orb.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        # Add an opening range
        strategy.opening_ranges["AAPL"] = OpeningRange(
            symbol="AAPL",
            high=150.00,
            low=148.00,
            range_size=2.00,
            vwap=149.00,
            timestamp=datetime.now()
        )
        return strategy

    def test_check_long_conditions_all_met(self, orb_strategy_with_range):
        """LONG signal when all conditions are met."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_long_conditions(
            price=151.00,      # Above ORB high (150.00)
            orb=orb,
            vwap=149.50,       # Price > VWAP
            rsi=55.0,          # Below overbought (70)
            rel_volume=2.0,    # Above min (1.5)
            macd_bullish=True,
            sentiment=0.2      # Above min (-0.3)
        )

        assert result is True

    def test_check_long_conditions_price_below_orb(self, orb_strategy_with_range):
        """LONG signal should fail if price below ORB high."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_long_conditions(
            price=149.50,      # BELOW ORB high (150.00)
            orb=orb,
            vwap=149.00,
            rsi=55.0,
            rel_volume=2.0,
            macd_bullish=True,
            sentiment=0.2
        )

        assert result is False

    def test_check_long_conditions_below_vwap(self, orb_strategy_with_range):
        """LONG signal should fail if price below VWAP."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_long_conditions(
            price=151.00,
            orb=orb,
            vwap=152.00,       # Price BELOW VWAP
            rsi=55.0,
            rel_volume=2.0,
            macd_bullish=True,
            sentiment=0.2
        )

        assert result is False

    def test_check_long_conditions_low_volume(self, orb_strategy_with_range):
        """LONG signal should fail if relative volume too low."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_long_conditions(
            price=151.00,
            orb=orb,
            vwap=149.00,
            rsi=55.0,
            rel_volume=1.1,    # BELOW min (1.2 for MODERATE)
            macd_bullish=True,
            sentiment=0.2
        )

        assert result is False

    def test_check_long_conditions_overbought_rsi(self, orb_strategy_with_range):
        """LONG signal should fail if RSI is overbought."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_long_conditions(
            price=151.00,
            orb=orb,
            vwap=149.00,
            rsi=76.0,          # ABOVE overbought (75 for MODERATE)
            rel_volume=2.0,
            macd_bullish=True,
            sentiment=0.2
        )

        assert result is False

    def test_check_long_conditions_macd_not_bullish(self, orb_strategy_with_range):
        """LONG signal should fail if MACD not bullish."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_long_conditions(
            price=151.00,
            orb=orb,
            vwap=149.00,
            rsi=55.0,
            rel_volume=2.0,
            macd_bullish=False,  # NOT bullish
            sentiment=0.2
        )

        assert result is False

    def test_check_long_conditions_negative_sentiment(self, orb_strategy_with_range):
        """LONG signal should fail if sentiment too negative."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_long_conditions(
            price=151.00,
            orb=orb,
            vwap=149.00,
            rsi=55.0,
            rel_volume=2.0,
            macd_bullish=True,
            sentiment=-0.6     # BELOW min (-0.5 for MODERATE)
        )

        assert result is False


# ============================================================================
# SHORT Breakout Condition Tests (7 tests)
# ============================================================================

class TestShortBreakoutConditions:
    """Tests for SHORT breakout signal conditions."""

    @pytest.fixture
    def orb_strategy_with_range(self):
        """Create ORBStrategy with pre-calculated opening range."""
        with patch('strategy.orb.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        strategy.opening_ranges["AAPL"] = OpeningRange(
            symbol="AAPL",
            high=150.00,
            low=148.00,
            range_size=2.00,
            vwap=149.00,
            timestamp=datetime.now()
        )
        return strategy

    def test_check_short_conditions_all_met(self, orb_strategy_with_range):
        """SHORT signal when all conditions are met."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_short_conditions(
            price=147.00,      # Below ORB low (148.00)
            orb=orb,
            vwap=149.50,       # Price < VWAP
            rsi=45.0,          # Above oversold (30)
            rel_volume=2.0,    # Above min (1.5)
            macd_bearish=True,
            sentiment=-0.2     # Below max (0.3)
        )

        assert result is True

    def test_check_short_conditions_price_above_orb(self, orb_strategy_with_range):
        """SHORT signal should fail if price above ORB low."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_short_conditions(
            price=149.00,      # ABOVE ORB low (148.00)
            orb=orb,
            vwap=150.00,
            rsi=45.0,
            rel_volume=2.0,
            macd_bearish=True,
            sentiment=-0.2
        )

        assert result is False

    def test_check_short_conditions_above_vwap(self, orb_strategy_with_range):
        """SHORT signal should fail if price above VWAP."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_short_conditions(
            price=147.00,
            orb=orb,
            vwap=146.00,       # Price ABOVE VWAP
            rsi=45.0,
            rel_volume=2.0,
            macd_bearish=True,
            sentiment=-0.2
        )

        assert result is False

    def test_check_short_conditions_oversold_rsi(self, orb_strategy_with_range):
        """SHORT signal should fail if RSI is oversold."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_short_conditions(
            price=147.00,
            orb=orb,
            vwap=149.00,
            rsi=24.0,          # BELOW oversold (25 for MODERATE)
            rel_volume=2.0,
            macd_bearish=True,
            sentiment=-0.2
        )

        assert result is False

    def test_check_short_conditions_positive_sentiment(self, orb_strategy_with_range):
        """SHORT signal should fail if sentiment too positive."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_short_conditions(
            price=147.00,
            orb=orb,
            vwap=149.00,
            rsi=45.0,
            rel_volume=2.0,
            macd_bearish=True,
            sentiment=0.6      # ABOVE max (0.5 for MODERATE)
        )

        assert result is False

    def test_check_short_conditions_macd_not_bearish(self, orb_strategy_with_range):
        """SHORT signal should fail if MACD not bearish."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_short_conditions(
            price=147.00,
            orb=orb,
            vwap=149.00,
            rsi=45.0,
            rel_volume=2.0,
            macd_bearish=False,  # NOT bearish
            sentiment=-0.2
        )

        assert result is False

    def test_check_short_conditions_low_volume(self, orb_strategy_with_range):
        """SHORT signal should fail if relative volume too low."""
        orb = orb_strategy_with_range.opening_ranges["AAPL"]

        result = orb_strategy_with_range._check_short_conditions(
            price=147.00,
            orb=orb,
            vwap=149.00,
            rsi=45.0,
            rel_volume=1.1,    # BELOW min (1.2 for MODERATE)
            macd_bearish=True,
            sentiment=-0.2
        )

        assert result is False


# ============================================================================
# Kelly Criterion Tests (4 tests)
# ============================================================================

class TestKellyCriterion:
    """Tests for Kelly Criterion position sizing."""

    @pytest.fixture
    def orb_strategy(self):
        """Create fresh ORBStrategy instance."""
        with patch('strategy.orb.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()
        return strategy

    def test_calculate_position_size_kelly_default(self, orb_strategy):
        """Position size with default Kelly (no history)."""
        position_size, risk_amount = orb_strategy._calculate_position_size_kelly(
            entry_price=100.0,
            stop_loss=98.0
        )

        # With default 2% risk and $25k capital
        # Risk per share = $2, Max risk = $500
        # Position size = 500 / 2 = 250 shares
        assert position_size > 0
        assert risk_amount > 0
        assert risk_amount <= 25000 * 0.02  # Max 2% of capital

    def test_calculate_position_size_kelly_with_history(self, orb_strategy, sample_trade_history):
        """Position size should adjust based on trade history."""
        orb_strategy.trade_history = sample_trade_history

        position_size, risk_amount = orb_strategy._calculate_position_size_kelly(
            entry_price=100.0,
            stop_loss=98.0
        )

        # Should still produce valid results
        assert position_size > 0
        assert risk_amount > 0

    def test_calculate_position_size_kelly_zero_risk(self, orb_strategy):
        """Position size should be (0, 0.0) when risk per share is zero."""
        position_size, risk_amount = orb_strategy._calculate_position_size_kelly(
            entry_price=100.0,
            stop_loss=100.0  # Same as entry = zero risk
        )

        assert position_size == 0
        assert risk_amount == 0.0

    def test_calculate_kelly_fraction_minimum_history(self, orb_strategy):
        """Kelly fraction should use default when history < 5 trades."""
        # Less than 5 trades
        orb_strategy.trade_history = [
            TradeResult(symbol="AAPL", entry_price=150, exit_price=155, pnl=5, pnl_pct=0.033, won=True),
            TradeResult(symbol="TSLA", entry_price=250, exit_price=245, pnl=-5, pnl_pct=-0.02, won=False),
        ]

        kelly = orb_strategy._calculate_kelly_fraction()

        # Should return default risk_per_trade (0.02)
        assert kelly == orb_strategy.config.risk_per_trade


# ============================================================================
# Trade Signal Creation Tests
# ============================================================================

class TestTradeSignalCreation:
    """Tests for trade signal creation."""

    @pytest.fixture
    def orb_strategy(self):
        """Create fresh ORBStrategy instance."""
        with patch('strategy.orb.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()
        return strategy

    def test_create_long_signal(self, orb_strategy, sample_opening_range):
        """LONG signal should be created with correct values."""
        with patch('strategy.orb.market_data') as mock_market_data:
            # Return empty bars so hybrid stop falls back to ORB-based stop
            mock_market_data.get_bars.return_value = pd.DataFrame()

            signal = orb_strategy._create_long_signal(
                symbol="AAPL",
                entry_price=151.00,
                orb=sample_opening_range,
                vwap=149.50,
                rsi=55.0,
                rel_volume=2.0,
                macd=0.5,
                macd_signal=0.3,
                macd_histogram=0.2,
                sentiment=0.3
            )

            assert isinstance(signal, TradeSignal)
            assert signal.symbol == "AAPL"
            assert signal.signal_type == SignalType.LONG
            assert signal.entry_price == 151.00
            assert signal.stop_loss == sample_opening_range.low  # Stop at ORB low
            assert signal.take_profit > signal.entry_price  # Target above entry
            assert signal.position_size > 0
            assert signal.rsi == 55.0
            assert signal.sentiment_score == 0.3

    def test_create_short_signal(self, orb_strategy, sample_opening_range):
        """SHORT signal should be created with correct values."""
        with patch('strategy.orb.market_data') as mock_market_data:
            # Return empty bars so hybrid stop falls back to ORB-based stop
            mock_market_data.get_bars.return_value = pd.DataFrame()

            signal = orb_strategy._create_short_signal(
                symbol="AAPL",
                entry_price=147.00,
                orb=sample_opening_range,
                vwap=149.50,
                rsi=45.0,
                rel_volume=2.0,
                macd=-0.5,
                macd_signal=-0.3,
                macd_histogram=-0.2,
                sentiment=-0.3
            )

            assert isinstance(signal, TradeSignal)
            assert signal.symbol == "AAPL"
            assert signal.signal_type == SignalType.SHORT
            assert signal.entry_price == 147.00
            assert signal.stop_loss == sample_opening_range.high  # Stop at ORB high
            assert signal.take_profit < signal.entry_price  # Target below entry
            assert signal.position_size > 0
            assert signal.sentiment_score == -0.3


# ============================================================================
# Trade Recording Tests
# ============================================================================

class TestTradeRecording:
    """Tests for trade result recording."""

    @pytest.fixture
    def orb_strategy(self):
        """Create fresh ORBStrategy instance."""
        with patch('strategy.orb.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()
        return strategy

    def test_record_winning_trade(self, orb_strategy):
        """Winning trade should be recorded correctly."""
        orb_strategy.record_trade_result(
            symbol="AAPL",
            entry_price=100.0,
            exit_price=110.0,
            is_long=True
        )

        assert len(orb_strategy.trade_history) == 1
        trade = orb_strategy.trade_history[0]
        assert trade.won is True
        assert trade.pnl == 10.0
        assert trade.pnl_pct == 0.1

    def test_record_losing_trade(self, orb_strategy):
        """Losing trade should be recorded correctly."""
        orb_strategy.record_trade_result(
            symbol="TSLA",
            entry_price=200.0,
            exit_price=190.0,
            is_long=True
        )

        assert len(orb_strategy.trade_history) == 1
        trade = orb_strategy.trade_history[0]
        assert trade.won is False
        assert trade.pnl == -10.0
        assert trade.pnl_pct == -0.05


# ============================================================================
# Reset and Utility Tests
# ============================================================================

class TestStrategyUtilities:
    """Tests for strategy utility methods."""

    @pytest.fixture
    def orb_strategy(self):
        """Create fresh ORBStrategy instance."""
        with patch('strategy.orb.settings') as mock_settings:
            mock_settings.trading = TradingConfig()
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()
        return strategy

    def test_reset_daily(self, orb_strategy, sample_opening_range, long_trade_signal):
        """reset_daily should clear all daily data."""
        # Add some data
        orb_strategy.opening_ranges["AAPL"] = sample_opening_range
        orb_strategy.signals_today.append(long_trade_signal)
        orb_strategy.sentiment_cache["AAPL"] = 0.5

        # Reset
        orb_strategy.reset_daily()

        assert len(orb_strategy.opening_ranges) == 0
        assert len(orb_strategy.signals_today) == 0
        assert len(orb_strategy.sentiment_cache) == 0

    def test_update_sentiment(self, orb_strategy):
        """update_sentiment should cache sentiment score."""
        orb_strategy.update_sentiment("AAPL", 0.5)
        assert orb_strategy.sentiment_cache["AAPL"] == 0.5

        # Test clamping
        orb_strategy.update_sentiment("TSLA", 1.5)  # Above max
        assert orb_strategy.sentiment_cache["TSLA"] == 1.0

        orb_strategy.update_sentiment("NVDA", -1.5)  # Below min
        assert orb_strategy.sentiment_cache["NVDA"] == -1.0

    def test_get_kelly_stats(self, orb_strategy, sample_trade_history):
        """get_kelly_stats should return current statistics."""
        orb_strategy.trade_history = sample_trade_history

        stats = orb_strategy.get_kelly_stats()

        assert 'win_rate' in stats
        assert 'avg_win' in stats
        assert 'avg_loss' in stats
        assert 'kelly_fraction' in stats
        assert 'trade_count' in stats
        assert stats['trade_count'] == len(sample_trade_history)
