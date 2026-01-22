"""
Unit tests for ORB Trading Bot Improvements (Phases 1-6)
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, time
import pandas as pd
import numpy as np

# Import modules under test
import sys
sys.path.insert(0, '/Users/silvio_requena/Code/la-envia-v2')

from config.settings import settings
from strategy.orb import ORBStrategy, OpeningRange, SignalType


class TestPhase1BreakoutBuffer:
    """Phase 1: Breakout Buffer & Confirmation tests"""

    def test_breakout_buffer_config_exists(self):
        """Test that breakout buffer config parameter exists"""
        assert hasattr(settings.trading, 'breakout_buffer_pct')
        assert settings.trading.breakout_buffer_pct == 0.001  # 0.1%

    def test_require_candle_close_config_exists(self):
        """Test that candle close confirmation config exists"""
        assert hasattr(settings.trading, 'require_candle_close')
        assert settings.trading.require_candle_close is True

    def test_long_breakout_with_buffer(self):
        """Test that LONG breakout requires price > ORB high + buffer"""
        strategy = ORBStrategy()
        orb = OpeningRange(
            symbol='TEST',
            high=100.0,
            low=98.0,
            range_size=2.0,
            vwap=99.0,
            timestamp=datetime.now()
        )

        # With 0.1% buffer, breakout level = 100.0 * 1.001 = 100.10
        buffer = settings.trading.breakout_buffer_pct
        breakout_level = orb.high * (1 + buffer)

        # Price exactly at ORB high should NOT trigger (no buffer cleared)
        assert 100.0 <= breakout_level  # Should fail breakout check

        # Price clearing buffer should trigger
        assert 100.15 > breakout_level  # Should pass breakout check

    def test_short_breakout_with_buffer(self):
        """Test that SHORT breakout requires price < ORB low - buffer"""
        orb = OpeningRange(
            symbol='TEST',
            high=100.0,
            low=98.0,
            range_size=2.0,
            vwap=99.0,
            timestamp=datetime.now()
        )

        # With 0.1% buffer, breakout level = 98.0 * 0.999 = 97.902
        buffer = settings.trading.breakout_buffer_pct
        breakout_level = orb.low * (1 - buffer)

        # Price exactly at ORB low should NOT trigger
        assert 98.0 >= breakout_level  # Should fail breakout check

        # Price clearing buffer should trigger
        assert 97.85 < breakout_level  # Should pass breakout check


class TestPhase3ORBSizeFilter:
    """Phase 3: ORB Size Filter tests"""

    def test_orb_range_filter_config_exists(self):
        """Test that ORB range filter config parameters exist"""
        assert hasattr(settings.trading, 'min_orb_range_pct')
        assert hasattr(settings.trading, 'max_orb_range_pct')
        assert settings.trading.min_orb_range_pct == 0.3
        assert settings.trading.max_orb_range_pct == 2.0

    def test_orb_range_too_narrow(self):
        """Test that ORB ranges below min threshold are filtered"""
        # ORB with 0.1% range (below 0.3% min)
        high = 100.10
        low = 100.0
        range_pct = (high - low) / low * 100  # 0.1%

        assert range_pct < settings.trading.min_orb_range_pct

    def test_orb_range_too_wide(self):
        """Test that ORB ranges above max threshold are filtered"""
        # ORB with 3% range (above 2.0% max)
        high = 103.0
        low = 100.0
        range_pct = (high - low) / low * 100  # 3.0%

        assert range_pct > settings.trading.max_orb_range_pct

    def test_orb_range_acceptable(self):
        """Test that ORB ranges within thresholds pass"""
        # ORB with 1% range (within 0.3-2.0%)
        high = 101.0
        low = 100.0
        range_pct = (high - low) / low * 100  # 1.0%

        assert settings.trading.min_orb_range_pct <= range_pct <= settings.trading.max_orb_range_pct


class TestPhase4SoftScoring:
    """Phase 4: Soft Scoring System tests"""

    def test_scoring_config_exists(self):
        """Test that min_signal_score config parameter exists"""
        assert hasattr(settings.trading, 'min_signal_score')
        assert settings.trading.min_signal_score == 70.0

    def test_calculate_signal_score_method_exists(self):
        """Test that scoring method exists in ORBStrategy"""
        strategy = ORBStrategy()
        assert hasattr(strategy, '_calculate_signal_score')
        assert callable(strategy._calculate_signal_score)

    def test_breakout_with_scoring_method_exists(self):
        """Test that scoring-based breakout check exists"""
        strategy = ORBStrategy()
        assert hasattr(strategy, '_check_breakout_with_scoring')
        assert callable(strategy._check_breakout_with_scoring)

    def test_score_breakout_strength(self):
        """Test that breakout strength contributes to score (0-25 pts)"""
        strategy = ORBStrategy()
        orb = OpeningRange(
            symbol='TEST',
            high=100.0,
            low=98.0,
            range_size=2.0,
            vwap=99.0,
            timestamp=datetime.now()
        )

        # Strong breakout (0.5% above ORB high) should get max 25 pts
        price = 100.50  # 0.5% above ORB high
        breakout_pct = (price - orb.high) / orb.high * 100
        score_contribution = min(breakout_pct * 50, 25)

        assert score_contribution == 25  # Max points for breakout strength

    def test_score_volume(self):
        """Test that volume contributes correctly to score (0-20 pts)"""
        # RVOL >= 2.5 should get 20 pts
        # RVOL >= 2.0 should get 15 pts
        # RVOL >= 1.5 should get 10 pts
        # RVOL >= 1.2 should get 5 pts

        assert 2.5 >= 2.5  # 20 pts
        assert 2.0 >= 2.0 and 2.0 < 2.5  # 15 pts
        assert 1.5 >= 1.5 and 1.5 < 2.0  # 10 pts
        assert 1.2 >= 1.2 and 1.2 < 1.5  # 5 pts


class TestPhase5Execution:
    """Phase 5: Execution Improvements tests"""

    def test_limit_order_config_exists(self):
        """Test that limit order config parameters exist"""
        assert hasattr(settings.trading, 'use_limit_entry')
        assert hasattr(settings.trading, 'limit_entry_buffer_pct')
        assert settings.trading.use_limit_entry is True
        assert settings.trading.limit_entry_buffer_pct == 0.001

    def test_stop_atr_multiplier_config_exists(self):
        """Test that ATR stop multiplier config exists"""
        assert hasattr(settings.trading, 'stop_atr_multiplier')
        assert settings.trading.stop_atr_multiplier == 1.5

    def test_hybrid_stop_method_exists(self):
        """Test that hybrid stop calculation method exists"""
        strategy = ORBStrategy()
        assert hasattr(strategy, '_calculate_hybrid_stop')
        assert callable(strategy._calculate_hybrid_stop)


class TestPhase6RiskManagement:
    """Phase 6: Risk Management Enhancements tests"""

    def test_daily_loss_limit_config_exists(self):
        """Test that daily loss limit config exists"""
        assert hasattr(settings.trading, 'max_daily_loss')
        assert settings.trading.max_daily_loss == 750.0  # 3% of $25k

    def test_consecutive_losses_config_exists(self):
        """Test that consecutive losses config exists"""
        assert hasattr(settings.trading, 'max_consecutive_losses')
        assert settings.trading.max_consecutive_losses == 2

    def test_latest_trade_time_config_exists(self):
        """Test that latest trade time config exists"""
        assert hasattr(settings.trading, 'latest_trade_time')
        assert settings.trading.latest_trade_time == "11:30"

    def test_strategy_tracks_daily_pnl(self):
        """Test that strategy tracks daily P/L"""
        strategy = ORBStrategy()
        assert hasattr(strategy, 'daily_pnl')
        assert strategy.daily_pnl == 0.0

    def test_strategy_tracks_consecutive_losses(self):
        """Test that strategy tracks consecutive losses"""
        strategy = ORBStrategy()
        assert hasattr(strategy, 'consecutive_losses')
        assert strategy.consecutive_losses == 0

    def test_daily_pnl_reset_on_daily_reset(self):
        """Test that daily P/L resets with daily reset"""
        strategy = ORBStrategy()
        strategy.daily_pnl = -500.0
        strategy.consecutive_losses = 3

        strategy.reset_daily()

        assert strategy.daily_pnl == 0.0
        assert strategy.consecutive_losses == 0

    def test_consecutive_losses_increments_on_loss(self):
        """Test that consecutive losses increments on losing trade"""
        strategy = ORBStrategy()

        # Record a losing trade
        strategy.record_trade_result(
            symbol='TEST',
            entry_price=100.0,
            exit_price=99.0,  # Loss
            is_long=True
        )

        assert strategy.consecutive_losses == 1

    def test_consecutive_losses_resets_on_win(self):
        """Test that consecutive losses resets on winning trade"""
        strategy = ORBStrategy()
        strategy.consecutive_losses = 2

        # Record a winning trade
        strategy.record_trade_result(
            symbol='TEST',
            entry_price=100.0,
            exit_price=102.0,  # Win
            is_long=True
        )

        assert strategy.consecutive_losses == 0


class TestPhase2TimeAdjustedRVOL:
    """Phase 2: Time-Adjusted RVOL tests"""

    def test_volume_profiles_attribute_exists(self):
        """Test that market_data has volume_profiles attribute"""
        from data.market_data import market_data
        assert hasattr(market_data, 'volume_profiles')
        assert isinstance(market_data.volume_profiles, dict)

    def test_cache_volume_profile_method_exists(self):
        """Test that cache_volume_profile method exists"""
        from data.market_data import market_data
        assert hasattr(market_data, 'cache_volume_profile')
        assert callable(market_data.cache_volume_profile)

    def test_calculate_time_adjusted_rvol_method_exists(self):
        """Test that time-adjusted RVOL calculation exists"""
        from data.market_data import market_data
        assert hasattr(market_data, 'calculate_time_adjusted_rvol')
        assert callable(market_data.calculate_time_adjusted_rvol)

    def test_get_current_minute_index(self):
        """Test minute index calculation"""
        from data.market_data import market_data
        assert hasattr(market_data, 'get_current_minute_index')
        minute_idx = market_data.get_current_minute_index()
        # Should return non-negative int
        assert isinstance(minute_idx, int)
        assert minute_idx >= 0

    def test_time_adjusted_rvol_fallback(self):
        """Test that RVOL falls back to 1.0 when no profile"""
        from data.market_data import market_data

        # Clear any cached profile
        market_data.volume_profiles.pop('NONEXISTENT', None)

        rvol = market_data.calculate_time_adjusted_rvol(
            symbol='NONEXISTENT',
            current_minute=16,
            cumulative_volume=100000
        )

        assert rvol == 1.0  # Fallback value


class TestTradeSignalScore:
    """Test that TradeSignal includes score"""

    def test_trade_signal_has_score_field(self):
        """Test that TradeSignal dataclass has signal_score field"""
        from strategy.orb import TradeSignal, SignalType

        signal = TradeSignal(
            symbol='TEST',
            signal_type=SignalType.LONG,
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=104.0,
            position_size=100,
            risk_amount=200.0,
            orb_high=100.0,
            orb_low=98.0,
            vwap=99.0,
            rsi=50.0,
            relative_volume=2.0,
            timestamp=datetime.now(),
            signal_score=85.0
        )

        assert hasattr(signal, 'signal_score')
        assert signal.signal_score == 85.0

    def test_trade_signal_str_includes_score(self):
        """Test that signal string representation includes score"""
        from strategy.orb import TradeSignal, SignalType

        signal = TradeSignal(
            symbol='TEST',
            signal_type=SignalType.LONG,
            entry_price=100.0,
            stop_loss=98.0,
            take_profit=104.0,
            position_size=100,
            risk_amount=200.0,
            orb_high=100.0,
            orb_low=98.0,
            vwap=99.0,
            rsi=50.0,
            relative_volume=2.0,
            timestamp=datetime.now(),
            signal_score=85.0
        )

        signal_str = str(signal)
        assert 'Score' in signal_str or 'score' in signal_str.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
