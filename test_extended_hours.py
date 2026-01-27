"""
Tests for Extended Hours Trading Functionality

Tests cover:
- Configuration settings
- Premarket strategy (Gap & Go)
- Postmarket strategy (Earnings/News)
- Order execution with extended_hours=True
- Session detection and trading windows
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, time
import pytz

from config.settings import (
    TradingMode, ExtendedHoursConfig, settings,
    get_session_params, SessionParams
)
from strategy.extended_hours import (
    PremarketStrategy, PostmarketStrategy,
    ExtendedHoursSignal, ExtendedSignalType,
    get_current_session, is_in_trading_window,
    premarket_strategy, postmarket_strategy
)


EST = pytz.timezone('US/Eastern')


class TestTradingMode:
    """Test TradingMode enum and configuration"""

    def test_trading_mode_values(self):
        """Test TradingMode enum has correct values"""
        assert TradingMode.REGULAR.value == "regular"
        assert TradingMode.PREMARKET.value == "premarket"
        assert TradingMode.POSTMARKET.value == "postmarket"
        assert TradingMode.ALL_SESSIONS.value == "all"

    def test_extended_hours_config_defaults(self):
        """Test ExtendedHoursConfig default values"""
        config = ExtendedHoursConfig()

        # Session enablement
        assert config.premarket_enabled is False
        assert config.postmarket_enabled is False

        # Trading windows
        assert config.premarket_trade_start == "08:00"
        assert config.premarket_trade_end == "09:25"
        assert config.postmarket_trade_start == "16:05"
        assert config.postmarket_trade_end == "18:00"
        assert config.postmarket_force_close == "19:30"

        # Position sizing
        assert config.premarket_position_size_mult == 0.50
        assert config.postmarket_position_size_mult == 0.25

        # Stop multipliers
        assert config.premarket_stop_atr_mult == 2.5
        assert config.postmarket_stop_atr_mult == 3.0

        # Volume requirements
        assert config.premarket_min_rvol == 2.5
        assert config.postmarket_min_rvol == 2.5

        # Trade limits
        assert config.premarket_max_trades == 2
        assert config.postmarket_max_trades == 2

        # Spread limits
        assert config.premarket_max_spread_pct == 0.005
        assert config.postmarket_max_spread_pct == 0.004


class TestSessionParams:
    """Test session-specific parameters"""

    def test_get_session_params_regular(self):
        """Test regular hours parameters"""
        config = ExtendedHoursConfig()
        params = get_session_params(TradingMode.REGULAR, config)

        assert params.position_size_mult == 1.0
        assert params.stop_atr_mult == 1.5
        assert params.min_rvol == 1.2
        assert params.max_trades == 3
        assert params.max_spread_pct is None
        assert params.trade_start == "09:30"
        assert params.trade_end == "16:00"

    def test_get_session_params_premarket(self):
        """Test premarket parameters"""
        config = ExtendedHoursConfig()
        params = get_session_params(TradingMode.PREMARKET, config)

        assert params.position_size_mult == 0.50
        assert params.stop_atr_mult == 2.5
        assert params.min_rvol == 2.5
        assert params.max_trades == 2
        assert params.max_spread_pct == 0.005
        assert params.trade_start == "08:00"
        assert params.trade_end == "09:25"

    def test_get_session_params_postmarket(self):
        """Test postmarket parameters"""
        config = ExtendedHoursConfig()
        params = get_session_params(TradingMode.POSTMARKET, config)

        assert params.position_size_mult == 0.25
        assert params.stop_atr_mult == 3.0
        assert params.min_rvol == 2.5
        assert params.max_trades == 2
        assert params.max_spread_pct == 0.004
        assert params.trade_start == "16:05"
        assert params.trade_end == "18:00"


class TestExtendedHoursSignal:
    """Test ExtendedHoursSignal dataclass"""

    def test_signal_creation(self):
        """Test creating an extended hours signal"""
        signal = ExtendedHoursSignal(
            symbol="AAPL",
            signal_type=ExtendedSignalType.GAP_LONG,
            entry_price=150.00,
            limit_price=150.30,
            stop_loss=147.00,
            take_profit=156.00,
            position_size=50,
            risk_amount=150.00,
            session='premarket',
            gap_pct=5.5,
            spread_pct=0.002,
            signal_score=75.0
        )

        assert signal.symbol == "AAPL"
        assert signal.signal_type == ExtendedSignalType.GAP_LONG
        assert signal.entry_price == 150.00
        assert signal.limit_price == 150.30
        assert signal.session == 'premarket'
        assert signal.gap_pct == 5.5

    def test_signal_string_representation(self):
        """Test signal string output"""
        signal = ExtendedHoursSignal(
            symbol="TSLA",
            signal_type=ExtendedSignalType.NEWS_SHORT,
            entry_price=200.00,
            limit_price=199.40,
            stop_loss=206.00,
            take_profit=188.00,
            position_size=25,
            risk_amount=150.00,
            session='postmarket',
            gap_pct=-7.5,
            spread_pct=0.003,
            signal_score=65.0
        )

        output = str(signal)
        assert "TSLA" in output
        assert "NEWS_SHORT" in output
        assert "postmarket" in output
        assert "200.00" in output


class TestPremarketStrategy:
    """Test PremarketStrategy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = PremarketStrategy()
        self.strategy.reset_daily()

    def test_initial_state(self):
        """Test initial strategy state"""
        assert self.strategy.trades_today == 0
        assert len(self.strategy.signals_today) == 0
        assert len(self.strategy.premarket_ranges) == 0

    def test_gap_score_calculation(self):
        """Test gap signal scoring"""
        # Strong gap with high volume should score high
        score = self.strategy._calculate_gap_score(
            gap_pct=8.0,
            rel_volume=4.0,
            spread_pct=0.001,
            direction='LONG'
        )
        assert score >= 60  # Should be a decent score

        # Weak gap with low volume should score lower
        weak_score = self.strategy._calculate_gap_score(
            gap_pct=3.0,
            rel_volume=2.0,
            spread_pct=0.004,
            direction='LONG'
        )
        assert weak_score < score

    def test_position_size_calculation(self):
        """Test position sizing for premarket"""
        # With 50% size multiplier
        size, risk = self.strategy._calculate_position_size(
            entry_price=100.00,
            stop_loss=97.00,
            size_multiplier=0.50
        )

        # Should be smaller than regular hours would be
        assert size > 0
        assert risk > 0
        # Risk should be controlled
        assert risk <= settings.trading.max_capital * settings.trading.risk_per_trade * 0.50

    def test_reset_daily(self):
        """Test daily reset"""
        self.strategy.trades_today = 5
        self.strategy.premarket_ranges['AAPL'] = {'high': 150}

        self.strategy.reset_daily()

        assert self.strategy.trades_today == 0
        assert len(self.strategy.premarket_ranges) == 0
        assert len(self.strategy.signals_today) == 0


class TestPostmarketStrategy:
    """Test PostmarketStrategy"""

    def setup_method(self):
        """Setup test fixtures"""
        self.strategy = PostmarketStrategy()
        self.strategy.reset_daily()

    def test_initial_state(self):
        """Test initial strategy state"""
        assert self.strategy.trades_today == 0
        assert len(self.strategy.signals_today) == 0

    def test_news_score_calculation(self):
        """Test news reaction signal scoring"""
        # Strong move with earnings catalyst
        score = self.strategy._calculate_news_score(
            move_pct=12.0,
            spread_pct=0.001,
            catalyst_type='earnings'
        )
        assert score >= 60

        # Weaker move with generic news
        weak_score = self.strategy._calculate_news_score(
            move_pct=5.0,
            spread_pct=0.003,
            catalyst_type='news'
        )
        assert weak_score < score

    def test_position_size_calculation(self):
        """Test position sizing for postmarket"""
        # With 25% size multiplier
        size, risk = self.strategy._calculate_position_size(
            entry_price=100.00,
            stop_loss=94.00,
            size_multiplier=0.25
        )

        assert size > 0
        assert risk > 0
        # Risk should be very controlled (25% of regular)
        assert risk <= settings.trading.max_capital * settings.trading.risk_per_trade * 0.25

    def test_reset_daily(self):
        """Test daily reset"""
        self.strategy.trades_today = 3

        self.strategy.reset_daily()

        assert self.strategy.trades_today == 0
        assert len(self.strategy.signals_today) == 0


class TestSessionDetection:
    """Test session detection functions"""

    @patch('strategy.extended_hours.datetime')
    def test_get_current_session_premarket(self, mock_datetime):
        """Test premarket session detection"""
        mock_now = datetime(2024, 1, 15, 7, 30, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        session = get_current_session()
        assert session == TradingMode.PREMARKET

    @patch('strategy.extended_hours.datetime')
    def test_get_current_session_regular(self, mock_datetime):
        """Test regular session detection"""
        mock_now = datetime(2024, 1, 15, 11, 30, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        session = get_current_session()
        assert session == TradingMode.REGULAR

    @patch('strategy.extended_hours.datetime')
    def test_get_current_session_postmarket(self, mock_datetime):
        """Test postmarket session detection"""
        mock_now = datetime(2024, 1, 15, 17, 30, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        session = get_current_session()
        assert session == TradingMode.POSTMARKET

    @patch('strategy.extended_hours.datetime')
    def test_get_current_session_closed(self, mock_datetime):
        """Test closed market detection"""
        mock_now = datetime(2024, 1, 15, 22, 0, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        session = get_current_session()
        assert session is None


class TestTradingWindow:
    """Test trading window validation"""

    @patch('strategy.extended_hours.datetime')
    def test_is_in_premarket_window(self, mock_datetime):
        """Test premarket trading window"""
        # 8:30 AM should be in window
        mock_now = datetime(2024, 1, 15, 8, 30, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        assert is_in_trading_window(TradingMode.PREMARKET) is True

        # 7:30 AM should be out of window
        mock_now = datetime(2024, 1, 15, 7, 30, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        assert is_in_trading_window(TradingMode.PREMARKET) is False

    @patch('strategy.extended_hours.datetime')
    def test_is_in_postmarket_window(self, mock_datetime):
        """Test postmarket trading window"""
        # 5:00 PM should be in window
        mock_now = datetime(2024, 1, 15, 17, 0, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        assert is_in_trading_window(TradingMode.POSTMARKET) is True

        # 7:00 PM should be out of window
        mock_now = datetime(2024, 1, 15, 19, 0, 0, tzinfo=EST)
        mock_datetime.now.return_value = mock_now

        assert is_in_trading_window(TradingMode.POSTMARKET) is False


class TestExtendedSignalTypes:
    """Test extended hours signal types"""

    def test_signal_type_values(self):
        """Test signal type enum values"""
        assert ExtendedSignalType.GAP_LONG.value == "GAP_LONG"
        assert ExtendedSignalType.GAP_SHORT.value == "GAP_SHORT"
        assert ExtendedSignalType.FADE_LONG.value == "FADE_LONG"
        assert ExtendedSignalType.FADE_SHORT.value == "FADE_SHORT"
        assert ExtendedSignalType.NEWS_LONG.value == "NEWS_LONG"
        assert ExtendedSignalType.NEWS_SHORT.value == "NEWS_SHORT"
        assert ExtendedSignalType.NONE.value == "NONE"

    def test_long_signals_contain_long(self):
        """Test that LONG signals contain 'LONG' in value"""
        long_types = [
            ExtendedSignalType.GAP_LONG,
            ExtendedSignalType.FADE_LONG,
            ExtendedSignalType.NEWS_LONG
        ]
        for sig_type in long_types:
            assert 'LONG' in sig_type.value

    def test_short_signals_contain_short(self):
        """Test that SHORT signals contain 'SHORT' in value"""
        short_types = [
            ExtendedSignalType.GAP_SHORT,
            ExtendedSignalType.FADE_SHORT,
            ExtendedSignalType.NEWS_SHORT
        ]
        for sig_type in short_types:
            assert 'SHORT' in sig_type.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
