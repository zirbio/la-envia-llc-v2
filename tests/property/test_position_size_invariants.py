"""
Property-based tests for position sizing invariants using Hypothesis.

These tests verify that _calculate_position_size_kelly() maintains invariants:
1. Position size is always >= 0 (and >= 1 when risk > 0)
2. Risk never exceeds max_capital * risk_per_trade
3. Position never exceeds max affordable shares

Run with: pytest tests/property/test_position_size_invariants.py -v
"""
import pytest
from unittest.mock import patch
from hypothesis import given, strategies as st, assume, settings

from strategy.orb import ORBStrategy, TradeResult
from config.settings import TradingConfig, SentimentConfig, SignalLevel


# ============================================================================
# Property 1: Position Size Non-Negative
# ============================================================================

@pytest.mark.property
class TestPositionSizeNonNegative:
    """Property tests for position size non-negativity."""

    @given(
        entry_price=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        stop_loss=st.floats(min_value=0.5, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_position_size_never_negative(self, entry_price, stop_loss):
        """Position size should never be negative."""
        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        position, risk = strategy._calculate_position_size_kelly(
            entry_price=entry_price,
            stop_loss=stop_loss
        )

        assert position >= 0, f"Position {position} is negative"
        assert risk >= 0, f"Risk {risk} is negative"

    @given(
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance=st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_positive_risk_gives_positive_position(self, entry_price, stop_distance):
        """When risk per share > 0, position should be >= 1."""
        assume(stop_distance < entry_price)  # Ensure valid stop

        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        stop_loss = entry_price - stop_distance  # Long position stop

        position, risk = strategy._calculate_position_size_kelly(
            entry_price=entry_price,
            stop_loss=stop_loss
        )

        assert position >= 1, f"Position should be >= 1 but got {position}"


# ============================================================================
# Property 2: Risk Within Limits
# ============================================================================

@pytest.mark.property
class TestRiskWithinLimits:
    """Property tests for risk limits."""

    @given(
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance=st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_risk_never_exceeds_max(self, entry_price, stop_distance):
        """Risk should never exceed max_capital * risk_per_trade."""
        assume(stop_distance < entry_price)

        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        stop_loss = entry_price - stop_distance

        position, risk = strategy._calculate_position_size_kelly(
            entry_price=entry_price,
            stop_loss=stop_loss
        )

        max_risk = strategy.config.max_capital * strategy.config.risk_per_trade
        assert risk <= max_risk * 1.01, f"Risk {risk} exceeds max {max_risk}"

    @given(
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance=st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_position_cost_within_capital(self, entry_price, stop_distance):
        """Total position cost should not exceed max_capital."""
        assume(stop_distance < entry_price)

        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        stop_loss = entry_price - stop_distance

        position, _ = strategy._calculate_position_size_kelly(
            entry_price=entry_price,
            stop_loss=stop_loss
        )

        total_cost = position * entry_price
        assert total_cost <= strategy.config.max_capital, \
            f"Total cost {total_cost} exceeds capital {strategy.config.max_capital}"


# ============================================================================
# Property 3: Kelly Fraction Bounds
# ============================================================================

@pytest.mark.property
class TestKellyFractionBounds:
    """Property tests for Kelly fraction bounds."""

    @given(
        n_trades=st.integers(min_value=5, max_value=50),
        win_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_kelly_fraction_always_bounded(self, n_trades, win_rate):
        """Kelly fraction should always be in [0, 0.25]."""
        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        # Create trade history with given win rate
        n_wins = int(n_trades * win_rate)
        n_losses = n_trades - n_wins

        history = []
        for i in range(n_wins):
            history.append(TradeResult(
                symbol=f"W{i}",
                entry_price=100.0,
                exit_price=103.0,
                pnl=3.0,
                pnl_pct=0.03,
                won=True
            ))
        for i in range(n_losses):
            history.append(TradeResult(
                symbol=f"L{i}",
                entry_price=100.0,
                exit_price=98.0,
                pnl=-2.0,
                pnl_pct=-0.02,
                won=False
            ))

        strategy.trade_history = history
        kelly = strategy._calculate_kelly_fraction()

        assert 0 <= kelly <= 0.25, f"Kelly {kelly} out of bounds [0, 0.25]"


# ============================================================================
# Property 4: Position Scaling with History
# ============================================================================

@pytest.mark.property
class TestPositionScaling:
    """Property tests for position scaling based on history."""

    @given(
        entry_price=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        stop_distance=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_good_history_increases_or_maintains_position(self, entry_price, stop_distance):
        """Better trade history should increase or maintain position size."""
        assume(stop_distance < entry_price)

        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy_bad = ORBStrategy()

        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy_good = ORBStrategy()

        # Bad history (all losses)
        bad_history = [
            TradeResult(symbol=f"L{i}", entry_price=100, exit_price=97, pnl=-3, pnl_pct=-0.03, won=False)
            for i in range(5)
        ]

        # Good history (all wins)
        good_history = [
            TradeResult(symbol=f"W{i}", entry_price=100, exit_price=104, pnl=4, pnl_pct=0.04, won=True)
            for i in range(5)
        ]

        strategy_bad.trade_history = bad_history
        strategy_good.trade_history = good_history

        stop_loss = entry_price - stop_distance

        pos_bad, _ = strategy_bad._calculate_position_size_kelly(entry_price, stop_loss)
        pos_good, _ = strategy_good._calculate_position_size_kelly(entry_price, stop_loss)

        # Good history should give equal or larger position
        assert pos_good >= pos_bad, f"Good history {pos_good} < bad history {pos_bad}"


# ============================================================================
# Property 5: Determinism
# ============================================================================

@pytest.mark.property
class TestPositionSizeDeterminism:
    """Property tests for position sizing determinism."""

    @given(
        entry_price=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        stop_distance=st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_same_inputs_same_position(self, entry_price, stop_distance):
        """Same inputs should always produce same position size."""
        assume(stop_distance < entry_price)

        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        stop_loss = entry_price - stop_distance

        pos1, risk1 = strategy._calculate_position_size_kelly(entry_price, stop_loss)
        pos2, risk2 = strategy._calculate_position_size_kelly(entry_price, stop_loss)
        pos3, risk3 = strategy._calculate_position_size_kelly(entry_price, stop_loss)

        assert pos1 == pos2 == pos3
        assert risk1 == risk2 == risk3


# ============================================================================
# Property 6: Risk Proportional to Stop Distance
# ============================================================================

@pytest.mark.property
class TestRiskProportionality:
    """Property tests for risk proportionality."""

    @given(
        entry_price=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        stop_distance1=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
        stop_distance2=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_wider_stop_smaller_position(self, entry_price, stop_distance1, stop_distance2):
        """Wider stop should result in smaller or equal position size."""
        assume(stop_distance1 < entry_price)
        assume(stop_distance2 < entry_price)
        assume(stop_distance2 > stop_distance1)

        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        stop_loss1 = entry_price - stop_distance1  # Tighter stop
        stop_loss2 = entry_price - stop_distance2  # Wider stop

        pos1, _ = strategy._calculate_position_size_kelly(entry_price, stop_loss1)
        pos2, _ = strategy._calculate_position_size_kelly(entry_price, stop_loss2)

        # Wider stop = larger risk per share = smaller position
        assert pos2 <= pos1, f"Wider stop {pos2} > tighter stop {pos1}"


# ============================================================================
# Property 7: Trade History Size Handling
# ============================================================================

@pytest.mark.property
class TestTradeHistorySize:
    """Property tests for trade history size handling."""

    @given(
        n_trades=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50)
    def test_any_history_size_works(self, n_trades):
        """Position sizing should work with any history size."""
        with patch('strategy.orb.settings') as mock_settings:
            config = TradingConfig()
            config.signal_level = SignalLevel.MODERATE
            mock_settings.trading = config
            mock_settings.sentiment = SentimentConfig(finnhub_api_key="", enabled=False)
            strategy = ORBStrategy()

        # Create history of n_trades
        for i in range(n_trades):
            is_win = i % 2 == 0
            strategy.trade_history.append(TradeResult(
                symbol=f"T{i}",
                entry_price=100.0,
                exit_price=103.0 if is_win else 97.0,
                pnl=3.0 if is_win else -3.0,
                pnl_pct=0.03 if is_win else -0.03,
                won=is_win
            ))

        # Should not crash or produce invalid values
        position, risk = strategy._calculate_position_size_kelly(
            entry_price=100.0,
            stop_loss=98.0
        )

        assert position >= 0
        assert risk >= 0
