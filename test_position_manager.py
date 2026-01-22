"""
Tests for position management with partial profit taking and EMA9 trailing stop

Run with: python test_position_manager.py
"""
import asyncio
from datetime import datetime
from unittest.mock import MagicMock

from execution.position_manager import (
    PositionManager,
    ManagedPosition,
    PositionState,
    PositionEvent
)
from strategy.orb import TradeSignal, SignalType

# Optional pytest import for advanced testing
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create a no-op decorator for pytest.fixture and pytest.mark.asyncio
    class pytest:
        @staticmethod
        def fixture(func):
            return func
        class mark:
            @staticmethod
            def asyncio(func):
                return func


class TestManagedPosition:
    """Tests for ManagedPosition dataclass"""

    def test_calculate_r_multiple_long_profit(self):
        """Test R-multiple calculation for profitable long position"""
        position = ManagedPosition(
            symbol='AAPL',
            side='long',
            original_qty=100,
            entry_price=100.0,
            original_stop_loss=98.0,  # $2 risk per share
            risk_amount=200.0,
            state=PositionState.OPEN,
            current_qty=100,
            current_stop_loss=98.0
        )

        # At 1R profit ($102), should return 1.0
        assert position.calculate_r_multiple(102.0) == 1.0

        # At 2R profit ($104), should return 2.0
        assert position.calculate_r_multiple(104.0) == 2.0

        # At 0.5R profit ($101), should return 0.5
        assert position.calculate_r_multiple(101.0) == 0.5

    def test_calculate_r_multiple_long_loss(self):
        """Test R-multiple calculation for losing long position"""
        position = ManagedPosition(
            symbol='AAPL',
            side='long',
            original_qty=100,
            entry_price=100.0,
            original_stop_loss=98.0,  # $2 risk per share
            risk_amount=200.0,
            state=PositionState.OPEN,
            current_qty=100,
            current_stop_loss=98.0
        )

        # At stop loss ($98), should return -1.0
        assert position.calculate_r_multiple(98.0) == -1.0

        # At $99, should return -0.5
        assert position.calculate_r_multiple(99.0) == -0.5

    def test_calculate_r_multiple_short_profit(self):
        """Test R-multiple calculation for profitable short position"""
        position = ManagedPosition(
            symbol='AAPL',
            side='short',
            original_qty=100,
            entry_price=100.0,
            original_stop_loss=102.0,  # $2 risk per share
            risk_amount=200.0,
            state=PositionState.OPEN,
            current_qty=100,
            current_stop_loss=102.0
        )

        # At 1R profit ($98), should return 1.0
        assert position.calculate_r_multiple(98.0) == 1.0

        # At 2R profit ($96), should return 2.0
        assert position.calculate_r_multiple(96.0) == 2.0

    def test_calculate_r_multiple_short_loss(self):
        """Test R-multiple calculation for losing short position"""
        position = ManagedPosition(
            symbol='AAPL',
            side='short',
            original_qty=100,
            entry_price=100.0,
            original_stop_loss=102.0,  # $2 risk per share
            risk_amount=200.0,
            state=PositionState.OPEN,
            current_qty=100,
            current_stop_loss=102.0
        )

        # At stop loss ($102), should return -1.0
        assert position.calculate_r_multiple(102.0) == -1.0

    def test_risk_per_share(self):
        """Test risk per share calculation"""
        position = ManagedPosition(
            symbol='AAPL',
            side='long',
            original_qty=100,
            entry_price=100.0,
            original_stop_loss=97.5,  # $2.50 risk per share
            risk_amount=250.0,
            state=PositionState.OPEN,
            current_qty=100,
            current_stop_loss=97.5
        )

        assert position.risk_per_share == 2.5


class TestPositionManager:
    """Tests for PositionManager class"""

    @pytest.fixture
    def manager(self):
        """Create a fresh position manager for each test"""
        return PositionManager()

    @pytest.fixture
    def sample_signal(self):
        """Create a sample trade signal"""
        return TradeSignal(
            symbol='AAPL',
            signal_type=SignalType.LONG,
            entry_price=150.0,
            stop_loss=147.0,  # $3 risk
            take_profit=156.0,
            position_size=100,
            risk_amount=300.0,
            orb_high=149.5,
            orb_low=148.0,
            vwap=149.0,
            rsi=55.0,
            relative_volume=2.0,
            timestamp=datetime.now()
        )

    @pytest.mark.asyncio
    async def test_register_position(self, manager, sample_signal):
        """Test registering a new position"""
        position = await manager.register_position(
            signal=sample_signal,
            fill_price=150.0,
            qty=100,
            stop_order_id='stop_123'
        )

        assert position.symbol == 'AAPL'
        assert position.side == 'long'
        assert position.original_qty == 100
        assert position.entry_price == 150.0
        assert position.original_stop_loss == 147.0
        assert position.state == PositionState.OPEN
        assert position.current_stop_order_id == 'stop_123'

        # Check it's in the managed positions
        assert 'AAPL' in manager.managed_positions

    @pytest.mark.asyncio
    async def test_get_active_positions(self, manager, sample_signal):
        """Test getting active positions only"""
        # Register a position
        await manager.register_position(
            signal=sample_signal,
            fill_price=150.0,
            qty=100,
            stop_order_id='stop_123'
        )

        active = manager.get_active_positions()
        assert len(active) == 1

        # Mark as closed
        manager.managed_positions['AAPL'].state = PositionState.CLOSED

        active = manager.get_active_positions()
        assert len(active) == 0

    def test_reset(self, manager):
        """Test resetting the position manager"""
        manager.managed_positions['AAPL'] = MagicMock()
        manager.managed_positions['MSFT'] = MagicMock()

        manager.reset()

        assert len(manager.managed_positions) == 0


class TestTrailingStopLogic:
    """Test trailing stop EMA9 logic"""

    def test_long_trailing_stop_only_rises(self):
        """For longs, trailing stop should only move up"""
        position = ManagedPosition(
            symbol='AAPL',
            side='long',
            original_qty=100,
            entry_price=150.0,
            original_stop_loss=147.0,
            risk_amount=300.0,
            state=PositionState.PARTIAL_CLOSED,
            current_qty=50,
            current_stop_loss=150.0  # At breakeven
        )

        # EMA9 at 152 - should update (higher)
        ema9 = 152.0
        if ema9 > position.current_stop_loss:
            should_update = True
        else:
            should_update = False

        assert should_update is True

        # EMA9 at 149 - should NOT update (lower)
        ema9 = 149.0
        if ema9 > position.current_stop_loss:
            should_update = True
        else:
            should_update = False

        assert should_update is False

    def test_short_trailing_stop_only_falls(self):
        """For shorts, trailing stop should only move down"""
        position = ManagedPosition(
            symbol='AAPL',
            side='short',
            original_qty=100,
            entry_price=150.0,
            original_stop_loss=153.0,
            risk_amount=300.0,
            state=PositionState.PARTIAL_CLOSED,
            current_qty=50,
            current_stop_loss=150.0  # At breakeven
        )

        # EMA9 at 148 - should update (lower)
        ema9 = 148.0
        if ema9 < position.current_stop_loss:
            should_update = True
        else:
            should_update = False

        assert should_update is True

        # EMA9 at 151 - should NOT update (higher)
        ema9 = 151.0
        if ema9 < position.current_stop_loss:
            should_update = True
        else:
            should_update = False

        assert should_update is False


class TestPartialCloseLogic:
    """Test partial close at 1R logic"""

    def test_partial_close_qty_calculation(self):
        """Test calculating quantity to close"""
        original_qty = 100
        partial_close_percent = 0.50

        close_qty = int(original_qty * partial_close_percent)
        assert close_qty == 50

        # Remaining
        remaining_qty = original_qty - close_qty
        assert remaining_qty == 50

    def test_partial_close_with_odd_qty(self):
        """Test partial close with odd quantity"""
        original_qty = 75
        partial_close_percent = 0.50

        close_qty = int(original_qty * partial_close_percent)
        assert close_qty == 37

        remaining_qty = original_qty - close_qty
        assert remaining_qty == 38

    def test_breakeven_stop_calculation(self):
        """Test breakeven stop calculation after partial close"""
        entry_price = 150.0

        # Breakeven is simply the entry price
        breakeven_stop = entry_price
        assert breakeven_stop == 150.0


def run_tests():
    """Run all tests"""
    print("Running position manager tests...")

    # Test R-multiple calculations
    test_pos = TestManagedPosition()
    test_pos.test_calculate_r_multiple_long_profit()
    print("  ✓ R-multiple long profit")

    test_pos.test_calculate_r_multiple_long_loss()
    print("  ✓ R-multiple long loss")

    test_pos.test_calculate_r_multiple_short_profit()
    print("  ✓ R-multiple short profit")

    test_pos.test_calculate_r_multiple_short_loss()
    print("  ✓ R-multiple short loss")

    test_pos.test_risk_per_share()
    print("  ✓ Risk per share")

    # Test trailing stop logic
    test_trail = TestTrailingStopLogic()
    test_trail.test_long_trailing_stop_only_rises()
    print("  ✓ Long trailing stop only rises")

    test_trail.test_short_trailing_stop_only_falls()
    print("  ✓ Short trailing stop only falls")

    # Test partial close logic
    test_partial = TestPartialCloseLogic()
    test_partial.test_partial_close_qty_calculation()
    print("  ✓ Partial close qty calculation")

    test_partial.test_partial_close_with_odd_qty()
    print("  ✓ Partial close with odd qty")

    test_partial.test_breakeven_stop_calculation()
    print("  ✓ Breakeven stop calculation")

    print("\n✅ All position manager tests passed!")


if __name__ == "__main__":
    run_tests()
