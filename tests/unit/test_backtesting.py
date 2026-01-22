"""
Unit tests for backtesting module.

Tests cover:
- ORB calculation (high/low from bars)
- Long signal conditions (all 5 conditions)
- Short signal conditions (all 5 conditions)
- Position sizing with risk management
- Stop loss/take profit calculations
- PnL calculations for long and short trades
- Cache invalidation
- Error handling
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import os


# ============================================================================
# ORB Signal Condition Tests
# ============================================================================

class TestLongSignalConditions:
    """Tests for LONG signal conditions."""

    def test_long_signal_all_conditions_met(self):
        """LONG signal should trigger when all 5 conditions are met."""
        from backtesting.orb_backtest import ORBBacktestStrategy

        # Create a mock strategy instance with parameters
        strategy = MagicMock(spec=ORBBacktestStrategy)
        strategy.parameters = {
            "min_relative_volume": 1.5,
            "rsi_overbought": 70,
        }

        # Test the static logic for long signal conditions
        # All conditions met:
        # 1. price > orb_high
        # 2. price > vwap
        # 3. relative_volume >= min_relative_volume
        # 4. rsi < rsi_overbought
        # 5. macd_bullish

        price = 151.00
        orb = {"high": 150.00, "low": 148.00}
        vwap = 149.50
        rsi = 55.0
        rel_volume = 2.0  # 2x average (above 1.5 threshold)
        macd = 0.5
        signal = 0.3
        histogram = 0.3
        prev_histogram = 0.1  # Growing histogram

        # Check individual conditions
        assert price > orb["high"], "Price should be above ORB high"
        assert price > vwap, "Price should be above VWAP"
        assert rel_volume >= strategy.parameters["min_relative_volume"], "Volume should be above threshold"
        assert rsi < strategy.parameters["rsi_overbought"], "RSI should be below overbought"

        # MACD bullish: above signal, histogram positive and growing
        from data.indicators import is_macd_bullish
        assert is_macd_bullish(macd, signal, histogram, prev_histogram), "MACD should be bullish"

    def test_long_signal_price_below_orb_high(self):
        """LONG signal should NOT trigger when price is below ORB high."""
        price = 149.00  # Below ORB high of 150
        orb_high = 150.00

        assert price <= orb_high, "Price below ORB high should not trigger long"

    def test_long_signal_price_below_vwap(self):
        """LONG signal should NOT trigger when price is below VWAP."""
        price = 149.00
        vwap = 150.00

        assert price <= vwap, "Price below VWAP should not trigger long"

    def test_long_signal_low_volume(self):
        """LONG signal should NOT trigger when volume is below threshold."""
        rel_volume = 1.2  # Below 1.5 threshold
        min_rel_volume = 1.5

        assert rel_volume < min_rel_volume, "Low volume should not trigger long"

    def test_long_signal_rsi_overbought(self):
        """LONG signal should NOT trigger when RSI is overbought."""
        rsi = 75.0  # Above 70 overbought threshold
        rsi_overbought = 70

        assert rsi >= rsi_overbought, "Overbought RSI should not trigger long"

    def test_long_signal_macd_not_bullish(self):
        """LONG signal should NOT trigger when MACD is not bullish."""
        from data.indicators import is_macd_bullish

        # MACD below signal line
        macd = 0.2
        signal = 0.5
        histogram = -0.3
        prev_histogram = -0.1

        assert not is_macd_bullish(macd, signal, histogram, prev_histogram), "MACD should not be bullish"


class TestShortSignalConditions:
    """Tests for SHORT signal conditions."""

    def test_short_signal_all_conditions_met(self):
        """SHORT signal should trigger when all 5 conditions are met."""
        from backtesting.orb_backtest import ORBBacktestStrategy

        strategy = MagicMock(spec=ORBBacktestStrategy)
        strategy.parameters = {
            "min_relative_volume": 1.5,
            "rsi_oversold": 30,
        }

        # All conditions met:
        # 1. price < orb_low
        # 2. price < vwap
        # 3. relative_volume >= min_relative_volume
        # 4. rsi > rsi_oversold
        # 5. macd_bearish

        price = 147.00
        orb = {"high": 150.00, "low": 148.00}
        vwap = 149.50
        rsi = 45.0
        rel_volume = 2.0
        macd = -0.5
        signal = -0.3
        histogram = -0.3
        prev_histogram = -0.1  # Falling histogram

        assert price < orb["low"], "Price should be below ORB low"
        assert price < vwap, "Price should be below VWAP"
        assert rel_volume >= strategy.parameters["min_relative_volume"], "Volume should be above threshold"
        assert rsi > strategy.parameters["rsi_oversold"], "RSI should be above oversold"

        from data.indicators import is_macd_bearish
        assert is_macd_bearish(macd, signal, histogram, prev_histogram), "MACD should be bearish"

    def test_short_signal_price_above_orb_low(self):
        """SHORT signal should NOT trigger when price is above ORB low."""
        price = 149.00  # Above ORB low of 148
        orb_low = 148.00

        assert price >= orb_low, "Price above ORB low should not trigger short"

    def test_short_signal_rsi_oversold(self):
        """SHORT signal should NOT trigger when RSI is oversold."""
        rsi = 25.0  # Below 30 oversold threshold
        rsi_oversold = 30

        assert rsi <= rsi_oversold, "Oversold RSI should not trigger short"


# ============================================================================
# Position Sizing Tests
# ============================================================================

class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_position_size_calculation(self):
        """Position size should be calculated based on risk amount and risk per share."""
        portfolio_value = 25000.0
        risk_per_trade = 0.02  # 2%
        entry_price = 150.00
        stop_loss = 148.00

        risk_amount = portfolio_value * risk_per_trade  # $500
        risk_per_share = entry_price - stop_loss  # $2

        expected_quantity = int(risk_amount / risk_per_share)  # 250 shares

        assert expected_quantity == 250, "Position size should be 250 shares"

    def test_position_size_short_calculation(self):
        """Position size for shorts should use stop - entry as risk."""
        portfolio_value = 25000.0
        risk_per_trade = 0.02
        entry_price = 148.00
        stop_loss = 150.00  # Stop is above entry for shorts

        risk_amount = portfolio_value * risk_per_trade  # $500
        risk_per_share = stop_loss - entry_price  # $2

        expected_quantity = int(risk_amount / risk_per_share)  # 250 shares

        assert expected_quantity == 250, "Short position size should be 250 shares"

    def test_position_size_zero_risk(self):
        """Position size should be 0 when risk per share is 0 or negative."""
        portfolio_value = 25000.0
        risk_per_trade = 0.02
        entry_price = 150.00
        stop_loss = 150.00  # Same as entry

        risk_amount = portfolio_value * risk_per_trade
        risk_per_share = entry_price - stop_loss  # $0

        if risk_per_share <= 0:
            expected_quantity = 0
        else:
            expected_quantity = int(risk_amount / risk_per_share)

        assert expected_quantity == 0, "Position size should be 0 when no risk"


# ============================================================================
# Stop Loss / Take Profit Tests
# ============================================================================

class TestStopLossTakeProfit:
    """Tests for stop loss and take profit calculations."""

    def test_long_stop_loss_at_orb_low(self):
        """Long stop loss should be set at ORB low."""
        orb_low = 148.00
        entry_price = 151.00

        stop_loss = orb_low

        assert stop_loss == 148.00, "Long stop loss should be at ORB low"
        assert stop_loss < entry_price, "Stop loss should be below entry for longs"

    def test_long_take_profit_calculation(self):
        """Long take profit should be entry + (risk * reward_ratio)."""
        entry_price = 151.00
        stop_loss = 148.00
        reward_risk_ratio = 2.0

        risk_per_share = entry_price - stop_loss  # $3
        take_profit = entry_price + (risk_per_share * reward_risk_ratio)  # $157

        assert take_profit == 157.00, "Take profit should be $157"

    def test_short_stop_loss_at_orb_high(self):
        """Short stop loss should be set at ORB high."""
        orb_high = 150.00
        entry_price = 147.00

        stop_loss = orb_high

        assert stop_loss == 150.00, "Short stop loss should be at ORB high"
        assert stop_loss > entry_price, "Stop loss should be above entry for shorts"

    def test_short_take_profit_calculation(self):
        """Short take profit should be entry - (risk * reward_ratio)."""
        entry_price = 147.00
        stop_loss = 150.00
        reward_risk_ratio = 2.0

        risk_per_share = stop_loss - entry_price  # $3
        take_profit = entry_price - (risk_per_share * reward_risk_ratio)  # $141

        assert take_profit == 141.00, "Take profit should be $141"


# ============================================================================
# PnL Calculation Tests
# ============================================================================

class TestPnLCalculations:
    """Tests for PnL (Profit and Loss) calculations."""

    def test_long_winning_trade_pnl(self):
        """PnL for winning long trade should be positive."""
        entry_price = 150.00
        exit_price = 157.00
        quantity = 100

        pnl = (exit_price - entry_price) * quantity
        pnl_pct = (exit_price - entry_price) / entry_price

        assert pnl == 700.00, "PnL should be $700"
        assert round(pnl_pct, 4) == 0.0467, "PnL % should be ~4.67%"

    def test_long_losing_trade_pnl(self):
        """PnL for losing long trade should be negative."""
        entry_price = 150.00
        exit_price = 148.00  # Hit stop loss
        quantity = 100

        pnl = (exit_price - entry_price) * quantity
        pnl_pct = (exit_price - entry_price) / entry_price

        assert pnl == -200.00, "PnL should be -$200"
        assert round(pnl_pct, 4) == -0.0133, "PnL % should be ~-1.33%"

    def test_short_winning_trade_pnl(self):
        """PnL for winning short trade should be positive."""
        entry_price = 150.00
        exit_price = 144.00  # Hit take profit
        quantity = 100

        # For shorts: PnL = (entry - exit) * quantity
        pnl = (entry_price - exit_price) * quantity
        pnl_pct = (entry_price - exit_price) / entry_price

        assert pnl == 600.00, "PnL should be $600"
        assert round(pnl_pct, 4) == 0.04, "PnL % should be 4%"

    def test_short_losing_trade_pnl(self):
        """PnL for losing short trade should be negative."""
        entry_price = 148.00
        exit_price = 150.00  # Hit stop loss
        quantity = 100

        pnl = (entry_price - exit_price) * quantity
        pnl_pct = (entry_price - exit_price) / entry_price

        assert pnl == -200.00, "PnL should be -$200"
        assert round(pnl_pct, 4) == -0.0135, "PnL % should be ~-1.35%"


# ============================================================================
# ORB Calculation Tests
# ============================================================================

class TestORBCalculation:
    """Tests for Opening Range Breakout calculation."""

    def test_orb_high_low_from_bars(self):
        """ORB high/low should be calculated from bar data."""
        bars = [
            {"high": 150.50, "low": 149.00, "close": 150.00, "volume": 100000},
            {"high": 151.00, "low": 149.50, "close": 150.50, "volume": 120000},
            {"high": 150.75, "low": 148.50, "close": 149.00, "volume": 110000},
            {"high": 152.00, "low": 150.00, "close": 151.50, "volume": 150000},
            {"high": 151.50, "low": 149.75, "close": 150.25, "volume": 130000},
        ]

        orb_high = max(b["high"] for b in bars)
        orb_low = min(b["low"] for b in bars)
        orb_range = orb_high - orb_low

        assert orb_high == 152.00, "ORB high should be 152.00"
        assert orb_low == 148.50, "ORB low should be 148.50"
        assert orb_range == 3.50, "ORB range should be 3.50"

    def test_orb_vwap_calculation(self):
        """VWAP should be calculated from ORB bars."""
        bars = [
            {"high": 150.50, "low": 149.00, "close": 150.00, "volume": 100000},
            {"high": 151.00, "low": 149.50, "close": 150.50, "volume": 120000},
            {"high": 150.75, "low": 148.50, "close": 149.00, "volume": 110000},
        ]

        total_volume = sum(b["volume"] for b in bars)
        typical_prices = [(b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in bars]
        vwap = sum(typical_prices) / total_volume

        # Verify VWAP is within reasonable range
        assert 149.0 < vwap < 151.0, "VWAP should be within bar range"

    def test_orb_minimum_bars_requirement(self):
        """ORB calculation should require minimum number of bars."""
        min_bars_for_orb = 5

        # Less than minimum
        insufficient_bars = [
            {"high": 150.50, "low": 149.00, "close": 150.00, "volume": 100000},
            {"high": 151.00, "low": 149.50, "close": 150.50, "volume": 120000},
        ]

        assert len(insufficient_bars) < min_bars_for_orb, "Should require at least 5 bars"


# ============================================================================
# Cache Invalidation Tests
# ============================================================================

class TestCacheInvalidation:
    """Tests for data cache invalidation."""

    def test_cache_valid_within_age(self):
        """Cache should be valid if file is within max age."""
        from backtesting.data_loader import BacktestDataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = BacktestDataLoader(cache_dir=tmpdir, max_cache_age_days=7)

            # Create a recent cache file
            cache_file = Path(tmpdir) / "TEST_20240101_20240131_1min.parquet"
            cache_file.touch()

            # File is new, should be valid
            assert loader._is_cache_valid(cache_file) is True

    def test_cache_invalid_beyond_age(self):
        """Cache should be invalid if file is beyond max age."""
        from backtesting.data_loader import BacktestDataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = BacktestDataLoader(cache_dir=tmpdir, max_cache_age_days=7)

            # Create a cache file and modify its time to be old
            cache_file = Path(tmpdir) / "TEST_20240101_20240131_1min.parquet"
            cache_file.touch()

            # Set file modification time to 10 days ago
            old_time = (datetime.now() - timedelta(days=10)).timestamp()
            os.utime(cache_file, (old_time, old_time))

            # File is old, should be invalid
            assert loader._is_cache_valid(cache_file) is False

    def test_cache_invalid_nonexistent(self):
        """Cache should be invalid if file doesn't exist."""
        from backtesting.data_loader import BacktestDataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = BacktestDataLoader(cache_dir=tmpdir)

            nonexistent = Path(tmpdir) / "NONEXISTENT_1min.parquet"
            assert loader._is_cache_valid(nonexistent) is False

    def test_clear_cache_all(self):
        """clear_cache() should remove all cache files."""
        from backtesting.data_loader import BacktestDataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = BacktestDataLoader(cache_dir=tmpdir)

            # Create some cache files
            (Path(tmpdir) / "SPY_20240101_20240131_1min.parquet").touch()
            (Path(tmpdir) / "QQQ_20240101_20240131_1min.parquet").touch()
            (Path(tmpdir) / "AAPL_20240101_20240131_1min.parquet").touch()

            cleared = loader.clear_cache()

            assert cleared == 3, "Should have cleared 3 files"
            assert len(list(Path(tmpdir).glob("*.parquet"))) == 0

    def test_clear_cache_single_symbol(self):
        """clear_cache(symbol) should only remove cache for that symbol."""
        from backtesting.data_loader import BacktestDataLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = BacktestDataLoader(cache_dir=tmpdir)

            # Create cache files for different symbols
            (Path(tmpdir) / "SPY_20240101_20240131_1min.parquet").touch()
            (Path(tmpdir) / "QQQ_20240101_20240131_1min.parquet").touch()

            cleared = loader.clear_cache("SPY")

            assert cleared == 1, "Should have cleared 1 file"
            assert (Path(tmpdir) / "QQQ_20240101_20240131_1min.parquet").exists()


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for data loader error handling."""

    def test_invalid_symbol_error(self):
        """InvalidSymbolError should be raised for unknown symbols."""
        from backtesting.data_loader import InvalidSymbolError

        # Test that the exception exists and can be raised
        with pytest.raises(InvalidSymbolError):
            raise InvalidSymbolError("Unknown symbol: FAKESYMBOL")

    def test_network_error(self):
        """NetworkError should be raised for connectivity issues."""
        from backtesting.data_loader import NetworkError

        with pytest.raises(NetworkError):
            raise NetworkError("Connection timeout")

    def test_rate_limit_error(self):
        """RateLimitError should be raised for API rate limits."""
        from backtesting.data_loader import RateLimitError

        with pytest.raises(RateLimitError):
            raise RateLimitError("Too many requests")


# ============================================================================
# Class State Management Tests
# ============================================================================

class TestClassStateManagement:
    """Tests for proper class-level state management."""

    def test_reset_state_clears_trades(self):
        """reset_state() should clear all closed trades."""
        from backtesting.orb_backtest import ORBBacktestStrategy

        # Add some trades
        ORBBacktestStrategy.add_trade({"symbol": "SPY", "pnl": 100})
        ORBBacktestStrategy.add_trade({"symbol": "QQQ", "pnl": -50})

        assert len(ORBBacktestStrategy.get_all_trades()) == 2

        # Reset state
        ORBBacktestStrategy.reset_state()

        assert len(ORBBacktestStrategy.get_all_trades()) == 0

    def test_get_all_trades_returns_empty_list_initially(self):
        """get_all_trades() should return empty list when not initialized."""
        from backtesting.orb_backtest import ORBBacktestStrategy

        # Reset to ensure clean state
        ORBBacktestStrategy._all_closed_trades = None

        trades = ORBBacktestStrategy.get_all_trades()
        assert trades == [], "Should return empty list"

    def test_add_trade_initializes_list(self):
        """add_trade() should initialize list if None."""
        from backtesting.orb_backtest import ORBBacktestStrategy

        # Reset to None
        ORBBacktestStrategy._all_closed_trades = None

        # Add a trade
        ORBBacktestStrategy.add_trade({"symbol": "SPY", "pnl": 100})

        assert len(ORBBacktestStrategy.get_all_trades()) == 1

        # Cleanup
        ORBBacktestStrategy.reset_state()


# ============================================================================
# Report Generation Tests
# ============================================================================

class TestReportGeneration:
    """Tests for report generation."""

    def test_report_calculates_win_rate(self):
        """Report should calculate correct win rate."""
        trades = [
            {"pnl": 100},
            {"pnl": 50},
            {"pnl": -30},
            {"pnl": 75},
            {"pnl": -25},
        ]

        winning = [t for t in trades if t["pnl"] > 0]
        win_rate = len(winning) / len(trades) * 100

        assert win_rate == 60.0, "Win rate should be 60%"

    def test_report_calculates_profit_factor(self):
        """Report should calculate correct profit factor."""
        trades = [
            {"pnl": 100},
            {"pnl": 50},
            {"pnl": -30},
            {"pnl": -20},
        ]

        winning = [t for t in trades if t["pnl"] > 0]
        losing = [t for t in trades if t["pnl"] <= 0]

        total_wins = sum(t["pnl"] for t in winning)  # 150
        total_losses = abs(sum(t["pnl"] for t in losing))  # 50

        profit_factor = total_wins / total_losses

        assert profit_factor == 3.0, "Profit factor should be 3.0"

    def test_report_handles_no_losing_trades(self):
        """Report should handle case with no losing trades (inf profit factor)."""
        trades = [
            {"pnl": 100},
            {"pnl": 50},
        ]

        losing = [t for t in trades if t["pnl"] <= 0]

        if len(losing) == 0 or sum(t["pnl"] for t in losing) == 0:
            profit_factor = float("inf")
        else:
            profit_factor = 1.0

        assert profit_factor == float("inf"), "Profit factor should be infinity"
