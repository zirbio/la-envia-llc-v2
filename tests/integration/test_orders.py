"""
Integration tests for execution/orders.py - Order Execution Module.

Tests cover:
- Signal execution (4 tests)
- Position management (6 tests)
- Market status (3 tests)
- Order management (3 tests)
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from execution.orders import OrderExecutor, OrderResult, Position
from strategy.orb import TradeSignal, SignalType


# ============================================================================
# Signal Execution Tests (4 tests)
# ============================================================================

class TestSignalExecution:
    """Tests for trade signal execution."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock TradingClient."""
        mock = MagicMock()

        # Mock order response
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.status.value = "accepted"
        mock.submit_order.return_value = mock_order

        return mock

    @pytest.fixture
    def executor(self, mock_trading_client):
        """Create OrderExecutor with mocked client."""
        with patch('execution.orders.TradingClient', return_value=mock_trading_client):
            with patch('execution.orders.settings') as mock_settings:
                mock_settings.alpaca.api_key = "test_key"
                mock_settings.alpaca.secret_key = "test_secret"
                mock_settings.alpaca.paper = True
                executor = OrderExecutor()
        executor.client = mock_trading_client
        return executor

    def test_execute_signal_long_success(self, executor, long_trade_signal):
        """LONG signal should be executed as BUY bracket order."""
        result = executor.execute_signal(long_trade_signal)

        assert result.success is True
        assert result.order_id == "order-123"
        assert result.symbol == long_trade_signal.symbol
        assert result.side == "buy"
        assert result.qty == long_trade_signal.position_size

        # Verify submit_order was called
        executor.client.submit_order.assert_called_once()

    def test_execute_signal_short_success(self, executor, short_trade_signal):
        """SHORT signal should be executed as SELL bracket order."""
        result = executor.execute_signal(short_trade_signal)

        assert result.success is True
        assert result.order_id == "order-123"
        assert result.symbol == short_trade_signal.symbol
        assert result.side == "sell"
        assert result.qty == short_trade_signal.position_size

    def test_execute_signal_bracket_order_format(self, executor, long_trade_signal):
        """Bracket order should include stop loss and take profit."""
        executor.execute_signal(long_trade_signal)

        # Get the order request that was passed to submit_order
        call_args = executor.client.submit_order.call_args
        order_request = call_args[0][0]

        # Verify bracket order components
        assert order_request.order_class == "bracket"
        assert order_request.stop_loss is not None
        assert order_request.take_profit is not None

    def test_execute_signal_api_error(self, executor, long_trade_signal):
        """API error should return failure result."""
        executor.client.submit_order.side_effect = Exception("API Error")

        result = executor.execute_signal(long_trade_signal)

        assert result.success is False
        assert result.error == "API Error"


# ============================================================================
# Position Management Tests (6 tests)
# ============================================================================

class TestPositionManagement:
    """Tests for position retrieval and management."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock TradingClient with positions."""
        mock = MagicMock()

        # Mock position
        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.side.value = "long"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_position.current_price = "155.00"
        mock_position.unrealized_pl = "500.00"
        mock_position.unrealized_plpc = "0.033"

        mock.get_all_positions.return_value = [mock_position]
        mock.get_open_position.return_value = mock_position

        # Mock close position response
        mock_close = MagicMock()
        mock_close.id = "close-order-123"
        mock.close_position.return_value = mock_close

        return mock

    @pytest.fixture
    def executor(self, mock_trading_client):
        """Create OrderExecutor with mocked client."""
        with patch('execution.orders.TradingClient', return_value=mock_trading_client):
            with patch('execution.orders.settings') as mock_settings:
                mock_settings.alpaca.api_key = "test_key"
                mock_settings.alpaca.secret_key = "test_secret"
                mock_settings.alpaca.paper = True
                executor = OrderExecutor()
        executor.client = mock_trading_client
        return executor

    def test_get_positions_returns_list(self, executor):
        """get_positions should return list of Position objects."""
        positions = executor.get_positions()

        assert len(positions) == 1
        assert isinstance(positions[0], Position)
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 100
        assert positions[0].entry_price == 150.00
        assert positions[0].unrealized_pl == 500.00

    def test_get_positions_empty(self, executor):
        """get_positions with no positions should return empty list."""
        executor.client.get_all_positions.return_value = []

        positions = executor.get_positions()

        assert positions == []

    def test_get_position_single_symbol(self, executor):
        """get_position for specific symbol should return Position."""
        position = executor.get_position("AAPL")

        assert position is not None
        assert position.symbol == "AAPL"
        executor.client.get_open_position.assert_called_with("AAPL")

    def test_get_position_not_found(self, executor):
        """get_position for non-existent symbol should return None."""
        executor.client.get_open_position.side_effect = Exception("Position not found")

        position = executor.get_position("UNKNOWN")

        assert position is None

    def test_close_position_success(self, executor):
        """close_position should close position and return result."""
        result = executor.close_position("AAPL")

        assert result.success is True
        assert result.symbol == "AAPL"
        assert result.status == "closed"
        executor.client.close_position.assert_called_with("AAPL")

    def test_close_all_positions_success(self, executor):
        """close_all_positions should close all and return results."""
        results = executor.close_all_positions()

        assert len(results) == 1
        assert results[0].success is True
        executor.client.close_all_positions.assert_called_with(cancel_orders=True)


# ============================================================================
# Market Status Tests (3 tests)
# ============================================================================

class TestMarketStatus:
    """Tests for market status checks."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock TradingClient with clock."""
        mock = MagicMock()

        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_clock.next_open = datetime.now() + timedelta(hours=12)
        mock_clock.next_close = datetime.now() + timedelta(hours=6)
        mock.get_clock.return_value = mock_clock

        return mock

    @pytest.fixture
    def executor(self, mock_trading_client):
        """Create OrderExecutor with mocked client."""
        with patch('execution.orders.TradingClient', return_value=mock_trading_client):
            with patch('execution.orders.settings') as mock_settings:
                mock_settings.alpaca.api_key = "test_key"
                mock_settings.alpaca.secret_key = "test_secret"
                mock_settings.alpaca.paper = True
                executor = OrderExecutor()
        executor.client = mock_trading_client
        return executor

    def test_is_market_open_true(self, executor):
        """is_market_open should return True when market is open."""
        result = executor.is_market_open()

        assert result is True

    def test_is_market_open_false(self, executor):
        """is_market_open should return False when market is closed."""
        executor.client.get_clock.return_value.is_open = False

        result = executor.is_market_open()

        assert result is False

    def test_get_next_market_times(self, executor):
        """get_next_market_times should return dict with times."""
        result = executor.get_next_market_times()

        assert 'is_open' in result
        assert 'next_open' in result
        assert 'next_close' in result
        assert result['is_open'] is True


# ============================================================================
# Order Management Tests (3 tests)
# ============================================================================

class TestOrderManagement:
    """Tests for order management operations."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock TradingClient."""
        mock = MagicMock()

        # Mock open orders
        mock_order = MagicMock()
        mock_order.id = "order-456"
        mock_order.symbol = "AAPL"
        mock_order.side.value = "buy"
        mock_order.qty = "100"
        mock_order.type.value = "market"
        mock_order.status.value = "new"
        mock_order.created_at = datetime.now()
        mock.get_orders.return_value = [mock_order]

        return mock

    @pytest.fixture
    def executor(self, mock_trading_client):
        """Create OrderExecutor with mocked client."""
        with patch('execution.orders.TradingClient', return_value=mock_trading_client):
            with patch('execution.orders.settings') as mock_settings:
                mock_settings.alpaca.api_key = "test_key"
                mock_settings.alpaca.secret_key = "test_secret"
                mock_settings.alpaca.paper = True
                executor = OrderExecutor()
        executor.client = mock_trading_client
        return executor

    def test_get_open_orders(self, executor):
        """get_open_orders should return list of order dicts."""
        orders = executor.get_open_orders()

        assert len(orders) == 1
        assert orders[0]['id'] == "order-456"
        assert orders[0]['symbol'] == "AAPL"
        assert orders[0]['side'] == "buy"

    def test_cancel_order_success(self, executor):
        """cancel_order should cancel specific order."""
        result = executor.cancel_order("order-456")

        assert result is True
        executor.client.cancel_order_by_id.assert_called_with("order-456")

    def test_cancel_all_orders_success(self, executor):
        """cancel_all_orders should cancel all open orders."""
        result = executor.cancel_all_orders()

        assert result is True
        executor.client.cancel_orders.assert_called_once()


# ============================================================================
# Account Tests
# ============================================================================

class TestAccountOperations:
    """Tests for account operations."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock TradingClient with account."""
        mock = MagicMock()

        mock_account = MagicMock()
        mock_account.equity = "25000.00"
        mock_account.cash = "10000.00"
        mock_account.buying_power = "50000.00"
        mock_account.daytrade_count = 2
        mock_account.pattern_day_trader = False
        mock.get_account.return_value = mock_account

        return mock

    @pytest.fixture
    def executor(self, mock_trading_client):
        """Create OrderExecutor with mocked client."""
        with patch('execution.orders.TradingClient', return_value=mock_trading_client):
            with patch('execution.orders.settings') as mock_settings:
                mock_settings.alpaca.api_key = "test_key"
                mock_settings.alpaca.secret_key = "test_secret"
                mock_settings.alpaca.paper = True
                executor = OrderExecutor()
        executor.client = mock_trading_client
        return executor

    def test_get_account(self, executor):
        """get_account should return account information dict."""
        account = executor.get_account()

        assert 'equity' in account
        assert 'cash' in account
        assert 'buying_power' in account
        assert 'day_trade_count' in account
        assert account['equity'] == 25000.00
        assert account['cash'] == 10000.00
        assert account['day_trade_count'] == 2

    def test_get_account_error(self, executor):
        """get_account should return empty dict on error."""
        executor.client.get_account.side_effect = Exception("API Error")

        account = executor.get_account()

        assert account == {}


# ============================================================================
# Format Methods Tests
# ============================================================================

class TestFormatMethods:
    """Tests for message formatting methods."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock TradingClient with positions."""
        mock = MagicMock()

        mock_position = MagicMock()
        mock_position.symbol = "AAPL"
        mock_position.side.value = "long"
        mock_position.qty = "100"
        mock_position.avg_entry_price = "150.00"
        mock_position.current_price = "155.00"
        mock_position.unrealized_pl = "500.00"
        mock_position.unrealized_plpc = "0.033"

        mock.get_all_positions.return_value = [mock_position]

        return mock

    @pytest.fixture
    def executor(self, mock_trading_client):
        """Create OrderExecutor with mocked client."""
        with patch('execution.orders.TradingClient', return_value=mock_trading_client):
            with patch('execution.orders.settings') as mock_settings:
                mock_settings.alpaca.api_key = "test_key"
                mock_settings.alpaca.secret_key = "test_secret"
                mock_settings.alpaca.paper = True
                executor = OrderExecutor()
        executor.client = mock_trading_client
        return executor

    def test_format_positions_message_with_positions(self, executor):
        """format_positions_message should format positions nicely."""
        message = executor.format_positions_message()

        assert "POSITIONS" in message
        assert "AAPL" in message
        assert "100" in message
        assert "$150.00" in message or "150.0" in message

    def test_format_positions_message_no_positions(self, executor):
        """format_positions_message with no positions."""
        executor.client.get_all_positions.return_value = []

        message = executor.format_positions_message()

        assert "No open positions" in message


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in order operations."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create mock TradingClient that raises errors."""
        mock = MagicMock()
        mock.submit_order.side_effect = Exception("Insufficient buying power")
        mock.close_position.side_effect = Exception("Position not found")
        mock.cancel_order_by_id.side_effect = Exception("Order not found")
        return mock

    @pytest.fixture
    def executor(self, mock_trading_client):
        """Create OrderExecutor with mocked client."""
        with patch('execution.orders.TradingClient', return_value=mock_trading_client):
            with patch('execution.orders.settings') as mock_settings:
                mock_settings.alpaca.api_key = "test_key"
                mock_settings.alpaca.secret_key = "test_secret"
                mock_settings.alpaca.paper = True
                executor = OrderExecutor()
        executor.client = mock_trading_client
        return executor

    def test_execute_signal_handles_error(self, executor, long_trade_signal):
        """execute_signal should handle and return error."""
        result = executor.execute_signal(long_trade_signal)

        assert result.success is False
        assert "Insufficient buying power" in result.error

    def test_close_position_handles_error(self, executor):
        """close_position should handle and return error."""
        result = executor.close_position("AAPL")

        assert result.success is False
        assert "Position not found" in result.error

    def test_cancel_order_handles_error(self, executor):
        """cancel_order should handle error and return False."""
        result = executor.cancel_order("order-123")

        assert result is False
