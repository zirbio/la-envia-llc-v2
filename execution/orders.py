"""
Order execution and position management via Alpaca API
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest,
    StopOrderRequest,
    ReplaceOrderRequest
)
from alpaca.trading.enums import (
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    QueryOrderStatus
)
from loguru import logger

from config.settings import settings
from strategy.orb import TradeSignal, SignalType


@dataclass
class Position:
    """Active position data"""
    symbol: str
    side: str
    qty: int
    entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_pl_pct: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class OrderResult:
    """Result of order execution"""
    success: bool
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    side: Optional[str] = None
    qty: Optional[int] = None
    filled_price: Optional[float] = None
    status: Optional[str] = None
    error: Optional[str] = None


class OrderExecutor:
    """Execute and manage orders via Alpaca"""

    def __init__(self):
        self.client = TradingClient(
            api_key=settings.alpaca.api_key,
            secret_key=settings.alpaca.secret_key,
            paper=settings.alpaca.paper
        )
        self.active_orders: dict[str, str] = {}  # symbol -> order_id

    def get_account(self) -> dict:
        """Get account information"""
        try:
            account = self.client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'day_trade_count': account.daytrade_count,
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {}

    def execute_signal(self, signal: TradeSignal, use_limit: bool = None) -> OrderResult:
        """
        Execute a trade signal

        Args:
            signal: TradeSignal to execute
            use_limit: Override for limit order preference (defaults to settings)

        Returns:
            OrderResult with execution details
        """
        try:
            side = OrderSide.BUY if signal.signal_type == SignalType.LONG else OrderSide.SELL

            # Phase 5: Determine if using limit or market order
            if use_limit is None:
                use_limit = settings.trading.use_limit_entry

            if use_limit:
                # Calculate limit price with buffer
                buffer_pct = settings.trading.limit_entry_buffer_pct

                if signal.signal_type == SignalType.LONG:
                    # Limit slightly above current price for long
                    limit_price = signal.entry_price * (1 + buffer_pct)
                else:
                    # Limit slightly below current price for short
                    limit_price = signal.entry_price * (1 - buffer_pct)

                order_request = LimitOrderRequest(
                    symbol=signal.symbol,
                    qty=signal.position_size,
                    side=side,
                    time_in_force=TimeInForce.IOC,  # Immediate-or-cancel
                    limit_price=round(limit_price, 2),
                    order_class="bracket",
                    stop_loss=StopLossRequest(stop_price=round(signal.stop_loss, 2)),
                    take_profit=TakeProfitRequest(limit_price=round(signal.take_profit, 2))
                )

                logger.info(
                    f"Order submitted: {side.value} {signal.position_size} {signal.symbol} "
                    f"@ limit ${limit_price:.2f}, SL=${signal.stop_loss:.2f}, TP=${signal.take_profit:.2f}"
                )
            else:
                # Create market bracket order (entry + stop loss + take profit)
                order_request = MarketOrderRequest(
                    symbol=signal.symbol,
                    qty=signal.position_size,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    order_class="bracket",
                    stop_loss=StopLossRequest(stop_price=round(signal.stop_loss, 2)),
                    take_profit=TakeProfitRequest(limit_price=round(signal.take_profit, 2))
                )

                logger.info(
                    f"Order submitted: {side.value} {signal.position_size} {signal.symbol} "
                    f"@ market, SL=${signal.stop_loss:.2f}, TP=${signal.take_profit:.2f}"
                )

            order = self.client.submit_order(order_request)

            self.active_orders[signal.symbol] = order.id

            return OrderResult(
                success=True,
                order_id=order.id,
                symbol=signal.symbol,
                side=side.value,
                qty=signal.position_size,
                status=order.status.value
            )

        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return OrderResult(
                success=False,
                error=str(e)
            )

    def execute_market_order(
        self,
        symbol: str,
        qty: int,
        side: str
    ) -> OrderResult:
        """
        Execute a simple market order

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'

        Returns:
            OrderResult
        """
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )

            order = self.client.submit_order(order_request)

            return OrderResult(
                success=True,
                order_id=order.id,
                symbol=symbol,
                side=side,
                qty=qty,
                status=order.status.value
            )

        except Exception as e:
            logger.error(f"Error executing market order: {e}")
            return OrderResult(success=False, error=str(e))

    def get_positions(self) -> list[Position]:
        """Get all open positions"""
        try:
            positions = self.client.get_all_positions()
            return [
                Position(
                    symbol=pos.symbol,
                    side=pos.side.value,
                    qty=int(pos.qty),
                    entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_pl_pct=float(pos.unrealized_plpc) * 100
                )
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        try:
            pos = self.client.get_open_position(symbol)
            return Position(
                symbol=pos.symbol,
                side=pos.side.value,
                qty=int(pos.qty),
                entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                unrealized_pl=float(pos.unrealized_pl),
                unrealized_pl_pct=float(pos.unrealized_plpc) * 100
            )
        except Exception as e:
            logger.debug(f"No position for {symbol}: {e}")
            return None

    def close_position(self, symbol: str) -> OrderResult:
        """Close position for a symbol"""
        try:
            order = self.client.close_position(symbol)
            logger.info(f"Closed position for {symbol}")
            return OrderResult(
                success=True,
                order_id=order.id if hasattr(order, 'id') else None,
                symbol=symbol,
                status="closed"
            )
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return OrderResult(success=False, error=str(e))

    def close_all_positions(self) -> list[OrderResult]:
        """Close all open positions"""
        results = []
        try:
            self.client.close_all_positions(cancel_orders=True)
            logger.info("Closed all positions")
            results.append(OrderResult(success=True, status="all_closed"))
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            results.append(OrderResult(success=False, error=str(e)))
        return results

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order"""
        try:
            self.client.cancel_order_by_id(order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders"""
        try:
            self.client.cancel_orders()
            logger.info("Cancelled all orders")
            return True
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return False

    def get_open_orders(self) -> list[dict]:
        """Get all open orders"""
        try:
            request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = self.client.get_orders(request)
            return [
                {
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'qty': order.qty,
                    'type': order.type.value,
                    'status': order.status.value,
                    'created_at': order.created_at
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []

    def execute_entry_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        use_limit: bool = False,
        limit_price: float = None
    ) -> OrderResult:
        """
        Execute an entry order without bracket (standalone)

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            use_limit: Whether to use limit order
            limit_price: Limit price if using limit order

        Returns:
            OrderResult with order details
        """
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            if use_limit and limit_price:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(limit_price, 2)
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY
                )

            order = self.client.submit_order(order_request)
            self.active_orders[symbol] = order.id

            logger.info(f"Entry order submitted: {side} {qty} {symbol}")

            return OrderResult(
                success=True,
                order_id=order.id,
                symbol=symbol,
                side=side,
                qty=qty,
                status=order.status.value
            )

        except Exception as e:
            logger.error(f"Error executing entry order: {e}")
            return OrderResult(success=False, error=str(e))

    def create_stop_loss_order(
        self,
        symbol: str,
        qty: int,
        stop_price: float,
        position_side: str
    ) -> OrderResult:
        """
        Create a standalone stop loss order

        Args:
            symbol: Stock symbol
            qty: Number of shares
            stop_price: Stop trigger price
            position_side: 'long' or 'short' (determines sell/buy)

        Returns:
            OrderResult with order details
        """
        try:
            # For long positions, stop is a SELL order
            # For short positions, stop is a BUY order
            order_side = OrderSide.SELL if position_side.lower() == 'long' else OrderSide.BUY

            order_request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                stop_price=round(stop_price, 2)
            )

            order = self.client.submit_order(order_request)

            logger.info(
                f"Stop loss order created: {symbol} qty={qty} @ ${stop_price:.2f}"
            )

            return OrderResult(
                success=True,
                order_id=order.id,
                symbol=symbol,
                side=order_side.value,
                qty=qty,
                status=order.status.value
            )

        except Exception as e:
            logger.error(f"Error creating stop loss order: {e}")
            return OrderResult(success=False, error=str(e))

    def cancel_and_replace_stop(
        self,
        order_id: str,
        new_stop_price: float
    ) -> OrderResult:
        """
        Cancel existing stop order and create new one with updated price

        Args:
            order_id: Existing order ID to replace
            new_stop_price: New stop price

        Returns:
            OrderResult with new order details
        """
        try:
            # Use Alpaca's replace order endpoint
            replace_request = ReplaceOrderRequest(
                stop_price=round(new_stop_price, 2)
            )

            new_order = self.client.replace_order_by_id(
                order_id=order_id,
                order_data=replace_request
            )

            logger.info(
                f"Stop order replaced: {order_id} -> {new_order.id} @ ${new_stop_price:.2f}"
            )

            return OrderResult(
                success=True,
                order_id=new_order.id,
                symbol=new_order.symbol,
                status=new_order.status.value
            )

        except Exception as e:
            logger.error(f"Error replacing stop order: {e}")
            return OrderResult(success=False, error=str(e))

    def close_partial_position(
        self,
        symbol: str,
        qty: int,
        position_side: str
    ) -> OrderResult:
        """
        Close a partial position

        Args:
            symbol: Stock symbol
            qty: Number of shares to close
            position_side: 'long' or 'short'

        Returns:
            OrderResult with execution details
        """
        try:
            # To close, we do the opposite: sell for longs, buy for shorts
            close_side = 'sell' if position_side.lower() == 'long' else 'buy'
            return self.execute_market_order(symbol, qty, close_side)

        except Exception as e:
            logger.error(f"Error closing partial position: {e}")
            return OrderResult(success=False, error=str(e))

    def get_order_by_id(self, order_id: str) -> Optional[dict]:
        """
        Get order details by ID

        Args:
            order_id: Order ID

        Returns:
            Dict with order details or None
        """
        try:
            order = self.client.get_order_by_id(order_id)
            return {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side.value,
                'qty': order.qty,
                'filled_qty': order.filled_qty,
                'type': order.type.value,
                'status': order.status.value,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'created_at': order.created_at
            }
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            clock = self.client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False

    def get_next_market_times(self) -> dict:
        """Get next market open/close times"""
        try:
            clock = self.client.get_clock()
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close
            }
        except Exception as e:
            logger.error(f"Error getting market times: {e}")
            return {}

    def format_positions_message(self) -> str:
        """Format positions for Telegram message"""
        positions = self.get_positions()

        if not positions:
            return "📊 No open positions"

        lines = ["📊 *POSITIONS*\n"]

        for pos in positions:
            emoji = "🟢" if pos.unrealized_pl >= 0 else "🔴"
            direction = "+" if pos.unrealized_pl >= 0 else ""

            lines.append(
                f"{emoji} *{pos.symbol}* ({pos.side})\n"
                f"   {pos.qty} shares @ ${pos.entry_price:.2f}\n"
                f"   P/L: {direction}${pos.unrealized_pl:.2f} "
                f"({direction}{pos.unrealized_pl_pct:.1f}%)"
            )

        return "\n".join(lines)


# Global executor instance
order_executor = OrderExecutor()
