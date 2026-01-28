"""
Order execution and position management via Alpaca API
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from enum import Enum
import time as time_module
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

from config.settings import settings, TradingMode, get_session_params
from strategy.orb import TradeSignal, SignalType

# Constants for extended hours trading
EXTENDED_HOURS_ORDER_TIMEOUT_SECONDS = 120  # 2 minutes timeout for extended hours orders
QUOTE_CACHE_TTL_SECONDS = 5  # Cache quotes for 5 seconds to reduce API calls


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


@dataclass
class BracketOrderResult:
    """Result of bracket order execution (entry + stop loss + take profit)"""
    success: bool
    order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    take_profit_order_id: Optional[str] = None
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
        # Quote cache for spread validation: {symbol: {'quote': {...}, 'timestamp': datetime}}
        self._quote_cache: dict[str, dict] = {}

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

    async def execute_oto_entry(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_price: float,
        use_limit: bool = False,
        limit_price: float = None
    ) -> BracketOrderResult:
        """
        Execute entry with OTO (One-Triggers-Other) - entry triggers stop loss only.
        No take profit order - compatible with partial close strategy.

        This avoids Alpaca's wash trade detection and allows stop loss modification
        (which bracket order legs do not support).

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            stop_price: Stop loss trigger price
            use_limit: Whether to use limit order for entry
            limit_price: Limit price if using limit order

        Returns:
            BracketOrderResult with order details including stop_order_id
        """
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            if use_limit and limit_price:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=round(limit_price, 2),
                    order_class="oto",  # OTO instead of bracket
                    stop_loss=StopLossRequest(stop_price=round(stop_price, 2))
                    # No take_profit - managed separately for partial close strategy
                )
                logger.info(
                    f"OTO order submitted: {side} {qty} {symbol} "
                    f"@ limit ${limit_price:.2f}, SL=${stop_price:.2f}"
                )
            else:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    order_class="oto",  # OTO instead of bracket
                    stop_loss=StopLossRequest(stop_price=round(stop_price, 2))
                    # No take_profit - managed separately for partial close strategy
                )
                logger.info(
                    f"OTO order submitted: {side} {qty} {symbol} "
                    f"@ market, SL=${stop_price:.2f}"
                )

            order = self.client.submit_order(order_request)
            self.active_orders[symbol] = order.id

            # Extract stop order ID from OTO order legs
            stop_order_id = None

            if hasattr(order, 'legs') and order.legs:
                for leg in order.legs:
                    if leg.order_type == OrderType.STOP:
                        stop_order_id = leg.id

            # VALIDATION: Fail if stop_order_id could not be extracted
            if not stop_order_id:
                logger.error(f"Failed to extract stop_order_id from OTO order {order.id}")
                return BracketOrderResult(
                    success=False,
                    order_id=order.id,
                    symbol=symbol,
                    error="Could not extract stop_order_id from OTO order legs"
                )

            # Wait for entry fill using async sleep
            # Use longer timeout for limit orders since they may not fill immediately
            max_wait = settings.trading.limit_order_fill_timeout if use_limit else 10
            filled_price = None
            for _ in range(max_wait):
                # Use asyncio.to_thread to avoid blocking the event loop
                order_status = await asyncio.to_thread(self.client.get_order_by_id, order.id)
                if order_status.status == OrderStatus.FILLED:
                    filled_price = float(order_status.filled_avg_price) if order_status.filled_avg_price else None
                    break
                elif order_status.status in (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
                    return BracketOrderResult(
                        success=False,
                        order_id=order.id,
                        symbol=symbol,
                        error=f"Order {order_status.status.value}"
                    )
                await asyncio.sleep(1)  # Non-blocking async sleep

            # If order didn't fill within timeout, cancel it and return failure
            if filled_price is None:
                logger.warning(
                    f"OTO entry timeout: {symbol} not filled in {max_wait}s, cancelling..."
                )
                try:
                    await asyncio.to_thread(self.client.cancel_order_by_id, order.id)
                    logger.info(f"Cancelled unfilled OTO entry order for {symbol}")
                except Exception as cancel_err:
                    logger.error(f"Error cancelling unfilled OTO entry: {cancel_err}")

                return BracketOrderResult(
                    success=False,
                    order_id=order.id,
                    symbol=symbol,
                    error=f"Entry order not filled within {max_wait}s timeout - order cancelled"
                )

            logger.info(
                f"OTO entry executed: {symbol} filled @ ${filled_price:.2f}, "
                f"stop_order_id={stop_order_id}"
            )

            return BracketOrderResult(
                success=True,
                order_id=order.id,
                stop_order_id=stop_order_id,
                take_profit_order_id=None,  # OTO has no take profit
                symbol=symbol,
                side=side,
                qty=qty,
                filled_price=filled_price,
                status=order.status.value
            )

        except Exception as e:
            logger.error(f"Error executing OTO entry: {e}")
            return BracketOrderResult(success=False, error=str(e))

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

    async def close_partial_position(
        self,
        symbol: str,
        qty: int,
        position_side: str
    ) -> OrderResult:
        """
        Close a partial position and wait for fill to get actual fill price.

        Args:
            symbol: Stock symbol
            qty: Number of shares to close
            position_side: 'long' or 'short'

        Returns:
            OrderResult with execution details including filled_price
        """
        try:
            # To close, we do the opposite: sell for longs, buy for shorts
            close_side = 'sell' if position_side.lower() == 'long' else 'buy'
            order_side = OrderSide.SELL if close_side == 'sell' else OrderSide.BUY

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY
            )

            order = self.client.submit_order(order_request)

            # Wait for fill (max 10 seconds) to get actual fill price using async sleep
            for _ in range(10):
                # Use asyncio.to_thread to avoid blocking the event loop
                order_status = await asyncio.to_thread(self.client.get_order_by_id, order.id)
                if order_status.status == OrderStatus.FILLED:
                    filled_price = float(order_status.filled_avg_price) if order_status.filled_avg_price else None
                    price_str = f"${filled_price:.2f}" if filled_price is not None else "N/A"
                    logger.info(
                        f"Partial close filled: {symbol} {close_side} {qty} @ {price_str}"
                    )
                    return OrderResult(
                        success=True,
                        order_id=order.id,
                        symbol=symbol,
                        side=close_side,
                        qty=qty,
                        filled_price=filled_price,
                        status='filled'
                    )
                elif order_status.status in (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
                    return OrderResult(
                        success=False,
                        order_id=order.id,
                        symbol=symbol,
                        error=f"Order {order_status.status.value}"
                    )
                await asyncio.sleep(1)  # Non-blocking async sleep

            # If not filled in 10s, return without fill price
            logger.warning(f"Partial close order not filled in 10s: {symbol}")
            return OrderResult(
                success=True,
                order_id=order.id,
                symbol=symbol,
                side=close_side,
                qty=qty,
                status=order.status.value
            )

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

    def execute_extended_hours_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        stop_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        timeout_seconds: int = EXTENDED_HOURS_ORDER_TIMEOUT_SECONDS
    ) -> OrderResult:
        """
        Execute an order during extended hours (premarket/postmarket).

        Extended hours orders MUST be limit orders with extended_hours=True.
        Includes automatic timeout and cancellation for unfilled orders.

        IMPORTANT: Stop orders do NOT execute during extended hours in Alpaca.
        They will only trigger once regular market hours begin. Consider using
        manual price monitoring for stop protection during extended hours.

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            limit_price: Limit price (required for extended hours)
            stop_price: Optional stop loss price (WARNING: won't trigger until regular hours)
            take_profit_price: Optional take profit price
            timeout_seconds: Max seconds to wait for fill before cancelling (default 120)

        Returns:
            OrderResult with execution details
        """
        try:
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL

            # Extended hours requires limit order with extended_hours=True
            # TimeInForce.DAY works with extended_hours=True
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
                extended_hours=True  # Critical for extended hours trading
            )

            logger.info(
                f"Extended hours order submitted: {side} {qty} {symbol} "
                f"@ limit ${limit_price:.2f} (extended_hours=True, timeout={timeout_seconds}s)"
            )

            order = self.client.submit_order(order_request)
            self.active_orders[symbol] = order.id

            # Wait for fill with timeout
            fill_price = None
            start_time = time_module.time()
            while time_module.time() - start_time < timeout_seconds:
                order_status = self.client.get_order_by_id(order.id)
                if order_status.status == OrderStatus.FILLED:
                    fill_price = float(order_status.filled_avg_price) if order_status.filled_avg_price else limit_price
                    logger.info(f"Extended hours order filled: {symbol} @ ${fill_price:.2f}")
                    break
                elif order_status.status in (OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED):
                    logger.warning(f"Extended hours order {order_status.status.value}: {symbol}")
                    return OrderResult(
                        success=False,
                        order_id=order.id,
                        symbol=symbol,
                        error=f"Order {order_status.status.value}"
                    )
                time_module.sleep(2)  # Check every 2 seconds

            # If not filled, cancel the order
            if fill_price is None:
                logger.warning(f"Extended hours order timeout ({timeout_seconds}s): {symbol}, cancelling...")
                try:
                    self.client.cancel_order_by_id(order.id)
                    logger.info(f"Extended hours order cancelled due to timeout: {symbol}")
                except Exception as cancel_err:
                    logger.error(f"Error cancelling timed-out order: {cancel_err}")
                return OrderResult(
                    success=False,
                    order_id=order.id,
                    symbol=symbol,
                    error=f"Order timeout after {timeout_seconds}s - cancelled"
                )

            # CRITICAL WARNING: Stop orders do NOT execute during extended hours
            # Create stop order but warn about limitation
            stop_order_id = None
            stop_warning = None
            if stop_price:
                position_side = 'long' if side.lower() == 'buy' else 'short'
                stop_result = self.create_stop_loss_order(
                    symbol=symbol,
                    qty=qty,
                    stop_price=stop_price,
                    position_side=position_side
                )
                if stop_result.success:
                    stop_order_id = stop_result.order_id
                    stop_warning = (
                        "⚠️ CRITICAL: Stop loss order created but will NOT trigger during "
                        "extended hours. It will only activate once regular market opens. "
                        "Consider manual price monitoring for risk management."
                    )
                    logger.warning(stop_warning)

            return OrderResult(
                success=True,
                order_id=order.id,
                symbol=symbol,
                side=side,
                qty=qty,
                filled_price=fill_price,
                status='filled'
            )

        except Exception as e:
            logger.error(f"Error executing extended hours order: {e}")
            return OrderResult(success=False, error=str(e))

    def _get_cached_quote(self, symbol: str) -> Optional[dict]:
        """
        Get quote from cache if valid, otherwise fetch fresh quote.

        Args:
            symbol: Stock symbol

        Returns:
            Quote dict or None
        """
        from data.market_data import market_data

        now = datetime.now()
        cached = self._quote_cache.get(symbol)

        if cached:
            age = (now - cached['timestamp']).total_seconds()
            if age < QUOTE_CACHE_TTL_SECONDS:
                logger.debug(f"Using cached quote for {symbol} (age: {age:.1f}s)")
                return cached['quote']

        # Fetch fresh quote
        quote = market_data.get_latest_quote(symbol)
        if quote:
            self._quote_cache[symbol] = {
                'quote': quote,
                'timestamp': now
            }
        return quote

    def invalidate_quote_cache(self, symbol: Optional[str] = None):
        """
        Invalidate quote cache.

        Args:
            symbol: Specific symbol to invalidate, or None for all
        """
        if symbol:
            self._quote_cache.pop(symbol, None)
        else:
            self._quote_cache.clear()

    def validate_spread(
        self,
        symbol: str,
        max_spread_pct: float
    ) -> tuple[bool, float, Optional[str]]:
        """
        Validate bid-ask spread is within acceptable limits.

        Uses cached quotes to reduce API calls when checking multiple symbols.
        Cache TTL is QUOTE_CACHE_TTL_SECONDS (default 5 seconds).

        Args:
            symbol: Stock symbol
            max_spread_pct: Maximum acceptable spread as decimal (e.g., 0.005 for 0.5%)

        Returns:
            Tuple of (is_valid, spread_pct, error_message)
        """
        try:
            quote = self._get_cached_quote(symbol)

            if not quote:
                return False, 0.0, "Unable to get quote"

            bid = quote.get('bid', 0)
            ask = quote.get('ask', 0)

            if bid <= 0 or ask <= 0:
                return False, 0.0, "Invalid bid/ask prices"

            spread = ask - bid
            mid_price = (bid + ask) / 2
            spread_pct = spread / mid_price

            if spread_pct > max_spread_pct:
                return (
                    False,
                    spread_pct,
                    f"Spread {spread_pct:.2%} exceeds max {max_spread_pct:.2%}"
                )

            return True, spread_pct, None

        except Exception as e:
            logger.error(f"Error validating spread for {symbol}: {e}")
            return False, 0.0, str(e)

    def get_extended_hours_status(self) -> dict:
        """
        Get current extended hours trading status.

        Returns:
            Dict with is_premarket, is_postmarket, is_regular flags and times
        """
        try:
            from datetime import time
            import pytz
            EST = pytz.timezone('US/Eastern')

            clock = self.client.get_clock()
            now = datetime.now(EST)
            current_time = now.time()

            # Define session windows
            premarket_start = time(4, 0)
            premarket_end = time(9, 30)
            regular_start = time(9, 30)
            regular_end = time(16, 0)
            postmarket_start = time(16, 0)
            postmarket_end = time(20, 0)

            is_premarket = premarket_start <= current_time < premarket_end
            is_regular = regular_start <= current_time < regular_end
            is_postmarket = postmarket_start <= current_time < postmarket_end

            return {
                'is_premarket': is_premarket,
                'is_regular': is_regular,
                'is_postmarket': is_postmarket,
                'is_market_open': clock.is_open,
                'current_time': current_time.strftime('%H:%M:%S'),
                'next_open': clock.next_open,
                'next_close': clock.next_close
            }

        except Exception as e:
            logger.error(f"Error getting extended hours status: {e}")
            return {
                'is_premarket': False,
                'is_regular': False,
                'is_postmarket': False,
                'is_market_open': False,
                'error': str(e)
            }

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
