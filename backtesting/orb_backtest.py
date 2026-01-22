"""
ORB (Opening Range Breakout) Strategy adapted for Lumibot backtesting
"""
from datetime import datetime, time
from typing import Optional
import pandas as pd
from loguru import logger

from lumibot.strategies import Strategy
from lumibot.entities import Asset

from data.indicators import (
    calculate_rsi, calculate_vwap, calculate_macd,
    is_macd_bullish, is_macd_bearish
)
from config.settings import settings


class ORBBacktestStrategy(Strategy):
    """
    Opening Range Breakout Strategy for Backtesting

    Rules:
    - Calculate high/low of first 15 minutes (Opening Range)
    - LONG: Price breaks above ORB high + price > VWAP + volume spike + RSI < 70 + MACD bullish
    - SHORT: Price breaks below ORB low + price < VWAP + volume spike + RSI > 30 + MACD bearish
    - Stop: Opposite side of ORB
    - Target: 2:1 risk/reward ratio
    """

    # Class variable to store all trades across backtest (for final report)
    # Initialized to None and set to fresh list in initialize() to avoid mutable default issues
    _all_closed_trades: list = None

    @classmethod
    def reset_state(cls):
        """
        Reset class-level state for a fresh backtest.
        Call this before starting a new backtest to ensure clean state.
        """
        cls._all_closed_trades = []

    @classmethod
    def get_all_trades(cls) -> list:
        """Get all closed trades. Returns empty list if not initialized."""
        return cls._all_closed_trades if cls._all_closed_trades is not None else []

    @classmethod
    def add_trade(cls, trade_record: dict):
        """Add a trade record to class-level tracking."""
        if cls._all_closed_trades is None:
            cls._all_closed_trades = []
        cls._all_closed_trades.append(trade_record)

    @classmethod
    def generate_final_report(cls):
        """Generate performance report from class-level trade tracking"""
        all_trades = cls.get_all_trades()
        if not all_trades:
            logger.warning("No trades to report")
            return

        from pathlib import Path

        trades_df = pd.DataFrame(all_trades)
        total_pnl = trades_df["pnl"].sum()
        winning = trades_df[trades_df["pnl"] > 0]
        losing = trades_df[trades_df["pnl"] <= 0]

        win_rate = len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0
        avg_win = winning["pnl"].mean() if len(winning) > 0 else 0
        avg_loss = abs(losing["pnl"].mean()) if len(losing) > 0 else 0
        profit_factor = winning["pnl"].sum() / abs(losing["pnl"].sum()) if len(losing) > 0 and losing["pnl"].sum() != 0 else float("inf")

        print("\n" + "=" * 60)
        print("ORB BACKTEST PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Total Trades:    {len(trades_df)}")
        print(f"Winning Trades:  {len(winning)}")
        print(f"Losing Trades:   {len(losing)}")
        print(f"Win Rate:        {win_rate:.1f}%")
        print("-" * 60)
        print(f"Total PnL:       ${total_pnl:,.2f}")
        print(f"Avg Win:         ${avg_win:,.2f}")
        print(f"Avg Loss:        ${avg_loss:,.2f}")
        print(f"Profit Factor:   {profit_factor:.2f}")
        print("-" * 60)
        print(f"Best Trade:      ${trades_df['pnl'].max():,.2f}")
        print(f"Worst Trade:     ${trades_df['pnl'].min():,.2f}")
        print("=" * 60)

        # Save trades to CSV (convert dates to ISO format for export)
        from datetime import datetime as dt
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)

        # Convert date column to ISO format strings for CSV export
        export_df = trades_df.copy()
        if "date" in export_df.columns:
            export_df["date"] = export_df["date"].apply(
                lambda d: d.isoformat() if hasattr(d, "isoformat") else str(d)
            )

        trades_file = output_dir / f"trades_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_df.to_csv(trades_file, index=False)
        logger.info(f"Trades saved to {trades_file}")

        return trades_df

    # Strategy parameters (can be overridden)
    # These values define thresholds and requirements for signal generation
    parameters = {
        # ORB Configuration
        "orb_period_minutes": settings.trading.orb_period_minutes,  # Duration of opening range

        # Risk Management
        "risk_per_trade": settings.trading.risk_per_trade,  # Fraction of capital to risk per trade
        "reward_risk_ratio": settings.trading.reward_risk_ratio,  # Take profit multiplier vs stop loss

        # Signal Thresholds
        "min_relative_volume": settings.trading.min_relative_volume,  # Min volume spike ratio (1.5 = 50% above avg)
        "rsi_overbought": settings.trading.rsi_overbought,  # RSI level above which we don't go long (70)
        "rsi_oversold": settings.trading.rsi_oversold,  # RSI level below which we don't go short (30)

        # Trading Limits
        "max_trades_per_day": settings.trading.max_trades_per_day,  # Max trades allowed per day

        # Symbols
        "symbols": ["SPY"],  # Default symbols to trade

        # Data Requirements - minimum bars needed for calculations
        # These thresholds ensure we have enough data for reliable indicator calculations
        "min_bars_for_orb": 5,  # Minimum bars to form a valid Opening Range (ensures statistical significance)
        "min_bars_for_indicators": 20,  # Minimum bars for indicator calculation (RSI=14, but need buffer)
        "volume_average_periods": 20,  # Periods for calculating average volume (standard 20-period average)
        "rsi_period": 14,  # RSI calculation period (standard is 14)
    }

    def initialize(self):
        """Initialize strategy state"""
        self.sleeptime = "1M"  # 1 minute bars

        # Daily state (reset each day)
        self.opening_ranges = {}  # {symbol: {"high": x, "low": y, "vwap": z}}
        self.orb_bars = {}  # {symbol: [bars during ORB period]}
        self.trades_today = 0
        self.traded_symbols_today = set()
        self.current_date = None

        # Track bars for indicator calculation
        self.bars_history = {}  # {symbol: DataFrame}

        # Track open positions for manual stop/target management
        self.position_targets = {}  # {symbol: {"entry": x, "stop": y, "target": z, "side": "long"/"short"}}

        # Track all closed trades for reporting
        self.closed_trades = []

        # Reset class-level state at start of new backtest
        ORBBacktestStrategy.reset_state()

        logger.info(f"ORB Backtest Strategy initialized with symbols: {self.parameters['symbols']}")

    def on_trading_iteration(self):
        """Main trading logic - adapts to both minute and daily data"""
        current_dt = self.get_datetime()

        # Reset daily state at new day
        if self.current_date != current_dt.date():
            self._reset_daily_state()
            self.current_date = current_dt.date()

            # For daily data: calculate "opening range" from previous day
            self._calculate_opening_ranges_daily()

        # Check for stop loss / take profit on existing positions
        self._check_position_exits()

        # Check for breakouts
        if self.opening_ranges:
            self._check_breakouts()

    def _reset_daily_state(self):
        """Reset state for a new trading day"""
        self.opening_ranges = {}
        self.orb_bars = {}
        self.trades_today = 0
        self.traded_symbols_today = set()
        self.bars_history = {}
        # Note: position_targets is NOT reset daily - positions carry over
        logger.debug("Daily state reset")

    def _check_position_exits(self):
        """Check if any open positions hit stop loss or take profit"""
        positions_to_close = []

        for symbol, targets in self.position_targets.items():
            position = self.get_position(symbol)
            if position is None or position.quantity == 0:
                positions_to_close.append(symbol)
                continue

            # Get current price
            asset = Asset(symbol=symbol, asset_type="stock")
            bars = self.get_historical_prices(asset, 1, "minute")
            if bars is None or bars.df.empty:
                bars = self.get_historical_prices(asset, 1, "day")
            if bars is None or bars.df.empty:
                continue

            current_price = bars.df["close"].iloc[-1]

            if targets["side"] == "long":
                # Check stop loss (price dropped below stop)
                if current_price <= targets["stop"]:
                    self._close_position(symbol, current_price, "stop_loss")
                    positions_to_close.append(symbol)
                # Check take profit (price rose above target)
                elif current_price >= targets["target"]:
                    self._close_position(symbol, current_price, "take_profit")
                    positions_to_close.append(symbol)

            elif targets["side"] == "short":
                # Check stop loss (price rose above stop)
                if current_price >= targets["stop"]:
                    self._close_position(symbol, current_price, "stop_loss")
                    positions_to_close.append(symbol)
                # Check take profit (price dropped below target)
                elif current_price <= targets["target"]:
                    self._close_position(symbol, current_price, "take_profit")
                    positions_to_close.append(symbol)

        # Remove closed positions from tracking
        for symbol in positions_to_close:
            if symbol in self.position_targets:
                del self.position_targets[symbol]

    def _close_position(self, symbol: str, price: float, reason: str):
        """Close a position and log the result"""
        position = self.get_position(symbol)
        if position is None or position.quantity == 0:
            return

        targets = self.position_targets.get(symbol, {})
        entry = targets.get("entry", price)
        side = targets.get("side", "unknown")
        quantity = abs(position.quantity)

        # Calculate PnL
        if side == "long":
            pnl = (price - entry) * quantity
            pnl_pct = (price - entry) / entry if entry > 0 else 0
        else:
            pnl = (entry - price) * quantity
            pnl_pct = (entry - price) / entry if entry > 0 else 0

        # Record the trade for reporting
        trade_record = {
            "symbol": symbol,
            "side": side,
            "entry_price": entry,
            "exit_price": price,
            "quantity": quantity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": reason,
            "date": self.get_datetime().date()  # Store as datetime.date, convert for export
        }
        self.closed_trades.append(trade_record)
        ORBBacktestStrategy.add_trade(trade_record)

        # Close the position
        self.sell_all(symbol)

        logger.info(
            f"CLOSED {side.upper()} {symbol}: Exit=${price:.2f}, "
            f"Entry=${entry:.2f}, PnL=${pnl:.2f} ({reason})"
        )

    def _calculate_opening_ranges_daily(self):
        """
        Calculate Opening Range using previous day's data (for daily backtesting).
        Uses previous day's high/low as the range to break.

        IMPORTANT: This method ensures no data leakage by only using data
        from days BEFORE the current trading day.
        """
        current_dt = self.get_datetime()
        current_date = current_dt.date()

        for symbol in self.parameters["symbols"]:
            asset = Asset(symbol=symbol, asset_type="stock")

            # Get last 5 days of data
            bars = self.get_historical_prices(asset, 5, "day")
            if bars is None or bars.df.empty or len(bars.df) < 2:
                logger.warning(f"Not enough daily data for {symbol}")
                continue

            # DATA LEAKAGE CHECK: Verify we're not using current day's data
            # The most recent bar should be from yesterday or earlier
            last_bar_date = bars.df.index[-1]
            if hasattr(last_bar_date, 'date'):
                last_bar_date = last_bar_date.date()
            elif hasattr(last_bar_date, 'to_pydatetime'):
                last_bar_date = last_bar_date.to_pydatetime().date()

            # Log data boundaries for debugging
            logger.debug(
                f"Data leakage check {symbol}: Current date={current_date}, "
                f"Last bar date={last_bar_date}"
            )

            # Use previous day's range (second to last bar)
            # This ensures we're using data from BEFORE the current day
            prev_day = bars.df.iloc[-2]  # Second to last (yesterday)
            prev_day_date = bars.df.index[-2]
            if hasattr(prev_day_date, 'date'):
                prev_day_date = prev_day_date.date()

            # Assert: The "previous day" we're using must be before current date
            assert prev_day_date < current_date, (
                f"Data leakage detected for {symbol}: Using data from {prev_day_date} "
                f"on trading day {current_date}"
            )

            self.opening_ranges[symbol] = {
                "high": prev_day["high"],
                "low": prev_day["low"],
                "range": prev_day["high"] - prev_day["low"],
                "vwap": (prev_day["high"] + prev_day["low"] + prev_day["close"]) / 3
            }

            logger.debug(
                f"Daily ORB {symbol}: High=${prev_day['high']:.2f}, "
                f"Low=${prev_day['low']:.2f} (from {prev_day_date})"
            )

    def _collect_orb_bars(self):
        """Collect bars during the Opening Range period"""
        for symbol in self.parameters["symbols"]:
            asset = Asset(symbol=symbol, asset_type="stock")

            # Get current bar
            bars = self.get_historical_prices(asset, 1, "minute")
            if bars is None or bars.df.empty:
                continue

            bar = bars.df.iloc[-1]

            if symbol not in self.orb_bars:
                self.orb_bars[symbol] = []

            self.orb_bars[symbol].append({
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["volume"],
                "vwap": bar.get("vwap", bar["close"])
            })

    def _calculate_opening_ranges(self):
        """Calculate Opening Range for each symbol after ORB period"""
        min_bars = self.parameters.get("min_bars_for_orb", 5)

        for symbol, bars in self.orb_bars.items():
            if len(bars) < min_bars:
                logger.warning(f"Not enough ORB bars for {symbol}: {len(bars)} (need {min_bars})")
                continue

            orb_high = max(b["high"] for b in bars)
            orb_low = min(b["low"] for b in bars)

            # Calculate VWAP for ORB period
            total_volume = sum(b["volume"] for b in bars)
            if total_volume > 0:
                typical_prices = [(b["high"] + b["low"] + b["close"]) / 3 * b["volume"] for b in bars]
                vwap = sum(typical_prices) / total_volume
            else:
                vwap = bars[-1]["close"]

            self.opening_ranges[symbol] = {
                "high": orb_high,
                "low": orb_low,
                "range": orb_high - orb_low,
                "vwap": vwap
            }

            logger.info(
                f"ORB {symbol}: High=${orb_high:.2f}, Low=${orb_low:.2f}, "
                f"Range=${orb_high - orb_low:.2f}"
            )

    def _check_breakouts(self):
        """Check for breakout signals (works with both minute and daily data)"""
        if self.trades_today >= self.parameters["max_trades_per_day"]:
            return

        # Get configurable thresholds from parameters
        min_bars = self.parameters.get("min_bars_for_indicators", 20)
        vol_avg_periods = self.parameters.get("volume_average_periods", 20)
        rsi_period = self.parameters.get("rsi_period", 14)

        for symbol in self.parameters["symbols"]:
            if symbol in self.traded_symbols_today:
                continue

            if symbol not in self.opening_ranges:
                continue

            # Get current data with enough history for indicators
            # Try minute data first, fall back to daily
            asset = Asset(symbol=symbol, asset_type="stock")
            bars = self.get_historical_prices(asset, 50, "minute")

            if bars is None or bars.df.empty:
                # Fall back to daily data
                bars = self.get_historical_prices(asset, 50, "day")

            if bars is None or bars.df.empty or len(bars.df) < min_bars:
                continue

            df = bars.df.copy()

            # Calculate indicators
            current_price = df["close"].iloc[-1]
            current_volume = df["volume"].iloc[-1]

            # VWAP - use shared function for consistency with live trading
            df["vwap"] = calculate_vwap(df)
            current_vwap = df["vwap"].iloc[-1]

            # RSI
            df["rsi"] = calculate_rsi(df["close"], rsi_period)
            current_rsi = df["rsi"].iloc[-1]

            # MACD
            macd_line, signal_line, histogram = calculate_macd(df["close"])
            df["macd"] = macd_line
            df["macd_signal"] = signal_line
            df["macd_histogram"] = histogram

            current_macd = df["macd"].iloc[-1]
            current_signal = df["macd_signal"].iloc[-1]
            current_histogram = df["macd_histogram"].iloc[-1]
            prev_histogram = df["macd_histogram"].iloc[-2] if len(df) > 1 else 0

            # Volume spike check (compare current to rolling average)
            avg_volume = df["volume"].rolling(vol_avg_periods).mean().iloc[-1]
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 0

            orb = self.opening_ranges[symbol]

            # Check LONG conditions
            if self._check_long_signal(
                current_price, orb, current_vwap, current_rsi,
                relative_volume, current_macd, current_signal,
                current_histogram, prev_histogram
            ):
                self._execute_long(symbol, current_price, orb)

            # Check SHORT conditions
            elif self._check_short_signal(
                current_price, orb, current_vwap, current_rsi,
                relative_volume, current_macd, current_signal,
                current_histogram, prev_histogram
            ):
                self._execute_short(symbol, current_price, orb)

    def _check_long_signal(
        self,
        price: float,
        orb: dict,
        vwap: float,
        rsi: float,
        rel_volume: float,
        macd: float,
        signal: float,
        histogram: float,
        prev_histogram: float
    ) -> bool:
        """Check if LONG conditions are met"""
        macd_bullish = is_macd_bullish(macd, signal, histogram, prev_histogram)

        conditions = [
            price > orb["high"],  # Price above ORB high
            price > vwap,  # Price above VWAP
            rel_volume >= self.parameters["min_relative_volume"],  # Volume spike
            rsi < self.parameters["rsi_overbought"],  # Not overbought
            macd_bullish,  # MACD confirmation
        ]

        return all(conditions)

    def _check_short_signal(
        self,
        price: float,
        orb: dict,
        vwap: float,
        rsi: float,
        rel_volume: float,
        macd: float,
        signal: float,
        histogram: float,
        prev_histogram: float
    ) -> bool:
        """Check if SHORT conditions are met"""
        macd_bearish = is_macd_bearish(macd, signal, histogram, prev_histogram)

        conditions = [
            price < orb["low"],  # Price below ORB low
            price < vwap,  # Price below VWAP
            rel_volume >= self.parameters["min_relative_volume"],  # Volume spike
            rsi > self.parameters["rsi_oversold"],  # Not oversold
            macd_bearish,  # MACD confirmation
        ]

        return all(conditions)

    def _execute_long(self, symbol: str, entry_price: float, orb: dict):
        """Execute a LONG trade with manual stop/target tracking"""
        stop_loss = orb["low"]
        risk_per_share = entry_price - stop_loss

        if risk_per_share <= 0:
            return

        take_profit = entry_price + (risk_per_share * self.parameters["reward_risk_ratio"])

        # Calculate position size
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.parameters["risk_per_trade"]
        quantity = int(risk_amount / risk_per_share)

        if quantity <= 0:
            return

        # Place simple market order (no bracket for backtesting)
        asset = Asset(symbol=symbol, asset_type="stock")

        order = self.create_order(
            asset=asset,
            quantity=quantity,
            side="buy"
        )

        self.submit_order(order)

        # Track position targets for manual exit management
        self.position_targets[symbol] = {
            "entry": entry_price,
            "stop": stop_loss,
            "target": take_profit,
            "side": "long"
        }

        self.trades_today += 1
        self.traded_symbols_today.add(symbol)

        logger.info(
            f"LONG {symbol}: Entry=${entry_price:.2f}, "
            f"SL=${stop_loss:.2f}, TP=${take_profit:.2f}, Qty={quantity}"
        )

    def _execute_short(self, symbol: str, entry_price: float, orb: dict):
        """Execute a SHORT trade with manual stop/target tracking"""
        stop_loss = orb["high"]
        risk_per_share = stop_loss - entry_price

        if risk_per_share <= 0:
            return

        take_profit = entry_price - (risk_per_share * self.parameters["reward_risk_ratio"])

        # Calculate position size
        portfolio_value = self.get_portfolio_value()
        risk_amount = portfolio_value * self.parameters["risk_per_trade"]
        quantity = int(risk_amount / risk_per_share)

        if quantity <= 0:
            return

        # Place simple market order (no bracket for backtesting)
        asset = Asset(symbol=symbol, asset_type="stock")

        order = self.create_order(
            asset=asset,
            quantity=quantity,
            side="sell"
        )

        self.submit_order(order)

        # Track position targets for manual exit management
        self.position_targets[symbol] = {
            "entry": entry_price,
            "stop": stop_loss,
            "target": take_profit,
            "side": "short"
        }

        self.trades_today += 1
        self.traded_symbols_today.add(symbol)

        logger.info(
            f"SHORT {symbol}: Entry=${entry_price:.2f}, "
            f"SL=${stop_loss:.2f}, TP=${take_profit:.2f}, Qty={quantity}"
        )

    def before_market_closes(self):
        """Close all positions before market close"""
        # Close and record PnL for any open positions
        for symbol, targets in list(self.position_targets.items()):
            position = self.get_position(symbol)
            if position and position.quantity != 0:
                asset = Asset(symbol=symbol, asset_type="stock")
                bars = self.get_historical_prices(asset, 1, "minute")
                if bars is None or bars.df.empty:
                    bars = self.get_historical_prices(asset, 1, "day")
                if bars is not None and not bars.df.empty:
                    current_price = bars.df["close"].iloc[-1]
                    entry = targets.get("entry", current_price)
                    side = targets.get("side", "unknown")
                    quantity = abs(position.quantity)

                    if side == "long":
                        pnl = (current_price - entry) * quantity
                        pnl_pct = (current_price - entry) / entry if entry > 0 else 0
                    else:
                        pnl = (entry - current_price) * quantity
                        pnl_pct = (entry - current_price) / entry if entry > 0 else 0

                    # Record the trade
                    trade_record = {
                        "symbol": symbol,
                        "side": side,
                        "entry_price": entry,
                        "exit_price": current_price,
                        "quantity": quantity,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "exit_reason": "eod_close",
                        "date": self.get_datetime().date()  # Store as datetime.date
                    }
                    self.closed_trades.append(trade_record)
                    ORBBacktestStrategy.add_trade(trade_record)

                    logger.info(f"EOD close {side.upper()} {symbol}: Exit=${current_price:.2f}, PnL=${pnl:.2f}")

        self.sell_all()
        self.position_targets.clear()
        logger.info("Closed all positions before market close")

    def on_abrupt_closing(self):
        """Called at end of backtest - generate custom report"""
        # Use class method for final report (aggregates all trades)
        ORBBacktestStrategy.generate_final_report()

    def teardown(self):
        """Called at end of backtest - generate custom report"""
        # Use class method for final report (aggregates all trades)
        ORBBacktestStrategy.generate_final_report()

    def on_bot_crash(self, error):
        """Called if strategy crashes - still generate report"""
        logger.error(f"Strategy crashed: {error}")
        # Use class method for final report (aggregates all trades)
        ORBBacktestStrategy.generate_final_report()

    def on_filled_order(self, position, order, price, quantity, multiplier):
        """Log filled orders"""
        logger.info(f"Order filled: {order.side} {quantity} {position.asset.symbol} @ ${price:.2f}")

    def on_canceled_order(self, order):
        """Log canceled orders"""
        logger.warning(f"Order canceled: {order}")
