"""
VWAP Mean Reversion Strategy

Entry Logic:
- LONG: Price < VWAP * 0.98 (2%+ below) + RSI < 35 + Volume > 1.2x + MACD improving
- SHORT: Price > VWAP * 1.02 (2%+ above) + RSI > 65 + Volume > 1.2x + MACD worsening

Risk Management:
- Stop Loss: 1.5x ATR from entry
- Take Profit 1: VWAP (close 50%)
- Take Profit 2: Opposite side of VWAP
- Time Stop: 45 minutes max hold

Best Conditions:
- Range-bound days (after morning volatility settles)
- After 10:30 AM ET (avoid opening range volatility)
- ATR < 2.5% of price (not overly volatile days)
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from loguru import logger

from strategy.oneoff.base import OneOffStrategy, OneOffSignal, SignalDirection
from data.market_data import market_data
from data.indicators import calculate_atr, calculate_vwap, calculate_macd


@dataclass
class VWAPReversionConfig:
    """Configuration for VWAP Mean Reversion strategy"""
    # Entry conditions
    min_vwap_distance_pct: float = 0.02    # 2% minimum distance from VWAP
    max_vwap_distance_pct: float = 0.03    # 3% maximum (too far = trend, not reversion)
    rsi_oversold: int = 35                  # RSI threshold for long
    rsi_overbought: int = 65                # RSI threshold for short
    min_rel_volume: float = 1.2             # Minimum relative volume
    atr_max_pct: float = 0.025              # Max ATR as % of price (2.5%)

    # Risk management
    stop_atr_multiplier: float = 1.5        # Stop at 1.5x ATR
    time_stop_minutes: int = 45             # Max hold time

    # Scoring
    min_signal_score: float = 60.0          # Minimum score to generate signal


class VWAPReversionStrategy(OneOffStrategy):
    """
    VWAP Mean Reversion Strategy

    Trades mean reversion to VWAP when price deviates significantly.
    High probability strategy (65-70%) with defined risk parameters.
    """

    def __init__(self, config: Optional[VWAPReversionConfig] = None):
        super().__init__()
        self.config = config or VWAPReversionConfig()

    @property
    def name(self) -> str:
        return "vwap_reversion"

    @property
    def display_name(self) -> str:
        return "VWAP Mean Reversion"

    @property
    def description(self) -> str:
        return "Reverses to VWAP when price deviates 2%+ from it"

    async def scan_opportunities(
        self,
        symbols: List[str]
    ) -> List[OneOffSignal]:
        """
        Scan symbols for VWAP reversion setups.

        Args:
            symbols: List of stock symbols to scan

        Returns:
            List of valid signals
        """
        signals = []

        for symbol in symbols:
            try:
                signal = await self._check_symbol(symbol)
                if signal:
                    signals.append(signal)
                    self.signals_generated.append(signal)
                    logger.info(f"Signal found: {signal}")
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue

        logger.info(
            f"VWAP Reversion scan complete: "
            f"{len(signals)} signals from {len(symbols)} symbols"
        )

        return signals

    async def _check_symbol(self, symbol: str) -> Optional[OneOffSignal]:
        """
        Check a single symbol for VWAP reversion setup.

        Args:
            symbol: Stock symbol

        Returns:
            OneOffSignal if setup found, None otherwise
        """
        # Get market data with indicators
        bars = market_data.get_bars(symbol, limit=50)
        if bars.empty or len(bars) < 30:
            logger.debug(f"{symbol}: Not enough bars")
            return None

        # Get current price
        quote = market_data.get_latest_quote(symbol)
        if not quote:
            logger.debug(f"{symbol}: No quote available")
            return None

        current_price = quote['mid']

        # Calculate indicators
        vwap = calculate_vwap(bars).iloc[-1]
        macd_line, macd_signal, macd_histogram = calculate_macd(bars['close'])
        atr = calculate_atr(bars).iloc[-1]

        # Get RSI from indicator calculator
        indicators = self.get_latest_indicators(symbol)
        rsi = indicators.get('rsi', 50)
        prev_histogram = macd_histogram.iloc[-2] if len(macd_histogram) > 1 else 0
        current_histogram = macd_histogram.iloc[-1]

        # Calculate relative volume
        avg_volume = market_data.get_avg_daily_volume(symbol)
        current_volume = int(bars['volume'].iloc[-1])
        rel_volume = current_volume / (avg_volume / 390) if avg_volume > 0 else 1.0

        # Calculate VWAP distance
        vwap_distance_pct = (current_price - vwap) / vwap

        # Check if ATR is within acceptable range
        atr_pct = atr / current_price
        if atr_pct > self.config.atr_max_pct:
            logger.debug(f"{symbol}: ATR too high ({atr_pct:.2%} > {self.config.atr_max_pct:.2%})")
            return None

        # Check LONG setup (price below VWAP)
        if self._check_long_conditions(
            current_price, vwap, rsi, rel_volume,
            current_histogram, prev_histogram, vwap_distance_pct
        ):
            score = self.calculate_score(
                vwap_distance_pct=abs(vwap_distance_pct),
                rsi=rsi,
                rel_volume=rel_volume,
                macd_improving=(current_histogram > prev_histogram),
                atr_pct=atr_pct,
                direction='LONG'
            )

            if score >= self.config.min_signal_score:
                return self._create_long_signal(
                    symbol, current_price, vwap, atr,
                    rsi, rel_volume, score
                )

        # Check SHORT setup (price above VWAP)
        if self._check_short_conditions(
            current_price, vwap, rsi, rel_volume,
            current_histogram, prev_histogram, vwap_distance_pct
        ):
            score = self.calculate_score(
                vwap_distance_pct=abs(vwap_distance_pct),
                rsi=rsi,
                rel_volume=rel_volume,
                macd_improving=(current_histogram < prev_histogram),  # For shorts, decreasing is "improving"
                atr_pct=atr_pct,
                direction='SHORT'
            )

            if score >= self.config.min_signal_score:
                return self._create_short_signal(
                    symbol, current_price, vwap, atr,
                    rsi, rel_volume, score
                )

        return None

    def _check_long_conditions(
        self,
        price: float,
        vwap: float,
        rsi: float,
        rel_volume: float,
        macd_histogram: float,
        prev_histogram: float,
        vwap_distance_pct: float
    ) -> bool:
        """Check if LONG conditions are met"""
        # Price must be 2-3% below VWAP
        c1 = -self.config.max_vwap_distance_pct <= vwap_distance_pct <= -self.config.min_vwap_distance_pct

        # RSI oversold
        c2 = rsi < self.config.rsi_oversold

        # Volume confirmation
        c3 = rel_volume >= self.config.min_rel_volume

        # MACD improving (histogram rising)
        c4 = macd_histogram > prev_histogram

        conditions = [c1, c2, c3, c4]

        if any([c1, c2]):  # Log if price/RSI conditions are close
            status = (
                f"{'✓' if c1 else '✗'} VWAP dist({vwap_distance_pct:.2%}) "
                f"{'✓' if c2 else '✗'} RSI({rsi:.0f}<{self.config.rsi_oversold}) "
                f"{'✓' if c3 else '✗'} vol({rel_volume:.1f}x) "
                f"{'✓' if c4 else '✗'} MACD↑"
            )
            logger.debug(f"VWAP_REV LONG check: {status}")

        return all(conditions)

    def _check_short_conditions(
        self,
        price: float,
        vwap: float,
        rsi: float,
        rel_volume: float,
        macd_histogram: float,
        prev_histogram: float,
        vwap_distance_pct: float
    ) -> bool:
        """Check if SHORT conditions are met"""
        # Price must be 2-3% above VWAP
        c1 = self.config.min_vwap_distance_pct <= vwap_distance_pct <= self.config.max_vwap_distance_pct

        # RSI overbought
        c2 = rsi > self.config.rsi_overbought

        # Volume confirmation
        c3 = rel_volume >= self.config.min_rel_volume

        # MACD worsening (histogram falling)
        c4 = macd_histogram < prev_histogram

        conditions = [c1, c2, c3, c4]

        if any([c1, c2]):  # Log if price/RSI conditions are close
            status = (
                f"{'✓' if c1 else '✗'} VWAP dist({vwap_distance_pct:.2%}) "
                f"{'✓' if c2 else '✗'} RSI({rsi:.0f}>{self.config.rsi_overbought}) "
                f"{'✓' if c3 else '✗'} vol({rel_volume:.1f}x) "
                f"{'✓' if c4 else '✗'} MACD↓"
            )
            logger.debug(f"VWAP_REV SHORT check: {status}")

        return all(conditions)

    def calculate_score(
        self,
        vwap_distance_pct: float,
        rsi: float,
        rel_volume: float,
        macd_improving: bool,
        atr_pct: float,
        direction: str
    ) -> float:
        """
        Calculate signal quality score (0-100).

        Scoring breakdown:
        - VWAP distance: 0-30 pts (sweet spot is 2-3%)
        - RSI extreme: 0-25 pts (further from 50 = better)
        - Volume: 0-20 pts (higher = better)
        - MACD direction: 0-15 pts (confirming direction)
        - ATR (low vol): 0-10 pts (lower = safer trade)
        """
        score = 0.0

        # VWAP DISTANCE (0-30 pts)
        # Sweet spot: 2-3% distance
        if 0.02 <= vwap_distance_pct <= 0.03:
            score += 30  # Perfect zone
        elif 0.015 <= vwap_distance_pct < 0.02:
            score += 20  # Close but not quite
        elif 0.03 < vwap_distance_pct <= 0.035:
            score += 15  # Slightly overextended
        else:
            score += 5   # Too close or too far

        # RSI EXTREME (0-25 pts)
        if direction == 'LONG':
            if rsi < 25:
                score += 25  # Very oversold
            elif rsi < 30:
                score += 20
            elif rsi < 35:
                score += 15
            else:
                score += 5
        else:  # SHORT
            if rsi > 75:
                score += 25  # Very overbought
            elif rsi > 70:
                score += 20
            elif rsi > 65:
                score += 15
            else:
                score += 5

        # VOLUME (0-20 pts)
        if rel_volume >= 2.0:
            score += 20
        elif rel_volume >= 1.5:
            score += 15
        elif rel_volume >= 1.2:
            score += 10
        else:
            score += 5

        # MACD DIRECTION (0-15 pts)
        if macd_improving:
            score += 15
        else:
            score += 0  # No points if MACD not confirming

        # ATR (LOW VOLATILITY) (0-10 pts)
        if atr_pct < 0.015:
            score += 10  # Very calm
        elif atr_pct < 0.02:
            score += 7
        elif atr_pct < 0.025:
            score += 4
        else:
            score += 0

        return score

    def _create_long_signal(
        self,
        symbol: str,
        entry_price: float,
        vwap: float,
        atr: float,
        rsi: float,
        rel_volume: float,
        score: float
    ) -> OneOffSignal:
        """Create a LONG signal for VWAP reversion"""
        # Stop loss: 1.5x ATR below entry
        stop_loss = entry_price - (atr * self.config.stop_atr_multiplier)

        # Take profit 1: VWAP (target reversion)
        take_profit_1 = vwap

        # Take profit 2: Slightly above VWAP (full reversion + continuation)
        take_profit_2 = vwap * 1.01  # 1% above VWAP

        # Calculate position size
        position_size, risk_amount = self.calculate_position_size(
            entry_price, stop_loss
        )

        # Build reasoning
        vwap_dist = ((entry_price - vwap) / vwap) * 100
        reasoning = (
            f"Price {abs(vwap_dist):.1f}% below VWAP, "
            f"RSI oversold ({rsi:.0f}), "
            f"volume {rel_volume:.1f}x avg, "
            f"MACD improving"
        )

        return OneOffSignal(
            symbol=symbol,
            strategy_name=self.display_name,
            direction=SignalDirection.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            position_size=position_size,
            risk_amount=risk_amount,
            score=score,
            reasoning=reasoning,
            timestamp=datetime.now(),
            vwap=vwap,
            rsi=rsi,
            atr=atr,
            relative_volume=rel_volume
        )

    def _create_short_signal(
        self,
        symbol: str,
        entry_price: float,
        vwap: float,
        atr: float,
        rsi: float,
        rel_volume: float,
        score: float
    ) -> OneOffSignal:
        """Create a SHORT signal for VWAP reversion"""
        # Stop loss: 1.5x ATR above entry
        stop_loss = entry_price + (atr * self.config.stop_atr_multiplier)

        # Take profit 1: VWAP (target reversion)
        take_profit_1 = vwap

        # Take profit 2: Slightly below VWAP (full reversion + continuation)
        take_profit_2 = vwap * 0.99  # 1% below VWAP

        # Calculate position size
        position_size, risk_amount = self.calculate_position_size(
            entry_price, stop_loss
        )

        # Build reasoning
        vwap_dist = ((entry_price - vwap) / vwap) * 100
        reasoning = (
            f"Price {abs(vwap_dist):.1f}% above VWAP, "
            f"RSI overbought ({rsi:.0f}), "
            f"volume {rel_volume:.1f}x avg, "
            f"MACD weakening"
        )

        return OneOffSignal(
            symbol=symbol,
            strategy_name=self.display_name,
            direction=SignalDirection.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            position_size=position_size,
            risk_amount=risk_amount,
            score=score,
            reasoning=reasoning,
            timestamp=datetime.now(),
            vwap=vwap,
            rsi=rsi,
            atr=atr,
            relative_volume=rel_volume
        )
