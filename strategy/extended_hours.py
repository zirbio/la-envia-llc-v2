"""
Extended Hours Trading Strategies

Premarket (4:00 AM - 9:30 AM EST): Gap & Go momentum strategy
Postmarket (4:00 PM - 8:00 PM EST): Earnings/news reaction strategy
"""
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional
from enum import Enum
import pytz
from loguru import logger

from config.settings import settings, TradingMode, get_session_params
from data.market_data import market_data
from data.indicators import calculate_atr, calculate_vwap, calculate_rsi


EST = pytz.timezone('US/Eastern')


class ExtendedSignalType(Enum):
    """Extended hours signal types"""
    GAP_LONG = "GAP_LONG"       # Gap up momentum continuation
    GAP_SHORT = "GAP_SHORT"     # Gap down momentum continuation
    FADE_LONG = "FADE_LONG"     # Fade gap down (reversal)
    FADE_SHORT = "FADE_SHORT"   # Fade gap up (reversal)
    NEWS_LONG = "NEWS_LONG"     # Bullish news reaction
    NEWS_SHORT = "NEWS_SHORT"   # Bearish news reaction
    NONE = "NONE"


@dataclass
class ExtendedHoursSignal:
    """Trading signal for extended hours sessions"""
    symbol: str
    signal_type: ExtendedSignalType
    entry_price: float
    limit_price: float  # Limit price for entry (required for extended hours)
    stop_loss: float
    take_profit: float
    position_size: int
    risk_amount: float
    session: str  # 'premarket' or 'postmarket'
    # Gap metrics
    gap_pct: float = 0.0
    premarket_high: float = 0.0
    premarket_low: float = 0.0
    premarket_volume: int = 0
    # Spread validation
    spread_pct: float = 0.0
    # News catalyst (postmarket)
    has_catalyst: bool = False
    catalyst_type: str = ""  # 'earnings', 'news', 'guidance', etc.
    # Signal quality
    signal_score: float = 0.0
    relative_volume: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(EST)

    def __str__(self) -> str:
        session_emoji = "🌅" if self.session == 'premarket' else "🌙"
        direction_emoji = "🟢" if 'LONG' in self.signal_type.value else "🔴"
        return (
            f"{session_emoji}{direction_emoji} {self.signal_type.value} {self.symbol}\n"
            f"Entry: ${self.entry_price:.2f} (limit ${self.limit_price:.2f})\n"
            f"Stop: ${self.stop_loss:.2f}\n"
            f"Target: ${self.take_profit:.2f}\n"
            f"Size: {self.position_size} shares\n"
            f"Gap: {self.gap_pct:+.1f}%\n"
            f"Spread: {self.spread_pct:.2%}\n"
            f"Score: {self.signal_score:.0f}/100"
        )


class PremarketStrategy:
    """
    Gap & Go Strategy for Premarket Trading (8:00 - 9:25 EST)

    Strategy:
    - Focus on stocks with significant gaps (3%+)
    - Trade momentum in direction of gap
    - Higher volume requirements (2.5x RVOL)
    - Wider stops (2.5x ATR)
    - 50% position size vs regular hours
    - Only limit orders with extended_hours=True
    """

    def __init__(self):
        self.config = settings.extended_hours
        self.trading_config = settings.trading
        self.premarket_ranges: dict[str, dict] = {}
        self.signals_today: list[ExtendedHoursSignal] = []
        self.trades_today: int = 0

    def calculate_premarket_range(self, symbol: str) -> Optional[dict]:
        """
        Calculate premarket high/low/volume for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with premarket range data or None
        """
        try:
            # Get premarket bars (from 4:00 AM)
            bars = market_data.get_bars(symbol, timeframe='1Min', limit=240)  # 4 hours

            if bars.empty:
                logger.warning(f"No premarket data for {symbol}")
                return None

            # Calculate range from all premarket bars
            # Note: bars are already filtered for premarket hours by the API
            high = bars['high'].max()
            low = bars['low'].min()
            volume = bars['volume'].sum()
            vwap = calculate_vwap(bars).iloc[-1] if len(bars) > 0 else 0

            # Get previous close for gap calculation
            prev_close = market_data.get_previous_close(symbol)
            if not prev_close:
                logger.warning(f"No previous close for {symbol}")
                return None

            gap_pct = ((high - prev_close) / prev_close) * 100

            range_data = {
                'symbol': symbol,
                'high': high,
                'low': low,
                'range': high - low,
                'volume': int(volume),
                'vwap': vwap,
                'prev_close': prev_close,
                'gap_pct': gap_pct,
                'timestamp': datetime.now(EST)
            }

            self.premarket_ranges[symbol] = range_data

            logger.info(
                f"Premarket range {symbol}: High=${high:.2f}, Low=${low:.2f}, "
                f"Gap={gap_pct:+.1f}%, Volume={volume:,.0f}"
            )

            return range_data

        except Exception as e:
            logger.error(f"Error calculating premarket range for {symbol}: {e}")
            return None

    def check_liquidity(self, symbol: str) -> tuple[bool, float, str]:
        """
        Check if symbol has sufficient liquidity for premarket trading.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (is_liquid, spread_pct, reason)
        """
        try:
            from execution.orders import order_executor
            is_valid, spread_pct, error = order_executor.validate_spread(
                symbol, self.config.premarket_max_spread_pct
            )

            if not is_valid:
                return False, spread_pct, error or "Spread too wide"

            # Check premarket volume
            range_data = self.premarket_ranges.get(symbol)
            if range_data:
                min_vol = settings.trading.min_premarket_volume
                if range_data['volume'] < min_vol:
                    return False, spread_pct, f"Volume {range_data['volume']:,} < {min_vol:,}"

            return True, spread_pct, "OK"

        except Exception as e:
            logger.error(f"Error checking liquidity for {symbol}: {e}")
            return False, 0.0, str(e)

    def check_gap_momentum(
        self,
        symbol: str,
        current_price: float,
        avg_volume: int
    ) -> Optional[ExtendedHoursSignal]:
        """
        Check for Gap & Go momentum signal.

        Strategy:
        - Price holding above VWAP for longs (gap up)
        - Price holding below VWAP for shorts (gap down)
        - High relative volume (2.5x+)
        - Gap of 3%+ from previous close

        Args:
            symbol: Stock symbol
            current_price: Current price
            avg_volume: Average daily volume

        Returns:
            ExtendedHoursSignal if signal detected, None otherwise
        """
        # Check trade limit
        if self.trades_today >= self.config.premarket_max_trades:
            logger.debug("Premarket trade limit reached")
            return None

        # Get premarket range
        range_data = self.premarket_ranges.get(symbol)
        if not range_data:
            self.calculate_premarket_range(symbol)
            range_data = self.premarket_ranges.get(symbol)
            if not range_data:
                return None

        gap_pct = range_data['gap_pct']
        vwap = range_data['vwap']
        pm_high = range_data['high']
        pm_low = range_data['low']
        pm_volume = range_data['volume']

        # Check minimum gap
        if abs(gap_pct) < self.config.premarket_min_gap_pct:
            logger.debug(f"{symbol}: Gap {gap_pct:.1f}% below minimum {self.config.premarket_min_gap_pct}%")
            return None

        # Check volume
        rel_volume = pm_volume / max(avg_volume * 0.1, 1)  # 10% of daily as PM benchmark
        if rel_volume < self.config.premarket_min_rvol:
            logger.debug(f"{symbol}: RVOL {rel_volume:.1f}x below minimum {self.config.premarket_min_rvol}x")
            return None

        # Check liquidity/spread
        is_liquid, spread_pct, reason = self.check_liquidity(symbol)
        if not is_liquid:
            logger.debug(f"{symbol}: Liquidity check failed - {reason}")
            return None

        # Determine signal direction
        signal_type = ExtendedSignalType.NONE
        score = 0.0

        if gap_pct > 0 and current_price > vwap:
            # Gap up, holding above VWAP - bullish
            signal_type = ExtendedSignalType.GAP_LONG
            score = self._calculate_gap_score(gap_pct, rel_volume, spread_pct, 'LONG')
        elif gap_pct < 0 and current_price < vwap:
            # Gap down, holding below VWAP - bearish
            signal_type = ExtendedSignalType.GAP_SHORT
            score = self._calculate_gap_score(abs(gap_pct), rel_volume, spread_pct, 'SHORT')

        if signal_type == ExtendedSignalType.NONE:
            return None

        # Check minimum score
        min_score = settings.trading.min_signal_score * 0.8  # 80% of regular threshold
        if score < min_score:
            logger.debug(f"{symbol}: Score {score:.0f} below minimum {min_score:.0f}")
            return None

        # Create signal
        return self._create_premarket_signal(
            symbol=symbol,
            signal_type=signal_type,
            current_price=current_price,
            range_data=range_data,
            rel_volume=rel_volume,
            spread_pct=spread_pct,
            score=score
        )

    def _calculate_gap_score(
        self,
        gap_pct: float,
        rel_volume: float,
        spread_pct: float,
        direction: str
    ) -> float:
        """
        Calculate signal quality score for premarket gap trade.

        Scoring thresholds are configurable via ExtendedHoursConfig.
        See config/settings.py for justification of each threshold.
        """
        score = 0.0
        cfg = self.config

        # Gap strength (0-30 pts)
        if gap_pct >= 10:
            score += cfg.gap_score_10pct
        elif gap_pct >= 7:
            score += cfg.gap_score_7pct
        elif gap_pct >= 5:
            score += cfg.gap_score_5pct
        elif gap_pct >= 3:
            score += cfg.gap_score_3pct

        # Volume (0-30 pts)
        if rel_volume >= 5.0:
            score += cfg.rvol_score_5x
        elif rel_volume >= 3.5:
            score += cfg.rvol_score_3_5x
        elif rel_volume >= 2.5:
            score += cfg.rvol_score_2_5x
        elif rel_volume >= 2.0:
            score += cfg.rvol_score_2x

        # Spread tightness (0-20 pts)
        if spread_pct <= 0.001:
            score += cfg.spread_score_0_1pct
        elif spread_pct <= 0.002:
            score += cfg.spread_score_0_2pct
        elif spread_pct <= 0.003:
            score += cfg.spread_score_0_3pct
        elif spread_pct <= 0.005:
            score += cfg.spread_score_0_5pct

        # Time bonus (0-20 pts) - better signals closer to market open
        now = datetime.now(EST)
        minutes_to_open = (9 * 60 + 30) - (now.hour * 60 + now.minute)
        if minutes_to_open <= 15:
            score += cfg.time_score_15min
        elif minutes_to_open <= 30:
            score += cfg.time_score_30min
        elif minutes_to_open <= 60:
            score += cfg.time_score_60min

        return score

    def _create_premarket_signal(
        self,
        symbol: str,
        signal_type: ExtendedSignalType,
        current_price: float,
        range_data: dict,
        rel_volume: float,
        spread_pct: float,
        score: float
    ) -> ExtendedHoursSignal:
        """Create a premarket trading signal."""
        # Get session params
        session_params = get_session_params(TradingMode.PREMARKET, self.config)

        # Calculate stop loss using ATR with premarket multiplier
        bars = market_data.get_bars(symbol, limit=20)
        atr = 0.0
        if not bars.empty and len(bars) >= 14:
            atr_series = calculate_atr(bars)
            atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0

        is_long = 'LONG' in signal_type.value

        if atr > 0:
            if is_long:
                stop_loss = current_price - (atr * session_params.stop_atr_mult)
            else:
                stop_loss = current_price + (atr * session_params.stop_atr_mult)
        else:
            # Fallback: use premarket range
            if is_long:
                stop_loss = range_data['low']
            else:
                stop_loss = range_data['high']

        risk_per_share = abs(current_price - stop_loss)
        take_profit = current_price + (risk_per_share * 2.0) if is_long else current_price - (risk_per_share * 2.0)

        # Calculate position size (reduced for extended hours)
        position_size, risk_amount = self._calculate_position_size(
            current_price, stop_loss, session_params.position_size_mult
        )

        # Limit price: slightly worse than current for execution probability
        if is_long:
            limit_price = current_price * 1.002  # 0.2% above for buy
        else:
            limit_price = current_price * 0.998  # 0.2% below for sell

        signal = ExtendedHoursSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=current_price,
            limit_price=round(limit_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            position_size=position_size,
            risk_amount=risk_amount,
            session='premarket',
            gap_pct=range_data['gap_pct'],
            premarket_high=range_data['high'],
            premarket_low=range_data['low'],
            premarket_volume=range_data['volume'],
            spread_pct=spread_pct,
            signal_score=score,
            relative_volume=rel_volume
        )

        self.signals_today.append(signal)
        self.trades_today += 1

        logger.info(f"Premarket signal: {signal}")
        return signal

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        size_multiplier: float
    ) -> tuple[int, float]:
        """Calculate position size for extended hours (reduced)."""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0, 0.0

        # Base risk from config, adjusted for extended hours
        max_risk = self.trading_config.max_capital * self.trading_config.risk_per_trade * size_multiplier
        position_size = int(max_risk / risk_per_share)

        # Ensure within capital limits
        max_shares = int(self.trading_config.max_capital * size_multiplier / entry_price)
        position_size = min(position_size, max_shares)
        position_size = max(position_size, 1)

        actual_risk = position_size * risk_per_share
        return position_size, actual_risk

    def reset_daily(self):
        """Reset daily data."""
        self.premarket_ranges.clear()
        self.signals_today.clear()
        self.trades_today = 0
        logger.info("Premarket strategy reset")


class PostmarketStrategy:
    """
    Earnings/News Reaction Strategy for Postmarket Trading (16:05 - 18:00 EST)

    Strategy:
    - Focus on stocks with earnings or significant news
    - Require catalyst (earnings beat/miss, guidance, news)
    - Higher movement threshold (5%+ move)
    - Very small position size (25% of regular)
    - Wider stops (3x ATR)
    - Force close by 19:30
    """

    def __init__(self):
        self.config = settings.extended_hours
        self.trading_config = settings.trading
        self.postmarket_data: dict[str, dict] = {}
        self.signals_today: list[ExtendedHoursSignal] = []
        self.trades_today: int = 0

    def check_earnings_play(
        self,
        symbol: str,
        current_price: float,
        prev_close: float,
        catalyst_type: str = "earnings"
    ) -> Optional[ExtendedHoursSignal]:
        """
        Check for earnings reaction trading opportunity.

        Args:
            symbol: Stock symbol
            current_price: Current postmarket price
            prev_close: Previous day close
            catalyst_type: Type of catalyst ('earnings', 'guidance', 'news')

        Returns:
            ExtendedHoursSignal if opportunity detected, None otherwise
        """
        if self.trades_today >= self.config.postmarket_max_trades:
            logger.debug("Postmarket trade limit reached")
            return None

        # Calculate move percentage
        move_pct = ((current_price - prev_close) / prev_close) * 100

        # Check minimum movement
        if abs(move_pct) < self.config.postmarket_min_move_pct:
            logger.debug(f"{symbol}: Move {move_pct:.1f}% below minimum {self.config.postmarket_min_move_pct}%")
            return None

        # Check liquidity
        from execution.orders import order_executor
        is_valid, spread_pct, error = order_executor.validate_spread(
            symbol, self.config.postmarket_max_spread_pct
        )
        if not is_valid:
            logger.debug(f"{symbol}: Spread check failed - {error}")
            return None

        # Determine direction
        if move_pct > 0:
            signal_type = ExtendedSignalType.NEWS_LONG
        else:
            signal_type = ExtendedSignalType.NEWS_SHORT

        # Calculate score
        score = self._calculate_news_score(abs(move_pct), spread_pct, catalyst_type)

        min_score = settings.trading.min_signal_score * 0.7  # 70% of regular for postmarket
        if score < min_score:
            logger.debug(f"{symbol}: Score {score:.0f} below minimum {min_score:.0f}")
            return None

        return self._create_postmarket_signal(
            symbol=symbol,
            signal_type=signal_type,
            current_price=current_price,
            prev_close=prev_close,
            move_pct=move_pct,
            spread_pct=spread_pct,
            score=score,
            catalyst_type=catalyst_type
        )

    def check_news_momentum(
        self,
        symbol: str,
        current_price: float,
        news_sentiment: float
    ) -> Optional[ExtendedHoursSignal]:
        """
        Check for news-driven momentum in postmarket.

        Args:
            symbol: Stock symbol
            current_price: Current price
            news_sentiment: Sentiment score from news (-1 to 1)

        Returns:
            ExtendedHoursSignal if opportunity detected
        """
        # Require catalyst if configured
        if self.config.postmarket_require_catalyst and abs(news_sentiment) < 0.5:
            logger.debug(f"{symbol}: News sentiment {news_sentiment:.2f} not strong enough")
            return None

        # Get previous close
        prev_close = market_data.get_previous_close(symbol)
        if not prev_close:
            return None

        move_pct = ((current_price - prev_close) / prev_close) * 100

        # Check if movement aligns with sentiment
        if (news_sentiment > 0 and move_pct < 0) or (news_sentiment < 0 and move_pct > 0):
            logger.debug(f"{symbol}: Movement doesn't align with sentiment")
            return None

        return self.check_earnings_play(symbol, current_price, prev_close, "news")

    def _calculate_news_score(
        self,
        move_pct: float,
        spread_pct: float,
        catalyst_type: str
    ) -> float:
        """
        Calculate signal quality score for postmarket news trade.

        Scoring thresholds are configurable via ExtendedHoursConfig.
        See config/settings.py for justification of each threshold.
        """
        score = 0.0
        cfg = self.config

        # Movement strength (0-40 pts)
        if move_pct >= 15:
            score += cfg.move_score_15pct
        elif move_pct >= 10:
            score += cfg.move_score_10pct
        elif move_pct >= 7:
            score += cfg.move_score_7pct
        elif move_pct >= 5:
            score += cfg.move_score_5pct

        # Spread tightness (0-30 pts)
        # Reuse spread scoring from premarket config (similar principles)
        if spread_pct <= 0.001:
            score += 30  # Postmarket needs even tighter spreads
        elif spread_pct <= 0.002:
            score += 20
        elif spread_pct <= 0.003:
            score += 10
        elif spread_pct <= 0.004:
            score += 5

        # Catalyst type (0-30 pts)
        # Catalyst scoring reflects reliability of the price move
        catalyst_scores = {
            'earnings': 30,      # Most reliable - actual financial results
            'guidance': 25,      # Forward-looking, affects sentiment
            'acquisition': 25,   # Major corporate event
            'fda': 25,           # Regulatory approval/rejection
            'upgrade': 20,       # Analyst action
            'downgrade': 20,     # Analyst action
            'news': 15,          # General news (less reliable)
        }
        score += catalyst_scores.get(catalyst_type.lower(), 10)

        return score

    def _create_postmarket_signal(
        self,
        symbol: str,
        signal_type: ExtendedSignalType,
        current_price: float,
        prev_close: float,
        move_pct: float,
        spread_pct: float,
        score: float,
        catalyst_type: str
    ) -> ExtendedHoursSignal:
        """Create a postmarket trading signal."""
        session_params = get_session_params(TradingMode.POSTMARKET, self.config)

        # Calculate stop using ATR
        bars = market_data.get_bars(symbol, limit=20)
        atr = 0.0
        if not bars.empty and len(bars) >= 14:
            atr_series = calculate_atr(bars)
            atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0

        is_long = 'LONG' in signal_type.value

        if atr > 0:
            if is_long:
                stop_loss = current_price - (atr * session_params.stop_atr_mult)
            else:
                stop_loss = current_price + (atr * session_params.stop_atr_mult)
        else:
            # Fallback: percentage-based stop
            stop_pct = 0.03  # 3%
            if is_long:
                stop_loss = current_price * (1 - stop_pct)
            else:
                stop_loss = current_price * (1 + stop_pct)

        risk_per_share = abs(current_price - stop_loss)
        take_profit = current_price + (risk_per_share * 1.5) if is_long else current_price - (risk_per_share * 1.5)

        # Calculate position size (very reduced for postmarket)
        position_size, risk_amount = self._calculate_position_size(
            current_price, stop_loss, session_params.position_size_mult
        )

        # Limit price
        if is_long:
            limit_price = current_price * 1.003  # 0.3% above for buy
        else:
            limit_price = current_price * 0.997  # 0.3% below for sell

        signal = ExtendedHoursSignal(
            symbol=symbol,
            signal_type=signal_type,
            entry_price=current_price,
            limit_price=round(limit_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            position_size=position_size,
            risk_amount=risk_amount,
            session='postmarket',
            gap_pct=move_pct,
            spread_pct=spread_pct,
            has_catalyst=True,
            catalyst_type=catalyst_type,
            signal_score=score
        )

        self.signals_today.append(signal)
        self.trades_today += 1

        logger.info(f"Postmarket signal: {signal}")
        return signal

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        size_multiplier: float
    ) -> tuple[int, float]:
        """Calculate position size for postmarket (very reduced)."""
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0, 0.0

        max_risk = self.trading_config.max_capital * self.trading_config.risk_per_trade * size_multiplier
        position_size = int(max_risk / risk_per_share)

        max_shares = int(self.trading_config.max_capital * size_multiplier / entry_price)
        position_size = min(position_size, max_shares)
        position_size = max(position_size, 1)

        actual_risk = position_size * risk_per_share
        return position_size, actual_risk

    def reset_daily(self):
        """Reset daily data."""
        self.postmarket_data.clear()
        self.signals_today.clear()
        self.trades_today = 0
        logger.info("Postmarket strategy reset")


def get_current_session() -> Optional[TradingMode]:
    """
    Determine current trading session based on time.

    Returns:
        TradingMode for current session, or None if outside trading hours
    """
    now = datetime.now(EST)
    current_time = now.time()

    premarket_start = time(4, 0)
    market_open = time(9, 30)
    market_close = time(16, 0)
    postmarket_end = time(20, 0)

    if premarket_start <= current_time < market_open:
        return TradingMode.PREMARKET
    elif market_open <= current_time < market_close:
        return TradingMode.REGULAR
    elif market_close <= current_time < postmarket_end:
        return TradingMode.POSTMARKET

    return None


def is_in_trading_window(mode: TradingMode) -> bool:
    """
    Check if current time is within the trading window for a mode.

    Args:
        mode: Trading mode to check

    Returns:
        True if within trading window
    """
    now = datetime.now(EST)
    current_time = now.time()
    ext_config = settings.extended_hours

    if mode == TradingMode.PREMARKET:
        start = time(*map(int, ext_config.premarket_trade_start.split(':')))
        end = time(*map(int, ext_config.premarket_trade_end.split(':')))
        return start <= current_time <= end

    elif mode == TradingMode.POSTMARKET:
        start = time(*map(int, ext_config.postmarket_trade_start.split(':')))
        end = time(*map(int, ext_config.postmarket_trade_end.split(':')))
        return start <= current_time <= end

    elif mode == TradingMode.REGULAR:
        return time(9, 30) <= current_time < time(16, 0)

    elif mode == TradingMode.ALL_SESSIONS:
        # In any tradeable window
        return (
            is_in_trading_window(TradingMode.PREMARKET) or
            is_in_trading_window(TradingMode.REGULAR) or
            is_in_trading_window(TradingMode.POSTMARKET)
        )

    return False


# Global strategy instances
premarket_strategy = PremarketStrategy()
postmarket_strategy = PostmarketStrategy()
