"""
Pre-market scanner to find trading candidates
"""
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from config.settings import settings
from data.market_data import market_data
from data.sentiment import sentiment_analyzer, SentimentResult


@dataclass
class PremktContext:
    """Premarket context data for context score calculation"""
    symbol: str
    premarket_high: float
    premarket_low: float
    prev_close: float
    open_price: float = 0.0  # Updated at 9:30

    def has_resistance_above(self, price: float, threshold_pct: float = 0.003) -> bool:
        """Check if premarket high is close above price (resistance)"""
        if self.premarket_high <= price:
            return False
        distance_pct = (self.premarket_high - price) / price
        return distance_pct <= threshold_pct

    def has_support_below(self, price: float, threshold_pct: float = 0.003) -> bool:
        """Check if premarket low is close below price (support)"""
        if self.premarket_low >= price:
            return False
        distance_pct = (price - self.premarket_low) / price
        return distance_pct <= threshold_pct


@dataclass
class ScanResult:
    """Result of scanning a single symbol"""
    symbol: str
    prev_close: float
    current_price: float
    gap_percent: float
    premarket_volume: int
    avg_daily_volume: int
    score: float = 0.0
    sentiment_score: float = 0.0
    sentiment_label: str = "Neutral"  # Alcista, Bajista, Neutral
    sentiment_news_count: int = 0
    # Premarket context for context score
    premarket_high: float = 0.0
    premarket_low: float = 0.0

    def __str__(self) -> str:
        direction = "+" if self.gap_percent > 0 else ""
        return (
            f"{self.symbol}: {direction}{self.gap_percent:.1f}% | "
            f"Vol: {self.premarket_volume:,} | Score: {self.score:.0f} | "
            f"Sent: {self.sentiment_label}"
        )


class PremarketScanner:
    """Scan for pre-market gap candidates"""

    def __init__(self):
        self.config = settings.trading
        self.sentiment_config = settings.sentiment
        self.candidates: list[ScanResult] = []
        # Cache for premarket context (used by strategy/orb.py for context score)
        self.premarket_context: dict[str, PremktContext] = {}

        # Configure sentiment analyzer with API key
        if self.sentiment_config.finnhub_api_key:
            sentiment_analyzer.finnhub_api_key = self.sentiment_config.finnhub_api_key
            logger.info("Sentiment analysis enabled with Finnhub API")
        else:
            logger.warning("Finnhub API key not configured - sentiment analysis limited")

    def scan_symbol(self, symbol: str) -> Optional[ScanResult]:
        """
        Scan a single symbol for pre-market criteria

        Args:
            symbol: Stock symbol to scan

        Returns:
            ScanResult if meets criteria, None otherwise
        """
        try:
            # Get pre-market data
            premarket = market_data.get_premarket_data(symbol)
            if not premarket:
                return None

            # Get average daily volume
            avg_volume = market_data.get_avg_daily_volume(symbol, days=20)

            # Apply filters
            gap_pct = abs(premarket['gap_percent'])
            price = premarket['current_price']
            pm_volume = premarket['premarket_volume']

            # Check minimum gap
            if gap_pct < self.config.min_gap_percent:
                return None

            # Check price range
            if price < self.config.min_price or price > self.config.max_price:
                return None

            # Check pre-market volume
            if pm_volume < self.config.min_premarket_volume:
                return None

            # Check average daily volume
            if avg_volume < self.config.min_avg_volume:
                return None

            # Calculate score (higher is better)
            score = self._calculate_score(
                gap_pct=gap_pct,
                pm_volume=pm_volume,
                avg_volume=avg_volume,
                price=price
            )

            # Get premarket High/Low for context score
            premarket_high = premarket.get('premarket_high', price)
            premarket_low = premarket.get('premarket_low', price)

            # Cache premarket context for later use in strategy
            self.premarket_context[symbol] = PremktContext(
                symbol=symbol,
                premarket_high=premarket_high,
                premarket_low=premarket_low,
                prev_close=premarket['prev_close'],
                open_price=0.0  # Will be updated at 9:30
            )

            return ScanResult(
                symbol=symbol,
                prev_close=premarket['prev_close'],
                current_price=price,
                gap_percent=premarket['gap_percent'],
                premarket_volume=pm_volume,
                avg_daily_volume=avg_volume,
                score=score,
                premarket_high=premarket_high,
                premarket_low=premarket_low
            )

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None

    def _calculate_score(
        self,
        gap_pct: float,
        pm_volume: int,
        avg_volume: int,
        price: float
    ) -> float:
        """
        Calculate a composite score for ranking candidates

        Higher score = better candidate
        """
        # Gap score (max 30 points)
        gap_score = min(gap_pct * 5, 30)

        # Pre-market volume score (max 30 points)
        volume_ratio = pm_volume / max(avg_volume * 0.1, 1)  # Compare to 10% of avg
        volume_score = min(volume_ratio * 10, 30)

        # Liquidity score based on avg volume (max 20 points)
        liquidity_score = min(avg_volume / 1_000_000 * 5, 20)

        # Price score - prefer mid-range prices (max 20 points)
        if 20 <= price <= 200:
            price_score = 20
        elif 10 <= price < 20 or 200 < price <= 300:
            price_score = 10
        else:
            price_score = 5

        return gap_score + volume_score + liquidity_score + price_score

    def scan_watchlist(self, symbols: list[str]) -> list[ScanResult]:
        """
        Scan a list of symbols and return sorted candidates

        Args:
            symbols: List of symbols to scan

        Returns:
            List of ScanResult sorted by score (highest first)
        """
        self.candidates = []

        logger.info(f"Scanning {len(symbols)} symbols for pre-market gaps...")

        for symbol in symbols:
            result = self.scan_symbol(symbol)
            if result:
                self.candidates.append(result)
                logger.debug(f"Found candidate: {result}")

        # Sort by score (highest first)
        self.candidates.sort(key=lambda x: x.score, reverse=True)

        # Limit to max watchlist size
        self.candidates = self.candidates[:self.config.max_watchlist_size]

        logger.info(f"Found {len(self.candidates)} candidates")
        return self.candidates

    async def scan_watchlist_with_sentiment(self, symbols: list[str]) -> list[ScanResult]:
        """
        Scan symbols and add sentiment analysis (async version)

        Args:
            symbols: List of symbols to scan

        Returns:
            List of ScanResult with sentiment scores, sorted by score
        """
        # First do the regular scan
        self.candidates = []

        logger.info(f"Scanning {len(symbols)} symbols for pre-market gaps...")

        for symbol in symbols:
            result = self.scan_symbol(symbol)
            if result:
                self.candidates.append(result)

        if not self.candidates:
            logger.info("No candidates found")
            return self.candidates

        # Add sentiment analysis if enabled
        if self.sentiment_config.enabled:
            logger.info(f"Analyzing sentiment for {len(self.candidates)} candidates...")

            try:
                candidate_symbols = [c.symbol for c in self.candidates]
                sentiment_results = await sentiment_analyzer.get_batch_sentiment(candidate_symbols)

                # Update candidates with sentiment
                for candidate in self.candidates:
                    if candidate.symbol in sentiment_results:
                        sent = sentiment_results[candidate.symbol]
                        candidate.sentiment_score = sent.score
                        candidate.sentiment_news_count = sent.news_count

                        # Set label (in Spanish) using signal level thresholds
                        signal_config = settings.trading.signal_config
                        if sent.score >= signal_config.max_sentiment_short:
                            candidate.sentiment_label = "Alcista"
                        elif sent.score <= signal_config.min_sentiment_long:
                            candidate.sentiment_label = "Bajista"
                        else:
                            candidate.sentiment_label = "Neutral"

                        # Boost/penalize score based on sentiment alignment with gap
                        sentiment_boost = self._calculate_sentiment_boost(
                            candidate.gap_percent, sent.score
                        )
                        candidate.score += sentiment_boost
            except Exception as e:
                logger.error(f"Error in batch sentiment analysis: {e}")
                # Continue with neutral sentiment for all candidates

        # Sort by score (highest first)
        self.candidates.sort(key=lambda x: x.score, reverse=True)

        # Limit to max watchlist size
        self.candidates = self.candidates[:self.config.max_watchlist_size]

        logger.info(f"Found {len(self.candidates)} candidates with sentiment")
        return self.candidates

    def _calculate_sentiment_boost(self, gap_percent: float, sentiment_score: float) -> float:
        """
        Calculate score boost/penalty based on sentiment alignment

        Args:
            gap_percent: Gap percentage (+ for gap up, - for gap down)
            sentiment_score: Sentiment score (-1 to 1)

        Returns:
            Score adjustment (positive = boost, negative = penalty)
        """
        # Gap up + bullish sentiment = good alignment (boost)
        # Gap up + bearish sentiment = bad alignment (penalty)
        # Gap down + bearish sentiment = good alignment (boost)
        # Gap down + bullish sentiment = bad alignment (penalty)

        if gap_percent > 0:
            # Gap up - want bullish sentiment
            return sentiment_score * 10  # Max +/- 10 points
        else:
            # Gap down - want bearish sentiment
            return -sentiment_score * 10  # Inverted: bearish = boost

    def get_top_candidates(self, n: int = 5) -> list[ScanResult]:
        """Get top N candidates by score"""
        return self.candidates[:n]

    def get_gappers_up(self) -> list[ScanResult]:
        """Get candidates with positive gap (gap up)"""
        return [c for c in self.candidates if c.gap_percent > 0]

    def get_gappers_down(self) -> list[ScanResult]:
        """Get candidates with negative gap (gap down)"""
        return [c for c in self.candidates if c.gap_percent < 0]

    def get_premarket_context(self, symbol: str) -> Optional[PremktContext]:
        """Get premarket context for a symbol (for context score calculation)"""
        return self.premarket_context.get(symbol)

    def update_open_prices(self):
        """Update open prices for all cached symbols (call at 9:30)"""
        for symbol, ctx in self.premarket_context.items():
            open_price = market_data.get_open_price(symbol)
            if open_price:
                ctx.open_price = open_price
                logger.debug(f"{symbol}: Updated open price to ${open_price:.2f}")

    def reset(self):
        """Reset scanner state for new day"""
        self.candidates.clear()
        self.premarket_context.clear()
        logger.info("Premarket scanner reset")

    async def scan_with_screener(self) -> list[ScanResult]:
        """
        Scan using Screener API as primary source (more efficient).

        Uses Alpaca's pre-calculated most active stocks and market movers,
        then gets snapshots for gap calculation. This reduces API calls from
        ~40 to ~3 while improving coverage.

        Flow:
        1. Get candidates from Screener API (most actives + market movers)
        2. Get snapshots for all candidates (ONE API call)
        3. Filter by gap and price criteria
        4. Add sentiment analysis
        5. Return sorted results

        Returns:
            List of ScanResult with sentiment scores, sorted by score
        """
        self.candidates = []
        screener_config = settings.universe

        logger.info("🔍 Using Screener API for candidate discovery...")

        try:
            # Step 1: Get candidates from Screener API
            screener_symbols = market_data.get_screener_candidates(
                min_price=self.config.min_price,
                max_price=self.config.max_price,
                top_actives=screener_config.screener_top_actives,
                top_movers=screener_config.screener_top_movers
            )

            if not screener_symbols:
                logger.warning("Screener API returned no candidates, falling back to tiered filtering")
                return await self._fallback_to_tiered_scan()

            logger.info(f"Screener: {len(screener_symbols)} candidates from most actives + movers")

            # Step 2: Get snapshots for all candidates (ONE API call)
            snapshots = market_data.get_snapshots_batch(screener_symbols)

            if not snapshots:
                logger.warning("Snapshot API returned no data, falling back to tiered filtering")
                return await self._fallback_to_tiered_scan()

            # Step 3: Filter by gap criteria and build candidates
            for symbol, data in snapshots.items():
                gap_pct = data['gap_pct']

                # Check minimum gap
                if abs(gap_pct) < self.config.min_gap_percent:
                    continue

                # Check price range (already filtered by screener, but double-check)
                price = data['price']
                if price < self.config.min_price or price > self.config.max_price:
                    continue

                # Skip if spread is too wide (liquidity concern)
                if data['spread_pct'] > 1.0:  # >1% spread
                    logger.debug(f"{symbol}: Spread too wide ({data['spread_pct']:.2f}%)")
                    continue

                # Calculate score
                score = self._calculate_score(
                    gap_pct=abs(gap_pct),
                    pm_volume=data['daily_volume'],
                    avg_volume=data['daily_volume'],  # Using daily volume as proxy
                    price=price
                )

                # Cache premarket context (we don't have premarket H/L from snapshots,
                # so use price as placeholder - will be refined later)
                self.premarket_context[symbol] = PremktContext(
                    symbol=symbol,
                    premarket_high=price * 1.01,  # Estimate +1%
                    premarket_low=price * 0.99,   # Estimate -1%
                    prev_close=data['prev_close'],
                    open_price=0.0
                )

                self.candidates.append(ScanResult(
                    symbol=symbol,
                    prev_close=data['prev_close'],
                    current_price=price,
                    gap_percent=gap_pct,
                    premarket_volume=data['daily_volume'],
                    avg_daily_volume=data['daily_volume'],
                    score=score,
                    premarket_high=price * 1.01,
                    premarket_low=price * 0.99
                ))

            logger.info(f"Gap filter: {len(self.candidates)} candidates with >= {self.config.min_gap_percent}% gap")

        except Exception as e:
            logger.error(f"Error in screener scan: {e}")
            return await self._fallback_to_tiered_scan()

        if not self.candidates:
            logger.info("No candidates found via screener")
            return self.candidates

        # Step 4: Add sentiment analysis if enabled
        if self.sentiment_config.enabled:
            logger.info(f"Analyzing sentiment for {len(self.candidates)} candidates...")
            try:
                candidate_symbols = [c.symbol for c in self.candidates]
                sentiment_results = await sentiment_analyzer.get_batch_sentiment(candidate_symbols)

                for candidate in self.candidates:
                    if candidate.symbol in sentiment_results:
                        sent = sentiment_results[candidate.symbol]
                        candidate.sentiment_score = sent.score
                        candidate.sentiment_news_count = sent.news_count

                        # Set label
                        signal_config = settings.trading.signal_config
                        if sent.score >= signal_config.max_sentiment_short:
                            candidate.sentiment_label = "Alcista"
                        elif sent.score <= signal_config.min_sentiment_long:
                            candidate.sentiment_label = "Bajista"
                        else:
                            candidate.sentiment_label = "Neutral"

                        # Boost score based on sentiment alignment
                        sentiment_boost = self._calculate_sentiment_boost(
                            candidate.gap_percent, sent.score
                        )
                        candidate.score += sentiment_boost
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")

        # Step 5: Sort and limit results
        self.candidates.sort(key=lambda x: x.score, reverse=True)
        self.candidates = self.candidates[:self.config.max_watchlist_size]

        logger.info(f"✅ Screener scan complete: {len(self.candidates)} final candidates")
        return self.candidates

    async def _fallback_to_tiered_scan(self) -> list[ScanResult]:
        """
        Fallback to tiered filtering when Screener API fails.

        Uses UniverseManager's cached universe or fallback list.
        """
        logger.warning("Falling back to tiered universe scan...")

        from scanner.universe import universe_manager
        universe = universe_manager.get_cached_universe()

        if not universe:
            logger.warning("No universe available, using fallback")
            universe = SCAN_UNIVERSE_FALLBACK

        return await self.scan_watchlist_with_sentiment(universe)

    async def scan_dynamic_universe(self) -> list[ScanResult]:
        """
        Scan the dynamic universe with automatic method selection.

        Uses Screener API when enabled (more efficient), otherwise falls back
        to tiered filtering from UniverseManager.

        Returns:
            List of ScanResult with sentiment scores, sorted by score
        """
        screener_config = settings.universe

        # Use Screener API if enabled (default: True)
        if screener_config.use_screener_api:
            return await self.scan_with_screener()

        # Fallback to tiered filtering
        from scanner.universe import universe_manager

        universe = universe_manager.get_cached_universe()

        if not universe:
            logger.warning("No universe available, using fallback")
            universe = SCAN_UNIVERSE_FALLBACK

        logger.info(f"Scanning dynamic universe: {len(universe)} symbols")

        return await self.scan_watchlist_with_sentiment(universe)

    def format_watchlist_message(self) -> str:
        """Format watchlist for Telegram message"""
        if not self.candidates:
            return "No candidates found matching criteria."

        lines = ["📋 *WATCHLIST*\n"]

        for i, c in enumerate(self.candidates, 1):
            emoji = "🟢" if c.gap_percent > 0 else "🔴"
            direction = "+" if c.gap_percent > 0 else ""

            # Sentiment emoji
            if c.sentiment_score >= 0.3:
                sent_emoji = "📈"
            elif c.sentiment_score <= -0.3:
                sent_emoji = "📉"
            else:
                sent_emoji = "➖"

            lines.append(
                f"{emoji} *{c.symbol}* {direction}{c.gap_percent:.1f}%\n"
                f"   Vol: {c.premarket_volume:,} | Score: {c.score:.0f}\n"
                f"   {sent_emoji} Sentimiento: {c.sentiment_label} ({c.sentiment_score:+.2f})"
            )

        return "\n".join(lines)


# Legacy static universe - kept for backward compatibility
# Use universe_manager.get_cached_universe() instead for dynamic universe
# Updated 2024: Removed delisted symbols (BBBY) and low-volume tickers
SCAN_UNIVERSE_FALLBACK = [
    # Mega caps (highly liquid, always tradeable)
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    "UNH", "JNJ", "XOM", "JPM", "V", "PG", "MA", "HD", "CVX",
    "MRK", "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "MCD",
    "WMT", "CSCO", "ACN", "ABT", "DHR", "NEE", "LIN", "TXN", "PM",
    "VZ", "ADBE", "CRM", "NKE", "RTX", "HON", "QCOM", "UNP", "ORCL",

    # Popular trading stocks (verified active as of 2024)
    "AMD", "INTC", "PYPL", "SQ", "SHOP", "ROKU", "SNAP", "PINS",
    "UBER", "LYFT", "ABNB", "COIN", "HOOD", "RIVN", "LCID", "NIO",
    "PLTR", "SOFI", "AFRM", "UPST", "DKNG", "PENN", "MGM", "WYNN",
    "GME", "AMC", "PLUG", "FCEL",  # Removed BBBY (delisted), BB, SPCE, BLNK (low volume)

    # Tech growth
    "NET", "CRWD", "ZS", "DDOG", "SNOW", "MDB", "OKTA", "ZM",
    "DOCU", "TWLO", "U", "RBLX", "DASH", "PATH", "MNDY",

    # Biotech/Healthcare
    "MRNA", "BNTX", "PFE", "BMY", "GILD", "REGN", "VRTX",

    # Financial
    "GS", "MS", "BAC", "C", "WFC", "SCHW", "BLK", "AXP",

    # Energy
    "OXY", "SLB", "HAL", "DVN", "EOG", "MPC", "VLO",  # Removed PXD (acquired by XOM)

    # AI/Semiconductor (added for current market relevance)
    "ARM", "SMCI", "MRVL", "MU", "LRCX", "AMAT", "KLAC",

    # ETFs for reference (high liquidity benchmarks)
    "SPY", "QQQ", "IWM", "DIA"
]

# Backward compatibility alias
SCAN_UNIVERSE = SCAN_UNIVERSE_FALLBACK


# Global scanner instance
premarket_scanner = PremarketScanner()
