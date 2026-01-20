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

            return ScanResult(
                symbol=symbol,
                prev_close=premarket['prev_close'],
                current_price=price,
                gap_percent=premarket['gap_percent'],
                premarket_volume=pm_volume,
                avg_daily_volume=avg_volume,
                score=score
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

            candidate_symbols = [c.symbol for c in self.candidates]
            sentiment_results = await sentiment_analyzer.get_batch_sentiment(candidate_symbols)

            # Update candidates with sentiment
            for candidate in self.candidates:
                if candidate.symbol in sentiment_results:
                    sent = sentiment_results[candidate.symbol]
                    candidate.sentiment_score = sent.score
                    candidate.sentiment_news_count = sent.news_count

                    # Set label (in Spanish)
                    if sent.score >= 0.3:
                        candidate.sentiment_label = "Alcista"
                    elif sent.score <= -0.3:
                        candidate.sentiment_label = "Bajista"
                    else:
                        candidate.sentiment_label = "Neutral"

                    # Boost/penalize score based on sentiment alignment with gap
                    sentiment_boost = self._calculate_sentiment_boost(
                        candidate.gap_percent, sent.score
                    )
                    candidate.score += sentiment_boost

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


# Common stock universe for scanning
# Top traded stocks by volume - good for day trading
SCAN_UNIVERSE = [
    # Mega caps
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    "BRK.B", "UNH", "JNJ", "XOM", "JPM", "V", "PG", "MA", "HD", "CVX",
    "MRK", "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "MCD",
    "WMT", "CSCO", "ACN", "ABT", "DHR", "NEE", "LIN", "TXN", "PM",
    "VZ", "ADBE", "CRM", "NKE", "RTX", "HON", "QCOM", "UNP", "ORCL",

    # Popular trading stocks
    "AMD", "INTC", "PYPL", "SQ", "SHOP", "ROKU", "SNAP", "PINS",
    "UBER", "LYFT", "ABNB", "COIN", "HOOD", "RIVN", "LCID", "NIO",
    "PLTR", "SOFI", "AFRM", "UPST", "DKNG", "PENN", "MGM", "WYNN",
    "GME", "AMC", "BBBY", "BB", "SPCE", "PLUG", "FCEL", "BLNK",

    # Tech growth
    "NET", "CRWD", "ZS", "DDOG", "SNOW", "MDB", "OKTA", "ZM",
    "DOCU", "TWLO", "U", "RBLX", "DASH", "PATH", "MNDY",

    # Biotech/Healthcare
    "MRNA", "BNTX", "PFE", "JNJ", "BMY", "GILD", "REGN", "VRTX",

    # Financial
    "GS", "MS", "BAC", "C", "WFC", "SCHW", "BLK", "AXP",

    # Energy
    "OXY", "SLB", "HAL", "DVN", "PXD", "EOG", "MPC", "VLO",

    # ETFs for reference
    "SPY", "QQQ", "IWM", "DIA"
]


# Global scanner instance
premarket_scanner = PremarketScanner()
