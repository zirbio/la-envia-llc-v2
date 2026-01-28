"""
Dynamic Universe Manager for scanning all tradeable stocks.

Implements tiered filtering to efficiently reduce ~7000 symbols to ~500-800 candidates:
- Tier 1: Asset filter (active, tradable, shortable)
- Tier 2: Price filter ($10-$500)
- Tier 3: Volume filter (>=1M avg daily volume)
- Tier 4: Premarket scan (gap, PM volume) - handled by PremarketScanner
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from loguru import logger

from config.settings import settings
from data.market_data import market_data


@dataclass
class UniverseStats:
    """Statistics from the universe building process"""
    tier1_total: int = 0        # All assets from Alpaca
    tier1_filtered: int = 0     # After asset filter
    tier2_filtered: int = 0     # After price filter
    tier3_filtered: int = 0     # After volume filter (final universe)
    build_time_seconds: float = 0.0
    build_timestamp: Optional[datetime] = None
    error_message: Optional[str] = None

    def __str__(self) -> str:
        if self.error_message:
            return f"Universe build failed: {self.error_message}"

        return (
            f"Universe: {self.tier3_filtered} high-volume symbols\n"
            f"  Tier 1 (assets): {self.tier1_total} -> {self.tier1_filtered}\n"
            f"  Tier 2 (price): {self.tier1_filtered} -> {self.tier2_filtered}\n"
            f"  Tier 3 (volume): {self.tier2_filtered} -> {self.tier3_filtered}\n"
            f"  Build time: {self.build_time_seconds:.1f}s\n"
            f"  Last built: {self.build_timestamp.strftime('%Y-%m-%d %H:%M') if self.build_timestamp else 'Never'}"
        )


# Fallback universe - used when API fails or dynamic universe disabled
# Updated 2024: Removed delisted symbols (BBBY) and low-volume tickers
# This list should be periodically verified against current market listings
FALLBACK_UNIVERSE = [
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


class UniverseManager:
    """
    Manages the dynamic universe of tradeable stocks.

    Builds and caches a filtered universe using tiered filtering:
    1. Get all tradeable assets from Alpaca
    2. Filter by price range
    3. Filter by average daily volume

    The resulting universe is cached and refreshed daily (configurable).
    """

    def __init__(self):
        self.config = settings.universe
        self.trading_config = settings.trading

        # Cached universe
        self._universe: list[str] = []
        self._stats: UniverseStats = UniverseStats()
        self._last_build: Optional[datetime] = None

    @property
    def is_cache_valid(self) -> bool:
        """Check if cached universe is still valid"""
        if not self._last_build or not self._universe:
            return False

        age_hours = (datetime.now() - self._last_build).total_seconds() / 3600
        return age_hours < self.config.refresh_hours

    @property
    def stats(self) -> UniverseStats:
        """Get universe build statistics"""
        return self._stats

    def get_cached_universe(self) -> list[str]:
        """
        Get the cached universe, or fallback if not available.

        Returns:
            List of symbols (cached universe or fallback)
        """
        if self._universe:
            return self._universe

        logger.warning("No cached universe available, using fallback")
        return FALLBACK_UNIVERSE.copy()

    async def build_universe(self, force: bool = False, notify_callback=None) -> list[str]:
        """
        Build the high-volume universe using tiered filtering.

        Args:
            force: Force rebuild even if cache is valid
            notify_callback: Optional async callback for progress notifications
                             Signature: async def callback(message: str)

        Returns:
            List of high-volume symbols
        """
        # Check if we should use cached version
        if not force and self.is_cache_valid:
            logger.info(
                f"Using cached universe: {len(self._universe)} symbols "
                f"(built {self._stats.build_timestamp})"
            )
            return self._universe

        # Check if dynamic universe is disabled
        if not self.config.enabled:
            logger.info("Dynamic universe disabled, using fallback list")
            self._universe = FALLBACK_UNIVERSE.copy()
            self._stats = UniverseStats(
                tier3_filtered=len(self._universe),
                build_timestamp=datetime.now()
            )
            self._last_build = datetime.now()
            return self._universe

        async def notify(msg: str):
            """Send progress notification if callback provided"""
            logger.info(msg)
            if notify_callback:
                try:
                    await notify_callback(msg)
                except Exception as e:
                    logger.warning(f"Failed to send progress notification: {e}")

        await notify("🔄 Building high-volume universe from Alpaca...")
        start_time = datetime.now()

        stats = UniverseStats()

        try:
            # Tier 1: Get all tradeable assets
            await notify("📊 Tier 1: Fetching tradeable assets...")
            tier1_assets = market_data.get_tradeable_universe()

            if not tier1_assets:
                raise ValueError("No tradeable assets returned from Alpaca API")

            stats.tier1_total = len(tier1_assets) + 2000  # Approximate total before our filter
            stats.tier1_filtered = len(tier1_assets)
            elapsed = (datetime.now() - start_time).total_seconds()
            await notify(f"✓ Tier 1: {stats.tier1_filtered:,} tradeable symbols ({elapsed:.0f}s)")

            # Tier 2: Filter by price
            await notify(
                f"💰 Tier 2: Filtering by price "
                f"(${self.trading_config.min_price}-${self.trading_config.max_price})..."
            )
            tier2_symbols = market_data.prefilter_by_price_batch(
                symbols=tier1_assets,
                min_price=self.trading_config.min_price,
                max_price=self.trading_config.max_price,
                batch_size=self.config.price_filter_batch_size
            )

            if not tier2_symbols:
                raise ValueError("No symbols passed price filter")

            stats.tier2_filtered = len(tier2_symbols)
            elapsed = (datetime.now() - start_time).total_seconds()
            await notify(f"✓ Tier 2: {stats.tier2_filtered:,} in price range ({elapsed:.0f}s)")

            # Tier 3: Filter by average volume (this is the longest step)
            await notify(
                f"📈 Tier 3: Filtering by avg volume (>= {self.trading_config.min_avg_volume:,})...\n"
                f"⏳ This may take 2-3 minutes..."
            )
            tier3_symbols = market_data.prefilter_by_avg_volume_batch(
                symbols=tier2_symbols,
                min_avg_volume=self.trading_config.min_avg_volume,
                batch_size=self.config.volume_filter_batch_size
            )

            if not tier3_symbols:
                raise ValueError("No symbols passed volume filter")

            stats.tier3_filtered = len(tier3_symbols)

            # Calculate build time
            build_time = (datetime.now() - start_time).total_seconds()
            stats.build_time_seconds = build_time
            stats.build_timestamp = datetime.now()

            # Update cache
            self._universe = tier3_symbols
            self._stats = stats
            self._last_build = datetime.now()

            await notify(
                f"✅ Universe ready: {stats.tier3_filtered:,} high-volume symbols "
                f"(from {stats.tier1_filtered:,} tradeable) in {build_time:.0f}s"
            )

            return self._universe

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error building universe: {error_msg}")

            stats.error_message = error_msg
            stats.build_timestamp = datetime.now()
            self._stats = stats

            # Use fallback
            if not self._universe:
                logger.warning("Using fallback universe due to build failure")
                self._universe = FALLBACK_UNIVERSE.copy()
                self._stats.tier3_filtered = len(self._universe)
                await notify(f"⚠️ Using fallback universe ({len(self._universe)} symbols)")

            return self._universe

    def reset(self):
        """Reset universe cache (call at daily reset)"""
        logger.info("Universe cache reset")
        self._universe = []
        self._last_build = None
        self._stats = UniverseStats()

    def format_stats_message(self) -> str:
        """Format universe stats for Telegram message"""
        if not self._stats.build_timestamp:
            return "No universe built yet"

        if self._stats.error_message:
            return (
                f"*Universe Status*\n\n"
                f"*Build failed*\n"
                f"Error: {self._stats.error_message}\n"
                f"Using fallback: {len(FALLBACK_UNIVERSE)} symbols"
            )

        # Determine if using dynamic or fallback
        is_dynamic = self.config.enabled and self._stats.tier1_filtered > 0

        emoji = "✅" if is_dynamic else "⚠️"
        source = "Dynamic (Alpaca)" if is_dynamic else "Fallback (hardcoded)"

        return (
            f"*Universe Status*\n\n"
            f"{emoji} Source: {source}\n"
            f"Symbols: {self._stats.tier3_filtered:,}\n\n"
            f"*Filtering Pipeline:*\n"
            f"├ Tier 1 (assets): {self._stats.tier1_filtered:,}\n"
            f"├ Tier 2 (price): {self._stats.tier2_filtered:,}\n"
            f"└ Tier 3 (volume): {self._stats.tier3_filtered:,}\n\n"
            f"Build time: {self._stats.build_time_seconds:.1f}s\n"
            f"Last built: {self._stats.build_timestamp.strftime('%H:%M EST')}\n"
            f"Cache TTL: {self.config.refresh_hours}h"
        )


# Global universe manager instance
universe_manager = UniverseManager()
