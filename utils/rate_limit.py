"""
Rate limiting and retry utilities for API calls.

Implements:
- Exponential backoff with jitter for retries
- Rate limit header parsing (X-RateLimit-*)
- Automatic rate limit waiting

Alpaca API Rate Limits (as of 2024):
- Data API: 200 requests/minute for free tier, higher for paid
- Trading API: 200 requests/minute
- Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
"""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable, TypeVar, Any

from loguru import logger

# Type variable for generic retry function
T = TypeVar('T')

# Default configuration for exponential backoff
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 60.0  # seconds
DEFAULT_MAX_RETRIES = 5
DEFAULT_JITTER_FACTOR = 0.5  # Random factor between 0.5x and 1.5x


@dataclass
class RateLimitInfo:
    """
    Parsed rate limit information from API response headers.

    Attributes:
        limit: Maximum requests allowed in the window
        remaining: Requests remaining in current window
        reset_at: Unix timestamp when the window resets
        retry_after: Seconds to wait before retrying (if rate limited)
    """
    limit: Optional[int] = None
    remaining: Optional[int] = None
    reset_at: Optional[float] = None
    retry_after: Optional[float] = None

    @property
    def is_rate_limited(self) -> bool:
        """Check if currently rate limited (remaining = 0)"""
        return self.remaining is not None and self.remaining <= 0

    @property
    def seconds_until_reset(self) -> Optional[float]:
        """Seconds until rate limit window resets"""
        if self.reset_at is None:
            return None
        now = datetime.now(timezone.utc).timestamp()
        return max(0, self.reset_at - now)

    def __str__(self) -> str:
        if self.remaining is not None and self.limit is not None:
            pct = (self.remaining / self.limit) * 100 if self.limit > 0 else 0
            return f"RateLimit: {self.remaining}/{self.limit} ({pct:.0f}% remaining)"
        return "RateLimit: unknown"


def parse_rate_limit_headers(headers: dict) -> RateLimitInfo:
    """
    Parse rate limit information from HTTP response headers.

    Supports common header formats:
    - X-RateLimit-Limit / X-RateLimit-Remaining / X-RateLimit-Reset (Alpaca, GitHub)
    - RateLimit-Limit / RateLimit-Remaining / RateLimit-Reset (standard)
    - Retry-After (429 responses)

    Args:
        headers: HTTP response headers dict

    Returns:
        RateLimitInfo with parsed values
    """
    info = RateLimitInfo()

    # Try both X-RateLimit-* and RateLimit-* formats
    for prefix in ['X-RateLimit-', 'RateLimit-', 'x-ratelimit-', 'ratelimit-']:
        limit_key = f"{prefix}Limit" if prefix.startswith('X-') else f"{prefix}limit"
        remaining_key = f"{prefix}Remaining" if prefix.startswith('X-') else f"{prefix}remaining"
        reset_key = f"{prefix}Reset" if prefix.startswith('X-') else f"{prefix}reset"

        # Case-insensitive header lookup
        headers_lower = {k.lower(): v for k, v in headers.items()}

        if limit_key.lower() in headers_lower:
            try:
                info.limit = int(headers_lower[limit_key.lower()])
            except (ValueError, TypeError):
                pass

        if remaining_key.lower() in headers_lower:
            try:
                info.remaining = int(headers_lower[remaining_key.lower()])
            except (ValueError, TypeError):
                pass

        if reset_key.lower() in headers_lower:
            try:
                # Reset can be Unix timestamp or seconds until reset
                reset_val = float(headers_lower[reset_key.lower()])
                # If value is small, it's likely seconds; if large, it's a timestamp
                if reset_val < 86400 * 365:  # Less than a year in seconds
                    info.reset_at = datetime.now(timezone.utc).timestamp() + reset_val
                else:
                    info.reset_at = reset_val
            except (ValueError, TypeError):
                pass

    # Check for Retry-After header (used in 429 responses)
    retry_after = headers.get('Retry-After') or headers.get('retry-after')
    if retry_after:
        try:
            info.retry_after = float(retry_after)
        except (ValueError, TypeError):
            pass

    return info


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter_factor: float = DEFAULT_JITTER_FACTOR
) -> float:
    """
    Calculate exponential backoff delay with jitter.

    Formula: min(max_delay, base_delay * 2^attempt * random_jitter)

    Jitter prevents "thundering herd" when multiple clients retry simultaneously.

    Args:
        attempt: Current retry attempt (0-based)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        jitter_factor: Random factor (0.5 = between 0.5x and 1.5x)

    Returns:
        Delay in seconds with jitter applied
    """
    # Exponential backoff: base_delay * 2^attempt
    delay = base_delay * (2 ** attempt)

    # Apply jitter: random value between (1 - jitter_factor) and (1 + jitter_factor)
    jitter = 1 + (random.random() * 2 - 1) * jitter_factor
    delay *= jitter

    # Cap at max_delay
    return min(delay, max_delay)


def exponential_backoff_with_jitter(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter_factor: float = DEFAULT_JITTER_FACTOR
) -> float:
    """
    Alias for calculate_backoff_delay for backward compatibility.

    See calculate_backoff_delay for full documentation.
    """
    return calculate_backoff_delay(attempt, base_delay, max_delay, jitter_factor)


def should_retry(
    exception: Exception,
    status_code: Optional[int] = None
) -> bool:
    """
    Determine if a request should be retried based on the exception or status code.

    Retryable conditions:
    - Status codes: 429 (rate limit), 500-599 (server errors), 408 (timeout)
    - Connection errors, timeout errors
    - NOT retryable: 400, 401, 403, 404 (client errors)

    Args:
        exception: The exception that was raised
        status_code: HTTP status code if available

    Returns:
        True if the request should be retried
    """
    # Check status code first if available
    if status_code is not None:
        # Retryable status codes
        if status_code == 429:  # Rate limit
            return True
        if status_code == 408:  # Request timeout
            return True
        if 500 <= status_code < 600:  # Server errors
            return True
        # Client errors are not retryable
        if 400 <= status_code < 500:
            return False

    # Check exception type
    exception_str = str(type(exception).__name__).lower()
    error_msg = str(exception).lower()

    # Retryable exception types
    retryable_patterns = [
        'timeout', 'connection', 'network', 'temporary',
        'unavailable', 'overloaded', 'rate', 'throttl'
    ]

    for pattern in retryable_patterns:
        if pattern in exception_str or pattern in error_msg:
            return True

    return False


async def wait_for_rate_limit(
    rate_limit_info: RateLimitInfo,
    max_wait: float = 60.0
) -> float:
    """
    Wait until rate limit window resets (async version).

    Args:
        rate_limit_info: Parsed rate limit information
        max_wait: Maximum seconds to wait

    Returns:
        Actual seconds waited
    """
    wait_time = 0.0

    # Check retry_after first (most authoritative)
    if rate_limit_info.retry_after is not None:
        wait_time = min(rate_limit_info.retry_after, max_wait)
    # Fall back to seconds_until_reset
    elif rate_limit_info.seconds_until_reset is not None:
        wait_time = min(rate_limit_info.seconds_until_reset + 1, max_wait)  # +1 for safety
    # Default wait if rate limited but no timing info
    elif rate_limit_info.is_rate_limited:
        wait_time = min(10.0, max_wait)  # Conservative default

    if wait_time > 0:
        logger.warning(f"Rate limited, waiting {wait_time:.1f}s before retry")
        await asyncio.sleep(wait_time)

    return wait_time


def wait_for_rate_limit_sync(
    rate_limit_info: RateLimitInfo,
    max_wait: float = 60.0
) -> float:
    """
    Wait until rate limit window resets (sync version).

    Args:
        rate_limit_info: Parsed rate limit information
        max_wait: Maximum seconds to wait

    Returns:
        Actual seconds waited
    """
    wait_time = 0.0

    if rate_limit_info.retry_after is not None:
        wait_time = min(rate_limit_info.retry_after, max_wait)
    elif rate_limit_info.seconds_until_reset is not None:
        wait_time = min(rate_limit_info.seconds_until_reset + 1, max_wait)
    elif rate_limit_info.is_rate_limited:
        wait_time = min(10.0, max_wait)

    if wait_time > 0:
        logger.warning(f"Rate limited, waiting {wait_time:.1f}s before retry")
        time.sleep(wait_time)

    return wait_time


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter_factor: float = DEFAULT_JITTER_FACTOR,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs
) -> T:
    """
    Execute a function with automatic retry and exponential backoff.

    Args:
        func: Function to execute (can be sync or async)
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Initial backoff delay
        max_delay: Maximum backoff delay
        jitter_factor: Jitter randomization factor
        on_retry: Optional callback called on each retry with (attempt, exception, delay)
        **kwargs: Keyword arguments for func

    Returns:
        Result of func

    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            # Check if func is async
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            # Check if we should retry
            if attempt >= max_retries:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
                raise

            if not should_retry(e):
                logger.error(f"Non-retryable error: {e}")
                raise

            # Calculate backoff delay
            delay = calculate_backoff_delay(
                attempt, base_delay, max_delay, jitter_factor
            )

            # Log and optionally call retry callback
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )

            if on_retry:
                on_retry(attempt, e, delay)

            await asyncio.sleep(delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry_with_backoff")


def retry_with_backoff_sync(
    func: Callable[..., T],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter_factor: float = DEFAULT_JITTER_FACTOR,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs
) -> T:
    """
    Execute a sync function with automatic retry and exponential backoff.

    See retry_with_backoff for full documentation.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            last_exception = e

            if attempt >= max_retries:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
                raise

            if not should_retry(e):
                logger.error(f"Non-retryable error: {e}")
                raise

            delay = calculate_backoff_delay(
                attempt, base_delay, max_delay, jitter_factor
            )

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )

            if on_retry:
                on_retry(attempt, e, delay)

            time.sleep(delay)

    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected state in retry_with_backoff_sync")


class RateLimitTracker:
    """
    Track rate limit state across multiple API calls.

    Usage:
        tracker = RateLimitTracker()

        # After each API call
        tracker.update_from_headers(response.headers)

        # Before next call
        if tracker.should_wait():
            await tracker.wait_if_needed()
    """

    def __init__(
        self,
        safety_margin: float = 0.1,  # Keep 10% buffer
        min_remaining_warn: int = 10  # Warn when < 10 requests remaining
    ):
        self.current_info: Optional[RateLimitInfo] = None
        self.safety_margin = safety_margin
        self.min_remaining_warn = min_remaining_warn
        self._last_update = None

    def update_from_headers(self, headers: dict) -> RateLimitInfo:
        """Update rate limit state from response headers"""
        self.current_info = parse_rate_limit_headers(headers)
        self._last_update = datetime.now(timezone.utc)

        # Log warning if running low
        if (self.current_info.remaining is not None and
            self.current_info.remaining <= self.min_remaining_warn):
            logger.warning(
                f"Rate limit running low: {self.current_info}"
            )

        return self.current_info

    def should_wait(self) -> bool:
        """Check if we should wait before next request"""
        if self.current_info is None:
            return False

        # Wait if rate limited
        if self.current_info.is_rate_limited:
            return True

        # Wait if below safety margin
        if (self.current_info.remaining is not None and
            self.current_info.limit is not None and
            self.current_info.limit > 0):
            remaining_pct = self.current_info.remaining / self.current_info.limit
            if remaining_pct < self.safety_margin:
                return True

        return False

    async def wait_if_needed(self, max_wait: float = 60.0) -> float:
        """Wait if rate limited (async)"""
        if not self.should_wait() or self.current_info is None:
            return 0.0
        return await wait_for_rate_limit(self.current_info, max_wait)

    def wait_if_needed_sync(self, max_wait: float = 60.0) -> float:
        """Wait if rate limited (sync)"""
        if not self.should_wait() or self.current_info is None:
            return 0.0
        return wait_for_rate_limit_sync(self.current_info, max_wait)
