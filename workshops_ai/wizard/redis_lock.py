# workshops_ai/wizard/redis_lock.py
"""
Redis distributed lock implementation for preventing race conditions.

Used to prevent duplicate work when multiple requests try to scrape
the same URL simultaneously.
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class LockAcquisitionError(Exception):
    """Raised when Redis lock cannot be acquired."""

    def __init__(self, lock_key: str, timeout: int):
        self.lock_key = lock_key
        self.timeout = timeout
        super().__init__(
            f"Failed to acquire lock '{lock_key}' within {timeout}s"
        )


@asynccontextmanager
async def redis_lock(
    redis_client: Redis,
    lock_key: str,
    timeout: int = 30,
    blocking_timeout: int | None = None,
) -> AsyncGenerator[str, None]:
    """
    Distributed lock using Redis SET with NX and EX.

    Prevents duplicate work (e.g., multiple requests scraping same URL).

    Args:
        redis_client: Redis client instance
        lock_key: Redis key for the lock (e.g., "wizard:lock:scrape:example.com")
        timeout: Lock expiration time in seconds (prevents deadlock if holder crashes)
        blocking_timeout: How long to wait for lock acquisition (None = fail immediately)

    Yields:
        lock_value: Unique lock identifier (UUID) for safe release

    Raises:
        LockAcquisitionError: If lock cannot be acquired within blocking_timeout

    Example:
        ```python
        async with redis_lock(redis, "wizard:lock:scrape:example.com", timeout=30) as lock_value:
            # Critical section - only one request can be here at a time
            content = await scrape_website(url)
            await cache_content(url, content)
        # Lock automatically released here
        ```

    Implementation notes:
        - Uses SET with NX (only set if not exists) and EX (expiration)
        - Lock value is a UUID to prevent accidental release by wrong holder
        - Lock auto-expires after `timeout` seconds to prevent deadlock
        - Blocking mode polls every 100ms if blocking_timeout is set
    """
    # Generate unique lock value to prevent accidental release by wrong holder
    lock_value = str(uuid.uuid4())

    # Try to acquire lock
    acquired = False
    wait_time = 0.0

    while not acquired:
        # SET key value NX EX timeout
        # Returns True if key was set (lock acquired), False if key already exists
        acquired = await redis_client.set(
            lock_key,
            lock_value,
            nx=True,  # Only set if not exists
            ex=timeout,  # Expire after timeout seconds
        )

        if acquired:
            logger.info(
                "redis_lock_acquired",
                extra={
                    "lock_key": lock_key,
                    "lock_value": lock_value,
                    "timeout_sec": timeout,
                }
            )
            break

        # Lock not acquired
        if blocking_timeout is None:
            # Fail immediately if not blocking
            logger.warning(
                "redis_lock_acquisition_failed",
                extra={
                    "lock_key": lock_key,
                    "reason": "lock_already_held",
                }
            )
            raise LockAcquisitionError(lock_key, timeout)

        # Blocking mode - wait and retry
        if wait_time >= blocking_timeout:
            logger.warning(
                "redis_lock_acquisition_timeout",
                extra={
                    "lock_key": lock_key,
                    "blocking_timeout_sec": blocking_timeout,
                    "wait_time_sec": wait_time,
                }
            )
            raise LockAcquisitionError(lock_key, blocking_timeout)

        # Wait 100ms before retry
        import asyncio
        await asyncio.sleep(0.1)
        wait_time += 0.1

    try:
        # Lock acquired - yield control to caller
        yield lock_value

    finally:
        # Release lock only if we still own it
        # Use Lua script to ensure atomic check-and-delete
        release_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            released = await redis_client.eval(
                release_script,
                1,  # Number of keys
                lock_key,  # KEYS[1]
                lock_value,  # ARGV[1]
            )

            if released:
                logger.info(
                    "redis_lock_released",
                    extra={
                        "lock_key": lock_key,
                        "lock_value": lock_value,
                    }
                )
            else:
                # Lock was already released (probably expired)
                logger.warning(
                    "redis_lock_already_released",
                    extra={
                        "lock_key": lock_key,
                        "lock_value": lock_value,
                        "reason": "lock_expired_or_stolen",
                    }
                )

        except Exception as e:
            logger.error(
                "redis_lock_release_failed",
                extra={
                    "lock_key": lock_key,
                    "lock_value": lock_value,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            # Don't raise - lock will expire naturally
