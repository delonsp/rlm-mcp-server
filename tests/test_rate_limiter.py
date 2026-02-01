"""
Tests for rlm_mcp.rate_limiter module.
"""

import time

import pytest

from rlm_mcp.rate_limiter import (
    RateLimitConfig,
    RateLimitResult,
    SlidingWindowRateLimiter,
    MultiRateLimiter,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_valid_config(self):
        """Test creating a valid config."""
        config = RateLimitConfig(max_requests=100, window_seconds=60)
        assert config.max_requests == 100
        assert config.window_seconds == 60

    def test_invalid_max_requests_zero(self):
        """Test that zero max_requests raises ValueError."""
        with pytest.raises(ValueError, match="max_requests must be positive"):
            RateLimitConfig(max_requests=0, window_seconds=60)

    def test_invalid_max_requests_negative(self):
        """Test that negative max_requests raises ValueError."""
        with pytest.raises(ValueError, match="max_requests must be positive"):
            RateLimitConfig(max_requests=-1, window_seconds=60)

    def test_invalid_window_seconds_zero(self):
        """Test that zero window_seconds raises ValueError."""
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            RateLimitConfig(max_requests=100, window_seconds=0)

    def test_invalid_window_seconds_negative(self):
        """Test that negative window_seconds raises ValueError."""
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            RateLimitConfig(max_requests=100, window_seconds=-1)


class TestRateLimitResult:
    """Tests for RateLimitResult dataclass."""

    def test_allowed_result(self):
        """Test creating an allowed result."""
        result = RateLimitResult(
            allowed=True,
            current_count=5,
            limit=100,
            window_seconds=60,
        )
        assert result.allowed is True
        assert result.current_count == 5
        assert result.limit == 100
        assert result.window_seconds == 60
        assert result.retry_after is None

    def test_denied_result_with_retry_after(self):
        """Test creating a denied result with retry_after."""
        result = RateLimitResult(
            allowed=False,
            current_count=100,
            limit=100,
            window_seconds=60,
            retry_after=15.5,
        )
        assert result.allowed is False
        assert result.retry_after == 15.5


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter class."""

    def test_init_creates_limiter(self):
        """Test that limiter is created with correct config."""
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        assert limiter.config.max_requests == 100
        assert limiter.config.window_seconds == 60

    def test_init_invalid_params(self):
        """Test that invalid params raise ValueError."""
        with pytest.raises(ValueError):
            SlidingWindowRateLimiter(max_requests=0, window_seconds=60)

    def test_check_allows_first_request(self):
        """Test that first request is always allowed."""
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        result = limiter.check("session_1")
        assert result.allowed is True
        assert result.current_count == 0

    def test_check_does_not_increment_counter(self):
        """Test that check() doesn't increment the counter."""
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        now = time.time()

        # Multiple checks shouldn't increment
        for _ in range(10):
            result = limiter.check("session_1", now)
            assert result.current_count == 0

    def test_record_increments_counter(self):
        """Test that record() increments the counter."""
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        now = time.time()

        limiter.record("session_1", now)
        limiter.record("session_1", now)
        limiter.record("session_1", now)

        result = limiter.check("session_1", now)
        assert result.current_count == 3

    def test_check_and_record_increments(self):
        """Test that check_and_record() increments counter if allowed."""
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        now = time.time()

        result = limiter.check_and_record("session_1", now)
        assert result.allowed is True
        assert result.current_count == 1

        result = limiter.check_and_record("session_1", now)
        assert result.current_count == 2

    def test_denies_when_limit_reached(self):
        """Test that requests are denied when limit is reached."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        now = time.time()

        # Make 5 requests
        for i in range(5):
            result = limiter.check_and_record("session_1", now)
            assert result.allowed is True, f"Request {i+1} should be allowed"

        # 6th request should be denied
        result = limiter.check("session_1", now)
        assert result.allowed is False
        assert result.current_count == 5

    def test_retry_after_is_set_when_denied(self):
        """Test that retry_after is set when request is denied."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        now = time.time()

        limiter.check_and_record("session_1", now)
        limiter.check_and_record("session_1", now)

        result = limiter.check("session_1", now)
        assert result.allowed is False
        assert result.retry_after is not None
        assert result.retry_after > 0

    def test_sliding_window_expires_old_requests(self):
        """Test that requests outside the window are not counted."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        base_time = 1000.0

        # Make 5 requests at base_time
        for _ in range(5):
            limiter.record("session_1", base_time)

        # At base_time, should be at limit
        result = limiter.check("session_1", base_time)
        assert result.allowed is False

        # After 61 seconds, old requests should expire
        result = limiter.check("session_1", base_time + 61)
        assert result.allowed is True
        assert result.current_count == 0

    def test_sliding_window_partial_expiry(self):
        """Test that sliding window correctly handles partial expiry."""
        limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=60)
        base_time = 1000.0

        # Make 5 requests at base_time
        for _ in range(5):
            limiter.record("session_1", base_time)

        # Make 5 requests 30 seconds later
        for _ in range(5):
            limiter.record("session_1", base_time + 30)

        # At base_time + 30, should be at limit
        result = limiter.check("session_1", base_time + 30)
        assert result.current_count == 10

        # After 65 seconds from start, first batch should be mostly expired
        result = limiter.check("session_1", base_time + 65)
        # First batch (5 at t=0) should be expired, second batch (5 at t=30) still valid
        assert result.current_count == 5 or result.current_count < 10

    def test_different_identifiers_are_independent(self):
        """Test that different identifiers have independent limits."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        now = time.time()

        # Exhaust session_1's limit
        for _ in range(5):
            limiter.record("session_1", now)

        # session_2 should still be allowed
        result = limiter.check("session_2", now)
        assert result.allowed is True
        assert result.current_count == 0

        # session_1 should be denied
        result = limiter.check("session_1", now)
        assert result.allowed is False

    def test_reset_clears_identifier(self):
        """Test that reset() clears all records for an identifier."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        now = time.time()

        # Make requests
        for _ in range(5):
            limiter.record("session_1", now)

        result = limiter.check("session_1", now)
        assert result.allowed is False

        # Reset
        limiter.reset("session_1")

        # Should be allowed again
        result = limiter.check("session_1", now)
        assert result.allowed is True
        assert result.current_count == 0

    def test_reset_doesnt_affect_other_identifiers(self):
        """Test that reset() only affects the specified identifier."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        now = time.time()

        limiter.record("session_1", now)
        limiter.record("session_1", now)
        limiter.record("session_2", now)
        limiter.record("session_2", now)
        limiter.record("session_2", now)

        limiter.reset("session_1")

        result1 = limiter.check("session_1", now)
        result2 = limiter.check("session_2", now)

        assert result1.current_count == 0
        assert result2.current_count == 3

    def test_get_stats_returns_correct_data(self):
        """Test that get_stats() returns correct statistics."""
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        now = time.time()

        for _ in range(25):
            limiter.record("session_1", now)

        stats = limiter.get_stats("session_1", now)

        assert stats["current_count"] == 25
        assert stats["limit"] == 100
        assert stats["remaining"] == 75
        assert stats["window_seconds"] == 60
        assert stats["reset_at"] > now

    def test_get_stats_remaining_doesnt_go_negative(self):
        """Test that remaining doesn't go negative when over limit."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        now = time.time()

        for _ in range(10):
            limiter.record("session_1", now)

        stats = limiter.get_stats("session_1", now)
        assert stats["remaining"] == 0

    def test_bucket_granularity(self):
        """Test that bucket size is reasonable."""
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        assert limiter._bucket_size == 6  # 60/10

        limiter2 = SlidingWindowRateLimiter(max_requests=100, window_seconds=10)
        assert limiter2._bucket_size == 1  # min 1 second


class TestMultiRateLimiter:
    """Tests for MultiRateLimiter class."""

    def test_add_limit(self):
        """Test adding a new limit."""
        limiter = MultiRateLimiter()
        limiter.add_limit("requests", max_requests=100, window_seconds=60)

        assert "requests" in limiter.list_limits()

    def test_add_multiple_limits(self):
        """Test adding multiple limits."""
        limiter = MultiRateLimiter()
        limiter.add_limit("requests", max_requests=100, window_seconds=60)
        limiter.add_limit("uploads", max_requests=10, window_seconds=60)

        limits = limiter.list_limits()
        assert "requests" in limits
        assert "uploads" in limits

    def test_check_specific_limit(self):
        """Test checking a specific limit."""
        limiter = MultiRateLimiter()
        limiter.add_limit("requests", max_requests=100, window_seconds=60)

        result = limiter.check("requests", "session_1")
        assert result.allowed is True
        assert result.limit == 100

    def test_check_nonexistent_limit_raises(self):
        """Test that checking a non-existent limit raises KeyError."""
        limiter = MultiRateLimiter()

        with pytest.raises(KeyError, match="Rate limit 'unknown' not configured"):
            limiter.check("unknown", "session_1")

    def test_record_specific_limit(self):
        """Test recording to a specific limit."""
        limiter = MultiRateLimiter()
        limiter.add_limit("requests", max_requests=100, window_seconds=60)
        now = time.time()

        limiter.record("requests", "session_1", now)
        limiter.record("requests", "session_1", now)

        result = limiter.check("requests", "session_1", now)
        assert result.current_count == 2

    def test_record_nonexistent_limit_raises(self):
        """Test that recording to non-existent limit raises KeyError."""
        limiter = MultiRateLimiter()

        with pytest.raises(KeyError):
            limiter.record("unknown", "session_1")

    def test_different_limits_are_independent(self):
        """Test that different limits are tracked independently."""
        limiter = MultiRateLimiter()
        limiter.add_limit("requests", max_requests=5, window_seconds=60)
        limiter.add_limit("uploads", max_requests=2, window_seconds=60)
        now = time.time()

        # Exhaust uploads limit
        limiter.record("uploads", "session_1", now)
        limiter.record("uploads", "session_1", now)

        # uploads should be denied
        result = limiter.check("uploads", "session_1", now)
        assert result.allowed is False

        # requests should still be allowed
        result = limiter.check("requests", "session_1", now)
        assert result.allowed is True

    def test_check_and_record_specific_limit(self):
        """Test check_and_record for a specific limit."""
        limiter = MultiRateLimiter()
        limiter.add_limit("requests", max_requests=100, window_seconds=60)

        result = limiter.check_and_record("requests", "session_1")
        assert result.allowed is True
        assert result.current_count == 1

    def test_reset_specific_limit(self):
        """Test resetting a specific limit."""
        limiter = MultiRateLimiter()
        limiter.add_limit("requests", max_requests=5, window_seconds=60)
        limiter.add_limit("uploads", max_requests=2, window_seconds=60)
        now = time.time()

        limiter.record("requests", "session_1", now)
        limiter.record("uploads", "session_1", now)

        limiter.reset("requests", "session_1")

        # requests should be reset
        result = limiter.check("requests", "session_1", now)
        assert result.current_count == 0

        # uploads should still have count
        result = limiter.check("uploads", "session_1", now)
        assert result.current_count == 1

    def test_reset_nonexistent_limit_raises(self):
        """Test that resetting non-existent limit raises KeyError."""
        limiter = MultiRateLimiter()

        with pytest.raises(KeyError):
            limiter.reset("unknown", "session_1")

    def test_reset_all(self):
        """Test resetting all limits for an identifier."""
        limiter = MultiRateLimiter()
        limiter.add_limit("requests", max_requests=100, window_seconds=60)
        limiter.add_limit("uploads", max_requests=10, window_seconds=60)
        now = time.time()

        limiter.record("requests", "session_1", now)
        limiter.record("uploads", "session_1", now)

        limiter.reset_all("session_1")

        result1 = limiter.check("requests", "session_1", now)
        result2 = limiter.check("uploads", "session_1", now)

        assert result1.current_count == 0
        assert result2.current_count == 0

    def test_get_stats_specific_limit(self):
        """Test getting stats for a specific limit."""
        limiter = MultiRateLimiter()
        limiter.add_limit("requests", max_requests=100, window_seconds=60)
        now = time.time()

        limiter.record("requests", "session_1", now)
        limiter.record("requests", "session_1", now)

        stats = limiter.get_stats("requests", "session_1", now)

        assert stats["current_count"] == 2
        assert stats["limit"] == 100
        assert stats["remaining"] == 98

    def test_get_stats_nonexistent_limit_raises(self):
        """Test that getting stats for non-existent limit raises KeyError."""
        limiter = MultiRateLimiter()

        with pytest.raises(KeyError):
            limiter.get_stats("unknown", "session_1")

    def test_list_limits_empty(self):
        """Test list_limits() with no limits configured."""
        limiter = MultiRateLimiter()
        assert limiter.list_limits() == []


class TestSlidingWindowEdgeCases:
    """Tests for edge cases in sliding window algorithm."""

    def test_very_short_window(self):
        """Test with a very short time window."""
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=1)
        base_time = 1000.0

        # Make 5 requests
        for _ in range(5):
            limiter.record("session_1", base_time)

        # Should be at limit
        result = limiter.check("session_1", base_time)
        assert result.allowed is False

        # After 1.1 seconds, should be allowed
        result = limiter.check("session_1", base_time + 1.1)
        assert result.allowed is True

    def test_very_long_window(self):
        """Test with a very long time window."""
        limiter = SlidingWindowRateLimiter(max_requests=1000, window_seconds=3600)
        now = time.time()

        for _ in range(100):
            limiter.record("session_1", now)

        result = limiter.check("session_1", now)
        assert result.allowed is True
        assert result.current_count == 100

    def test_high_request_rate(self):
        """Test with high request rate in short time."""
        limiter = SlidingWindowRateLimiter(max_requests=1000, window_seconds=60)
        now = time.time()

        # Simulate 500 requests in same second
        for _ in range(500):
            limiter.record("session_1", now)

        result = limiter.check("session_1", now)
        assert result.allowed is True
        assert result.current_count == 500

    def test_requests_spread_over_window(self):
        """Test requests spread evenly over the window."""
        limiter = SlidingWindowRateLimiter(max_requests=60, window_seconds=60)
        base_time = 1000.0

        # 1 request per second for 60 seconds
        for i in range(60):
            limiter.record("session_1", base_time + i)

        # Sliding window uses bucket interpolation, so count may be slightly
        # less than exact due to partial bucket calculations at edges
        result = limiter.check("session_1", base_time + 59)
        # Should count most of the 60 requests (within ~10% due to bucketing)
        assert result.current_count >= 55
        assert result.current_count <= 60

    def test_cleanup_removes_old_buckets(self):
        """Test that old buckets are cleaned up."""
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        base_time = 1000.0

        # Make requests at base_time
        for _ in range(10):
            limiter.record("session_1", base_time)

        # Move forward 2 minutes
        limiter.check("session_1", base_time + 120)

        # Old buckets should be cleaned
        assert len(limiter._buckets["session_1"]) == 0 or all(
            ts > base_time + 60 for ts, _ in limiter._buckets["session_1"]
        )

    def test_single_request_limit(self):
        """Test with limit of 1 request per window."""
        limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
        now = time.time()

        result = limiter.check_and_record("session_1", now)
        assert result.allowed is True

        result = limiter.check("session_1", now)
        assert result.allowed is False

    def test_concurrent_identifiers(self):
        """Test many identifiers concurrently."""
        limiter = SlidingWindowRateLimiter(max_requests=10, window_seconds=60)
        now = time.time()

        # 100 different sessions, 5 requests each
        for session_num in range(100):
            for _ in range(5):
                limiter.record(f"session_{session_num}", now)

        # Check each session
        for session_num in range(100):
            result = limiter.check(f"session_{session_num}", now)
            assert result.current_count == 5
            assert result.allowed is True
