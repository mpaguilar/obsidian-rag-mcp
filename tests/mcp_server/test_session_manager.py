"""Tests for session_manager module."""

from unittest.mock import MagicMock, patch

import pytest

from obsidian_rag.mcp_server.session_manager import (
    SessionInfo,
    SessionManager,
    SessionMetrics,
)


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_session_info_creation(self):
        """Test SessionInfo creation with default values."""
        session = SessionInfo(session_id="test-123", client_ip="192.168.1.1")

        assert session.session_id == "test-123"
        assert session.client_ip == "192.168.1.1"
        assert session.request_count == 0

    def test_session_info_update_activity(self):
        """Test update_activity increments count and updates timestamp."""
        session = SessionInfo(session_id="test-123", client_ip="192.168.1.1")
        original_activity = session.last_activity

        session.update_activity()

        assert session.request_count == 1
        assert session.last_activity >= original_activity

    def test_session_info_duration_seconds(self):
        """Test duration_seconds property."""
        session = SessionInfo(session_id="test-123", client_ip="192.168.1.1")

        duration = session.duration_seconds

        assert duration >= 0


class TestSessionMetrics:
    """Tests for SessionMetrics dataclass."""

    def test_metrics_defaults(self):
        """Test SessionMetrics has correct default values."""
        metrics = SessionMetrics()

        assert metrics.total_created == 0
        assert metrics.total_destroyed == 0
        assert metrics.active_count == 0
        assert metrics.total_requests == 0
        assert metrics.peak_concurrent == 0

    def test_record_connection(self):
        """Test record_connection adds timestamp to history."""
        metrics = SessionMetrics()

        metrics.record_connection()

        assert len(metrics.connection_history) == 1

    def test_get_connection_rate_empty(self):
        """Test get_connection_rate returns 0 for empty history."""
        metrics = SessionMetrics()

        rate = metrics.get_connection_rate()

        assert rate == 0.0

    def test_get_connection_rate_with_connections(self):
        """Test get_connection_rate calculates correctly."""
        metrics = SessionMetrics()
        import time

        # Add connection 1 second ago
        metrics.connection_history.append(time.time() - 1)

        rate = metrics.get_connection_rate(window_seconds=60)

        assert rate > 0

    def test_get_connection_rate_old_connections(self):
        """Test get_connection_rate ignores old connections."""
        metrics = SessionMetrics()
        import time

        # Add connection 2 minutes ago (outside 60 second window)
        metrics.connection_history.append(time.time() - 120)

        rate = metrics.get_connection_rate(window_seconds=60)

        assert rate == 0.0


class TestSessionManager:
    """Tests for SessionManager class."""

    def test_session_manager_creation(self):
        """Test SessionManager initialization with defaults."""
        manager = SessionManager()

        assert manager.max_concurrent_sessions == 100
        assert manager.session_timeout_seconds == 300
        assert manager.rate_limit_per_second == 10.0
        assert manager.rate_limit_window == 60
        assert manager.metrics.total_created == 0

    def test_session_manager_custom_settings(self):
        """Test SessionManager initialization with custom settings."""
        manager = SessionManager(
            max_concurrent_sessions=50,
            session_timeout_seconds=120,
            rate_limit_per_second=5.0,
            rate_limit_window=30,
        )

        assert manager.max_concurrent_sessions == 50
        assert manager.session_timeout_seconds == 120
        assert manager.rate_limit_per_second == 5.0
        assert manager.rate_limit_window == 30

    def test_create_session_success(self):
        """Test successful session creation."""
        manager = SessionManager()

        result = manager.create_session("session-1", "192.168.1.1")

        assert result is True
        assert manager.metrics.total_created == 1
        assert manager.metrics.active_count == 1

    def test_create_session_max_concurrent(self):
        """Test session creation fails when max concurrent reached."""
        manager = SessionManager(max_concurrent_sessions=2)
        manager.create_session("session-1", "192.168.1.1")
        manager.create_session("session-2", "192.168.1.1")

        result = manager.create_session("session-3", "192.168.1.1")

        assert result is False

    def test_create_session_rate_limit(self):
        """Test session creation fails when rate limit exceeded."""
        manager = SessionManager(
            rate_limit_per_second=0.01,  # Very low rate: 1 per 100 seconds
            rate_limit_window=60,
        )
        manager.create_session("session-1", "192.168.1.1")

        # Immediately try again - should fail due to rate limit
        result = manager.create_session("session-2", "192.168.1.1")

        assert result is False

    def test_create_session_different_ips(self):
        """Test rate limiting is per IP address."""
        manager = SessionManager(
            rate_limit_per_second=0.01,  # Very low rate
            rate_limit_window=60,
        )
        manager.create_session("session-1", "192.168.1.1")

        # Different IP should succeed even when first IP is rate limited
        result = manager.create_session("session-2", "192.168.1.2")

        assert result is True

    def test_destroy_session_success(self):
        """Test successful session destruction."""
        manager = SessionManager()
        manager.create_session("session-1", "192.168.1.1")

        manager.destroy_session("session-1")

        assert manager.metrics.total_destroyed == 1
        assert manager.metrics.active_count == 0

    def test_destroy_session_not_found(self):
        """Test destroying non-existent session does not raise error."""
        manager = SessionManager()

        manager.destroy_session("non-existent")

        # Should not raise exception
        assert manager.metrics.total_destroyed == 0

    def test_record_request(self):
        """Test recording a request for a session."""
        manager = SessionManager()
        manager.create_session("session-1", "192.168.1.1")

        manager.record_request("session-1")

        assert manager.metrics.total_requests == 1

    def test_record_request_not_found(self):
        """Test recording request for non-existent session."""
        manager = SessionManager()

        manager.record_request("non-existent")

        # Should not raise exception
        assert manager.metrics.total_requests == 0

    def test_get_session_info(self):
        """Test retrieving session information."""
        manager = SessionManager()
        manager.create_session("session-1", "192.168.1.1")

        session = manager.get_session_info("session-1")

        assert session is not None
        assert session.session_id == "session-1"
        assert session.client_ip == "192.168.1.1"

    def test_get_session_info_not_found(self):
        """Test retrieving info for non-existent session."""
        manager = SessionManager()

        session = manager.get_session_info("non-existent")

        assert session is None

    def test_cleanup_expired_sessions(self):
        """Test cleanup of expired sessions."""
        manager = SessionManager(session_timeout_seconds=1)
        manager.create_session("session-1", "192.168.1.1")

        import time

        time.sleep(1.1)

        cleaned = manager.cleanup_expired_sessions()

        assert cleaned == 1
        assert manager.metrics.active_count == 0

    def test_cleanup_no_expired_sessions(self):
        """Test cleanup when no sessions are expired."""
        manager = SessionManager(session_timeout_seconds=300)
        manager.create_session("session-1", "192.168.1.1")

        cleaned = manager.cleanup_expired_sessions()

        assert cleaned == 0
        assert manager.metrics.active_count == 1

    def test_get_metrics(self):
        """Test retrieving session metrics."""
        manager = SessionManager()
        manager.create_session("session-1", "192.168.1.1")
        manager.create_session("session-2", "192.168.1.1")

        metrics = manager.get_metrics()

        assert metrics["total_created"] == 2
        assert metrics["active_count"] == 2
        assert metrics["active_sessions_by_ip"]["192.168.1.1"] == 2

    def test_peak_concurrent_tracking(self):
        """Test peak concurrent sessions tracking."""
        manager = SessionManager()

        manager.create_session("session-1", "192.168.1.1")
        manager.create_session("session-2", "192.168.1.1")

        assert manager.metrics.peak_concurrent == 2

        manager.destroy_session("session-1")
        manager.create_session("session-3", "192.168.1.1")

        # Peak should remain at 2
        assert manager.metrics.peak_concurrent == 2

    @patch("obsidian_rag.mcp_server.session_manager.log")
    def test_create_session_logs(self, mock_log: MagicMock):
        """Test that session creation logs appropriate messages."""
        manager = SessionManager()

        manager.create_session("session-1", "192.168.1.1")

        # Verify info log was called for session creation
        info_calls = [call for call in mock_log.info.call_args_list]
        assert len(info_calls) >= 1

    @patch("obsidian_rag.mcp_server.session_manager.log")
    def test_destroy_session_logs(self, mock_log: MagicMock):
        """Test that session destruction logs appropriate messages."""
        manager = SessionManager()
        manager.create_session("session-1", "192.168.1.1")

        manager.destroy_session("session-1")

        # Verify info log was called for session destruction
        info_calls = [call for call in mock_log.info.call_args_list]
        assert len(info_calls) >= 2

    @patch("obsidian_rag.mcp_server.session_manager.log")
    def test_rate_limit_warning_logs(self, mock_log: MagicMock):
        """Test that rate limit exceeded logs warning."""
        manager = SessionManager(rate_limit_per_second=0.01, rate_limit_window=60)
        manager.create_session("session-1", "192.168.1.1")

        # Immediately try again - should trigger rate limit
        manager.create_session("session-2", "192.168.1.1")

        # Verify warning log was called
        warning_calls = [call for call in mock_log.warning.call_args_list]
        assert len(warning_calls) >= 1

    @patch("obsidian_rag.mcp_server.session_manager.log")
    def test_max_concurrent_warning_logs(self, mock_log: MagicMock):
        """Test that max concurrent exceeded logs warning."""
        manager = SessionManager(max_concurrent_sessions=1)
        manager.create_session("session-1", "192.168.1.1")

        # Try to create second session - should trigger max concurrent
        manager.create_session("session-2", "192.168.1.1")

        # Verify warning log was called
        warning_calls = [call for call in mock_log.warning.call_args_list]
        assert len(warning_calls) >= 1
