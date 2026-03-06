"""Session management and metrics tracking for MCP server.

This module provides session tracking, metrics collection, and
rate limiting functionality for the MCP server to help diagnose
and manage client connection behavior.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about an active session.

    Attributes:
        session_id: Unique session identifier.
        client_ip: IP address of the connecting client.
        created_at: Timestamp when session was created.
        last_activity: Timestamp of last activity.
        request_count: Number of requests processed.

    """

    session_id: str
    client_ip: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    request_count: int = 0

    def update_activity(self) -> None:
        """Update the last activity timestamp and increment request count."""
        self.last_activity = time.time()
        self.request_count += 1

    @property
    def duration_seconds(self) -> float:
        """Calculate session duration in seconds.

        Returns:
            Total duration since session creation.
        """
        return time.time() - self.created_at


@dataclass
class SessionMetrics:
    """Metrics for session tracking.

    Attributes:
        total_created: Total number of sessions created.
        total_destroyed: Total number of sessions destroyed.
        active_count: Current number of active sessions.
        total_requests: Total number of requests processed.
        peak_concurrent: Peak number of concurrent sessions.
        connection_history: Recent connection timestamps.

    """

    total_created: int = 0
    total_destroyed: int = 0
    active_count: int = 0
    total_requests: int = 0
    peak_concurrent: int = 0
    connection_history: deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def record_connection(self) -> None:
        """Record a new connection in the history."""
        self.connection_history.append(time.time())

    def get_connection_rate(self, window_seconds: int = 60) -> float:
        """Calculate connection rate over a time window.

        Args:
            window_seconds: Time window in seconds (default: 60).

        Returns:
            Average connections per second over the window.
        """
        if not self.connection_history:
            return 0.0

        cutoff = time.time() - window_seconds
        recent = [ts for ts in self.connection_history if ts >= cutoff]

        if not recent:
            return 0.0

        actual_window = time.time() - min(recent)
        if actual_window <= 0:
            return 0.0

        return len(recent) / actual_window


class SessionManager:
    """Manages session lifecycle and metrics tracking.

    This class tracks active sessions, records metrics, and provides
    rate limiting functionality to prevent resource exhaustion.

    Attributes:
        max_concurrent_sessions: Maximum number of concurrent sessions allowed.
        session_timeout_seconds: Timeout for inactive sessions.
        rate_limit_per_second: Maximum connections per second per IP.
        metrics: SessionMetrics instance for tracking.
        _sessions: Dictionary of active sessions by ID.
        _lock: Thread lock for thread-safe operations.

    """

    def __init__(  # noqa: PLR0913
        self,
        max_concurrent_sessions: int = 100,
        session_timeout_seconds: int = 300,
        rate_limit_per_second: float = 10.0,
        rate_limit_window: int = 60,
    ) -> None:
        """Initialize the session manager.

        Args:
            max_concurrent_sessions: Maximum concurrent sessions (default: 100).
            session_timeout_seconds: Session timeout in seconds (default: 300).
            rate_limit_per_second: Max connections per second per IP (default: 10).
            rate_limit_window: Rate limit window in seconds (default: 60).

        """
        _msg = "SessionManager initializing"
        log.debug(_msg)

        self.max_concurrent_sessions = max_concurrent_sessions
        self.session_timeout_seconds = session_timeout_seconds
        self.rate_limit_per_second = rate_limit_per_second
        self.rate_limit_window = rate_limit_window
        self.metrics = SessionMetrics()
        self._sessions: dict[str, SessionInfo] = {}
        self._lock = Lock()
        self._rate_limit_history: dict[str, deque[float]] = {}

        _msg = "SessionManager initialized"
        log.debug(_msg)

    def create_session(self, session_id: str, client_ip: str) -> bool:
        """Create a new session.

        Args:
            session_id: Unique session identifier.
            client_ip: IP address of the connecting client.

        Returns:
            True if session was created successfully, False if rate limited.

        """
        _msg = f"Creating session {session_id} for {client_ip}"
        log.debug(_msg)

        with self._lock:
            # Check rate limit
            if not self._check_rate_limit(client_ip):
                _msg = (
                    f"Rate limit exceeded for {client_ip}: "
                    f"{self.rate_limit_per_second}/sec over {self.rate_limit_window}s"
                )
                log.warning(_msg)
                return False

            # Check max concurrent sessions
            if len(self._sessions) >= self.max_concurrent_sessions:
                _msg = (
                    f"Max concurrent sessions reached ({self.max_concurrent_sessions}): "
                    f"rejecting session {session_id}"
                )
                log.warning(_msg)
                return False

            # Record rate limit hit
            self._record_rate_limit_hit(client_ip)

            # Create session
            session = SessionInfo(
                session_id=session_id,
                client_ip=client_ip,
            )
            self._sessions[session_id] = session

            # Update metrics
            self.metrics.total_created += 1
            self.metrics.active_count = len(self._sessions)
            self.metrics.record_connection()

            if self.metrics.active_count > self.metrics.peak_concurrent:
                self.metrics.peak_concurrent = self.metrics.active_count

        _msg = (
            f"Session created: {session_id} from {client_ip} "
            f"(active: {self.metrics.active_count})"
        )
        log.info(_msg)

        return True

    def destroy_session(self, session_id: str) -> None:
        """Destroy a session and log its lifecycle.

        Args:
            session_id: The session ID to destroy.

        """
        _msg = f"Destroying session {session_id}"
        log.debug(_msg)

        with self._lock:
            session = self._sessions.pop(session_id, None)

            if session:
                duration = session.duration_seconds
                self.metrics.total_destroyed += 1
                self.metrics.active_count = len(self._sessions)

                _msg = (
                    f"Session destroyed: {session_id} "
                    f"(duration: {duration:.2f}s, requests: {session.request_count})"
                )
                log.info(_msg)
            else:
                _msg = f"Session {session_id} not found for destruction"
                log.debug(_msg)

    def record_request(self, session_id: str) -> None:
        """Record a request for a session.

        Args:
            session_id: The session ID that made the request.

        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.update_activity()
                self.metrics.total_requests += 1

    def get_session_info(self, session_id: str) -> SessionInfo | None:
        """Get information about a specific session.

        Args:
            session_id: The session ID to look up.

        Returns:
            SessionInfo if found, None otherwise.

        """
        with self._lock:
            return self._sessions.get(session_id)

    def cleanup_expired_sessions(self) -> int:
        """Clean up sessions that have exceeded the timeout.

        Returns:
            Number of sessions cleaned up.
        """
        _msg = "Cleaning up expired sessions"
        log.debug(_msg)

        cutoff = time.time() - self.session_timeout_seconds
        expired: list[str] = []

        with self._lock:
            for session_id, session in self._sessions.items():
                if session.last_activity < cutoff:
                    expired.append(session_id)

            for session_id in expired:
                session = self._sessions.pop(session_id)
                self.metrics.total_destroyed += 1
                self.metrics.active_count = len(self._sessions)

                _msg = (
                    f"Session expired and removed: {session_id} "
                    f"(inactive for {self.session_timeout_seconds}s)"
                )
                log.info(_msg)

        _msg = f"Cleaned up {len(expired)} expired sessions"
        log.debug(_msg)

        return len(expired)

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check if a client has exceeded the rate limit.

        Args:
            client_ip: The client IP address to check.

        Returns:
            True if within rate limit, False if exceeded.

        """
        history = self._rate_limit_history.get(client_ip)

        if not history:
            return True

        cutoff = time.time() - self.rate_limit_window
        recent = [ts for ts in history if ts >= cutoff]

        if not recent:
            return True

        rate = len(recent) / self.rate_limit_window

        return rate < self.rate_limit_per_second

    def _record_rate_limit_hit(self, client_ip: str) -> None:
        """Record a connection attempt for rate limiting.

        Args:
            client_ip: The client IP address.

        """
        if client_ip not in self._rate_limit_history:
            self._rate_limit_history[client_ip] = deque(maxlen=10000)

        self._rate_limit_history[client_ip].append(time.time())

    def get_metrics(self) -> dict[str, Any]:
        """Get current session metrics.

        Returns:
            Dictionary with session metrics.
        """
        with self._lock:
            active_ips: dict[str, int] = {}
            for session in self._sessions.values():
                active_ips[session.client_ip] = active_ips.get(session.client_ip, 0) + 1

            return {
                "total_created": self.metrics.total_created,
                "total_destroyed": self.metrics.total_destroyed,
                "active_count": self.metrics.active_count,
                "total_requests": self.metrics.total_requests,
                "peak_concurrent": self.metrics.peak_concurrent,
                "connection_rate": self.metrics.get_connection_rate(),
                "active_sessions_by_ip": active_ips,
            }
