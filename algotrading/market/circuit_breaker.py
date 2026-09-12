"""Circuit breaker pattern for external service calls.

Prevents hammering a failing service by tracking failures and temporarily
stopping requests when failure threshold is exceeded. Automatically recovers
after a timeout period.
"""
from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Any, Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation, requests go through
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenError(RuntimeError):
    """Raised when circuit breaker is open and rejects a call."""
    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message)


class CircuitBreaker:
    """Thread-safe circuit breaker for protecting external service calls.

    State transitions:
    - CLOSED -> OPEN: When failure_count >= fail_threshold
    - OPEN -> HALF_OPEN: After reset_timeout seconds
    - HALF_OPEN -> CLOSED: On successful call
    - HALF_OPEN -> OPEN: On failed call
    """

    def __init__(
        self,
        name: str,
        fail_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ) -> None:
        self.name = name
        self.fail_threshold = fail_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._check_state_transition()
            return self._state

    def is_open(self) -> bool:
        """Return True if circuit is open (rejecting calls)."""
        return self.state == CircuitState.OPEN

    def _check_state_transition(self) -> None:
        """Check and perform state transitions based on time."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                log.info("Circuit '%s': OPEN -> HALF_OPEN (timeout %.1fs elapsed)",
                         self.name, elapsed)
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute func with circuit breaker protection.

        Raises:
            CircuitOpenError: If circuit is open
            Any exception from func: Propagated after recording failure
        """
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open (failures: {self._failure_count}, "
                    f"next retry in {self.reset_timeout - (time.time() - self._last_failure_time):.0f}s)"
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' half-open limit reached"
                    )
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                log.info("Circuit '%s': HALF_OPEN -> CLOSED (successful call)", self.name)
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_calls = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # Reset on success

    def _on_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                log.warning("Circuit '%s': HALF_OPEN -> OPEN (call failed)", self.name)
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.fail_threshold:
                    log.warning(
                        "Circuit '%s': CLOSED -> OPEN (failures: %d >= threshold: %d)",
                        self.name, self._failure_count, self.fail_threshold
                    )
                    self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit to CLOSED state."""
        with self._lock:
            log.info("Circuit '%s': manually reset to CLOSED", self.name)
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self._last_failure_time = 0

    def get_status(self) -> dict[str, Any]:
        """Return current circuit status for monitoring."""
        with self._lock:
            self._check_state_transition()
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "fail_threshold": self.fail_threshold,
                "reset_timeout": self.reset_timeout,
                "time_since_last_failure": time.time() - self._last_failure_time
                if self._last_failure_time > 0 else None,
            }


# Global circuit breaker registry
_circuits: dict[str, CircuitBreaker] = {}
_circuits_lock = threading.Lock()


def get_circuit(name: str, **kwargs) -> CircuitBreaker:
    """Get or create a circuit breaker by name (singleton pattern)."""
    with _circuits_lock:
        if name not in _circuits:
            _circuits[name] = CircuitBreaker(name, **kwargs)
        return _circuits[name]


def reset_all_circuits() -> None:
    """Reset all registered circuit breakers (for testing)."""
    with _circuits_lock:
        for circuit in _circuits.values():
            circuit.reset()