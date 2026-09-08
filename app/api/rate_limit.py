"""In-memory sliding-window rate limiting for sensitive endpoints.

Applied to credential endpoints (``/auth/login``, ``/auth/token-info``) to
throttle brute-force / token-farming attempts. Each request that exceeds the
window budget is rejected with ``429 Too Many Requests`` plus a ``Retry-After``
header.

Deployment note: this is a *per-instance* throttle - sufficient for a single
replica or behind a sticky load balancer. Multi-node deployments should front
the API with a distributed limiter (nginx ``limit_req``, a Redis token bucket,
or an API gateway) and can disable this layer with
``AUTH_RATE_LIMIT_ENABLED=false``.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable
from threading import Lock

from fastapi import Depends, HTTPException, Request, status

from app.api.deps import get_container
from app.services.container import Container

_RETRY_GRANULARITY_SECONDS = 1


class RateLimiter:
    """Sliding-window request counter keyed by client identifier."""

    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> None:
        """Record an attempt for ``key``; raise 429 when over the budget."""
        now = time.monotonic()
        with self._lock:
            queue = self._hits[key]
            while queue and now - queue[0] > self._window:
                queue.popleft()
            if len(queue) >= self._max:
                retry_after = max(
                    _RETRY_GRANULARITY_SECONDS, int(self._window - (now - queue[0])) + 1
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Slow down and try again shortly.",
                    headers={"Retry-After": str(retry_after)},
                )
            queue.append(now)


def _limiter_for(container: Container, scope: str) -> RateLimiter:
    """Memoise one limiter per container, keeping state isolated per app."""
    attr = f"_rate_limiter_{scope}"
    limiter = getattr(container, attr, None)
    if limiter is None:
        settings = container.settings
        limiter = RateLimiter(
            max_requests=getattr(settings, "auth_rate_limit_max_requests", 10),
            window_seconds=getattr(settings, "auth_rate_limit_window_seconds", 60),
        )
        setattr(container, attr, limiter)
    return limiter


def rate_limit(scope: str) -> Callable:
    """Dependency factory: enforce the ``scope`` budget for the caller.

    Usage::

        @router.post("/login", dependencies=[Depends(rate_limit("login"))])
    """

    def dependency(request: Request, container: Container = Depends(get_container)) -> None:
        settings = container.settings
        if not getattr(settings, "auth_rate_limit_enabled", True):
            return
        client_host = request.client.host if request.client else "unknown"
        key = f"{scope}:{client_host}"
        _limiter_for(container, scope).check(key)

    return dependency
