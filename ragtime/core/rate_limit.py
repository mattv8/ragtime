"""
Rate limiting for authentication endpoints.

Uses slowapi to prevent brute-force attacks on login.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from ragtime.config.settings import settings

# Configure rate limiter using client IP address
# In production behind a reverse proxy, ensure X-Forwarded-For is set
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],  # No default limits - apply explicitly per endpoint
    enabled=not settings.debug_mode,  # Disable in debug mode for easier testing
)

# Rate limit constants
LOGIN_RATE_LIMIT = "5/minute"  # 5 login attempts per minute per IP
