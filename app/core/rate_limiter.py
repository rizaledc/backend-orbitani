"""
rate_limiter.py
In-memory role-based rate limiter for Orbitani API.

Rules:
  - role 'user'  : 5 requests per 60 seconds (5 RPM)
  - role 'admin' : unlimited
  - role 'superadmin': unlimited
"""
import time
import logging
from collections import deque
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# { user_id: deque([timestamp, timestamp, ...]) }
_request_log: dict[int, deque] = {}

USER_RPM_LIMIT = 5
WINDOW_SECONDS = 60


def check_rate_limit(user: dict) -> None:
    """
    Memeriksa apakah user boleh melakukan request.
    Melempar HTTP 429 jika limit tercapai.
    Dipanggil sebelum logika endpoint AI.
    """
    role = user.get("role", "user")

    # Admin dan superadmin tidak dibatasi
    if role in ("admin", "superadmin"):
        return

    user_id: int = user["id"]
    now = time.monotonic()

    if user_id not in _request_log:
        _request_log[user_id] = deque()

    # Buang timestamp yang sudah lebih dari 60 detik
    window = _request_log[user_id]
    while window and now - window[0] > WINDOW_SECONDS:
        window.popleft()

    if len(window) >= USER_RPM_LIMIT:
        oldest = window[0]
        retry_after = int(WINDOW_SECONDS - (now - oldest)) + 1
        logger.warning(
            "Rate limit hit: user_id=%s, requests=%d, retry_after=%ds",
            user_id, len(window), retry_after
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Batas permintaan tercapai (maks {USER_RPM_LIMIT} per menit untuk role '{role}'). "
                           f"Coba lagi dalam {retry_after} detik.",
                "retry_after_seconds": retry_after,
            },
            headers={"Retry-After": str(retry_after)},
        )

    # Rekam timestamp request ini
    window.append(now)
