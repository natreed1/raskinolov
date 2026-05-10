#!/usr/bin/env python3
"""Auth primitives for control-plane worker registration."""

from __future__ import annotations

import hashlib
import hmac
from typing import Optional


def hash_worker_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_worker_token(token: str, expected_hash: Optional[str]) -> bool:
    if not expected_hash:
        return False
    return hmac.compare_digest(hash_worker_token(token), expected_hash)

