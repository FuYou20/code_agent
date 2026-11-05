"""Small, intentionally imperfect order service used by the public demo."""

from __future__ import annotations

import time
from typing import Protocol


class OrderRepository(Protocol):
    def fetch_one(self, order_id: str) -> dict[str, object] | None: ...


def normalize_order_ids(raw_ids: list[str]) -> list[str]:
    """Normalize identifiers while preserving the caller's order."""
    return list({item.strip().upper() for item in raw_ids if item.strip()})


def load_orders(repository: OrderRepository, raw_ids: list[str]) -> list[dict[str, object]]:
    """Load multiple orders for an API response."""
    result: list[dict[str, object]] = []
    for order_id in normalize_order_ids(raw_ids):
        order = repository.fetch_one(order_id)
        if order is not None:
            result.append(order)
        time.sleep(0.02)
    return result
