"""Provide one concurrency-safe budget ledger shared by every agent."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
import uuid


class BudgetExceeded(Exception):
    """Raised when a reservation would exceed the shared global budget."""


class UnknownReservation(KeyError):
    """Raised when a reservation is missing or has already been settled."""


@dataclass(frozen=True)
class BudgetSnapshot:
    input_tokens: int
    output_tokens: int
    tool_calls: int


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    agent_id: str
    input_tokens: int
    output_tokens: int
    tool_calls: int


class GlobalBudgetLedger:
    """Atomically reserve and settle one budget shared across the agent tree."""

    def __init__(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        tool_calls: int,
    ) -> None:
        if min(input_tokens, output_tokens, tool_calls) < 0:
            raise ValueError("global budget values cannot be negative")
        self._remaining = BudgetSnapshot(input_tokens, output_tokens, tool_calls)
        self._reservations: dict[str, BudgetReservation] = {}
        self._lock = RLock()

    @staticmethod
    def _validate_amounts(input_tokens: int, output_tokens: int, tool_calls: int) -> None:
        if min(input_tokens, output_tokens, tool_calls) < 0:
            raise ValueError("budget amounts cannot be negative")

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._remaining

    def reserve(
        self,
        agent_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0,
    ) -> BudgetReservation:
        """Reserve maximum usage before starting a model or tool operation."""
        self._validate_amounts(input_tokens, output_tokens, tool_calls)
        with self._lock:
            remaining = self._remaining
            if (
                input_tokens > remaining.input_tokens
                or output_tokens > remaining.output_tokens
                or tool_calls > remaining.tool_calls
            ):
                raise BudgetExceeded(
                    "global agent budget exhausted: "
                    f"remaining={remaining}, requested="
                    f"({input_tokens}, {output_tokens}, {tool_calls})"
                )
            reservation = BudgetReservation(
                reservation_id=f"budget-{uuid.uuid4().hex}",
                agent_id=agent_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
            )
            self._remaining = BudgetSnapshot(
                remaining.input_tokens - input_tokens,
                remaining.output_tokens - output_tokens,
                remaining.tool_calls - tool_calls,
            )
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    def settle(
        self,
        reservation_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        tool_calls: int,
    ) -> None:
        """Charge actual usage and return the unused portion to the ledger."""
        self._validate_amounts(input_tokens, output_tokens, tool_calls)
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                raise UnknownReservation(reservation_id)
            if (
                input_tokens > reservation.input_tokens
                or output_tokens > reservation.output_tokens
                or tool_calls > reservation.tool_calls
            ):
                self._reservations[reservation_id] = reservation
                raise BudgetExceeded("actual usage exceeds the reserved maximum")
            remaining = self._remaining
            self._remaining = BudgetSnapshot(
                remaining.input_tokens + reservation.input_tokens - input_tokens,
                remaining.output_tokens + reservation.output_tokens - output_tokens,
                remaining.tool_calls + reservation.tool_calls - tool_calls,
            )

    def release(self, reservation_id: str) -> None:
        """Return a reservation that never started or was cancelled."""
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                raise UnknownReservation(reservation_id)
            remaining = self._remaining
            self._remaining = BudgetSnapshot(
                remaining.input_tokens + reservation.input_tokens,
                remaining.output_tokens + reservation.output_tokens,
                remaining.tool_calls + reservation.tool_calls,
            )

    def release_for_agent(self, agent_id: str) -> tuple[str, ...]:
        """Atomically return every outstanding reservation owned by an agent."""
        with self._lock:
            reservation_ids = tuple(
                reservation_id
                for reservation_id, reservation in self._reservations.items()
                if reservation.agent_id == agent_id
            )
            if not reservation_ids:
                return ()
            input_tokens = output_tokens = tool_calls = 0
            for reservation_id in reservation_ids:
                reservation = self._reservations.pop(reservation_id)
                input_tokens += reservation.input_tokens
                output_tokens += reservation.output_tokens
                tool_calls += reservation.tool_calls
            remaining = self._remaining
            self._remaining = BudgetSnapshot(
                remaining.input_tokens + input_tokens,
                remaining.output_tokens + output_tokens,
                remaining.tool_calls + tool_calls,
            )
            return reservation_ids
