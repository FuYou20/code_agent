"""Expose tool registration, lookup, execution, and scope primitives."""

from .registry import execute, lookup, register, schemas
from .scope import Scope, ScopeEscalationDenied, ToolRule, allows, intersect

__all__ = [
    "Scope",
    "ScopeEscalationDenied",
    "ToolRule",
    "allows",
    "execute",
    "intersect",
    "lookup",
    "register",
    "schemas",
]

