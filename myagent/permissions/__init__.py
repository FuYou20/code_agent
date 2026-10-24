"""Expose atomic tool-file permissions and their delegation tree."""

from .model import FileGrant, PermissionAtom
from .tree import (
    AgentAlreadyAttached,
    PermissionEscalationDenied,
    PermissionNodeView,
    PermissionTree,
    RootRevocationDenied,
    UnknownAgent,
    UnknownNode,
)

__all__ = [
    "AgentAlreadyAttached",
    "FileGrant",
    "PermissionAtom",
    "PermissionEscalationDenied",
    "PermissionNodeView",
    "PermissionTree",
    "RootRevocationDenied",
    "UnknownAgent",
    "UnknownNode",
]
