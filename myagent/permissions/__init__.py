"""Expose atomic tool-file permissions and their delegation tree."""

from .model import FileGrant, PermissionAtom
from .broker import ToolPermissionAdapter, advertised_tools, check_and_execute
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
    "ToolPermissionAdapter",
    "advertised_tools",
    "check_and_execute",
    "PermissionEscalationDenied",
    "PermissionNodeView",
    "PermissionTree",
    "RootRevocationDenied",
    "UnknownAgent",
    "UnknownNode",
]
