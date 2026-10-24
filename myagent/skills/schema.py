"""Define validated skill frontmatter and loaded skill records."""

from pydantic import BaseModel, Field


class ScopeManifestItem(BaseModel):
    tool: str
    paths: list[str] | None = None
    command_pattern: str | None = None


class SkillFrontmatter(BaseModel):
    name: str
    description: str
    triggers: list[str] = Field(default_factory=list)
    scope_manifest: list[ScopeManifestItem]


class Skill(BaseModel):
    frontmatter: SkillFrontmatter
    body: str
    path: str
    version: int = 1

