"""Define Skill metadata; requirements describe needs and never grant authority."""

from pydantic import AliasChoices, BaseModel, Field


class PermissionRequirement(BaseModel):
    tool: str
    paths: list[str] | None = None
    command_pattern: str | None = None


class SkillFrontmatter(BaseModel):
    name: str
    description: str
    scene: str = "general"
    triggers: list[str] = Field(default_factory=list)
    permission_requirements: list[PermissionRequirement] = Field(
        default_factory=list,
        validation_alias=AliasChoices("permission_requirements", "scope_manifest"),
    )


class Skill(BaseModel):
    frontmatter: SkillFrontmatter
    body: str
    path: str
    version: int = 1

