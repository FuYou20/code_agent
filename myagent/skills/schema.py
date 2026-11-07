"""Define Skill metadata; requirements describe needs and never grant authority."""

from pydantic import AliasChoices, BaseModel, Field, model_validator


class PermissionRequirement(BaseModel):
    tool: str
    paths: list[str] | None = None
    command_pattern: str | None = None


class SkillFrontmatter(BaseModel):
    name: str
    description: str
    scene: str = "general"
    triggers: list[str] = Field(default_factory=list)
    always_on: bool = False
    permission_requirements: list[PermissionRequirement] = Field(
        default_factory=list,
        validation_alias=AliasChoices("permission_requirements", "scope_manifest"),
    )

    @model_validator(mode="before")
    @classmethod
    def expand_project_metadata(cls, value):
        """Accept standard Skill frontmatter with project fields in metadata."""
        if not isinstance(value, dict):
            return value
        expanded = dict(value)
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            return expanded
        for key in (
            "scene",
            "triggers",
            "always_on",
            "permission_requirements",
            "scope_manifest",
        ):
            if key not in expanded and key in metadata:
                expanded[key] = metadata[key]
        return expanded


class Skill(BaseModel):
    frontmatter: SkillFrontmatter
    body: str
    path: str
    version: int = 1

