from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

CAPTION_SCHEMA_VERSION = "caption_v1"


class CaptionStructuredOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level_3: str = Field(description="Dense reconstruction caption.")
    level_2: str = Field(description="Detailed temporal caption.")
    level_1: str = Field(description="Concise video caption.")
    level_0: str = Field(description="Shortest semantic gist.")
