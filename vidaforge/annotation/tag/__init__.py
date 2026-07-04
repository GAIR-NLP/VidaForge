"""Clip-level semantic tagging for Stage 4 annotation."""

from .client import build_openai_tag_messages, generate_tag_response
from .config import TagConfig, TagResult
from .orchestrator import TagOrchestrator
from .parse import parse_tag_response
from .prompt import TagPromptRequest, build_tag_prompt
from .schema import (
    TAG_ACTION_LABELS,
    TAG_DOMAIN_LABELS,
    TAG_LABEL_FIELDS,
    TAG_LABEL_GUIDE_TEXT,
    TAG_PROMPT_VERSION,
    TAG_SCENE_LABELS,
    TAG_SCHEMA_VERSION,
    TAG_STYLE_LABELS,
    TAG_SUBJECT_LABELS,
    TAG_TEXT_LABELS,
    TAG_WATERMARK_LABELS,
    TagLabels,
    TagStructuredOutput,
    tag_structured_output_json_schema,
    unknown_tag_labels,
)
from .worker import TagWorker

__all__ = [
    "TAG_ACTION_LABELS",
    "TAG_DOMAIN_LABELS",
    "TAG_LABEL_FIELDS",
    "TAG_LABEL_GUIDE_TEXT",
    "TAG_PROMPT_VERSION",
    "TAG_SCENE_LABELS",
    "TAG_SCHEMA_VERSION",
    "TAG_STYLE_LABELS",
    "TAG_SUBJECT_LABELS",
    "TAG_TEXT_LABELS",
    "TAG_WATERMARK_LABELS",
    "TagConfig",
    "TagLabels",
    "TagOrchestrator",
    "TagPromptRequest",
    "TagResult",
    "TagStructuredOutput",
    "TagWorker",
    "build_openai_tag_messages",
    "build_tag_prompt",
    "generate_tag_response",
    "parse_tag_response",
    "tag_structured_output_json_schema",
    "unknown_tag_labels",
]
