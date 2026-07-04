"""Stage 4 camera annotation schema and prompt utilities."""

from .client import (
    MediaInput,
    build_openai_camera_messages,
    generate_camera_response,
)
from .config import CameraConfig, CameraResult
from .orchestrator import CameraOrchestrator
from .parse import parse_camera_response
from .prompt import CameraPromptRequest, build_camera_prompt
from .schema import (
    CAMERA_LABEL_GUIDE_TEXT,
    CAMERA_LABEL_FIELDS,
    CAMERA_LABEL_VERSION,
    CAMERA_PROMPT_VERSION,
    CameraIntrinsic,
    CameraLabels,
    CameraObjectCentric,
    CameraRotation,
    CameraStructuredOutput,
    CameraTranslation,
    camera_structured_output_json_schema,
    unknown_camera_labels,
)
from .worker import CameraWorker

__all__ = [
    "CAMERA_LABEL_GUIDE_TEXT",
    "CAMERA_LABEL_FIELDS",
    "CAMERA_LABEL_VERSION",
    "CAMERA_PROMPT_VERSION",
    "CameraConfig",
    "CameraOrchestrator",
    "CameraIntrinsic",
    "CameraLabels",
    "CameraObjectCentric",
    "CameraPromptRequest",
    "CameraResult",
    "CameraRotation",
    "CameraStructuredOutput",
    "CameraTranslation",
    "CameraWorker",
    "MediaInput",
    "build_openai_camera_messages",
    "build_camera_prompt",
    "camera_structured_output_json_schema",
    "generate_camera_response",
    "parse_camera_response",
    "unknown_camera_labels",
]
