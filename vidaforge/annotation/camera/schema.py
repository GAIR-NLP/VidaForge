from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


CAMERA_LABEL_VERSION = "camera_v1"
CAMERA_PROMPT_VERSION = "camera_qa_v1"

CAMERA_LABEL_FIELDS = (
    "motion_type",
    "steadiness",
    "rotation",
    "translation",
    "intrinsic",
    "object_centric",
    "speed",
    "effects",
    "scene_dynamics",
)

MotionType = Literal[
    "no-motion",
    "minor-motion",
    "simple-motion",
    "complex-motion",
    "unknown",
]
Steadiness = Literal[
    "static",
    "no-shaking",
    "minimal-shaking",
    "unsteady",
    "very-unsteady",
    "unknown",
]
Pan = Literal["pan-left", "pan-right", "no-pan", "unknown"]
Tilt = Literal["tilt-up", "tilt-down", "no-tilt", "unknown"]
Roll = Literal["roll-CW", "roll-CCW", "no-roll", "unknown"]
Dolly = Literal["dolly-in", "dolly-out", "no-dolly", "unknown"]
Pedestal = Literal["pedestal-up", "pedestal-down", "no-pedestal", "unknown"]
Truck = Literal["truck-left", "truck-right", "no-truck", "unknown"]
Zoom = Literal["zoom-in", "zoom-out", "no-zoom", "unknown"]
Arc = Literal["arc-CW", "arc-CCW", "no-arc", "unknown"]
ArcTracking = Literal["arc-tracking", "no-arc-tracking", "unknown"]
LeadTracking = Literal["lead-tracking", "no-lead-tracking", "unknown"]
TailTracking = Literal["tail-tracking", "no-tail-tracking", "unknown"]
SideTracking = Literal["side-tracking", "no-side-tracking", "unknown"]
AerialTracking = Literal["aerial-tracking", "no-aerial-tracking", "unknown"]
PanTracking = Literal["pan-tracking", "no-pan-tracking", "unknown"]
TiltTracking = Literal["tilt-tracking", "no-tilt-tracking", "unknown"]
SubjectSizeChange = Literal["subject-larger", "subject-smaller", "no-subject-change", "unknown"]
CameraSpeed = Literal["slow", "regular", "fast", "unknown"]
CinematicMotionEffect = Literal["frame-freezing", "dolly-zoom", "motion-blur", "none", "unknown"]
SceneDynamics = Literal["static", "mostly-static", "dynamic", "unknown"]


class CameraSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CameraRotation(CameraSchemaModel):
    pan: Pan = Field(description="Camera horizontal rotation around its own axis.")
    tilt: Tilt = Field(description="Camera vertical rotation around its own axis.")
    roll: Roll = Field(description="Camera clockwise/counterclockwise optical-axis rotation.")


class CameraTranslation(CameraSchemaModel):
    dolly: Dolly = Field(description="Camera physical forward/backward movement.")
    pedestal: Pedestal = Field(description="Camera physical upward/downward movement.")
    truck: Truck = Field(description="Camera physical left/right movement.")


class CameraIntrinsic(CameraSchemaModel):
    zoom: Zoom = Field(description="Focal length zoom change, not physical camera movement.")


class CameraObjectCentric(CameraSchemaModel):
    arc: Arc = Field(description="Circular or semi-circular camera movement around a subject or center.")
    arc_tracking: ArcTracking = Field(description="Circular tracking around a moving subject.")
    lead_tracking: LeadTracking = Field(description="Camera moves ahead of a moving subject.")
    tail_tracking: TailTracking = Field(description="Camera follows behind a moving subject.")
    side_tracking: SideTracking = Field(description="Camera moves parallel to a moving subject.")
    aerial_tracking: AerialTracking = Field(description="Camera tracks a subject from a high vantage point.")
    pan_tracking: PanTracking = Field(description="Camera pivots horizontally to follow a subject.")
    tilt_tracking: TiltTracking = Field(description="Camera pivots vertically to follow a subject.")
    subject_size_change: SubjectSizeChange = Field(
        description="Tracked subject becomes larger/smaller due to camera movement or zoom."
    )


class CameraLabels(CameraSchemaModel):
    motion_type: MotionType
    steadiness: Steadiness
    rotation: CameraRotation
    translation: CameraTranslation
    intrinsic: CameraIntrinsic
    object_centric: CameraObjectCentric
    speed: CameraSpeed
    effects: list[CinematicMotionEffect] = Field(
        min_length=1,
        description=(
            "Visible cinematic motion effects. Use only ['none'] when no effect is visible, "
            "or only ['unknown'] when evidence is insufficient."
        ),
    )
    scene_dynamics: SceneDynamics

    @field_validator("effects")
    @classmethod
    def validate_effects(cls, effects: list[CinematicMotionEffect]) -> list[CinematicMotionEffect]:
        unique_effects = list(dict.fromkeys(effects))
        if ("none" in unique_effects or "unknown" in unique_effects) and len(unique_effects) > 1:
            raise ValueError("effects cannot mix 'none' or 'unknown' with other labels")
        return unique_effects


class CameraStructuredOutput(CameraLabels):
    """Pydantic contract for VLM structured output."""

    label_version: Literal["camera_v1"] = Field(description="Camera label schema version.")
    prompt_version: Literal["camera_qa_v1"] = Field(description="Camera prompt version.")


CAMERA_LABEL_GUIDE_TEXT = """Label definitions and guidelines:

Motion type:
- no-motion: The camera remains stationary with no intentional movement. Note: Unintentional shaking belongs to no-motion.
- minor-motion: The camera moves slightly and intentionally, such as a gentle pan or zoom. The motion is noticeable but remains subtle and not significant.
- simple-motion: The camera moves significantly in a straightforward manner, such as a steady pan, tilt, arc, or simple tracking shot. Select this even if the video combines two or more motions, as long as they occur simultaneously at roughly the same speed.
- complex-motion: The camera exhibits complex movements that are difficult to classify. This includes conflicting motion, sequential motion, simultaneous motions at different speeds, or unclear motion / missing background information due to motion blur or lack of background cues.

Steadiness:
- static: The camera remains completely stationary with no movement or vibration.
- no-shaking: The camera moves smoothly with no detectable shake, typically using high-end stabilizers. Select only if the camera is moving and no unintended motion is present.
- minimal-shaking: The camera exhibits slight shaking, whether stationary or moving, maintaining a mostly stable shot.
- unsteady: The camera shows moderate shaking, whether stationary or in motion, introducing noticeable but controlled instability.
- very-unsteady: The camera shakes consistently, typical of unstabilized handheld or action footage. Select only if shaking is consistent throughout the video.

Translation:
- dolly-in / dolly-out: The camera moves forward or backward relative to the ground plane and the initial frame.
- no-dolly: The camera does not move forward/backward during the shot.
- pedestal-up / pedestal-down: Select this when the camera moves upward or downward clearly and consistently relative to the ground or the orientation of the initial frame.
- no-pedestal: The camera does not move upward or downward during the shot.
- truck-left / truck-right: The camera physically moves to the left or right, changing its position relative to the initial frame.
- no-truck: The camera does not move to the left or right during the shot.

Rotation:
- pan-left / pan-right: The camera rotates its angle by pivoting left or right with respect to the initial frame.
- no-pan: The camera does not pan left or right.
- tilt-up / tilt-down: The camera rotates its angle up or down vertically with respect to the initial frame.
- no-tilt: The camera does not tilt up or down.
- roll-CW / roll-CCW: The camera performs a clear and consistent clockwise (CW) or counterclockwise (CCW) roll by rotating around its own optical center.
- no-roll: The camera does not roll clockwise/counterclockwise.

Intrinsic change:
- zoom-in / zoom-out: The camera adjusts its focal length to zoom in or out, changing the frame size. This differs from physical camera movement.
- no-zoom: The camera does not adjust its focal length during the video.

Object-centric movement:
- arc-CW / arc-CCW: The camera moves in a circular or semi-circular motion around the subject or the frame center in a clockwise or counterclockwise direction.
- no-arc: The camera does not move in a circular or semi-circular motion during the video.
- arc-tracking: The camera moves in a circular or semi-circular path around the moving subject, often referred to as an orbit or circular tracking shot.
- no-arc-tracking: The camera does not track or does not move in a circular or semi-circular path around the moving subject.
- lead-tracking: The camera moves ahead of the moving subject, capturing their face or front as they follow the camera's path. This is also referred to as a leading shot.
- no-lead-tracking: The camera does not track or does not move ahead of the moving subject.
- tail-tracking: The camera follows directly behind the moving subject, keeping their back in view as they move forward. This is also known as a follow shot or chase shot.
- no-tail-tracking: The camera does not track or does not move behind the moving subject.
- side-tracking: The camera moves parallel to the moving subject, following them from the side as they move through the scene.
- no-side-tracking: The camera does not track or does not move parallel to the moving subject.
- aerial-tracking: The camera tracks the moving subject from a high vantage point, often using a drone or crane to follow their movement.
- no-aerial-tracking: The camera either does not track the moving subject or is not positioned at a high vantage point.
- pan-tracking: The camera remains in a fixed position but pivots horizontally to follow the subject as they move.
- no-pan-tracking: The camera does not track the subject or does not pivot horizontally to follow their movement.
- tilt-tracking: The camera tilts up or down to follow the vertical movement of the subject.
- no-tilt-tracking: The camera does not track the subject or does not pivot vertically to follow their movement.
- subject-larger: The camera moves or zooms in towards the tracked subject, making them appear larger in the frame.
- subject-smaller: The camera moves or zooms away from the tracked subject, making them appear smaller in the frame.
- no-subject-change: The camera neither moves towards nor away from the subject.

Camera movement speed:
- slow: The camera moves at a noticeably slow pace.
- regular: The camera moves at a regular pace. If the speed does not stand out as particularly slow or fast, it is considered regular.
- fast: The camera moves quickly, such as in a crash zoom or whip pan.

Cinematic motion effects:
- frame-freezing: A visual effect where scene motion is paused or frozen mid-action, creating a still frame within a moving sequence.
- dolly-zoom: A camera effect where the background appears to compress or stretch while the subject stays the same size, often used to create a sense of unease.
- motion-blur: A visual effect where moving objects blur due to slow shutter speed or camera movement, often used to emphasize speed and fluid motion in action scenes.

Scene dynamics:
- static: The entire scene, including all subjects and background, remains completely motionless throughout the video.
- mostly-static: The scene is largely still, with only minor elements or small parts exhibiting movement.
- dynamic: A significant portion of the frame is occupied by dynamic movement of subjects or scene elements, excluding camera motion, that visibly alters the scene.

Schema-specific fallback labels:
- unknown: Use only when the visual evidence is insufficient for that field. This fallback is added by this project; it is not listed in the source table.
- effects none: Use only when no cinematic motion effect is visible. This fallback is added by this project; it is not listed in the source table.
- effects is a multi-label field in this project. Output every visible cinematic motion effect, or output only ["none"] / only ["unknown"]. Do not mix none or unknown with other effect labels."""


def camera_structured_output_json_schema() -> dict[str, Any]:
    """Return the JSON schema to pass to vLLM structured outputs."""
    return CameraStructuredOutput.model_json_schema()


def unknown_camera_labels() -> dict[str, object]:
    return CameraLabels(
        motion_type="unknown",
        steadiness="unknown",
        rotation=CameraRotation(pan="unknown", tilt="unknown", roll="unknown"),
        translation=CameraTranslation(dolly="unknown", pedestal="unknown", truck="unknown"),
        intrinsic=CameraIntrinsic(zoom="unknown"),
        object_centric=CameraObjectCentric(
            arc="unknown",
            arc_tracking="unknown",
            lead_tracking="unknown",
            tail_tracking="unknown",
            side_tracking="unknown",
            aerial_tracking="unknown",
            pan_tracking="unknown",
            tilt_tracking="unknown",
            subject_size_change="unknown",
        ),
        speed="unknown",
        effects=["unknown"],
        scene_dynamics="unknown",
    ).model_dump()

