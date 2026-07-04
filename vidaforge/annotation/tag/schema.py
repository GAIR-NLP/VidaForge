from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator


TAG_SCHEMA_VERSION = "tag_v1"
TAG_PROMPT_VERSION = "tag_prompt_v1"

TAG_LABEL_FIELDS = (
    "domain",
    "scene",
    "subjects",
    "actions",
    "style",
    "text",
    "watermark",
)

TagDomain = Literal[
    "real_world",
    "animation",
    "game",
    "screen_recording",
    "synthetic_render",
    "mixed",
    "unknown",
]
TagScene = Literal[
    "general_indoor",
    "general_outdoor",
    "urban",
    "nature",
    "driving",
    "sports",
    "food",
    "product",
    "portrait",
    "screen",
    "other",
    "unknown",
]
TagSubject = Literal[
    "person",
    "vehicle",
    "animal",
    "object",
    "food",
    "landscape",
    "building",
    "text",
    "screen",
    "robot",
    "other",
    "unknown",
]
TagAction = Literal[
    "talking",
    "locomotion",
    "driving",
    "sports",
    "cooking",
    "object_manipulation",
    "natural_motion",
    "camera_motion_only",
    "timelapse",
    "none",
    "other",
    "unknown",
]
TagStyle = Literal[
    "photorealistic",
    "cinematic",
    "documentary",
    "anime",
    "cartoon",
    "cg_render",
    "gameplay",
    "graphic",
    "unknown",
]
TagText = Literal[
    "none",
    "incidental",
    "subtitle",
    "screen_ui",
    "document",
    "signage",
    "overlay_text",
    "unknown",
]
TagWatermark = Literal[
    "none",
    "logo",
    "text_watermark",
    "platform_watermark",
    "unknown",
]

TAG_DOMAIN_LABELS = get_args(TagDomain)
TAG_SCENE_LABELS = get_args(TagScene)
TAG_SUBJECT_LABELS = get_args(TagSubject)
TAG_ACTION_LABELS = get_args(TagAction)
TAG_STYLE_LABELS = get_args(TagStyle)
TAG_TEXT_LABELS = get_args(TagText)
TAG_WATERMARK_LABELS = get_args(TagWatermark)


class TagSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _dedupe_labels(labels: list[str]) -> list[str]:
    return list(dict.fromkeys(labels))


def _validate_mutually_exclusive_labels(
    *,
    field_name: str,
    labels: list[str],
    mutually_exclusive_labels: set[str],
) -> list[str]:
    unique_labels = _dedupe_labels(labels)
    for label in mutually_exclusive_labels:
        if label in unique_labels and len(unique_labels) > 1:
            raise ValueError(
                f"{field_name} cannot mix {label!r} with other labels"
            )
    return unique_labels


class TagLabels(TagSchemaModel):
    domain: TagDomain = Field(description="Primary data form of the clip.")
    scene: TagScene = Field(
        description="Dominant scene/content bucket, not a strict physical location."
    )
    subjects: list[TagSubject] = Field(
        min_length=1,
        description=(
            "Visible major subjects. Use only ['unknown'] when evidence is insufficient."
        ),
    )
    actions: list[TagAction] = Field(
        min_length=1,
        description=(
            "Main visible actions or motion. Use only ['none'] when no meaningful "
            "subject/scene action is visible, or only ['unknown'] when evidence is insufficient."
        ),
    )
    style: TagStyle = Field(description="Primary visual appearance style.")
    text: TagText = Field(description="Semantic role of visible text.")
    watermark: TagWatermark = Field(description="Semantic role of visible watermark or logo.")

    @field_validator("subjects")
    @classmethod
    def validate_subjects(cls, subjects: list[TagSubject]) -> list[TagSubject]:
        return _validate_mutually_exclusive_labels(
            field_name="subjects",
            labels=subjects,
            mutually_exclusive_labels={"unknown"},
        )

    @field_validator("actions")
    @classmethod
    def validate_actions(cls, actions: list[TagAction]) -> list[TagAction]:
        return _validate_mutually_exclusive_labels(
            field_name="actions",
            labels=actions,
            mutually_exclusive_labels={"none", "unknown"},
        )


class TagStructuredOutput(TagLabels):
    """Pydantic contract for VLM structured output."""

    schema_version: Literal["tag_v1"] = Field(description="Tag schema version.")
    prompt_version: Literal["tag_prompt_v1"] = Field(description="Tag prompt version.")


TAG_LABEL_GUIDE_TEXT = """Label definitions and guidelines:

Domain:
- real_world: Camera-captured real-world footage, including phones, professional cameras, dashcams, drones, and surveillance-like footage. If a physical screen is filmed by a camera, use real_world and include screen in subjects.
- animation: Authored animated video, including 2D/3D animation, anime, cartoons, animated characters, and stop-motion-like animation.
- game: Video game content or game engine gameplay, including first-person, third-person, and game replay footage, even when captured from a screen.
- screen_recording: Direct digital capture of software UI, websites, documents, slides, terminals, phone/computer interfaces, or recorded presentations. If the dominant content is gameplay, use game instead. If a physical screen is filmed by a camera, use real_world instead.
- synthetic_render: Non-game CGI, 3D render, simulation, product render, or synthetic scene that is not authored animation or gameplay.
- mixed: Multiple domains are visually central or occupy a substantial portion of the clip, such as real footage with large animation overlays or split-screen mixed sources. Do not use mixed for small overlays, stickers, captions, subtitles, logos, or minor inserted graphics.
- unknown: The data form cannot be determined from the evidence.

Scene:
- Scene is single-label. Choose the dominant scene/content bucket. Use general_indoor/general_outdoor only when no more specific scene/content label dominates.
- general_indoor: Main scene is inside a building or enclosed space, and no more specific scene label dominates.
- general_outdoor: Main scene is outside, and no more specific scene label dominates.
- urban: City, street, road-side pedestrian, architecture, crowd, or built-environment scene where driving/navigation is not the main focus.
- nature: Landscape, forest, mountain, ocean, sky, plants, wildlife habitat, or natural environment.
- driving: Road, traffic, vehicle-mounted, dashcam, cockpit, or vehicle-centered driving/navigation scene where the road, vehicle, or navigation context is the main focus.
- sports: Organized sport, exercise, fitness, competition, or athletic activity scene.
- food: Food preparation, eating, restaurant, ingredients, or food-focused scene.
- product: Product/object showcase, commercial-like object shot, package, appliance, tool, or item demonstration.
- portrait: Person-centered close-up, selfie, talking-head, interview, face/body portrait, or presenter-dominant scene.
- screen: Screen/UI/document/slides dominate the visual content.
- other: A clear scene exists but does not fit the listed labels.
- unknown: The scene cannot be determined from the evidence.

Subjects:
- person: One or more visible humans or human-like characters.
- vehicle: Car, truck, bus, train, bicycle, motorcycle, aircraft, boat, or similar vehicle.
- animal: Any visible animal.
- object: A salient foreground physical object, tool, toy, product, device, furniture, or item. Do not add object for ordinary background clutter.
- food: Food, drink, ingredients, dishes, or cooking material.
- landscape: Natural scenery, sky, water, mountain, forest, plants, or terrain as a main subject.
- building: Buildings, rooms, streetscape structures, bridges, monuments, or architecture.
- text: Text itself is a salient visible subject, such as a sign, poster, title card, slide text, document text, or large overlay text.
- screen: A visible screen or screen-captured interface is a salient subject, such as a monitor, phone display, TV, software UI, webpage, document, slide, or game HUD.
- robot: Robot, robotic arm, autonomous machine, or embodied AI platform.
- other: A clear major subject exists but does not fit the listed labels.
- unknown: Use only when the visible subject cannot be determined.

Actions:
- talking: A person or character appears to speak, present, sing, or address the camera/audience.
- locomotion: Walking, running, hiking, dancing steps, jumping, or human/character movement through space.
- driving: Vehicle driving, riding, traffic movement, or road navigation.
- sports: Sport, exercise, fitness, competition, or athletic motion.
- cooking: Cooking, food preparation, plating, eating preparation, or kitchen action.
- object_manipulation: Hands, tools, robots, or subjects manipulate, assemble, open, move, or operate objects.
- natural_motion: Natural motion such as flowing water, fire, smoke, weather, plants moving, or animal/environment motion.
- camera_motion_only: Camera movement is the main visible change while subjects/scene are mostly static. Do not use this for minor camera shake when another action is visible.
- timelapse: Time-lapse, accelerated growth, sunset, traffic trails, construction progress, or clearly sped-up temporal change.
- none: No meaningful subject or scene action is visible.
- other: A clear action exists but does not fit the listed labels.
- unknown: The action cannot be determined from the evidence.

Style:
- photorealistic: Realistic-looking footage or synthetic imagery that is not clearly documentary, cinematic, gameplay, graphic, anime/cartoon, or cg_render.
- cinematic: Deliberately stylized camera, lighting, color grading, composition, or film-like visual treatment. Use only when the visual treatment clearly stands out.
- documentary: Factual, vlog, news, tutorial, documentation-like, or observational real-world footage, without strong cinematic stylization.
- anime: Anime-style visual content.
- cartoon: Cartoon, 2D animation, western animation, simple animated characters, or non-anime cartoon style.
- cg_render: CGI, 3D render, simulation, digital twin, product render, or synthetic 3D scene.
- gameplay: Game visual style, including HUD/game engine aesthetics.
- graphic: Slides, diagrams, charts, infographic, UI-heavy graphic design, or mostly flat graphical content.
- unknown: The visual style cannot be determined from the evidence.

Text:
- Text is single-label. Choose the dominant visible text role.
- none: No visible text.
- incidental: Small or background text that is not central, such as labels, packaging, signs in the distance, or minor UI text.
- subtitle: Subtitles, captions, karaoke lyrics, or dialogue text overlaid on video. Use this when subtitles/captions are persistent and central.
- screen_ui: Software UI, websites, apps, menus, code editors, terminals, game HUDs, interface controls, or phone/computer screen text.
- document: PPT/PDF/slides, paper pages, books, posters, forms, presentation pages, or page-like document content, even when viewed on a screen.
- signage: Signs, billboards, road signs, storefront signs, placards, or wayfinding text.
- overlay_text: Large title text, meme text, sticker text, callouts, or other prominent text overlaid on the video that is not subtitle, UI, document, or signage.
- unknown: Visible text role cannot be determined.

Watermark:
- none: No visible watermark or logo-like ownership/platform mark.
- logo: A visible overlay/source/channel logo mark. Do not use this for ordinary logos printed on physical products or signs.
- text_watermark: Repeated, semi-transparent, or ownership text watermark.
- platform_watermark: Platform/app/source watermark, such as social media or stock-media marks.
- unknown: Watermark/logo status cannot be determined.

General rules:
- Single-label fields: domain, scene, style, text, watermark.
- Multi-label fields: subjects, actions.
- Choose the dominant label for single-label fields.
- Dominant means sustained across the clip and visually/semantically central, not a brief or background appearance.
- Multi-label fields should include all major visible labels, but stay concise.
- Use other when the evidence is clear but does not fit the listed labels.
- Use unknown only when the evidence is insufficient.
- Do not mix unknown with any other label in multi-label fields.
- Do not mix none with any other action label."""


def tag_structured_output_json_schema() -> dict[str, Any]:
    """Return the JSON schema to pass to vLLM structured outputs."""
    return TagStructuredOutput.model_json_schema()


def unknown_tag_labels() -> dict[str, object]:
    return TagLabels(
        domain="unknown",
        scene="unknown",
        subjects=["unknown"],
        actions=["unknown"],
        style="unknown",
        text="unknown",
        watermark="unknown",
    ).model_dump()
