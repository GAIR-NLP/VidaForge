from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .encoder import AutoModelEncodedSample


WAN_VAE_SPATIAL_COMPRESSION_RATIO = 8
WAN_TRANSFORMER_SPATIAL_PATCH_SIZE = 2
WAN_INPUT_SIZE_MULTIPLE = (
    WAN_VAE_SPATIAL_COMPRESSION_RATIO * WAN_TRANSFORMER_SPATIAL_PATCH_SIZE
)
WAN_TEMPORAL_STRIDE = 4
MAX_TEMPORAL_REPEAT_PAD_FRAMES = 1


@dataclass(frozen=True, slots=True)
class WanEncoderConfig:
    model_name: str | Path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    device: str = "cuda"
    max_sequence_length: int = 512
    resize_mode: str = "bilinear"
    seek_mode: str = "exact"
    center_crop: bool = True
    deterministic_latents: bool = True


class WanAutoModelEncoder:
    """Wan2.1 encoder for AutoModel `.meta` tensor cache generation."""

    input_size_multiple = WAN_INPUT_SIZE_MULTIPLE
    temporal_stride = WAN_TEMPORAL_STRIDE

    def __init__(
        self,
        *,
        model_name: str | Path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        device: str = "cuda",
        max_sequence_length: int = 512,
        resize_mode: str = "bilinear",
        seek_mode: str = "exact",
        center_crop: bool = True,
        deterministic_latents: bool = True,
    ) -> None:
        self.config = WanEncoderConfig(
            model_name=model_name,
            device=device,
            max_sequence_length=max_sequence_length,
            resize_mode=resize_mode,
            seek_mode=seek_mode,
            center_crop=center_crop,
            deterministic_latents=deterministic_latents,
        )
        self.validate_config()

        from diffusers import AutoencoderKLWan
        from transformers import AutoTokenizer, UMT5EncoderModel

        self.device = torch.device(
            "cuda"
            if self.config.device == "auto" and torch.cuda.is_available()
            else self.config.device
        )
        model_name_text = str(self.config.model_name)
        vae_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        text_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_name_text,
            subfolder="text_encoder",
            torch_dtype=text_dtype,
        )
        if (
            hasattr(self.text_encoder, "shared")
            and hasattr(self.text_encoder.encoder, "embed_tokens")
            and self.text_encoder.encoder.embed_tokens.weight.data_ptr()
            != self.text_encoder.shared.weight.data_ptr()
        ):
            self.text_encoder.encoder.embed_tokens.weight = self.text_encoder.shared.weight
        self.text_encoder.to(self.device)
        self.text_encoder.eval()

        self.vae = AutoencoderKLWan.from_pretrained(
            model_name_text,
            subfolder="vae",
            torch_dtype=vae_dtype,
        )
        self.vae.to(self.device)
        self.vae.eval()
        self.vae_dtype = vae_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_text,
            subfolder="tokenizer",
        )

    def validate_config(self) -> None:
        if self.config.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be > 0")
        if self.config.resize_mode != "bilinear":
            raise ValueError(
                "Wan TorchCodec decoding currently supports only resize_mode=bilinear"
            )
        if self.config.seek_mode not in {"exact", "approximate"}:
            raise ValueError("seek_mode must be one of: exact, approximate")

    def encode_batch(
        self,
        *,
        bucket_frame_count: int,
        bucket_resolution: tuple[int, int],
        video_paths: list[Path],
        source_resolutions: list[tuple[int, int]],
        source_fps: list[float],
        captions: list[str],
    ) -> list[AutoModelEncodedSample]:
        if (
            len(video_paths) != len(source_resolutions)
            or len(video_paths) != len(source_fps)
            or len(video_paths) != len(captions)
        ):
            raise ValueError(
                "video_paths, source_resolutions, source_fps, and captions "
                "must have the same length"
            )
        frame_count = int(bucket_frame_count)
        if frame_count <= 0:
            raise ValueError("bucket_frame_count must be > 0")
        if (frame_count - 1) % WAN_TEMPORAL_STRIDE != 0:
            raise ValueError(
                f"Wan bucket_frame_count must satisfy 4n+1, got {frame_count}"
            )

        video_tensors: list[torch.Tensor] = []
        video_infos: list[dict[str, object]] = []
        for video_path, source_resolution, fps in zip(
            video_paths,
            source_resolutions,
            source_fps,
        ):
            video_tensor, video_info = self.load_video_tensor(
                video_path,
                frame_count=frame_count,
                bucket_resolution=bucket_resolution,
                source_resolution=source_resolution,
                fps=fps,
            )
            video_tensors.append(video_tensor)
            video_infos.append(video_info)

        if not video_tensors:
            return []

        video_batch = torch.cat(video_tensors, dim=0)
        video_latents_batch = self.encode_video(video_batch)
        caption_token_lengths = self.caption_token_lengths(captions)
        text_embeddings_batch = self.encode_text_batch(captions)
        if video_latents_batch.shape[0] != len(video_paths):
            raise ValueError(
                "Wan VAE returned unexpected batch size: "
                f"expected={len(video_paths)}, actual={video_latents_batch.shape[0]}"
            )
        if text_embeddings_batch.shape[0] != len(captions):
            raise ValueError(
                "Wan text encoder returned unexpected batch size: "
                f"expected={len(captions)}, actual={text_embeddings_batch.shape[0]}"
            )
        if len(caption_token_lengths) != len(captions):
            raise ValueError(
                "Wan tokenizer returned unexpected batch size: "
                f"expected={len(captions)}, actual={len(caption_token_lengths)}"
            )

        samples: list[AutoModelEncodedSample] = []
        for index, video_info in enumerate(video_infos):
            width, height = tuple(int(value) for value in video_info["bucket_resolution"])
            caption_token_length = int(caption_token_lengths[index])
            metadata: dict[str, object] = {
                "model_type": "wan",
                "model_name": str(self.config.model_name),
                "frame_count": frame_count,
                "bucket_frame_count": frame_count,
                "max_sequence_length": self.config.max_sequence_length,
                "caption_token_length": caption_token_length,
                "caption_token_max_length": self.config.max_sequence_length,
                "caption_token_truncated": (
                    caption_token_length > self.config.max_sequence_length
                ),
                "resize_mode": self.config.resize_mode,
                "seek_mode": self.config.seek_mode,
                "center_crop": self.config.center_crop,
                "deterministic_latents": self.config.deterministic_latents,
                "original_resolution": list(video_info["original_resolution"]),
                "bucket_resolution": [width, height],
                "device": video_info["device"],
                "cpu_fallback": video_info["cpu_fallback"],
                "fps": video_info["fps"],
                "input_frame_count": video_info["input_frame_count"],
            }
            samples.append(
                AutoModelEncodedSample(
                    video_latents=video_latents_batch[index : index + 1],
                    text_embeddings=text_embeddings_batch[index : index + 1],
                    bucket_resolution=(width, height),
                    num_frames=int(video_info["frame_count"]),
                    metadata=metadata,
                )
            )
        return samples

    def load_video_tensor(
        self,
        video_path: str | Path,
        *,
        frame_count: int,
        bucket_resolution: tuple[int, int],
        source_resolution: tuple[int, int],
        fps: float,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        frames, video_info = self.load_video_frames(
            video_path,
            frame_count=frame_count,
            bucket_resolution=bucket_resolution,
            source_resolution=source_resolution,
            fps=fps,
        )
        tensor = frames.float()
        tensor = tensor / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
        return tensor, video_info

    def load_video_frames(
        self,
        video_path: str | Path,
        *,
        frame_count: int,
        bucket_resolution: tuple[int, int],
        source_resolution: tuple[int, int],
        fps: float,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        from torchcodec.decoders import VideoDecoder

        path = Path(video_path).expanduser().resolve()
        frame_count = int(frame_count)
        if frame_count <= 0:
            raise ValueError("frame_count must be > 0")
        if (frame_count - 1) % WAN_TEMPORAL_STRIDE != 0:
            raise ValueError(f"Wan frame_count must satisfy 4n+1, got {frame_count}")

        bucket_width, bucket_height = (
            int(bucket_resolution[0]),
            int(bucket_resolution[1]),
        )
        if bucket_width <= 0 or bucket_height <= 0:
            raise ValueError(
                f"bucket_resolution must contain positive width and height: "
                f"{bucket_resolution}"
            )
        if (
            bucket_width % WAN_INPUT_SIZE_MULTIPLE != 0
            or bucket_height % WAN_INPUT_SIZE_MULTIPLE != 0
        ):
            raise ValueError(
                "Wan bucket resolution must be divisible by "
                f"{WAN_INPUT_SIZE_MULTIPLE}: {bucket_resolution}"
            )
        bucket_resolution = (bucket_width, bucket_height)

        source_width, source_height = (
            int(source_resolution[0]),
            int(source_resolution[1]),
        )
        if source_width <= 0 or source_height <= 0:
            raise ValueError(
                f"source_resolution must contain positive width and height: "
                f"path={path}, source_resolution={source_resolution}"
            )
        fps = float(fps)
        if fps <= 0:
            raise ValueError(f"video fps must be > 0: path={path}, fps={fps}")

        decoder_kwargs = {
            "dimension_order": "NCHW",
            "device": str(self.device),
            "seek_mode": self.config.seek_mode,
            "num_ffmpeg_threads": 1,
        }
        if self.device.type != "cuda":
            decoder_kwargs["transforms"] = self.build_decode_transforms(
                source_resolution=(source_width, source_height),
                bucket_resolution=bucket_resolution,
            )
        if self.device.type == "cuda":
            from torchcodec.decoders import set_cuda_backend

            with set_cuda_backend("beta"):
                decoder = VideoDecoder(path, **decoder_kwargs)
        else:
            decoder = VideoDecoder(path, **decoder_kwargs)

        input_frame_count = int(len(decoder))
        if input_frame_count <= 0:
            raise ValueError(f"video frame count must be > 0: {path}")
        missing_frame_count = frame_count - input_frame_count
        # FFmpeg/ffprobe duration metadata and TorchCodec decoder length can
        # differ by one frame near clip boundaries. Allow exactly one repeated
        # sampled frame so Wan 4n+1 buckets remain usable without stretching
        # genuinely short clips.
        if missing_frame_count > MAX_TEMPORAL_REPEAT_PAD_FRAMES:
            raise ValueError(
                f"video frame count must be >= frame_count: "
                f"path={path}, input_frame_count={input_frame_count}, "
                f"frame_count={frame_count}"
            )

        frame_indices = torch.linspace(
            0,
            input_frame_count - 1,
            steps=frame_count,
            dtype=torch.float64,
        )
        frame_indices = frame_indices.round().to(dtype=torch.int64)
        frame_batch = decoder.get_frames_at(frame_indices.tolist())
        frames = frame_batch.data.contiguous()
        if self.device.type == "cuda":
            frames = self.resize_frames_to_bucket(
                frames,
                source_resolution=(source_width, source_height),
                bucket_resolution=bucket_resolution,
            ).contiguous()
        if frames.shape[0] != frame_count:
            raise ValueError(
                f"decoded frame count mismatch: path={path}, "
                f"decoded={frames.shape[0]}, expected={frame_count}"
            )
        if tuple(frames.shape[1:]) != (3, bucket_height, bucket_width):
            raise ValueError(
                f"decoded frame tensor shape mismatch: path={path}, "
                f"actual={tuple(frames.shape)}, "
                f"expected=({frame_count}, 3, {bucket_height}, {bucket_width})"
            )

        cpu_fallback = bool(decoder.cpu_fallback)
        return frames, {
            "device": str(frames.device),
            "cpu_fallback": cpu_fallback,
            "fps": fps,
            "input_frame_count": input_frame_count,
            "frame_count": int(frames.shape[0]),
            "original_resolution": (source_width, source_height),
            "bucket_resolution": bucket_resolution,
        }

    def build_decode_transforms(
        self,
        *,
        source_resolution: tuple[int, int],
        bucket_resolution: tuple[int, int],
    ) -> list[object]:
        from torchcodec.transforms import CenterCrop, Resize

        source_width, source_height = (
            int(source_resolution[0]),
            int(source_resolution[1]),
        )
        target_width, target_height = bucket_resolution
        if source_width <= 0 or source_height <= 0:
            raise ValueError(
                f"source_resolution must contain positive width and height: "
                f"{source_resolution}"
            )
        if target_width <= 0 or target_height <= 0:
            raise ValueError(
                f"bucket_resolution must contain positive width and height: "
                f"{bucket_resolution}"
            )

        if self.config.center_crop:
            source_ratio = source_width / source_height
            target_ratio = target_width / target_height
            if source_ratio > target_ratio:
                crop_height = source_height
                crop_width = max(1, int(math.floor(source_height * target_ratio)))
            else:
                crop_width = source_width
                crop_height = max(1, int(math.floor(source_width / target_ratio)))
            return [
                CenterCrop((crop_height, crop_width)),
                Resize((target_height, target_width)),
            ]

        return [Resize((target_height, target_width))]

    def resize_frames_to_bucket(
        self,
        frames: torch.Tensor,
        *,
        source_resolution: tuple[int, int],
        bucket_resolution: tuple[int, int],
    ) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"expected NCHW frames, got shape={tuple(frames.shape)}")

        source_width, source_height = (
            int(source_resolution[0]),
            int(source_resolution[1]),
        )
        target_width, target_height = (
            int(bucket_resolution[0]),
            int(bucket_resolution[1]),
        )
        if source_width <= 0 or source_height <= 0:
            raise ValueError(
                f"source_resolution must contain positive width and height: "
                f"{source_resolution}"
            )
        if target_width <= 0 or target_height <= 0:
            raise ValueError(
                f"bucket_resolution must contain positive width and height: "
                f"{bucket_resolution}"
            )

        if tuple(frames.shape[-2:]) != (source_height, source_width):
            raise ValueError(
                f"decoded frame resolution mismatch: "
                f"actual={tuple(frames.shape[-2:])}, "
                f"expected=({source_height}, {source_width})"
            )

        transformed = frames
        if self.config.center_crop:
            source_ratio = source_width / source_height
            target_ratio = target_width / target_height
            if source_ratio > target_ratio:
                crop_height = source_height
                crop_width = max(1, int(math.floor(source_height * target_ratio)))
                top = 0
                left = (source_width - crop_width) // 2
            else:
                crop_width = source_width
                crop_height = max(1, int(math.floor(source_width / target_ratio)))
                top = (source_height - crop_height) // 2
                left = 0
            transformed = transformed[
                ...,
                top : top + crop_height,
                left : left + crop_width,
            ]

        return F.interpolate(
            transformed.float(),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        video_tensor = video_tensor.to(device=self.device, dtype=self.vae_dtype)
        with torch.inference_mode():
            latent_dist = self.vae.encode(video_tensor)
            if self.config.deterministic_latents:
                video_latents = latent_dist.latent_dist.mean
            else:
                video_latents = latent_dist.latent_dist.sample()

        latents_mean = getattr(self.vae.config, "latents_mean", None)
        latents_std = getattr(self.vae.config, "latents_std", None)
        if latents_mean is None or latents_std is None:
            raise ValueError("Wan VAE config must contain latents_mean and latents_std")

        mean = torch.tensor(
            latents_mean,
            device=self.device,
            dtype=self.vae_dtype,
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            latents_std,
            device=self.device,
            dtype=self.vae_dtype,
        ).view(1, -1, 1, 1, 1)
        latents = (video_latents - mean) / std
        return latents.detach().cpu().to(torch.float16)

    def encode_text(self, prompt: str) -> torch.Tensor:
        return self.encode_text_batch([prompt])

    def caption_token_lengths(self, prompts: list[str]) -> list[int]:
        prompts = [self.clean_prompt(prompt) for prompt in prompts]
        tokenized = self.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )
        return [len(input_ids) for input_ids in tokenized["input_ids"]]

    def encode_text_batch(self, prompts: list[str]) -> torch.Tensor:
        prompts = [self.clean_prompt(prompt) for prompt in prompts]
        inputs = self.tokenizer(
            prompts,
            max_length=self.config.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        sequence_lengths = attention_mask.gt(0).sum(dim=1).long()

        with torch.inference_mode():
            prompt_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        padding_mask = torch.arange(
            self.config.max_sequence_length,
            device=self.device,
        )[None, :] >= sequence_lengths[:, None]
        prompt_embeds = prompt_embeds.masked_fill(padding_mask[:, :, None], 0)
        return prompt_embeds.detach().cpu()

    def clean_prompt(self, prompt: str) -> str:
        text = str(prompt)
        from diffusers.utils import is_ftfy_available

        if is_ftfy_available():
            import ftfy

            text = ftfy.fix_text(text)

        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text)
        return text.strip()


__all__ = [
    "WAN_INPUT_SIZE_MULTIPLE",
    "WAN_TEMPORAL_STRIDE",
    "WAN_TRANSFORMER_SPATIAL_PATCH_SIZE",
    "WAN_VAE_SPATIAL_COMPRESSION_RATIO",
    "WanAutoModelEncoder",
    "WanEncoderConfig",
]
