from __future__ import annotations

from typing import ClassVar

from vidaforge.common import join_data_dir
from vidaforge.filters import exp_saturation, linear_decay
from vidaforge.media import run_ffmpeg_motion

from .base import FilterBase
from .config import MotionFilterConfig


class MotionFilter(FilterBase[MotionFilterConfig]):
    filter_name: ClassVar[str] = "motion"
    config_type: ClassVar[type[MotionFilterConfig]] = MotionFilterConfig

    def __init__(self, config: MotionFilterConfig) -> None:
        if not isinstance(config, MotionFilterConfig):
            raise TypeError(f"expected MotionFilterConfig, got {type(config)!r}")
        self.config = config

    def apply(self, row: dict[str, object]) -> tuple[float, dict[str, object]]:
        clip_path = join_data_dir(str(row["clip_path"]))
        motion_result = run_ffmpeg_motion(
            clip_path,
            scdet_threshold=self.config.scdet.threshold,
            scdet_sample_fps=self.config.scdet.sample_fps,
            scdet_sample_width=self.config.scdet.sample_width,
            ffmpeg_bin=self.config.ffmpeg_bin,
        )
        vmafmotion_score = exp_saturation(
            motion_result.vmafmotion_mean,
            half_score_at=self.config.vmafmotion.mean_at_score_0_5,
        )
        scdet_score = linear_decay(
            motion_result.scdet_cut_ratio,
            a=self.config.scdet.cut_ratio_score_scale[0],
            b=self.config.scdet.cut_ratio_score_scale[1],
        )
        payload: dict[str, object] = {
            "vmafmotion": {
                "mean": motion_result.vmafmotion_mean,
                "frame_count": motion_result.vmafmotion_frame_count,
            },
            "scdet": {
                "score_max": motion_result.scdet_score_max,
                "cut_ratio": motion_result.scdet_cut_ratio,
                "frame_count": motion_result.scdet_frame_count,
            },
        }
        return round(vmafmotion_score * scdet_score, 6), payload
