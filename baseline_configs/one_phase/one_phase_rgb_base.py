from abc import ABC
from typing import Optional, Dict, Sequence

from allenact.base_abstractions.sensor import SensorSuite, Sensor, DepthSensor

from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from rearrange.sensors import (
    RGBRearrangeSensor,
    UnshuffledRGBRearrangeSensor,
)
from rearrange.tasks import RearrangeTaskSampler


class OnePhaseRGBBaseExperimentConfig(RearrangeBaseExperimentConfig, ABC):
    SENSORS = [
        RGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid=RearrangeBaseExperimentConfig.EGOCENTRIC_RGB_UUID,
        ),
        UnshuffledRGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid=RearrangeBaseExperimentConfig.UNSHUFFLED_RGB_UUID,
        ),
    ]

    @classmethod
    def make_sampler_fn(
        cls,
        stage: str,
        force_cache_reset: bool,
        allowed_scenes: Optional[Sequence[str]],
        seed: int,
        scene_to_allowed_rearrange_inds: Optional[Dict[str, Sequence[int]]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        **kwargs,
    ) -> RearrangeTaskSampler:
        """Return a RearrangeTaskSampler."""
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=False,
            run_unshuffle_phase=True,
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            rearrange_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **cls.REARRANGE_ENV_KWARGS,
                controller_kwargs={
                    "x_display": x_display,
                    **cls.THOR_CONTROLLER_KWARGS,
                    "renderDepthImage": any(
                        isinstance(s, DepthSensor) for s in cls.SENSORS
                    ),
                },
            ),
            seed=seed,
            sensors=SensorSuite(cls.SENSORS)
            if sensors is None
            else SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            **kwargs,
        )
