from typing import Optional, Sequence, Dict

from allenact.base_abstractions.sensor import SensorSuite, Sensor

try:
    from allenact.embodiedai.sensors.vision_sensors import DepthSensor
except ImportError:
    raise ImportError("Please update to allenact>=0.4.0.")

from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from rearrange.sensors import UnshuffledRGBRearrangeSensor
from rearrange.tasks import RearrangeTaskSampler


class WalkthroughBaseExperimentConfig(RearrangeBaseExperimentConfig):
    SENSORS = [
        UnshuffledRGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid=RearrangeBaseExperimentConfig.UNSHUFFLED_RGB_UUID,
        ),
    ]

    # Sensor info
    EGOCENTRIC_RGB_UUID = RearrangeBaseExperimentConfig.UNSHUFFLED_RGB_UUID
    EGOCENTRIC_RGB_RESNET_UUID = (
        RearrangeBaseExperimentConfig.UNSHUFFLED_RGB_RESNET_UUID
    )

    THOR_CONTROLLER_KWARGS = {
        **RearrangeBaseExperimentConfig.THOR_CONTROLLER_KWARGS,
        "snapToGrid": False,
    }
    FORCE_AXIS_ALIGNED_START = False
    RANDOMIZE_START_ROTATION_DURING_TRAINING = True

    @classmethod
    def actions(cls):
        other_move_actions = (
            tuple()
            if not cls.INCLUDE_OTHER_MOVE_ACTIONS
            else ("move_left", "move_right", "move_back",)
        )
        return (
            ("done", "move_ahead",)
            + other_move_actions
            + (
                "rotate_right",
                "rotate_left",
                "stand",
                "crouch",
                "look_up",
                "look_down",
            )
        )

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
        thor_controller_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> RearrangeTaskSampler:
        """Return an RearrangeTaskSampler."""
        sensors = cls.SENSORS if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=True,
            run_unshuffle_phase=False,
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
                        isinstance(s, DepthSensor) for s in sensors
                    ),
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    ),
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            randomize_start_rotation=stage == "train"
            and cls.RANDOMIZE_START_ROTATION_DURING_TRAINING,
            **kwargs,
        )
