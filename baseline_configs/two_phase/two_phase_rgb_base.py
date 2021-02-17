from abc import ABC
from typing import Optional, Sequence, Dict, Type

import gym
import gym.spaces
from allenact.base_abstractions.sensor import SensorSuite, Sensor, DepthSensor
from torch import nn

from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from rearrange.baseline_models import (
    TwoPhaseRearrangeActorCriticSimpleConvRNN,
    ResNetTwoPhaseRearrangeActorCriticRNN,
)
from rearrange.sensors import ClosestUnshuffledRGBRearrangeSensor
from rearrange.sensors import (
    RGBRearrangeSensor,
    InWalkthroughPhaseSensor,
)
from rearrange.tasks import RearrangeTaskSampler


class TwoPhaseRGBBaseExperimentConfig(RearrangeBaseExperimentConfig, ABC):
    SENSORS = [
        RGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid=RearrangeBaseExperimentConfig.EGOCENTRIC_RGB_UUID,
        ),
        ClosestUnshuffledRGBRearrangeSensor(
            height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid=RearrangeBaseExperimentConfig.UNSHUFFLED_RGB_UUID,
        ),
        InWalkthroughPhaseSensor(),
    ]

    TRAIN_UNSHUFFLE_REPEATS: int = 5
    IS_WALKTHROUGH_PHASE_EMBEDING_DIM: int = 32
    RNN_TYPE: str = "LSTM"

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
        disable_unshuffle_repeats: bool = False,
        **kwargs,
    ) -> RearrangeTaskSampler:
        """Return a RearrangeTaskSampler."""
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=True,
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
            unshuffle_repeats=cls.TRAIN_UNSHUFFLE_REPEATS
            if (not disable_unshuffle_repeats) and stage == "train"
            else None,
            **kwargs,
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        def get_sensor_uuid(stype: Type[Sensor]) -> Optional[str]:
            s = next((s for s in cls.SENSORS if isinstance(s, stype)), None,)
            return None if s is None else s.uuid

        walkthrougher_should_ignore_action_mask = [
            any(k in a for k in ["drop", "open", "pickup"]) for a in cls.actions()
        ]

        if not cls.USE_RESNET_CNN:
            return TwoPhaseRearrangeActorCriticSimpleConvRNN(
                action_space=gym.spaces.Discrete(len(cls.actions())),
                observation_space=SensorSuite(cls.SENSORS).observation_spaces,
                rgb_uuid=cls.EGOCENTRIC_RGB_UUID,
                unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_UUID,
                in_walkthrough_phase_uuid=get_sensor_uuid(InWalkthroughPhaseSensor),
                is_walkthrough_phase_embedding_dim=cls.IS_WALKTHROUGH_PHASE_EMBEDING_DIM,
                rnn_type=cls.RNN_TYPE,
                walkthrougher_should_ignore_action_mask=walkthrougher_should_ignore_action_mask,
                done_action_index=cls.actions().index("done"),
            )
        else:
            return ResNetTwoPhaseRearrangeActorCriticRNN(
                action_space=gym.spaces.Discrete(len(cls.actions())),
                observation_space=kwargs[
                    "sensor_preprocessor_graph"
                ].observation_spaces,
                rgb_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
                unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_RESNET_UUID,
                in_walkthrough_phase_uuid=get_sensor_uuid(InWalkthroughPhaseSensor),
                is_walkthrough_phase_embedding_dim=cls.IS_WALKTHROUGH_PHASE_EMBEDING_DIM,
                rnn_type=cls.RNN_TYPE,
                walkthrougher_should_ignore_action_mask=walkthrougher_should_ignore_action_mask,
                done_action_index=cls.actions().index("done"),
            )
