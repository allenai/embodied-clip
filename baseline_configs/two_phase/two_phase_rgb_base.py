from abc import ABC
from typing import Optional, Sequence, Dict, Type, Union

import gym
import gym.spaces
from torch import nn

from allenact.base_abstractions.sensor import SensorSuite, Sensor

try:
    from allenact.embodiedai.sensors.vision_sensors import DepthSensor
except ImportError:
    raise ImportError("Please update to allenact>=0.4.0.")

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

    TRAIN_UNSHUFFLE_RUNS_PER_WALKTHROUGH: int = 1
    IS_WALKTHROUGH_PHASE_EMBEDING_DIM: int = 32
    RNN_TYPE: str = "LSTM"

    @classmethod
    def make_sampler_fn(
        cls,
        stage: str,
        force_cache_reset: bool,
        allowed_scenes: Optional[Sequence[str]],
        seed: int,
        epochs: Union[str, float, int],
        scene_to_allowed_rearrange_inds: Optional[Dict[str, Sequence[int]]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        only_one_unshuffle_per_walkthrough: bool = False,
        thor_controller_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> RearrangeTaskSampler:
        """Return a RearrangeTaskSampler."""
        sensors = cls.SENSORS if sensors is None else sensors

        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING

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
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    ),
                    "renderDepthImage": any(
                        isinstance(s, DepthSensor) for s in sensors
                    ),
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            unshuffle_runs_per_walkthrough=cls.TRAIN_UNSHUFFLE_RUNS_PER_WALKTHROUGH
            if (not only_one_unshuffle_per_walkthrough) and stage == "train"
            else None,
            epochs=epochs,
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

        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None:
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
