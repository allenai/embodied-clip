import os
from typing import Type, Optional

import gym
import torch
from torch import nn

from allenact.base_abstractions.sensor import SensorSuite, Sensor, ExpertActionSensor
from allenact.embodiedai.mapping.mapping_models.active_neural_slam import (
    ActiveNeuralSLAM,
)
from allenact.utils.misc_utils import multiprocessing_safe_download_file_from_url
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RelativePositionChangeTHORSensor,
    ReachableBoundsTHORSensor,
)
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from baseline_configs.two_phase.two_phase_rgb_resnet_ppowalkthrough_ilunshuffle import (
    TwoPhaseRGBResNetPPOWalkthroughILUnshuffleExperimentConfig,
)
from rearrange.baseline_models import TwoPhaseRearrangeActorCriticFrozenMap
from rearrange.constants import (
    PICKUPABLE_OBJECTS,
    OPENABLE_OBJECTS,
)
from rearrange.sensors import (
    InWalkthroughPhaseSensor,
    RGBRearrangeSensor,
    ClosestUnshuffledRGBRearrangeSensor,
)
from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR


class TwoPhaseRGBResNetFrozenMapPPOWalkthroughILUnshuffleExperimentConfig(
    TwoPhaseRGBResNetPPOWalkthroughILUnshuffleExperimentConfig
):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = (
        None  # Not necessary as we're handling things in the model
    )
    IL_PIPELINE_TYPE: str = "40proc-longtf"

    ORDERED_OBJECT_TYPES = list(sorted(PICKUPABLE_OBJECTS + OPENABLE_OBJECTS))

    MAP_RANGE_SENSOR = ReachableBoundsTHORSensor(margin=1.0)

    MAP_INFO = dict(
        map_range_sensor=MAP_RANGE_SENSOR,
        vision_range_in_cm=40 * 5,
        map_size_in_cm=1050
        if isinstance(MAP_RANGE_SENSOR, ReachableBoundsTHORSensor)
        else 2200,
        resolution_in_cm=5,
    )

    SENSORS = [
        ExpertActionSensor(len(RearrangeBaseExperimentConfig.actions())),
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
        RelativePositionChangeTHORSensor(),
        MAP_RANGE_SENSOR,
    ]

    @classmethod
    def tag(cls) -> str:
        return f"TwoPhaseRGBResNetFrozenMapPPOWalkthroughILUnshuffle_{cls.IL_PIPELINE_TYPE}"

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        def get_sensor_uuid(stype: Type[Sensor]) -> Optional[str]:
            s = next((s for s in cls.SENSORS if isinstance(s, stype)), None,)
            return None if s is None else s.uuid

        walkthrougher_should_ignore_action_mask = [
            any(k in a for k in ["drop", "open", "pickup"]) for a in cls.actions()
        ]

        map_kwargs = dict(
            frame_height=224,
            frame_width=224,
            vision_range_in_cm=cls.MAP_INFO["vision_range_in_cm"],
            resolution_in_cm=cls.MAP_INFO["resolution_in_cm"],
            map_size_in_cm=cls.MAP_INFO["map_size_in_cm"],
        )

        observation_space = (
            SensorSuite(cls.SENSORS).observation_spaces
            if kwargs.get("sensor_preprocessor_graph") is None
            else kwargs["sensor_preprocessor_graph"].observation_spaces
        )

        semantic_map_channels = len(cls.ORDERED_OBJECT_TYPES)
        height_map_channels = 3
        map_kwargs["n_map_channels"] = height_map_channels + semantic_map_channels
        frozen_map = ActiveNeuralSLAM(**map_kwargs, use_resnet_layernorm=True)

        pretrained_map_ckpt_path = os.path.join(
            ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR,
            "pretrained_model_ckpts",
            "pretrained_active_neural_slam_via_walkthrough_75m.pt",
        )
        multiprocessing_safe_download_file_from_url(
            url="https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/rearrangement/walkthrough/pretrained_active_neural_slam_via_walkthrough_75m.pt",
            save_path=pretrained_map_ckpt_path,
        )
        frozen_map.load_state_dict(
            torch.load(pretrained_map_ckpt_path, map_location="cpu",)
        )

        return TwoPhaseRearrangeActorCriticFrozenMap(
            map=frozen_map,
            semantic_map_channels=semantic_map_channels,
            height_map_channels=height_map_channels,
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=observation_space,
            rgb_uuid=cls.EGOCENTRIC_RGB_UUID,
            in_walkthrough_phase_uuid=get_sensor_uuid(InWalkthroughPhaseSensor),
            is_walkthrough_phase_embedding_dim=cls.IS_WALKTHROUGH_PHASE_EMBEDING_DIM,
            rnn_type=cls.RNN_TYPE,
            walkthrougher_should_ignore_action_mask=walkthrougher_should_ignore_action_mask,
            done_action_index=cls.actions().index("done"),
        )
