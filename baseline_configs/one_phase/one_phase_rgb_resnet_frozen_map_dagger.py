import os
from typing import Sequence

import gym
import torch
from torch import nn

from allenact.base_abstractions.sensor import SensorSuite, Sensor
from allenact.embodiedai.mapping.mapping_models.active_neural_slam import (
    ActiveNeuralSLAM,
)
from allenact.utils.misc_utils import multiprocessing_safe_download_file_from_url
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RelativePositionChangeTHORSensor,
    ReachableBoundsTHORSensor,
)
from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)
from rearrange.baseline_models import OnePhaseRearrangeActorCriticFrozenMap
from rearrange.constants import (
    PICKUPABLE_OBJECTS,
    OPENABLE_OBJECTS,
)
from rearrange_constants import ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR


class OnePhaseRGBResNetFrozenMapDaggerExperimentConfig(
    OnePhaseRGBILBaseExperimentConfig
):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = (
        None  # Not necessary as we're handling things in the model
    )
    IL_PIPELINE_TYPE = "40proc"

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

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return list(
            super(OnePhaseRGBResNetFrozenMapDaggerExperimentConfig, cls).sensors()
        ) + [RelativePositionChangeTHORSensor(), cls.MAP_RANGE_SENSOR,]

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBResNetFrozenMapDagger_{cls.IL_PIPELINE_TYPE}"

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        map_kwargs = dict(
            frame_height=224,
            frame_width=224,
            vision_range_in_cm=cls.MAP_INFO["vision_range_in_cm"],
            resolution_in_cm=cls.MAP_INFO["resolution_in_cm"],
            map_size_in_cm=cls.MAP_INFO["map_size_in_cm"],
        )

        observation_space = (
            SensorSuite(cls.sensors()).observation_spaces
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

        return OnePhaseRearrangeActorCriticFrozenMap(
            map=frozen_map,
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=observation_space,
            rgb_uuid=cls.EGOCENTRIC_RGB_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_UUID,
            semantic_map_channels=semantic_map_channels,
            height_map_channels=height_map_channels,
        )
