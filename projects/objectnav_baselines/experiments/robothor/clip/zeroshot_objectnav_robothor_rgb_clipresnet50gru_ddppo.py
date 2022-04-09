from typing import Sequence, Union

import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.utils.experiment_utils import Builder, TrainingPipeline
from allenact_plugins.clip_plugin.clip_preprocessors import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import (
    GoalObjectTypeThorSensor,
    RGBSensorThor,
)
from projects.objectnav_baselines.experiments.clip.zeroshot_mixins import (
    ZeroshotClipResNetPreprocessGRUActorCriticMixin,
)
from projects.objectnav_baselines.experiments.robothor.zeroshot_objectnav_robothor_base import (
    ZeroshotObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.experiments.robothor.clip.objectnav_robothor_rgb_clipresnet50gru_ddppo import (
    ObjectNavRoboThorClipRGBPPOExperimentConfig,
)
from projects.objectnav_baselines.mixins import ObjectNavPPOMixin


class ZeroshotObjectNavRoboThorClipRGBPPOExperimentConfig(
    ZeroshotObjectNavRoboThorBaseConfig,
    ObjectNavRoboThorClipRGBPPOExperimentConfig
):
    """A Zeroshot CLIP Object Navigation experiment configuration in RoboThor
    with RGB input."""

    CLIP_MODEL_TYPE = "RN50"

    SENSORS = [
        RGBSensorThor(
            height=ZeroshotObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ZeroshotObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ZeroshotObjectNavRoboThorBaseConfig.TARGET_TYPES,
            uuid="goal_object_type_ind",
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.preprocessing_and_model = ZeroshotClipResNetPreprocessGRUActorCriticMixin(
            sensors=self.SENSORS,
            clip_model_type=self.CLIP_MODEL_TYPE,
            screen_size=self.SCREEN_SIZE,
            goal_sensor_type=GoalObjectTypeThorSensor,
            pool=True,
            pooling_type='attn',
            target_types=self.TARGET_TYPES
        )

    @classmethod
    def tag(cls):
        return "Zeroshot-ObjectNav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO"
