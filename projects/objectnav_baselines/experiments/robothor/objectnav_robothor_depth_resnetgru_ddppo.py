from torchvision import models

from plugins.habitat_plugin.habitat_preprocessors import ResnetPreProcessorHabitat
from plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_ddppo_base import (
    ObjectNavRoboThorPPOBaseExperimentConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_resnetgru_base import (
    ObjectNavRoboThorResNetGRUBaseExperimentConfig,
)
from utils.experiment_utils import Builder


class ObjectNavRoboThorRGBPPOExperimentConfig(
    ObjectNavRoboThorPPOBaseExperimentConfig,
    ObjectNavRoboThorResNetGRUBaseExperimentConfig,
):
    """An Object Navigation experiment configuration in RoboThor with Depth
    input."""

    SENSORS = (
        DepthSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_normalization=True,
            uuid="depth_lowres",
        ),
        GoalObjectTypeThorSensor(
            object_types=ObjectNavRoboThorBaseConfig.TARGET_TYPES,
        ),
    )

    PREPROCESSORS = [
        Builder(
            ResnetPreProcessorHabitat,
            {
                "input_height": ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
                "input_width": ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
                "output_width": 7,
                "output_height": 7,
                "output_dims": 512,
                "pool": False,
                "torchvision_resnet_model": models.resnet18,
                "input_uuids": ["depth_lowres"],
                "output_uuid": "depth_resnet",
                "parallel": False,
            },
        ),
    ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-Depth-ResNetGRU-DDPPO"
