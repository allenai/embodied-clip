from typing import Sequence, Union, Type, List

import attr
import gym
import numpy as np
import torch.nn as nn

from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.clip_plugin.clip_preprocessors import (
    ClipResNetPreprocessor,
    ClipTextPreprocessor,
)
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact_plugins.clip_plugin.clip_zeroshot_objectnav_models import (
    CLIPZeroshotObjectNavActorCritic,
)


@attr.s(kw_only=True)
class ZeroshotClipResNetPreprocessGRUActorCriticMixin:
    sensors: Sequence[Sensor] = attr.ib()
    clip_model_type: str = attr.ib()
    screen_size: int = attr.ib()
    goal_sensor_type: Type[Sensor] = attr.ib()
    pool: bool = attr.ib(default=False)
    pooling_type: str = attr.ib()
    target_types: List[str] = attr.ib()


    def preprocessors(self) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        rgb_sensor = next((s for s in self.sensors if isinstance(s, RGBSensor)), None)
        goal_sensor = next((s for s in self.sensors if isinstance(s, GoalObjectTypeThorSensor)), None)

        assert rgb_sensor is not None and goal_sensor is not None

        assert (
            np.linalg.norm(
                np.array(rgb_sensor._norm_means)
                - np.array(ClipResNetPreprocessor.CLIP_RGB_MEANS)
            )
            < 1e-5
        )
        assert (
            np.linalg.norm(
                np.array(rgb_sensor._norm_sds)
                - np.array(ClipResNetPreprocessor.CLIP_RGB_STDS)
            )
            < 1e-5
        )

        preprocessors = [
            ClipResNetPreprocessor(
                rgb_input_uuid=rgb_sensor.uuid,
                clip_model_type=self.clip_model_type,
                pool=self.pool,
                pooling_type=self.pooling_type,
                output_uuid="rgb_clip_resnet"
            ),
            ClipTextPreprocessor(
                goal_sensor_uuid=goal_sensor.uuid,
                clip_model_type=self.clip_model_type,
                object_types=self.target_types,
                output_uuid="goal_object_clip",
            )
        ]

        return preprocessors

    def create_model(self, num_actions: int, **kwargs) -> nn.Module:
        goal_sensor_uuid = next(
            (s.uuid for s in self.sensors if isinstance(s, self.goal_sensor_type)),
            None,
        )

        return CLIPZeroshotObjectNavActorCritic(
            action_space=gym.spaces.Discrete(num_actions),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid=goal_sensor_uuid,
            hidden_size=1024,
            clip_rgb_preprocessor_uuid='rgb_clip_resnet',
            clip_text_preprocessor_uuid='goal_object_clip',
            clip_embedding_dim = 1024
        )
