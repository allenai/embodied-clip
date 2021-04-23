from typing import Dict, Any, cast

import gym
import torch

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.sensor import SensorSuite
from allenact.embodiedai.mapping.mapping_losses import (
    BinnedPointCloudMapLoss,
    SemanticMapFocalLoss,
)
from allenact.utils.experiment_utils import LinearDecay, PipelineStage
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RelativePositionChangeTHORSensor,
    ReachableBoundsTHORSensor,
    BinnedPointCloudMapTHORSensor,
    SemanticMapTHORSensor,
)
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor
from baseline_configs.walkthrough.walkthrough_rgb_base import (
    WalkthroughBaseExperimentConfig,
)
from rearrange.baseline_models import WalkthroughActorCriticResNetWithPassiveMap
from rearrange.constants import (
    FOV,
    PICKUPABLE_OBJECTS,
    OPENABLE_OBJECTS,
)


class WalkthroughRGBMappingPPOExperimentConfig(WalkthroughBaseExperimentConfig):
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

    SENSORS = WalkthroughBaseExperimentConfig.SENSORS + [
        RelativePositionChangeTHORSensor(),
        MAP_RANGE_SENSOR,
        DepthSensorThor(
            height=WalkthroughBaseExperimentConfig.SCREEN_SIZE,
            width=WalkthroughBaseExperimentConfig.SCREEN_SIZE,
            use_normalization=False,
            uuid="depth",
        ),
        BinnedPointCloudMapTHORSensor(fov=FOV, **MAP_INFO),
        SemanticMapTHORSensor(
            fov=FOV, **MAP_INFO, ordered_object_types=ORDERED_OBJECT_TYPES,
        ),
    ]

    @classmethod
    def tag(cls) -> str:
        return "WalkthroughRGBMappingPPO"

    @classmethod
    def num_train_processes(cls) -> int:
        return max(1, torch.cuda.device_count() * 5)

    @classmethod
    def create_model(cls, **kwargs) -> WalkthroughActorCriticResNetWithPassiveMap:
        map_sensor = cast(
            BinnedPointCloudMapTHORSensor,
            next(
                s for s in cls.SENSORS if isinstance(s, BinnedPointCloudMapTHORSensor)
            ),
        )
        map_kwargs = dict(
            frame_height=224,
            frame_width=224,
            vision_range_in_cm=map_sensor.vision_range_in_cm,
            resolution_in_cm=map_sensor.resolution_in_cm,
            map_size_in_cm=map_sensor.map_size_in_cm,
        )

        observation_space = (
            SensorSuite(cls.SENSORS).observation_spaces
            if kwargs.get("sensor_preprocessor_graph") is None
            else kwargs["sensor_preprocessor_graph"].observation_spaces
        )

        return WalkthroughActorCriticResNetWithPassiveMap(
            action_space=gym.spaces.Discrete(len(cls.actions())),
            observation_space=observation_space,
            rgb_uuid=cls.EGOCENTRIC_RGB_UUID,
            unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_UUID,
            semantic_map_channels=len(cls.ORDERED_OBJECT_TYPES),
            height_map_channels=3,
            map_kwargs=map_kwargs,
        )

    @classmethod
    def _training_pipeline_info(cls, **kwargs) -> Dict[str, Any]:
        """Define how the model trains."""

        training_steps = cls.TRAINING_STEPS
        return dict(
            named_losses=dict(
                ppo_loss=PPO(clip_decay=LinearDecay(training_steps), **PPOConfig),
                binned_map_loss=BinnedPointCloudMapLoss(
                    binned_pc_uuid="binned_pc_map",
                    map_logits_uuid="ego_height_binned_map_logits",
                ),
                semantic_map_loss=SemanticMapFocalLoss(
                    semantic_map_uuid="semantic_map",
                    map_logits_uuid="ego_semantic_map_logits",
                ),
            ),
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss", "binned_map_loss", "semantic_map_loss"],
                    loss_weights=[1.0, 1.0, 100.0],
                    max_stage_steps=training_steps,
                )
            ],
            num_steps=32,
            num_mini_batch=1,
            update_repeats=3,
            use_lr_decay=True,
            lr=3e-4,
        )
