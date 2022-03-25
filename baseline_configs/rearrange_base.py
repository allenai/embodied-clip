import copy
import platform
from abc import abstractmethod
from typing import Optional, List, Sequence, Dict, Any, Tuple

import ai2thor.platform
import gym.spaces
import stringcase
import torch
import torchvision.models
from torch import nn, cuda, optim
from torch.optim.lr_scheduler import LambdaLR

import datagen.datagen_utils as datagen_utils
from allenact.base_abstractions.experiment_config import (
    ExperimentConfig,
    MachineParams,
    split_processes_onto_devices,
)
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import SensorSuite, Sensor, ExpertActionSensor
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.utils.experiment_utils import TrainingPipeline, LinearDecay, Builder
from allenact.utils.misc_utils import partition_sequence, md5_hash_str_as_int
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_sensors import (
    BinnedPointCloudMapTHORSensor,
    SemanticMapTHORSensor,
)
from allenact_plugins.ithor_plugin.ithor_util import get_open_x_displays
from rearrange.baseline_models import (
    RearrangeActorCriticSimpleConvRNN,
    ResNetRearrangeActorCriticRNN,
)
from rearrange.constants import (
    OBJECT_TYPES_WITH_PROPERTIES,
    THOR_COMMIT_ID,
)
from rearrange.environment import RearrangeMode


class RearrangeBaseExperimentConfig(ExperimentConfig):

    # Task parameters
    MAX_STEPS = {"walkthrough": 250, "unshuffle": 500}
    REQUIRE_DONE_ACTION = True
    FORCE_AXIS_ALIGNED_START = True
    RANDOMIZE_START_ROTATION_DURING_TRAINING = False

    # Environment parameters
    REARRANGE_ENV_KWARGS = dict(mode=RearrangeMode.SNAP,)
    SCREEN_SIZE = 224
    THOR_CONTROLLER_KWARGS = {
        "rotateStepDegrees": 90,
        "snapToGrid": True,
        "quality": "Very Low",
        "width": SCREEN_SIZE,
        "height": SCREEN_SIZE,
        "commit_id": THOR_COMMIT_ID,
        "fastActionEmit": True,
    }
    INCLUDE_OTHER_MOVE_ACTIONS = True

    # Training parameters
    TRAINING_STEPS = int(75e6)
    SAVE_INTERVAL = int(1e6)
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING: Optional[Tuple[str, str]] = None

    # Sensor info
    SENSORS: Optional[Sequence[Sensor]] = None
    EGOCENTRIC_RGB_UUID = "rgb"
    UNSHUFFLED_RGB_UUID = "unshuffled_rgb"
    EGOCENTRIC_RGB_RESNET_UUID = "rgb_resnet"
    UNSHUFFLED_RGB_RESNET_UUID = "unshuffled_rgb_resnet"

    # Actions
    PICKUP_ACTIONS = list(
        sorted(
            [
                f"pickup_{stringcase.snakecase(object_type)}"
                for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
                if properties["pickupable"]
            ]
        )
    )
    OPEN_ACTIONS = list(
        sorted(
            [
                f"open_by_type_{stringcase.snakecase(object_type)}"
                for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
                if properties["openable"] and not properties["pickupable"]
            ]
        )
    )

    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        return cls.SENSORS

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
                "drop_held_object_with_snap",
                *cls.OPEN_ACTIONS,
                *cls.PICKUP_ACTIONS,
            )
        )

    @classmethod
    def resnet_preprocessor_graph(cls, mode: str) -> SensorPreprocessorGraph:
        def create_resnet_builder(in_uuid: str, out_uuid: str):
            cnn_type, pretraining_type = cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING
            if pretraining_type == "imagenet":
                assert cnn_type in [
                    "RN18",
                    "RN50",
                ], "Only allow using RN18/RN50 with `imagenet` pretrained weights."
                return ResNetPreprocessor(
                    input_height=cls.THOR_CONTROLLER_KWARGS["height"],
                    input_width=cls.THOR_CONTROLLER_KWARGS["width"],
                    output_width=7,
                    output_height=7,
                    output_dims=512 if "18" in cnn_type else 2048,
                    pool=False,
                    torchvision_resnet_model=getattr(
                        torchvision.models, f"resnet{cnn_type.replace('RN', '')}"
                    ),
                    input_uuids=[in_uuid],
                    output_uuid=out_uuid,
                )
            elif pretraining_type == "clip":
                from allenact_plugins.clip_plugin.clip_preprocessors import (
                    ClipResNetPreprocessor,
                )
                import clip

                # Let's make sure we download the clip model now
                # so we don't download it on every spawned process
                clip.load(cnn_type, "cpu")

                return ClipResNetPreprocessor(
                    rgb_input_uuid=in_uuid,
                    clip_model_type=cnn_type,
                    pool=False,
                    output_uuid=out_uuid,
                )
            else:
                raise NotImplementedError

        img_uuids = [cls.EGOCENTRIC_RGB_UUID, cls.UNSHUFFLED_RGB_UUID]
        return SensorPreprocessorGraph(
            source_observation_spaces=SensorSuite(
                [
                    sensor
                    for sensor in cls.sensors()
                    if (mode == "train" or not isinstance(sensor, ExpertActionSensor))
                ]
            ).observation_spaces,
            preprocessors=[
                create_resnet_builder(sid, f"{sid}_resnet") for sid in img_uuids
            ],
        )

    @classmethod
    def get_lr_scheduler_builder(cls, use_lr_decay: bool):
        return (
            None
            if not use_lr_decay
            else Builder(
                LambdaLR,
                {
                    "lr_lambda": LinearDecay(
                        steps=cls.TRAINING_STEPS // 3, startp=1.0, endp=1.0 / 3
                    )
                },
            )
        )

    @classmethod
    def machine_params(cls, mode="train", **kwargs) -> MachineParams:
        """Return the number of processes and gpu_ids to use with training."""
        num_gpus = cuda.device_count()
        has_gpu = num_gpus != 0

        sampler_devices = None
        if mode == "train":
            nprocesses = cls.num_train_processes() if torch.cuda.is_available() else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )
        elif mode == "valid":
            devices = [num_gpus - 1] if has_gpu else [torch.device("cpu")]
            nprocesses = 2 if has_gpu else 0
        else:
            nprocesses = 20 if has_gpu else 1
            devices = (
                list(range(min(nprocesses, num_gpus)))
                if has_gpu
                else [torch.device("cpu")]
            )

        nprocesses = split_processes_onto_devices(
            nprocesses=nprocesses, ndevices=len(devices)
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices,
            sensor_preprocessor_graph=cls.resnet_preprocessor_graph(mode=mode)
            if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is not None
            else None,
        )

    @classmethod
    def stagewise_task_sampler_args(
        cls,
        stage: str,
        process_ind: int,
        total_processes: int,
        allowed_rearrange_inds_subset: Optional[Sequence[int]] = None,
        allowed_scenes: Sequence[str] = None,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        if allowed_scenes is not None:
            scenes = allowed_scenes
        elif stage == "combined":
            # Split scenes more evenly as the train scenes will have more episodes
            train_scenes = datagen_utils.get_scenes("train")
            other_scenes = datagen_utils.get_scenes("val") + datagen_utils.get_scenes(
                "test"
            )
            assert len(train_scenes) == 2 * len(other_scenes)
            scenes = []
            while len(train_scenes) != 0:
                scenes.append(train_scenes.pop())
                scenes.append(train_scenes.pop())
                scenes.append(other_scenes.pop())
            assert len(train_scenes) == len(other_scenes)
        else:
            scenes = datagen_utils.get_scenes(stage)

        if total_processes > len(scenes):
            assert stage == "train" and total_processes % len(scenes) == 0
            scenes = scenes * (total_processes // len(scenes))

        allowed_scenes = list(
            sorted(partition_sequence(seq=scenes, parts=total_processes,)[process_ind])
        )

        scene_to_allowed_rearrange_inds = None
        if allowed_rearrange_inds_subset is not None:
            allowed_rearrange_inds_subset = tuple(allowed_rearrange_inds_subset)
            assert stage in ["valid", "train_unseen"]
            scene_to_allowed_rearrange_inds = {
                scene: allowed_rearrange_inds_subset for scene in allowed_scenes
            }
        seed = md5_hash_str_as_int(str(allowed_scenes))

        device = (
            devices[process_ind % len(devices)]
            if devices is not None and len(devices) > 0
            else torch.device("cpu")
        )
        x_display: Optional[str] = None
        gpu_device: Optional[int] = None
        thor_platform: Optional[ai2thor.platform.BaseLinuxPlatform] = None
        if platform.system() == "Linux":
            try:
                x_displays = get_open_x_displays(throw_error_if_empty=True)

                if devices is not None and len(
                    [d for d in devices if d != torch.device("cpu")]
                ) > len(x_displays):
                    get_logger().warning(
                        f"More GPU devices found than X-displays (devices: `{x_displays}`, x_displays: `{x_displays}`)."
                        f" This is not necessarily a bad thing but may mean that you're not using GPU memory as"
                        f" efficiently as possible. Consider following the instructions here:"
                        f" https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
                        f" describing how to start an X-display on every GPU."
                    )
                x_display = x_displays[process_ind % len(x_displays)]
            except IOError:
                # Could not find an open `x_display`, use CloudRendering instead.
                assert all(
                    [d != torch.device("cpu") and d >= 0 for d in devices]
                ), "Cannot use CPU devices when there are no open x-displays as CloudRendering requires specifying a GPU."
                gpu_device = device
                thor_platform = ai2thor.platform.CloudRendering

        kwargs = {
            "stage": stage,
            "allowed_scenes": allowed_scenes,
            "scene_to_allowed_rearrange_inds": scene_to_allowed_rearrange_inds,
            "seed": seed,
            "x_display": x_display,
            "thor_controller_kwargs": {
                "gpu_device": gpu_device,
                "platform": thor_platform,
            },
        }

        sensors = kwargs.get("sensors", copy.deepcopy(cls.sensors()))
        kwargs["sensors"] = sensors

        sem_sensor = next(
            (s for s in kwargs["sensors"] if isinstance(s, SemanticMapTHORSensor)), None
        )
        binned_pc_sensor = next(
            (
                s
                for s in kwargs["sensors"]
                if isinstance(s, BinnedPointCloudMapTHORSensor)
            ),
            None,
        )

        if sem_sensor is not None:
            sem_sensor.device = torch.device(device)

        if binned_pc_sensor is not None:
            binned_pc_sensor.device = torch.device(device)

        if stage != "train":
            # Don't include several sensors during validation/testing
            kwargs["sensors"] = [
                s
                for s in kwargs["sensors"]
                if not isinstance(
                    s,
                    (
                        ExpertActionSensor,
                        SemanticMapTHORSensor,
                        BinnedPointCloudMapTHORSensor,
                    ),
                )
            ]
        return kwargs

    @classmethod
    def train_task_sampler_args(
        cls,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        return dict(
            force_cache_reset=False,
            epochs=float("inf"),
            **cls.stagewise_task_sampler_args(
                stage="train",
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    @classmethod
    def valid_task_sampler_args(
        cls,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ):
        return dict(
            force_cache_reset=True,
            epochs=1,
            **cls.stagewise_task_sampler_args(
                stage="valid",
                allowed_rearrange_inds_subset=tuple(range(10)),
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    @classmethod
    def test_task_sampler_args(
        cls,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
        task_spec_in_metrics: bool = False,
    ):
        task_spec_in_metrics = False

        # Train_unseen
        # stage = "train_unseen"
        # allowed_rearrange_inds_subset = list(range(15))

        # Val
        # stage = "val"
        # allowed_rearrange_inds_subset = None

        # Test
        # stage = "test"
        # allowed_rearrange_inds_subset = None

        # Combined (Will run inference on all datasets)
        stage = "combined"
        allowed_rearrange_inds_subset = None

        return dict(
            force_cache_reset=True,
            epochs=1,
            task_spec_in_metrics=task_spec_in_metrics,
            **cls.stagewise_task_sampler_args(
                stage=stage,
                allowed_rearrange_inds_subset=allowed_rearrange_inds_subset,
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            ),
        )

    @classmethod
    @abstractmethod
    def _training_pipeline_info(cls) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def num_train_processes(cls) -> int:
        raise NotImplementedError

    @classmethod
    def training_pipeline(cls, **kwargs) -> TrainingPipeline:
        info = cls._training_pipeline_info()

        return TrainingPipeline(
            gamma=info.get("gamma", 0.99),
            use_gae=info.get("use_gae", True),
            gae_lambda=info.get("gae_lambda", 0.95),
            num_steps=info["num_steps"],
            num_mini_batch=info["num_mini_batch"],
            update_repeats=info["update_repeats"],
            max_grad_norm=info.get("max_grad_norm", 0.5),
            save_interval=cls.SAVE_INTERVAL,
            named_losses=info["named_losses"],
            metric_accumulate_interval=cls.num_train_processes()
            * max(*cls.MAX_STEPS.values())
            if torch.cuda.is_available()
            else 1,
            optimizer_builder=Builder(optim.Adam, dict(lr=info["lr"])),
            advance_scene_rollout_period=None,
            pipeline_stages=info["pipeline_stages"],
            lr_scheduler_builder=cls.get_lr_scheduler_builder(
                use_lr_decay=info["use_lr_decay"]
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        if cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING is None:
            return RearrangeActorCriticSimpleConvRNN(
                action_space=gym.spaces.Discrete(len(cls.actions())),
                observation_space=SensorSuite(cls.sensors()).observation_spaces,
                rgb_uuid=cls.EGOCENTRIC_RGB_UUID,
                unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_UUID,
            )
        else:
            return ResNetRearrangeActorCriticRNN(
                action_space=gym.spaces.Discrete(len(cls.actions())),
                observation_space=kwargs[
                    "sensor_preprocessor_graph"
                ].observation_spaces,
                rgb_uuid=cls.EGOCENTRIC_RGB_RESNET_UUID,
                unshuffled_rgb_uuid=cls.UNSHUFFLED_RGB_RESNET_UUID,
            )
