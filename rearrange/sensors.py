from typing import Any, Optional, Union

import gym.spaces
import numpy as np
from allenact.base_abstractions.sensor import RGBSensor, Sensor
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.robothor_plugin.robothor_sensors import DepthSensorThor

from rearrange.constants import STEP_SIZE
from rearrange.environment import RearrangeTHOREnvironment
from rearrange.tasks import (
    UnshuffleTask,
    WalkthroughTask,
    AbstractRearrangeTask,
)


class RGBRearrangeSensor(
    RGBSensor[RearrangeTHOREnvironment, Union[WalkthroughTask, UnshuffleTask]]
):
    def frame_from_env(
        self, env: RearrangeTHOREnvironment, task: Union[WalkthroughTask, UnshuffleTask]
    ) -> np.ndarray:
        if isinstance(task, WalkthroughTask):
            return task.walkthrough_env.last_event.frame.copy()
        elif isinstance(task, UnshuffleTask):
            return task.unshuffle_env.last_event.frame.copy()
        else:
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `WalkthroughTask` or an `UnshuffleTask`."
            )


class DepthRearrangeSensor(DepthSensorThor):
    def frame_from_env(
        self, env: RearrangeTHOREnvironment, task: Union[WalkthroughTask, UnshuffleTask]
    ) -> np.ndarray:
        if isinstance(task, WalkthroughTask):
            return task.walkthrough_env.last_event.depth_frame.copy()
        elif isinstance(task, UnshuffleTask):
            return task.unshuffle_env.last_event.depth_frame.copy()
        else:
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `WalkthroughTask` or an `UnshuffleTask`."
            )


class UnshuffledRGBRearrangeSensor(
    RGBSensor[RearrangeTHOREnvironment, Union[WalkthroughTask, UnshuffleTask]]
):
    def frame_from_env(
        self, env: RearrangeTHOREnvironment, task: Union[WalkthroughTask, UnshuffleTask]
    ) -> np.ndarray:
        walkthrough_env = task.walkthrough_env
        if not isinstance(task, WalkthroughTask):
            unshuffle_loc = task.unshuffle_env.get_agent_location()
            walkthrough_agent_loc = walkthrough_env.get_agent_location()

            unshuffle_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                unshuffle_loc
            )
            walkthrough_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                walkthrough_agent_loc
            )

            if unshuffle_loc_tuple != walkthrough_loc_tuple:
                walkthrough_env.controller.step(
                    "TeleportFull",
                    x=unshuffle_loc["x"],
                    y=unshuffle_loc["y"],
                    z=unshuffle_loc["z"],
                    horizon=unshuffle_loc["horizon"],
                    rotation={"x": 0, "y": unshuffle_loc["rotation"], "z": 0},
                    standing=unshuffle_loc["standing"] == 1,
                    forceAction=True,
                )
        return walkthrough_env.last_event.frame.copy()


class ClosestUnshuffledRGBRearrangeSensor(
    RGBSensor[RearrangeTHOREnvironment, Union[WalkthroughTask, UnshuffleTask]]
):
    ROT_TO_FORWARD = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

    def frame_from_env(
        self, env: RearrangeTHOREnvironment, task: Union[WalkthroughTask, UnshuffleTask]
    ) -> np.ndarray:
        walkthrough_env = task.walkthrough_env
        if not isinstance(task, WalkthroughTask):
            walkthrough_visited_locs = (
                task.locations_visited_in_walkthrough
            )  # A (num unique visited) x 4 matrix
            assert walkthrough_visited_locs is not None

            current_loc = np.array(task.agent_location_tuple).reshape((1, -1))

            diffs = walkthrough_visited_locs - current_loc

            xz_dist = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)

            rot_diff = np.array(diffs[:, 2].round(), dtype=int) % 360
            rot_diff = np.minimum(rot_diff, 360 - rot_diff)
            rot_dist = 100 * (180 == rot_diff) + 2 * (90 == rot_diff)

            stand_dist = np.abs(diffs[:, 3]) * STEP_SIZE / 2

            horizon_dist = np.abs(diffs[:, 4]) * STEP_SIZE / 2

            x, z, rotation, standing, horizon = tuple(
                walkthrough_visited_locs[
                    np.argmin(xz_dist + rot_dist + stand_dist + horizon_dist), :
                ]
            )

            walkthrough_env = task.walkthrough_env
            assert task.unshuffle_env.scene == walkthrough_env.scene

            walkthrough_agent_loc = walkthrough_env.get_agent_location()
            walkthrough_loc_tuple = AbstractRearrangeTask.agent_location_to_tuple(
                walkthrough_agent_loc
            )
            if walkthrough_loc_tuple != (x, z, rotation, standing, horizon):
                walkthrough_env.controller.step(
                    "TeleportFull",
                    x=x,
                    y=walkthrough_agent_loc["y"],
                    z=z,
                    horizon=horizon,
                    rotation={"x": 0, "y": rotation, "z": 0},
                    standing=standing == 1,
                    forceAction=True,
                )
        return walkthrough_env.last_event.frame.copy()


class InWalkthroughPhaseSensor(
    Sensor[RearrangeTHOREnvironment, Union[UnshuffleTask, WalkthroughTask]]
):
    def __init__(self, uuid: str = "in_walkthrough_phase", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=False, high=True, shape=(1,), dtype=np.bool
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: RearrangeTHOREnvironment,
        task: Optional[UnshuffleTask],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(task, WalkthroughTask):
            return np.array([True], dtype=bool)
        elif isinstance(task, UnshuffleTask):
            return np.array([False], dtype=bool)
        else:
            raise NotImplementedError(
                f"Unknown task type {type(task)}, must be an `WalkthroughTask` or an `UnshuffleTask`."
            )
