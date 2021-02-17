"""Include the Task and TaskSampler to train on a single unshuffle instance."""
import copy
import os
import random
import traceback
from abc import ABC
from typing import Any, Tuple, Optional, Dict, Sequence, List, Union, cast, Set

import compress_pickle
import gym.spaces
import numpy as np
import stringcase
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor

from rearrange.constants import STARTER_DATA_DIR, STEP_SIZE
from rearrange.environment import (
    RearrangeTHOREnvironment,
    RearrangeTaskSpec,
)
from rearrange.expert import (
    GreedyUnshuffleExpert,
    ShortestPathNavigatorTHOR,
)
from rearrange.utils import (
    RearrangeActionSpace,
    include_object_data,
)


class AbstractRearrangeTask(Task, ABC):
    @staticmethod
    def agent_location_to_tuple(
        agent_loc: Dict[str, Union[Dict[str, float], bool, float, int]]
    ) -> Tuple[float, float, int, int, int]:
        if "position" in agent_loc:
            agent_loc = {
                "x": agent_loc["position"]["x"],
                "y": agent_loc["position"]["y"],
                "z": agent_loc["position"]["z"],
                "rotation": agent_loc["rotation"]["y"],
                "horizon": agent_loc["cameraHorizon"],
                "standing": agent_loc.get("isStanding"),
            }
        return (
            round(agent_loc["x"], 2),
            round(agent_loc["z"], 2),
            round_to_factor(agent_loc["rotation"], 90) % 360,
            1 * agent_loc["standing"],
            round_to_factor(agent_loc["horizon"], 30) % 360,
        )

    @property
    def agent_location_tuple(self) -> Tuple[float, float, int, int, int]:
        return self.agent_location_to_tuple(self.env.get_agent_location())


class UnshuffleTask(AbstractRearrangeTask):
    def __init__(
        self,
        sensors: SensorSuite,
        unshuffle_env: RearrangeTHOREnvironment,
        walkthrough_env: RearrangeTHOREnvironment,
        max_steps: int,
        discrete_actions: Tuple[str, ...],
        require_done_action: bool = False,
        locations_visited_in_walkthrough: Optional[np.ndarray] = None,
        object_names_seen_in_walkthrough: Set[str] = None,
        metrics_from_walkthrough: Optional[Dict[str, Any]] = None,
        task_spec_in_metrics: bool = False,
    ) -> None:
        """Create a new unshuffle task."""
        super().__init__(
            env=unshuffle_env, sensors=sensors, task_info=dict(), max_steps=max_steps
        )
        self.unshuffle_env = unshuffle_env
        self.walkthrough_env = walkthrough_env

        self.discrete_actions = discrete_actions
        self.require_done_action = require_done_action

        self.locations_visited_in_walkthrough = locations_visited_in_walkthrough
        self.object_names_seen_in_walkthrough = object_names_seen_in_walkthrough
        self.metrics_from_walkthrough = metrics_from_walkthrough
        self.task_spec_in_metrics = task_spec_in_metrics

        self._took_end_action: bool = False

        # TODO: add better typing to the dicts
        self._previous_state_trackers: Optional[Dict[str, Any]] = None
        self.states_visited: dict = dict(
            picked_up=dict(soap_bottle=False, pan=False, knife=False),
            opened_drawer=False,
            successfully_placed=dict(soap_bottle=False, pan=False, knife=False),
        )

        _, gps, cps = self.unshuffle_env.poses
        self.start_energies = self.unshuffle_env.pose_difference_energy(
            goal_pose=gps, cur_pose=cps
        )
        self.last_pose_energy = self.start_energies.sum()

        self.greedy_expert: Optional[GreedyUnshuffleExpert] = None
        self.actions_taken = []
        self.actions_taken_success = []
        self.agent_locs = [self.unshuffle_env.get_agent_location()]

    def query_expert(self, **kwargs) -> Tuple[Any, bool]:
        if self.greedy_expert is None:
            if not hasattr(self.unshuffle_env, "shortest_path_navigator"):
                # TODO: This is a bit hacky
                self.unshuffle_env.shortest_path_navigator = ShortestPathNavigatorTHOR(
                    controller=self.unshuffle_env.controller,
                    grid_size=STEP_SIZE,
                    include_move_left_right=all(
                        f"move_{k}" in self.action_names() for k in ["left", "right"]
                    ),
                )

            self.greedy_expert = GreedyUnshuffleExpert(
                task=self,
                shortest_path_navigator=self.unshuffle_env.shortest_path_navigator,
            )
            if self.object_names_seen_in_walkthrough is not None:
                # The expert shouldn't act on objects the walkthrougher hasn't seen!
                c = self.unshuffle_env.controller
                with include_object_data(c):
                    for o in c.last_event.metadata["objects"]:
                        if o["name"] not in self.object_names_seen_in_walkthrough:
                            self.greedy_expert.object_name_to_priority[o["name"]] = (
                                self.greedy_expert.max_priority_per_object + 1
                            )

        action = self.greedy_expert.expert_action
        if action is None:
            return 0, False
        else:
            return action, True

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """Return the simplified action space in RearrangeMode.SNAP mode."""
        return gym.spaces.Discrete(len(self.action_names()))

    def close(self) -> None:
        """Close the AI2-THOR rearrangement environment controllers."""
        try:
            self.unshuffle_env.stop()
        except Exception as _:
            pass

        try:
            self.walkthrough_env.stop()
        except Exception as _:
            pass

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        env = self.unshuffle_env
        ips, gps, cps = env.poses

        start_energies = self.start_energies
        end_energies = env.pose_difference_energy(gps, cps)
        start_energy = start_energies.sum()
        end_energy = end_energies.sum()

        start_misplaceds = start_energies > 0.0
        end_misplaceds = end_energies > 0.0

        num_broken = sum(cp["broken"] for cp in cps)
        num_initially_misplaced = start_misplaceds.sum()
        num_fixed = num_initially_misplaced - (start_misplaceds & end_misplaceds).sum()
        num_newly_misplaced = (end_misplaceds & np.logical_not(start_misplaceds)).sum()

        prop_fixed = (
            1.0 if num_initially_misplaced == 0 else num_fixed / num_initially_misplaced
        )
        metrics = {
            **super().metrics(),
            **{
                "start_energy": start_energy,
                "end_energy": end_energy,
                "success": float(end_energy == 0),
                "prop_fixed": prop_fixed,
                "prop_fixed_strict": float((num_newly_misplaced == 0) * prop_fixed),
                "num_misplaced": end_misplaceds.sum(),
                "num_newly_misplaced": num_newly_misplaced.sum(),
                "num_initially_misplaced": num_initially_misplaced,
                "num_fixed": num_fixed.sum(),
                "num_broken": num_broken,
            },
        }

        try:
            change_energies = env.pose_difference_energy(ips, cps)
            change_energy = change_energies.sum()
            changeds = change_energies > 0.0
            metrics["change_energy"] = change_energy
            metrics["num_changed"] = changeds.sum()
        except AssertionError as _:
            pass

        if num_initially_misplaced > 0:
            metrics["prop_misplaced"] = end_misplaceds.sum() / num_initially_misplaced

        if start_energy > 0:
            metrics["energy_prop"] = end_energy / start_energy

        task_info = metrics["task_info"]
        task_info["scene"] = self.unshuffle_env.scene
        task_info["index"] = self.unshuffle_env.current_task_spec.metrics.get("index")
        task_info["stage"] = self.unshuffle_env.current_task_spec.stage
        del metrics["task_info"]

        if self.task_spec_in_metrics:
            task_info["task_spec"] = {**self.unshuffle_env.current_task_spec.__dict__}
            task_info["poses"] = self.unshuffle_env.poses
            task_info["gps_vs_cps"] = self.unshuffle_env.compare_poses(gps, cps)
            task_info["ips_vs_cps"] = self.unshuffle_env.compare_poses(ips, cps)
            task_info["gps_vs_ips"] = self.unshuffle_env.compare_poses(gps, ips)
            task_info["actions"] = self.actions_taken
            task_info["actions_success"] = self.actions_taken_success
            task_info["agent_locs"] = self.agent_locs

        if self.metrics_from_walkthrough is not None:
            mes = {**self.metrics_from_walkthrough}
            del mes["task_info"]  # Already summarized by the unshuffle task info

            metrics = {
                "task_info": task_info,
                "ep_length": metrics["ep_length"] + mes["ep_length"],
                **{f"unshuffle/{k}": v for k, v in metrics.items()},
                **{f"walkthrough/{k}": v for k, v in mes.items()},
            }
        else:
            metrics = {
                "task_info": task_info,
                **{f"unshuffle/{k}": v for k, v in metrics.items()},
            }

        return metrics

    def class_action_names(self, **kwargs) -> Tuple[str, ...]:
        """Return the easy, simplified task's class names."""
        return self.discrete_actions

    def render(self, *args, **kwargs) -> Dict[str, Dict[str, np.array]]:
        """Return the rgb/depth obs from both walkthrough and unshuffle."""
        # TODO: eventually update when the phases are separated.
        # walkthrough_obs = self.walkthrough_env.observation
        unshuffle_obs = self.unshuffle_env.observation
        return {
            # "walkthrough": {"rgb": walkthrough_obs[0], "depth": walkthrough_obs[1]},
            "unshuffle": {"rgb": unshuffle_obs[0], "depth": unshuffle_obs[1]},
        }

    def reached_terminal_state(self) -> bool:
        """Return if end of current episode has been reached."""
        return (self.require_done_action and self._took_end_action) or (
            (not self.require_done_action)
            and self.unshuffle_env.all_rearranged_or_broken
        )

    def _judge(self, action_name: str) -> float:
        """Return the reward from a new (s, a, s')."""
        # TODO: Log reward scenarios.

        _, gps, cps = self.unshuffle_env.poses
        cur_pose_energy = self.unshuffle_env.pose_difference_energy(
            goal_pose=gps, cur_pose=cps
        ).sum()

        if self.is_done():
            return -cur_pose_energy

        energy_change = self.last_pose_energy - cur_pose_energy
        self.last_pose_energy = cur_pose_energy
        self.last_poses = cps
        return energy_change

    def _step(self, action: int) -> RLStepResult:
        """
        :action: is the index of the action from self.class_action_names()
        """
        # parse the action data
        action_name = self.class_action_names()[action]

        if action_name.startswith("pickup"):
            # NOTE: due to the object_id's not being in the metadata for speedups,
            # they cannot be targeted with interactible actions. Hence, why
            # we're resetting the object filter before targeting by object id.

            with include_object_data(self.unshuffle_env.controller):
                metadata = self.unshuffle_env.last_event.metadata

                if len(metadata["inventoryObjects"]) != 0:
                    action_success = False
                else:
                    object_type = stringcase.pascalcase(
                        action_name.replace("pickup_", "")
                    )
                    possible_objects = [
                        o
                        for o in metadata["objects"]
                        if o["visible"] and o["objectType"] == object_type
                    ]
                    object_before = None
                    if len(possible_objects) > 0:
                        object_before = random.choice(possible_objects)
                        object_id = object_before["objectId"]

                    if object_before is not None:
                        self.unshuffle_env.controller.step(
                            "PickupObject", objectId=object_id
                        )
                        action_success = self.unshuffle_env.last_event.metadata[
                            "lastActionSuccess"
                        ]
                    else:
                        action_success = False

                    if action_success:
                        cur_metadata = self.unshuffle_env.last_event.metadata
                        picked_up_objects = [
                            o for o in cur_metadata["objects"] if o["isPickedUp"]
                        ]
                        if len(picked_up_objects) == 0:
                            action_success = False
                        elif len(picked_up_objects) > 1:
                            # TODO: It seems possible to pickup more than a single object in rare cases.
                            #   Here we teleport any extraneously picked up objects back to their original locations.
                            get_logger().warning(
                                f"In scene {self.unshuffle_env.scene}:"
                                f" should not be able to hold more than one object."
                                f"\n\nCurrently holding {picked_up_objects}."
                                f"\n\nTask spec {self.unshuffle_env.current_task_spec}."
                            )
                            before_pickup_md = metadata
                            invalid_object_ids = {
                                o["objectId"]
                                for o in picked_up_objects
                                if o["objectId"] != object_id
                            }

                            def drop_objects_and_teleport_back_them_to_original_locations(
                                object_ids,
                            ):
                                cur_metadata = self.unshuffle_env.controller.step(
                                    "DropHandObject", forceAction=True
                                ).metadata
                                assert cur_metadata["lastActionSuccess"]
                                for obj in before_pickup_md["objects"]:
                                    if obj["objectId"] in object_ids:
                                        self.unshuffle_env.controller.step(
                                            "TeleportObject",
                                            objectId=obj["objectId"],
                                            position=obj["position"],
                                            rotation=obj["rotation"],
                                            forceAction=True,
                                        )

                            self.unshuffle_env.controller.step("PausePhysicsAutoSim")
                            drop_objects_and_teleport_back_them_to_original_locations(
                                invalid_object_ids
                            )
                            cur_metadata = self.unshuffle_env.controller.step(
                                "PickupObject", objectId=object_id, forceAction=True
                            ).metadata
                            held_object_ids = [
                                o["objectId"]
                                for o in cur_metadata["objects"]
                                if o["isPickedUp"]
                            ]
                            if len(held_object_ids) != 1:
                                drop_objects_and_teleport_back_them_to_original_locations(
                                    held_object_ids
                                )
                                action_success = False
                            self.unshuffle_env.controller.step("UnpausePhysicsAutoSim")

        elif action_name.startswith("open_by_type"):
            object_type = stringcase.pascalcase(
                action_name.replace("open_by_type_", "")
            )
            with include_object_data(self.unshuffle_env.controller):

                obj_name_to_goal_and_cur_poses = {
                    cur_pose["name"]: (goal_pose, cur_pose)
                    for _, goal_pose, cur_pose in zip(*self.unshuffle_env.poses)
                }

                goal_pose = None
                cur_pose = None
                for o in self.unshuffle_env.last_event.metadata["objects"]:
                    if (
                        o["visible"]
                        and o["objectType"] == object_type
                        and o["openable"]
                        and not self.unshuffle_env.are_poses_equal(
                            *obj_name_to_goal_and_cur_poses[o["name"]]
                        )
                    ):
                        goal_pose, cur_pose = obj_name_to_goal_and_cur_poses[o["name"]]
                        break

                if goal_pose is not None:
                    object_id = cur_pose["objectId"]
                    goal_openness = goal_pose["openness"]

                    if cur_pose["openness"] > 0.0:
                        self.unshuffle_env.controller.step(
                            "CloseObject", objectId=object_id,
                        )
                    # NOTE: 'moveMagnitude' is 'openness' with the open object action.
                    # This is not a typo here.
                    self.unshuffle_env.controller.step(
                        "OpenObject", objectId=object_id, moveMagnitude=goal_openness
                    )
                    action_success = self.unshuffle_env.last_event.metadata[
                        "lastActionSuccess"
                    ]
                else:
                    action_success = False

        elif action_name.startswith(("move", "rotate", "look", "stand", "crouch")):
            # apply to only the unshuffle env as the walkthrough agent's position
            # must now be managed by the whichever sensor is trying to read data from it.
            action_success = getattr(self.unshuffle_env, action_name)()
        elif action_name == "drop_held_object_with_snap":
            action_success = getattr(self.unshuffle_env, action_name)()
        elif action_name == "done":
            self._took_end_action = True
            action_success = True
        else:
            raise RuntimeError(
                f"Action '{action_name}' is not in the action space {RearrangeActionSpace}"
            )

        self.actions_taken.append(action_name)
        self.actions_taken_success.append(action_success)
        if self.task_spec_in_metrics:
            self.agent_locs.append(self.unshuffle_env.get_agent_location())
        return RLStepResult(
            observation=None,
            reward=self._judge(action_name),
            done=self.is_done(),
            info={"action_name": action_name, "action_success": action_success},
        )

    def step(self, action: int) -> RLStepResult:
        step_result = super().step(action=action)
        if self.greedy_expert is not None:
            self.greedy_expert.update(
                action_taken=action, action_success=step_result.info["action_success"]
            )
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=step_result.reward,
            done=step_result.done,
            info=step_result.info,
        )
        return step_result


class WalkthroughTask(AbstractRearrangeTask):
    def __init__(
        self,
        sensors: SensorSuite,
        walkthrough_env: RearrangeTHOREnvironment,
        max_steps: int,
        discrete_actions: Tuple[str, ...],
        disable_metrics: bool = False,
    ) -> None:
        """Create a new walkthrough task."""
        super().__init__(
            env=walkthrough_env, sensors=sensors, task_info=dict(), max_steps=max_steps
        )
        self.walkthrough_env = walkthrough_env
        self.discrete_actions = discrete_actions
        self.disable_metrics = disable_metrics

        self._took_end_action: bool = False

        self.visited_positions_xzrsh = {self.agent_location_tuple}
        self.visited_positions_xz = {self.agent_location_tuple[:2]}
        self.seen_pickupable_objects = set(
            o["name"] for o in self.pickupable_objects(visible_only=True)
        )
        self.seen_openable_objects = set(
            o["name"] for o in self.openable_not_pickupable_objects(visible_only=True)
        )
        self.total_pickupable_or_openable_objects = len(
            self.pickupable_or_openable_objects(visible_only=False)
        )

        self.walkthrough_env.controller.step("GetReachablePositions")
        assert self.walkthrough_env.last_event.metadata["lastActionSuccess"]

        self.reachable_positions = self.walkthrough_env.last_event.metadata[
            "actionReturn"
        ]

    def query_expert(self, **kwargs) -> Tuple[Any, bool]:
        return 0, False

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """Return the simplified action space in RearrangeMode.SNAP mode."""
        return gym.spaces.Discrete(len(self.action_names()))

    def close(self) -> None:
        """Close the AI2-THOR rearrangement environment controllers."""
        try:
            self.walkthrough_env.stop()
        except Exception as _:
            pass

    def metrics(self, force_return: bool = False) -> Dict[str, Any]:
        if (not force_return) and (self.disable_metrics or not self.is_done()):
            return {}

        nreachable = len(self.reachable_positions)
        prop_visited_xz = len(self.visited_positions_xz) / nreachable

        nreachable_xzr = 4 * nreachable  # 4 rotations
        visited_xzr = {p[:3] for p in self.visited_positions_xzrsh}
        prop_visited_xzr = len(visited_xzr) / nreachable_xzr

        n_obj_seen = len(self.seen_openable_objects) + len(self.seen_pickupable_objects)

        metrics = {
            **super().metrics(),
            **{
                "num_explored_xz": len(self.visited_positions_xz),
                "num_explored_xzr": len(visited_xzr),
                "prop_visited_xz": prop_visited_xz,
                "prop_visited_xzr": prop_visited_xzr,
                "num_obj_seen": n_obj_seen,
                "prop_obj_seen": n_obj_seen / self.total_pickupable_or_openable_objects,
            },
        }

        return metrics

    def class_action_names(self, **kwargs) -> Tuple[str, ...]:
        """Return the easy, simplified task's class names."""
        return self.discrete_actions

    def render(self, *args, **kwargs) -> Dict[str, Dict[str, np.array]]:
        """Return the rgb/depth obs from both walkthrough and unshuffle."""
        # TODO: eventually update when the phases are separated.
        walkthrough_obs = self.walkthrough_env.observation
        return {
            "walkthrough": {"rgb": walkthrough_obs[0], "depth": walkthrough_obs[1]},
        }

    def reached_terminal_state(self) -> bool:
        """Return if end of current episode has been reached."""
        return self._took_end_action

    def pickupable_objects(self, visible_only: bool = True):
        with include_object_data(self.walkthrough_env.controller):
            return [
                o
                for o in self.walkthrough_env.last_event.metadata["objects"]
                if ((o["visible"] or not visible_only) and o["pickupable"])
            ]

    def openable_not_pickupable_objects(self, visible_only: bool = True):
        with include_object_data(self.walkthrough_env.controller):
            return [
                o
                for o in self.walkthrough_env.last_event.metadata["objects"]
                if (
                    (o["visible"] or not visible_only)
                    and (o["openable"] and not o["pickupable"])
                )
            ]

    def pickupable_or_openable_objects(self, visible_only: bool = True):
        with include_object_data(self.walkthrough_env.controller):
            return [
                o
                for o in self.walkthrough_env.last_event.metadata["objects"]
                if (
                    (o["visible"] or not visible_only)
                    and (o["pickupable"] or (o["openable"] and not o["pickupable"]))
                )
            ]

    def _judge(self, action_name: str, action_success: bool) -> float:
        """Return the reward from a new (s, a, s')."""
        reward = 0

        if not action_success:
            reward -= 0.01

        # Seen openable obj reward
        seen_obj_reward = 0
        for obj in self.openable_not_pickupable_objects(visible_only=True):
            if obj["name"] not in self.seen_openable_objects:
                self.seen_openable_objects.add(obj["name"])
                seen_obj_reward += 0.1
        reward += min(0.3, seen_obj_reward)

        # Seen pickupable obj reward
        seen_obj_reward = 0
        for obj in self.pickupable_objects(visible_only=True):
            if obj["name"] not in self.seen_pickupable_objects:
                self.seen_pickupable_objects.add(obj["name"])
                seen_obj_reward += 0.1
        reward += min(0.3, seen_obj_reward)

        # Entered new location reward
        agent_loc_tuple = self.agent_location_tuple
        self.visited_positions_xzrsh.add(agent_loc_tuple)
        if agent_loc_tuple[:2] not in self.visited_positions_xz:
            self.visited_positions_xz.add(agent_loc_tuple[:2])
            reward += 0.01

        if self.is_done() or self.num_steps_taken() + 1 >= self.max_steps:
            prop_seen = (
                len(self.seen_pickupable_objects) + len(self.seen_openable_objects)
            ) / self.total_pickupable_or_openable_objects

            if prop_seen >= 0.95:
                reward += 10 if self._took_end_action else 5
            elif prop_seen >= 0.9:
                reward += 1
            elif prop_seen >= 0.8:
                reward -= 1
            else:
                reward -= 2

        return reward

    def _step(self, action: int) -> RLStepResult:
        """Take a step in the task.

        # Parameters
        action: is the index of the action from self.class_action_names()
        """
        # parse the action data
        action_name = self.class_action_names()[action]

        if action_name.startswith("pickup"):
            # Don't allow the exploration agent to pickup objects
            action_success = False

        elif action_name.startswith("open_by_type"):
            # Don't allow the exploration agent to open objects
            action_success = False

        elif action_name.startswith(("move", "rotate", "look", "stand", "crouch")):
            # take the movement action
            action_success = getattr(self.walkthrough_env, action_name)()

        elif action_name == "drop_held_object_with_snap":
            # Don't allow the exploration agent to drop objects (not that it can hold any)
            action_success = False

        elif action_name == "done":
            self._took_end_action = True
            action_success = True

        else:
            raise RuntimeError(
                f"Action '{action_name}' is not in the action space {RearrangeActionSpace}"
            )

        return RLStepResult(
            observation=self.get_observations(),
            reward=self._judge(action_name=action_name, action_success=action_success),
            done=self.is_done(),
            info={"action_name": action_name, "action_success": action_success},
        )


class RearrangeTaskSpecIterable:
    """Iterate through a collection of scenes and pose specifications for the
    rearrange task."""

    def __init__(
        self,
        scenes_to_task_spec_dicts: Dict[str, List[Dict]],
        seed: int,
        epochs: Union[int, float],
        shuffle: bool = True,
    ):
        assert epochs >= 1

        self.scenes_to_task_spec_dicts = {
            k: [*v] for k, v in scenes_to_task_spec_dicts.items()
        }
        assert len(self.scenes_to_task_spec_dicts) != 0 and all(
            len(self.scenes_to_task_spec_dicts[scene]) != 0
            for scene in self.scenes_to_task_spec_dicts
        )
        self._seed = seed
        self.random = random.Random(self.seed)
        self.start_epochs = epochs
        self.remaining_epochs = epochs
        self.shuffle = shuffle

        self.remaining_scenes: List[str] = []
        self.task_spec_dicts_for_current_scene: List[Dict[str, Any]] = []
        self.current_scene: Optional[str] = None

        self.reset()

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, seed: int):
        self._seed = seed
        self.random.seed(seed)

    @property
    def length(self):
        if self.remaining_epochs == float("inf"):
            return float("inf")

        return (
            len(self.task_spec_dicts_for_current_scene)
            + sum(
                len(self.scenes_to_task_spec_dicts[scene])
                for scene in self.remaining_scenes
            )
            + self.remaining_epochs
            * (sum(len(v) for v in self.scenes_to_task_spec_dicts.values()))
        )

    @property
    def total_unique(self):
        return sum(len(v) for v in self.scenes_to_task_spec_dicts.values())

    def reset(self):
        self.random.seed(self.seed)
        self.remaining_epochs = self.start_epochs
        self.remaining_scenes.clear()
        self.task_spec_dicts_for_current_scene.clear()
        self.current_scene = None

    def refresh_remaining_scenes(self):
        if self.remaining_epochs <= 0:
            raise StopIteration
        self.remaining_epochs -= 1

        self.remaining_scenes = list(sorted(self.scenes_to_task_spec_dicts.keys()))
        if self.shuffle:
            self.random.shuffle(self.remaining_scenes)
        return self.remaining_scenes

    def __next__(self) -> RearrangeTaskSpec:
        if len(self.task_spec_dicts_for_current_scene) == 0:
            if len(self.remaining_scenes) == 0:
                self.refresh_remaining_scenes()
            self.current_scene = self.remaining_scenes.pop()

            self.task_spec_dicts_for_current_scene = [
                *self.scenes_to_task_spec_dicts[self.current_scene]
            ]
            if self.shuffle:
                self.random.shuffle(self.task_spec_dicts_for_current_scene)

        new_task_spec_dict = self.task_spec_dicts_for_current_scene.pop()
        if "scene" not in new_task_spec_dict:
            new_task_spec_dict["scene"] = self.current_scene
        else:
            assert self.current_scene == new_task_spec_dict["scene"]

        return RearrangeTaskSpec(**new_task_spec_dict)


class RearrangeTaskSampler(TaskSampler):
    def __init__(
        self,
        run_walkthrough_phase: bool,
        run_unshuffle_phase: bool,
        stage: str,
        scenes_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
        rearrange_env_kwargs: Optional[Dict[str, Any]],
        sensors: SensorSuite,
        max_steps: Union[Dict[str, int], int],
        discrete_actions: Tuple[str, ...],
        require_done_action: bool,
        force_axis_aligned_start: bool,
        epochs: Union[int, float, str] = "default",
        seed: Optional[int] = None,
        unshuffle_repeats: Optional[int] = None,
        task_spec_in_metrics: bool = False,
    ) -> None:
        assert isinstance(run_walkthrough_phase, bool) and isinstance(
            run_unshuffle_phase, bool
        ), (
            f"Both `run_walkthrough_phase` (== {run_walkthrough_phase})"
            f" and `run_unshuffle_phase` (== {run_unshuffle_phase})"
            f" must be boolean valued."
        )
        assert (
            run_walkthrough_phase or run_unshuffle_phase
        ), "One of `run_walkthrough_phase` or `run_unshuffle_phase` must be `True`."

        assert (unshuffle_repeats is None) or (
            run_walkthrough_phase and run_unshuffle_phase
        ), (
            "`unshuffle_repeats` should be `None` if either `run_walkthrough_phase` or"
            " `run_unshuffle_phase` is `False`."
        )
        assert (
            unshuffle_repeats is None
        ) or unshuffle_repeats >= 1, (
            f"`unshuffle_repeats` (=={unshuffle_repeats}) must be >= 1."
        )

        self.run_walkthrough_phase = run_walkthrough_phase
        self.run_unshuffle_phase = run_unshuffle_phase

        self.sensors = sensors
        self.stage = stage
        self.main_seed = seed if seed is not None else random.randint(0, 2 * 30 - 1)

        self.unshuffle_repeats = 1 if unshuffle_repeats is None else unshuffle_repeats
        self.cur_unshuffle_repeat_count = 0

        self.task_spec_in_metrics = task_spec_in_metrics

        self.scenes_to_task_spec_dicts = copy.deepcopy(scenes_to_task_spec_dicts)

        if isinstance(epochs, str):
            if epochs.lower().strip() != "default":
                raise NotImplementedError(f"Unknown value for `epochs` (=={epochs})")
            epochs = float("inf") if stage == "train" else 1
        self.task_spec_iterator = RearrangeTaskSpecIterable(
            scenes_to_task_spec_dicts=self.scenes_to_task_spec_dicts,
            seed=self.main_seed,
            epochs=epochs,
            shuffle=epochs == float("inf"),
        )

        self.walkthrough_env = RearrangeTHOREnvironment(**rearrange_env_kwargs)

        self.unshuffle_env: Optional[RearrangeTHOREnvironment] = None
        if self.run_unshuffle_phase:
            self.unshuffle_env = RearrangeTHOREnvironment(**rearrange_env_kwargs)

        self.scenes = list(self.scenes_to_task_spec_dicts.keys())

        if isinstance(max_steps, int):
            max_steps = {"unshuffle": max_steps, "walkthrough": max_steps}
        self.max_steps: Dict[str, int] = max_steps
        self.discrete_actions = discrete_actions

        self.require_done_action = require_done_action
        self.force_axis_aligned_start = force_axis_aligned_start

        self._last_sampled_task: Optional[Union[UnshuffleTask, WalkthroughTask]] = None
        self._last_sampled_walkthrough_task: Optional[WalkthroughTask] = None
        self.was_in_exploration_phase: bool = False

    @classmethod
    def from_fixed_dataset(
        cls,
        run_walkthrough_phase: bool,
        run_unshuffle_phase: bool,
        stage: str,
        allowed_scenes: Optional[Sequence[str]] = None,
        scene_to_allowed_rearrange_inds: Optional[Dict[str, Sequence[int]]] = None,
        **init_kwargs,
    ):
        return cls(
            run_walkthrough_phase=run_walkthrough_phase,
            run_unshuffle_phase=run_unshuffle_phase,
            stage=stage,
            scenes_to_task_spec_dicts=cls._filter_scenes_to_task_spec_dicts(
                scenes_to_task_spec_dicts=cls.load_rearrange_data_from_path(
                    stage=stage, base_dir=STARTER_DATA_DIR
                ),
                allowed_scenes=allowed_scenes,
                scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            ),
            **init_kwargs,
        )

    @classmethod
    def from_scenes_at_runtime(
        cls,
        run_walkthrough_phase: bool,
        run_unshuffle_phase: bool,
        stage: str,
        allowed_scenes: Sequence[str],
        repeats_before_scene_change: int,
        **init_kwargs,
    ):
        assert "scene_to_allowed_rearrange_inds" not in init_kwargs
        assert repeats_before_scene_change >= 1
        return cls(
            run_walkthrough_phase=run_walkthrough_phase,
            run_unshuffle_phase=run_unshuffle_phase,
            stage=stage,
            scenes_to_task_spec_dicts={
                scene: tuple(
                    {scene: scene, "runtime_sample": True}
                    for _ in range(repeats_before_scene_change)
                )
                for scene in allowed_scenes
            },
            **init_kwargs,
        )

    @classmethod
    def _filter_scenes_to_task_spec_dicts(
        cls,
        scenes_to_task_spec_dicts: Dict[str, List[Dict[str, Any]]],
        allowed_scenes: Optional[Sequence[str]],
        scene_to_allowed_rearrange_inds: Optional[Dict[str, Sequence[int]]],
    ):
        if allowed_scenes is not None:
            scenes_to_task_spec_dicts = {
                scene: scenes_to_task_spec_dicts[scene] for scene in allowed_scenes
            }

        if scene_to_allowed_rearrange_inds is not None:
            scenes_to_task_spec_dicts = {
                scene: [
                    scenes_to_task_spec_dicts[scene][ind]
                    for ind in sorted(scene_to_allowed_rearrange_inds[scene])
                ]
                for scene in scene_to_allowed_rearrange_inds
                if scene in scenes_to_task_spec_dicts
            }
        return scenes_to_task_spec_dicts

    @classmethod
    def load_rearrange_data_from_path(
        cls, stage: str, base_dir: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        stage = stage.lower()

        if stage == "valid":
            stage = "val"

        data_path = os.path.join(base_dir, f"{stage}.pkl.gz")
        if not os.path.exists(data_path):
            raise RuntimeError(f"No data at path {data_path}")

        data = compress_pickle.load(path=data_path)
        for scene in data:
            for ind, task_spec_dict in enumerate(data[scene]):
                task_spec_dict["scene"] = scene

                if "index" not in task_spec_dict:
                    task_spec_dict["index"] = ind

                if "stage" not in task_spec_dict:
                    task_spec_dict["stage"] = stage
        return data

    @property
    def length(self) -> float:
        """Return the total number of allowable next_task calls."""
        count = self.run_walkthrough_phase + self.run_unshuffle_phase
        if count == 1:
            return self.task_spec_iterator.length
        elif count == 2:
            mult = self.unshuffle_repeats
            count = (1 + mult) * self.task_spec_iterator.length

            if self.last_sampled_task is not None and (
                isinstance(self.last_sampled_task, WalkthroughTask)
                or self.cur_unshuffle_repeat_count < mult
            ):
                count += mult - self.cur_unshuffle_repeat_count

            return count
        else:
            raise NotImplementedError

    @property
    def total_unique(self):
        return self.task_spec_iterator.total_unique

    @property
    def last_sampled_task(self) -> Optional[UnshuffleTask]:
        """Return the most recent sampled task."""
        return self._last_sampled_task

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Return if the observation space remains the same across steps."""
        return True

    def close(self) -> None:
        """Close the open AI2-THOR controllers."""
        try:
            self.unshuffle_env.stop()
        except Exception as _:
            pass

        try:
            self.walkthrough_env.stop()
        except Exception as _:
            pass

    def reset(self) -> None:
        """Restart the unshuffle iteration setup order."""
        self.task_spec_iterator.reset()
        self.cur_unshuffle_repeat_count = 0
        self._last_sampled_task = None
        self._last_sampled_walkthrough_task = None

    def set_seed(self, seed: int) -> None:
        self.task_spec_iterator.seed = seed
        self.main_seed = seed

    @property
    def current_task_spec(self) -> RearrangeTaskSpec:
        if self.run_unshuffle_phase:
            return self.unshuffle_env.current_task_spec
        else:
            return self.walkthrough_env.current_task_spec

    def next_task(self, **kwargs) -> Optional[UnshuffleTask]:
        """Return a fresh UnshuffleTask setup."""

        walkthrough_finished_and_should_run_unshuffle = (
            self.run_unshuffle_phase
            and self.run_walkthrough_phase
            and (
                self.was_in_exploration_phase
                or self.cur_unshuffle_repeat_count < self.unshuffle_repeats
            )
        )

        if (
            self.last_sampled_task is None
            or not walkthrough_finished_and_should_run_unshuffle
        ):
            self.cur_unshuffle_repeat_count = 0

            try:
                task_spec: RearrangeTaskSpec = next(self.task_spec_iterator)
            except StopIteration:
                return None

            runtime_sample = task_spec.runtime_sample

            try:
                if self.run_unshuffle_phase:
                    self.unshuffle_env.reset(
                        task_spec=task_spec,
                        force_axis_aligned_start=self.force_axis_aligned_start,
                    )
                    self.unshuffle_env.shuffle()

                    if runtime_sample:
                        unshuffle_task_spec = self.unshuffle_env.current_task_spec
                        starting_objects = unshuffle_task_spec.runtime_data[
                            "starting_objects"
                        ]
                        openable_data = [
                            {
                                "name": o["name"],
                                "objectName": o["name"],
                                "objectId": o["objectId"],
                                "start_openness": o["openness"],
                                "target_openness": o["openness"],
                            }
                            for o in starting_objects
                            if o["isOpen"] and not o["pickupable"]
                        ]
                        starting_poses = [
                            {
                                "name": o["name"],
                                "objectName": o["name"],
                                "position": o["position"],
                                "rotation": o["rotation"],
                            }
                            for o in starting_objects
                            if o["pickupable"]
                        ]
                        task_spec = RearrangeTaskSpec(
                            scene=unshuffle_task_spec.scene,
                            agent_position=task_spec.agent_position,
                            agent_rotation=task_spec.agent_rotation,
                            openable_data=openable_data,
                            starting_poses=starting_poses,
                            target_poses=starting_poses,
                        )

                self.walkthrough_env.reset(
                    task_spec=task_spec,
                    force_axis_aligned_start=self.force_axis_aligned_start,
                )

                if self.run_walkthrough_phase:
                    self.was_in_exploration_phase = True
                    self._last_sampled_task = WalkthroughTask(
                        sensors=self.sensors,
                        walkthrough_env=self.walkthrough_env,
                        max_steps=self.max_steps["walkthrough"],
                        discrete_actions=self.discrete_actions,
                        disable_metrics=self.run_unshuffle_phase,
                    )
                    self._last_sampled_walkthrough_task = self._last_sampled_task
                else:
                    self.cur_unshuffle_repeat_count += 1
                    self._last_sampled_task = UnshuffleTask(
                        sensors=self.sensors,
                        unshuffle_env=self.unshuffle_env,
                        walkthrough_env=self.walkthrough_env,
                        max_steps=self.max_steps["unshuffle"],
                        discrete_actions=self.discrete_actions,
                        require_done_action=self.require_done_action,
                        task_spec_in_metrics=self.task_spec_in_metrics,
                    )
            except Exception as e:
                if runtime_sample:
                    get_logger().error(
                        "Encountered exception while sampling a next task."
                        " As this next task was a 'runtime sample' we are"
                        " simply returning the next task."
                    )
                    get_logger().error(traceback.format_exc())
                    return self.next_task()
                else:
                    raise e
        else:
            self.cur_unshuffle_repeat_count += 1
            self.was_in_exploration_phase = False

            walkthrough_task = cast(
                WalkthroughTask, self._last_sampled_walkthrough_task
            )

            if self.cur_unshuffle_repeat_count != 1:
                self.unshuffle_env.reset(
                    task_spec=self.unshuffle_env.current_task_spec,
                    force_axis_aligned_start=self.force_axis_aligned_start,
                )
                self.unshuffle_env.shuffle()

            self._last_sampled_task = UnshuffleTask(
                sensors=self.sensors,
                unshuffle_env=self.unshuffle_env,
                walkthrough_env=self.walkthrough_env,
                max_steps=self.max_steps["unshuffle"],
                discrete_actions=self.discrete_actions,
                require_done_action=self.require_done_action,
                locations_visited_in_walkthrough=np.array(
                    tuple(walkthrough_task.visited_positions_xzrsh)
                ),
                object_names_seen_in_walkthrough=copy.copy(
                    walkthrough_task.seen_pickupable_objects
                    | walkthrough_task.seen_openable_objects
                ),
                metrics_from_walkthrough=walkthrough_task.metrics(force_return=True),
                task_spec_in_metrics=self.task_spec_in_metrics,
            )

        return self._last_sampled_task
