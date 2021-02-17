"""Definitions for a greedy expert for the `Unshuffle` task."""

import copy
import random
from collections import defaultdict
from typing import (
    Dict,
    Tuple,
    Any,
    Optional,
    Union,
    List,
    Sequence,
    TYPE_CHECKING,
)

import ai2thor.controller
import ai2thor.server
import networkx as nx
import stringcase
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor
from torch.distributions.utils import lazy_property

from rearrange.constants import STEP_SIZE
from rearrange.environment import (
    RearrangeTHOREnvironment,
    RearrangeMode,
)
from rearrange.utils import include_object_data
from rearrange.utils import save_frames_to_mp4

if TYPE_CHECKING:
    from rearrange.tasks import UnshuffleTask

AgentLocKeyType = Tuple[float, float, int, int]


class ShortestPathNavigatorTHOR:
    """Tracks shortest paths in AI2-THOR environments.

    Assumes 90 degree rotations and fixed step sizes.

    # Attributes
    controller : The AI2-THOR controller in which shortest paths are computed.
    """

    def __init__(
        self,
        controller: ai2thor.controller.Controller,
        grid_size: float,
        include_move_left_right: bool = False,
    ):
        """Create a `ShortestPathNavigatorTHOR` instance.

        # Parameters
        controller : An AI2-THOR controller which represents the environment in which shortest paths should be
            computed.
        grid_size : The distance traveled by an AI2-THOR agent when taking a single navigational step.
        include_move_left_right : If `True` the navigational actions will include `MoveLeft` and `MoveRight`, otherwise
            they wil not.
        """
        self._cached_graphs: Dict[str, nx.DiGraph] = {}

        self._current_scene: Optional[nx.DiGraph] = None
        self._current_graph: Optional[nx.DiGraph] = None

        self._grid_size = grid_size
        self.controller = controller

        self._include_move_left_right = include_move_left_right

    @lazy_property
    def nav_actions_set(self) -> frozenset:
        """Navigation actions considered when computing shortest paths."""
        nav_actions = [
            "LookUp",
            "LookDown",
            "RotateLeft",
            "RotateRight",
            "MoveAhead",
        ]
        if self._include_move_left_right:
            nav_actions.extend(["MoveLeft", "MoveRight"])
        return frozenset(nav_actions)

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.controller.last_event.metadata["sceneName"]

    @property
    def last_action_success(self) -> bool:
        """Was the last action taken by the agent a success?"""
        return self.controller.last_event.metadata["lastActionSuccess"]

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Last event returned by the controller."""
        return self.controller.last_event

    def on_reset(self):
        """Function that must be called whenever the AI2-THOR controller is
        reset."""
        self._current_scene = None

    @property
    def graph(self) -> nx.DiGraph:
        """A directed graph representing the navigation graph of the current
        scene."""
        if self._current_scene == self.scene_name:
            return self._current_graph

        if self.scene_name not in self._cached_graphs:
            g = nx.DiGraph()
            points = self.reachable_points_with_rotations_and_horizons()
            for p in points:
                self._add_node_to_graph(g, self.get_key(p))

            self._cached_graphs[self.scene_name] = g

        self._current_scene = self.scene_name
        self._current_graph = self._cached_graphs[self.scene_name].copy()
        return self._current_graph

    def reachable_points_with_rotations_and_horizons(
        self,
    ) -> List[Dict[str, Union[float, int]]]:
        """Get all the reaachable positions in the scene along with possible
        rotation/horizons."""
        self.controller.step(action="GetReachablePositions")
        assert self.last_action_success

        points_slim = self.last_event.metadata["actionReturn"]

        points = []
        for r in [0, 90, 180, 270]:
            for horizon in [-30, 0, 30, 60]:
                for p in points_slim:
                    p = copy.copy(p)
                    p["rotation"] = r
                    p["horizon"] = horizon
                    points.append(p)
        return points

    @staticmethod
    def location_for_key(key, y_value=0.0) -> Dict[str, Union[float, int]]:
        """Return a agent location dictionary given a graph node key."""
        x, z, rot, hor = key
        loc = dict(x=x, y=y_value, z=z, rotation=rot, horizon=hor)
        return loc

    @staticmethod
    def get_key(input_dict: Dict[str, Any], ndigits: int = 2) -> AgentLocKeyType:
        """Return a graph node key given an input agent location dictionary."""
        if "x" in input_dict:
            x = input_dict["x"]
            z = input_dict["z"]
            rot = input_dict["rotation"]
            hor = input_dict["horizon"]
        else:
            x = input_dict["position"]["x"]
            z = input_dict["position"]["z"]
            rot = input_dict["rotation"]["y"]
            hor = input_dict["cameraHorizon"]

        return (
            round(x, ndigits),
            round(z, ndigits),
            round_to_factor(rot, 90) % 360,
            round_to_factor(hor, 30) % 360,
        )

    def update_graph_with_failed_action(self, failed_action: str):
        """If an action failed, update the graph to let it know this happened
        so it won't try again."""
        if (
            self.scene_name not in self._cached_graphs
            or failed_action not in self.nav_actions_set
        ):
            return

        source_key = self.get_key(self.last_event.metadata["agent"])
        self._check_contains_key(source_key)

        edge_dict = self.graph[source_key]
        to_remove_key = None
        for target_key in self.graph[source_key]:
            if edge_dict[target_key]["action"] == failed_action:
                to_remove_key = target_key
                break
        if to_remove_key is not None:
            self.graph.remove_edge(source_key, to_remove_key)

    def _add_from_to_edge(
        self, g: nx.DiGraph, s: AgentLocKeyType, t: AgentLocKeyType,
    ):
        """Add an edge to the graph."""

        def ae(x, y):
            return abs(x - y) < 0.001

        s_x, s_z, s_rot, s_hor = s
        t_x, t_z, t_rot, t_hor = t

        l1_dist = round(abs(s_x - t_x) + abs(s_z - t_z), 2)
        angle_dist = (round_to_factor(t_rot - s_rot, 90) % 360) // 90
        horz_dist = (round_to_factor(t_hor - s_hor, 30) % 360) // 30

        # If source and target differ by more than one action, continue
        if sum(x != 0 for x in [l1_dist, angle_dist, horz_dist]) != 1:
            return

        grid_size = self._grid_size
        action = None
        if angle_dist != 0:
            if angle_dist == 1:
                action = "RotateRight"
            elif angle_dist == 3:
                action = "RotateLeft"

        elif horz_dist != 0:
            if horz_dist == 11:
                action = "LookUp"
            elif horz_dist == 1:
                action = "LookDown"
        elif ae(l1_dist, grid_size):

            if s_rot == 0:
                forward = round((t_z - s_z) / grid_size)
                right = round((t_x - s_x) / grid_size)
            elif s_rot == 90:
                forward = round((t_x - s_x) / grid_size)
                right = -round((t_z - s_z) / grid_size)
            elif s_rot == 180:
                forward = -round((t_z - s_z) / grid_size)
                right = -round((t_x - s_x) / grid_size)
            elif s_rot == 270:
                forward = -round((t_x - s_x) / grid_size)
                right = round((t_z - s_z) / grid_size)
            else:
                raise NotImplementedError(f"source rotation == {s_rot} unsupported.")

            if forward > 0:
                g.add_edge(s, t, action="MoveAhead")
            elif self._include_move_left_right:
                if forward < 0:
                    # Allowing MoveBack results in some really unintuitive
                    # expert trajectories (i.e. moving backwards to the goal and the
                    # rotating, for now it's disabled.
                    pass  # g.add_edge(s, t, action="MoveBack")
                elif right > 0:
                    g.add_edge(s, t, action="MoveRight")
                elif right < 0:
                    g.add_edge(s, t, action="MoveLeft")

        if action is not None:
            g.add_edge(s, t, action=action)

    @lazy_property
    def possible_neighbor_offsets(self) -> Tuple[AgentLocKeyType, ...]:
        """Offsets used to generate potential neighbors of a node."""
        grid_size = round(self._grid_size, 2)
        offsets = []
        for rot_diff in [-90, 0, 90]:
            for horz_diff in [-30, 0, 30, 60]:
                for x_diff in [-grid_size, 0, grid_size]:
                    for z_diff in [-grid_size, 0, grid_size]:
                        if (rot_diff != 0) + (horz_diff != 0) + (x_diff != 0) + (
                            z_diff != 0
                        ) == 1:
                            offsets.append((x_diff, z_diff, rot_diff, horz_diff))
        return tuple(offsets)

    def _add_node_to_graph(self, graph: nx.DiGraph, s: AgentLocKeyType):
        """Add a node to the graph along with any adjacent edges."""
        if s in graph:
            return

        existing_nodes = set(graph.nodes())
        graph.add_node(s)

        for x_diff, z_diff, rot_diff, horz_diff in self.possible_neighbor_offsets:
            t = (
                s[0] + x_diff,
                s[1] + z_diff,
                (s[2] + rot_diff) % 360,
                (s[3] + horz_diff) % 360,
            )
            if t in existing_nodes:
                self._add_from_to_edge(graph, s, t)
                self._add_from_to_edge(graph, t, s)

    def _check_contains_key(self, key: AgentLocKeyType, add_if_not=True) -> bool:
        """Check if a node key is in the graph.

        # Parameters
        key : The key to check.
        add_if_not : If the key doesn't exist and this is `True`, the key will be added along with
            edges to any adjacent nodes.
        """
        key_in_graph = key in self.graph
        if not key_in_graph:
            get_logger().debug(
                "{} was not in the graph for scene {}.".format(key, self.scene_name)
            )
            if add_if_not:
                self._add_node_to_graph(self.graph, key)
                if key not in self._cached_graphs[self.scene_name]:
                    self._add_node_to_graph(self._cached_graphs[self.scene_name], key)
        return key_in_graph

    def shortest_state_path(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ) -> Optional[Sequence[AgentLocKeyType]]:
        """Get the shortest path between node keys."""
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        # noinspection PyBroadException
        path = nx.shortest_path(
            G=self.graph, source=source_state_key, target=goal_state_key
        )
        return path

    def action_transitioning_between_keys(self, s: AgentLocKeyType, t: AgentLocKeyType):
        """Get the action that takes the agent from node s to node t."""
        self._check_contains_key(s)
        self._check_contains_key(t)
        if self.graph.has_edge(s, t):
            return self.graph.get_edge_data(s, t)["action"]
        else:
            return None

    def shortest_path_next_state(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the next node key on the shortest path from the source to the
        goal."""
        if source_state_key == goal_state_key:
            raise RuntimeError("called next state on the same source and goal state")
        state_path = self.shortest_state_path(source_state_key, goal_state_key)
        return state_path[1]

    def shortest_path_next_action(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the next action along the shortest path from the source to the
        goal."""
        next_state_key = self.shortest_path_next_state(source_state_key, goal_state_key)
        return self.graph.get_edge_data(source_state_key, next_state_key)["action"]

    def shortest_path_next_action_multi_target(
        self,
        source_state_key: AgentLocKeyType,
        goal_state_keys: Sequence[AgentLocKeyType],
    ):
        """Get the next action along the shortest path from the source to the
        closest goal."""
        self._check_contains_key(source_state_key)

        terminal_node = (-1.0, -1.0, -1, -1)
        self.graph.add_node(terminal_node)
        for gsk in goal_state_keys:
            self._check_contains_key(gsk)
            self.graph.add_edge(gsk, terminal_node, action=None)

        next_state_key = self.shortest_path_next_state(source_state_key, terminal_node)
        action = self.graph.get_edge_data(source_state_key, next_state_key)["action"]

        self.graph.remove_node(terminal_node)
        return action

    def shortest_path_length(
        self, source_state_key: AgentLocKeyType, goal_state_key: AgentLocKeyType
    ):
        """Get the path shorest path length between the source and the goal."""
        self._check_contains_key(source_state_key)
        self._check_contains_key(goal_state_key)
        try:
            return nx.shortest_path_length(self.graph, source_state_key, goal_state_key)
        except nx.NetworkXNoPath as _:
            return float("inf")


def _are_agent_locations_equal(
    ap0: Dict[str, Union[float, int, bool]],
    ap1: Dict[str, Union[float, int, bool]],
    ignore_standing: bool,
    tol=1e-2,
    ignore_y: bool = True,
):
    """Determines if two agent locations are equal up to some tolerance."""

    def rot_dist(r0: float, r1: float):
        diff = abs(r0 - r1) % 360
        return min(diff, 360 - diff)

    return (
        all(
            abs(ap0[k] - ap1[k]) <= tol
            for k in (["x", "z"] if ignore_y else ["x", "y", "z"])
        )
        and rot_dist(ap0["rotation"], ap1["rotation"]) <= tol
        and rot_dist(ap0["horizon"], ap1["horizon"]) <= tol
        and (ignore_standing or (ap0["standing"] == ap1["standing"]))
    )


class GreedyUnshuffleExpert:
    """An agent which greedily attempts to complete a given unshuffle task."""

    def __init__(
        self,
        task: "UnshuffleTask",
        shortest_path_navigator: ShortestPathNavigatorTHOR,
        max_priority_per_object: int = 3,
    ):
        """Initializes a `GreedyUnshuffleExpert` object.

        # Parameters
        task : An `UnshuffleTask` that the greedy expert should attempt to complete.
        shortest_path_navigator : A `ShortestPathNavigatorTHOR` object defined on the same
            AI2-THOR controller used by the `task`.
        max_priority_per_object : The maximum number of times we should try to unshuffle an object
            before giving up.
        """
        self.task = task
        self.shortest_path_navigator = shortest_path_navigator
        self.max_priority_per_object = max_priority_per_object

        assert self.task.num_steps_taken() == 0

        self.expert_action_list: List[int] = []

        self._last_held_object_name: Optional[str] = None
        self._last_to_interact_object_pose: Optional[Dict[str, Any]] = None
        self.object_name_to_priority: defaultdict = defaultdict(lambda: 0)

        self.shortest_path_navigator.on_reset()
        self.update(action_taken=None, action_success=None)

    @property
    def expert_action(self) -> int:
        """Get the current greedy expert action.

        # Returns An integer specifying the expert action in the current
        state. This corresponds to the order of actions in
        `self.task.action_names()`. For this action to be available the
        `update` function must be called after every step.
        """
        assert self.task.num_steps_taken() == len(self.expert_action_list) - 1
        return self.expert_action_list[-1]

    def update(self, action_taken: Optional[int], action_success: Optional[bool]):
        """Update the expert with the last action taken and whether or not that
        action succeeded."""
        if action_taken is not None:
            assert action_success is not None

            action_names = self.task.action_names()
            last_expert_action = self.expert_action_list[-1]
            action_str = action_names[action_taken]

            was_nav_action = any(k in action_str for k in ["move", "rotate", "look"])

            if (
                "drop_held_object_with_snap" in action_str
                and action_taken == last_expert_action
            ):
                self.object_name_to_priority[self._last_held_object_name] += 1

            if "open_by_type" in action_str and action_taken == last_expert_action:
                self.object_name_to_priority[
                    self._last_to_interact_object_pose["name"]
                ] += 1

            if not action_success:
                if was_nav_action:
                    self.shortest_path_navigator.update_graph_with_failed_action(
                        stringcase.pascalcase(action_str)
                    )
                elif (
                    ("pickup_" in action_str or "open_by_type_" in action_str)
                ) and action_taken == last_expert_action:
                    assert self._last_to_interact_object_pose is not None
                    self._invalidate_interactable_loc_for_pose(
                        location=self.task.unshuffle_env.get_agent_location(),
                        obj_pose=self._last_to_interact_object_pose,
                    )
                elif (
                    ("crouch" in action_str or "stand" in action_str)
                    and self.task.unshuffle_env.held_object is not None
                ) and action_taken == last_expert_action:
                    held_object_name = self.task.unshuffle_env.held_object["name"]
                    agent_loc = self.task.unshuffle_env.get_agent_location()
                    agent_loc["standing"] = not agent_loc["standing"]
                    self._invalidate_interactable_loc_for_pose(
                        location=agent_loc,
                        obj_pose=self.task.unshuffle_env.obj_name_to_walkthrough_start_pose[
                            held_object_name
                        ],
                    )
            else:
                # If the action succeeded and was not a move action then let's force an update
                # of our currently targeted object
                if not was_nav_action:
                    self._last_to_interact_object_pose = None

        held_object = self.task.unshuffle_env.held_object
        if self.task.unshuffle_env.held_object is not None:
            self._last_held_object_name = held_object["name"]

        self._generate_and_record_expert_action()

    def _expert_nav_action_to_obj(self, obj: Dict[str, Any]) -> Optional[str]:
        """Get the shortest path navigational action towards the object obj.

        The navigational action takes us to a position from which the
        object is interactable.
        """
        env: RearrangeTHOREnvironment = self.task.env
        agent_loc = env.get_agent_location()
        shortest_path_navigator = self.shortest_path_navigator

        interactable_positions = env._interactable_positions_cache.get(
            scene_name=env.scene, obj=obj, controller=env.controller,
        )

        target_keys = [
            shortest_path_navigator.get_key(loc) for loc in interactable_positions
        ]
        if len(target_keys) == 0:
            return None

        source_state_key = shortest_path_navigator.get_key(env.get_agent_location())

        action = "Pass"
        if source_state_key not in target_keys:
            try:
                action = shortest_path_navigator.shortest_path_next_action_multi_target(
                    source_state_key=source_state_key, goal_state_keys=target_keys,
                )
            except nx.NetworkXNoPath as _:
                # Could not find the expert actions
                return None

        if action != "Pass":
            return action
        else:
            agent_x = agent_loc["x"]
            agent_z = agent_loc["z"]
            for gdl in interactable_positions:
                d = round(abs(agent_x - gdl["x"]) + abs(agent_z - gdl["z"]), 2)
                if d <= 1e-2:
                    if _are_agent_locations_equal(agent_loc, gdl, ignore_standing=True):
                        if agent_loc["standing"] != gdl["standing"]:
                            return "Crouch" if agent_loc["standing"] else "Stand"
                        else:
                            # We are already at an interactable position
                            return "Pass"
            return None

    def _invalidate_interactable_loc_for_pose(
        self, location: Dict[str, Any], obj_pose: Dict[str, Any]
    ) -> bool:
        """Invalidate a given location in the `interactable_positions_cache` as
        we tried to interact but couldn't."""
        env = self.task.unshuffle_env

        interactable_positions = env._interactable_positions_cache.get(
            scene_name=env.scene, obj=obj_pose, controller=env.controller
        )
        for i, loc in enumerate([*interactable_positions]):
            if (
                self.shortest_path_navigator.get_key(loc)
                == self.shortest_path_navigator.get_key(location)
                and loc["standing"] == location["standing"]
            ):
                interactable_positions.pop(i)
                return True
        return False

    def _generate_expert_action_dict(self) -> Dict[str, Any]:
        """Generate a dictionary describing the next greedy expert action."""
        env = self.task.unshuffle_env

        if env.mode != RearrangeMode.SNAP:
            raise NotImplementedError(
                f"Expert only defined for 'easy' mode (current mode: {env.mode}"
            )

        held_object = env.held_object

        agent_loc = env.get_agent_location()

        if held_object is not None:
            self._last_to_interact_object_pose = None

            # Should navigate to a position where the held object can be placed
            expert_nav_action = self._expert_nav_action_to_obj(
                obj={
                    **held_object,
                    **{
                        k: env.obj_name_to_walkthrough_start_pose[held_object["name"]][
                            k
                        ]
                        for k in ["position", "rotation"]
                    },
                },
            )

            if expert_nav_action is None:
                # Could not find a path to the target, let's just immediately drop the held object
                return dict(action="DropHeldObjectWithSnap")
            elif expert_nav_action is "Pass":
                # We are in a position where we can drop the object, let's do that
                return dict(action="DropHeldObjectWithSnap")
            else:
                return dict(action=expert_nav_action)
        else:
            _, goal_poses, cur_poses = env.poses

            assert len(goal_poses) == len(cur_poses)

            failed_places_and_min_dist = (float("inf"), float("inf"))
            obj_pose_to_go_to = None
            goal_obj_pos = None
            for gp, cp in zip(goal_poses, cur_poses):
                if (
                    (gp["broken"] == cp["broken"] == False)
                    and self.object_name_to_priority[gp["name"]]
                    <= self.max_priority_per_object
                    and not RearrangeTHOREnvironment.are_poses_equal(gp, cp)
                ):
                    priority = self.object_name_to_priority[gp["name"]]
                    priority_and_dist_to_object = (
                        priority,
                        IThorEnvironment.position_dist(
                            agent_loc, gp["position"], ignore_y=True, l1_dist=True
                        ),
                    )
                    if (
                        self._last_to_interact_object_pose is not None
                        and self._last_to_interact_object_pose["name"] == gp["name"]
                    ):
                        # Set distance to -1 for the currently targeted object
                        priority_and_dist_to_object = (
                            priority_and_dist_to_object[0],
                            -1,
                        )

                    if priority_and_dist_to_object < failed_places_and_min_dist:
                        failed_places_and_min_dist = priority_and_dist_to_object
                        obj_pose_to_go_to = cp
                        goal_obj_pos = gp

            self._last_to_interact_object_pose = obj_pose_to_go_to

            if obj_pose_to_go_to is None:
                # There are no objects we need to change
                return dict(action="Done")

            expert_nav_action = self._expert_nav_action_to_obj(obj=obj_pose_to_go_to)
            if expert_nav_action is None:
                interactable_positions = self.task.env._interactable_positions_cache.get(
                    scene_name=env.scene,
                    obj=obj_pose_to_go_to,
                    controller=env.controller,
                )
                if len(interactable_positions) != 0:
                    # Could not find a path to the object, increment the place count of the object and
                    # try generating a new action.
                    get_logger().debug(
                        f"Could not find a path to {obj_pose_to_go_to}"
                        f" in scene {self.task.unshuffle_env.scene}"
                        f" when at position {self.task.unshuffle_env.get_agent_location()}."
                    )
                else:
                    get_logger().debug(
                        f"Object {obj_pose_to_go_to} in scene {self.task.unshuffle_env.scene}"
                        f" has no interactable positions."
                    )
                self.object_name_to_priority[obj_pose_to_go_to["name"]] += 1
                return self._generate_expert_action_dict()
            elif expert_nav_action == "Pass":
                with include_object_data(env.controller):
                    visible_objects = {
                        o["name"]
                        for o in env.last_event.metadata["objects"]
                        if o["visible"]
                    }

                if obj_pose_to_go_to["name"] not in visible_objects:
                    if self._invalidate_interactable_loc_for_pose(
                        location=agent_loc, obj_pose=obj_pose_to_go_to
                    ):
                        return self._generate_expert_action_dict()

                    raise RuntimeError("This should not be possible.")

                # The object of interest is interactable at the moment
                if (
                    obj_pose_to_go_to["openness"] is not None
                    and obj_pose_to_go_to["openness"] != goal_obj_pos["openness"]
                ):
                    return dict(
                        action="OpenByType",
                        objectId=obj_pose_to_go_to["objectId"],
                        moveMagnitude=goal_obj_pos["openness"],
                    )
                elif obj_pose_to_go_to["pickupable"]:
                    return dict(
                        action="Pickup", objectId=obj_pose_to_go_to["objectId"],
                    )
                else:
                    # We (likely) have an openable object which has been moved somehow but is not
                    # pickupable. We don't know what to do with such an object so we'll set its
                    # place count to a large value and try again.
                    get_logger().warning(
                        f"{obj_pose_to_go_to['name']} has moved but is not pickupable."
                    )
                    self.object_name_to_priority[goal_obj_pos["name"]] = (
                        self.max_priority_per_object + 1
                    )
                    return self._generate_expert_action_dict()
            else:
                # If we are not looking at the object to change, then we should navigate to it
                return dict(action=expert_nav_action)

    def _generate_and_record_expert_action(self):
        """Generate the next greedy expert action and save it to the
        `expert_action_list`."""
        if self.task.num_steps_taken() == len(self.expert_action_list) + 1:
            get_logger().warning(
                f"Already generated the expert action at step {self.task.num_steps_taken()}"
            )
            return

        assert self.task.num_steps_taken() == len(
            self.expert_action_list
        ), f"{self.task.num_steps_taken()} != {len(self.expert_action_list)}"
        expert_action_dict = self._generate_expert_action_dict()

        action_str = stringcase.snakecase(expert_action_dict["action"])
        if action_str not in self.task.action_names():
            obj_type = stringcase.snakecase(
                expert_action_dict["objectId"].split("|")[0]
            )
            action_str = f"{action_str}_{obj_type}"

        try:
            self.expert_action_list.append(self.task.action_names().index(action_str))
        except ValueError:
            get_logger().error(
                f"{action_str} is not a valid action for the given task."
            )
            self.expert_action_list.append(None)


def __test():
    from baseline_configs.one_phase.one_phase_rgb_base import (
        OnePhaseRGBBaseExperimentConfig,
    )
    from rearrange.tasks import RearrangeTaskSpecIterable

    task_sampler = OnePhaseRGBBaseExperimentConfig.make_sampler_fn(
        stage="train", seed=0, force_cache_reset=True, allowed_scenes=None
    )
    s = task_sampler.task_spec_iterator.scenes_to_task_spec_dicts
    task_sampler.task_spec_iterator = RearrangeTaskSpecIterable(
        scenes_to_task_spec_dicts={k: v[:1] for k, v in s.items()},
        seed=0,
        epochs=100000,
        shuffle=True,
    )
    random_action_prob = 0.0

    shortest_path_navigator = ShortestPathNavigatorTHOR(
        controller=task_sampler.unshuffle_env.controller, grid_size=STEP_SIZE
    )
    k = 0
    while True:
        print(k)
        random.seed(k)
        k += 1
        task = task_sampler.next_task()
        greedy_expert = GreedyUnshuffleExpert(
            task=task, shortest_path_navigator=shortest_path_navigator
        )
        controller = task_sampler.unshuffle_env.controller
        frames = [controller.last_event.frame]
        while not task.is_done():
            if random.random() < random_action_prob:
                assert task.action_names()[0] == "done"
                action_to_take = random.randint(1, len(task.action_names()) - 1)
            else:
                action_to_take = greedy_expert.expert_action

            # print(task.action_names()[action_to_take])
            step_result = task.step(action_to_take)
            task.unshuffle_env.controller.step("Pass")
            task.walkthrough_env.controller.step("Pass")

            greedy_expert.update(
                action_taken=action_to_take,
                action_success=step_result.info["action_success"],
            )

            frames.append(controller.last_event.frame)

        if task.metrics()["unshuffle/prop_fixed"] == 1:
            save_frames_to_mp4(frames=frames, file_name=f"rearrange_expert_{k}.mp4")

        print({k: v for k, v in task.metrics().items() if k != "task_info"})


if __name__ == "__main__":
    __test()
