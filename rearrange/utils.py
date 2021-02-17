import logging
import random
from contextlib import contextmanager
from typing import Dict, Callable, Tuple, Union, List, Any, Optional, Sequence

import ai2thor.controller
import lru
import numpy as np
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from scipy.spatial.qhull import ConvexHull, Delaunay

_UNIFORM_BOX_CACHE = {}


def save_frames_to_mp4(frames: Sequence[np.ndarray], file_name: str, fps=3):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import pylab

    h, w, _ = frames[0].shape
    aspect_ratio = w / h
    fig = plt.figure(figsize=(5 * aspect_ratio, 5))
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(frames[0], cmap="gray", interpolation="nearest")
    im.set_clim([0, 255])

    pylab.tight_layout()

    def update_img(n):
        if n >= len(frames):
            im.set_data(frames[-1])
        else:
            im.set_data(frames[n])
        return im

    ani = animation.FuncAnimation(fig, update_img, len(frames) - 1, interval=200)
    writer = animation.writers["ffmpeg"](fps=fps)

    ani.save(file_name, writer=writer, dpi=300)


def hand_in_initial_position(controller: ai2thor.controller.Controller):
    metadata = controller.last_event.metadata
    return (
        IThorEnvironment.position_dist(
            metadata["hand"]["localPosition"], {"x": 0, "y": -0.16, "z": 0.38},
        )
        < 1e-4
        and IThorEnvironment.angle_between_rotations(
            metadata["hand"]["localRotation"], {"x": 0, "y": 0, "z": 0}
        )
        < 1e-2
    )


class BoundedFloat(object):
    """Declare a bounded float placeholder variable."""

    def __init__(self, low: float, high: float):
        """High is the max float value, low is the min (both inclusive)."""
        self.types = {float, int, np.float64}
        if type(low) not in self.types or type(high) not in self.types:
            raise ValueError("Bounds must both be floats.")
        if low > high:
            raise ValueError("low must be less than high.")
        self.low = low
        self.high = high

    def sample(self) -> float:
        """Return a random float within the initialized range."""
        return random.random() * (self.high - self.low) + self.low

    def __contains__(self, n: float):
        """Assert n is within this classes bounded range."""
        if type(n) not in self.types:
            raise ValueError("n must be a float (or an int).")
        return n >= self.low and n <= self.high


class RearrangeActionSpace(object):
    """Control which actions with bounded variables can be executed."""

    def __init__(self, actions: Dict[Callable, Dict[str, BoundedFloat]]):
        """Build a new AI2-THOR action space.

        Attributes
        :actions (Dict[Callable, Dict[str, BoundedFloat]]) must be in the form
        {
            <Callable: e.g., controller.move_ahead>: {
                '<x>': <BoundedFloat(low=0.5, high=2.5)>,
                '<y>': <BoundedFloat(low=0.5, high=2.5)>,
                '<z>': <BoundedFloat(low=0.5, high=2.5)>,
                '<degrees>': <BoundedFloat(low=-90, high=90)>,
                ...
            },
            ...
        },
        where the action variables are in the value and the callable function
        is the key.
        """
        self.keys = list(actions.keys())
        self.actions = actions

    def execute_random_action(self, log_choice: bool = True) -> None:
        """Execute a random action within the specified action space."""
        action = random.choice(self.keys)
        kwargs = {
            name: bounds.sample() for name, bounds in self.actions[action].items()
        }

        # logging
        if log_choice:
            kwargs_str = str("".join(f"  {k}: {v},\n" for k, v in kwargs.items()))
            kwargs_str = "\n" + kwargs_str[:-2] if kwargs_str else ""
            logging.info(f"Executing {action.__name__}(" + kwargs_str + ")")

        action(**kwargs)

    def __contains__(
        self, action_fn_and_kwargs: Tuple[Callable, Dict[str, float]]
    ) -> bool:
        """Return if action_fn with variables is valid in this ActionSpace."""
        action_fn, variables = action_fn_and_kwargs

        # asserts the action is valid
        if action_fn not in self.actions:
            return False

        # asserts the variables are valid
        for name, x in variables.items():
            if x not in self.actions[action_fn][name]:
                return False

        return True

    def __str__(self) -> str:
        """Return a string representation of the action space."""
        return self.__repr__()

    def __repr__(self) -> str:
        """Return a string representation of the action space."""
        s = ""
        tab = " " * 2  # default tabs have like 8 spaces on shells
        for action_fn, vars in self.actions.items():
            fn_name = action_fn.__name__
            vstr = ""
            for i, (var_name, bound) in enumerate(vars.items()):
                low = bound.low
                high = bound.high
                vstr += f"{tab * 2}{var_name}: float(low={low}, high={high})"
                vstr += "\n" if i + 1 == len(vars) else ",\n"
            vstr = "\n" + vstr[:-1] if vstr else ""
            s += f"{tab}{fn_name}({vstr}),\n"
        s = s[:-2] if s else ""
        return "ActionSpace(\n" + s + "\n)"


def extract_obj_data(obj):
    """Return object evaluation metrics based on the env state."""
    if "type" in obj:
        return {
            "type": obj["type"],
            "position": obj["position"],
            "rotation": obj["rotation"],
            "openness": obj["openness"],
            "pickupable": obj["pickupable"],
            "broken": obj["broken"],
            "bounding_box": obj["bounding_box"],
            "objectId": obj["objectId"],
            "name": obj["name"],
            "parentReceptacles": obj.get("parentReceptacles", []),
        }
    return {
        "type": obj["objectType"],
        "position": obj["position"],
        "rotation": obj["rotation"],
        "openness": obj["openness"] if obj["openable"] else None,
        "pickupable": obj["pickupable"],
        "broken": obj["isBroken"],
        "objectId": obj["objectId"],
        "name": obj["name"],
        "parentReceptacles": obj.get("parentReceptacles", []),
        "bounding_box": obj["objectOrientedBoundingBox"]["cornerPoints"]
        if obj["objectOrientedBoundingBox"]
        else None,
    }


def get_pose_info(
    objs: Union[Sequence[Dict[str, Any]], Dict[str, Any]]
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Return data about each specified object.

    For each object, the return consists of its type, position,
    rotation, openness, and bounding box.
    """
    # list of objects
    if isinstance(objs, Sequence):
        return [extract_obj_data(obj) for obj in objs]
    # single object
    return extract_obj_data(objs)


def execute_action(
    controller: ai2thor.controller.Controller,
    action_space: RearrangeActionSpace,
    action_fn: Callable,
    thor_action: str,
    error_message: str = "",
    updated_kwarg_names: Optional[Dict[str, str]] = None,
    default_thor_kwargs: Optional[Dict[str, Any]] = None,
    preprocess_kwargs_inplace: Optional[Callable] = None,
    **kwargs: float,
) -> bool:
    """Execute a bounded action within the AI2-THOR controller."""
    if updated_kwarg_names is None:
        updated_kwarg_names = {}
    if default_thor_kwargs is None:
        default_thor_kwargs = {}

    if (action_fn, kwargs) not in action_space:  # Checks that values are in bounds
        raise ValueError(
            error_message
            + f" action_fn=={action_fn}, kwargs=={kwargs}, action_space=={action_space}."
        )

    if preprocess_kwargs_inplace is not None:
        if len(updated_kwarg_names) != 0:
            raise NotImplementedError(
                "Cannot have non-empty `updated_kwarg_names` and a non-None `preprocess_kwargs_inplace` argument."
            )
        preprocess_kwargs_inplace(kwargs)

    # get rid of bad variable names
    for better_kwarg, thor_kwarg in updated_kwarg_names.items():
        kwargs[thor_kwarg] = kwargs[better_kwarg]
        del kwargs[better_kwarg]

    for name, value in default_thor_kwargs.items():
        kwargs[name] = value

    event = controller.step(thor_action, **kwargs)
    return event.metadata["lastActionSuccess"]


def _iou_slow(
    b1: Sequence[Sequence[float]],
    b2: Sequence[Sequence[float]],
    num_points: int = 2197,
) -> float:
    """Calculate the IoU between 3d bounding boxes b1 and b2."""
    b1 = np.array(b1) if not isinstance(b1, np.ndarray) else b1
    b2 = np.array(b2) if not isinstance(b2, np.ndarray) else b2

    def _outer_bounds(
        points_1: np.ndarray, points_2: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Sample points from the outer bounds formed by points_1/2."""
        assert points_1.shape == points_2.shape
        bounds = dict()
        for i in range(points_1.shape[0]):
            x1, y1, z1 = points_1[i]
            x2, y2, z2 = points_2[i]
            points = [
                (x1, "x"),
                (x2, "x"),
                (y1, "y"),
                (y2, "y"),
                (z1, "z"),
                (z2, "z"),
            ]
            for val, d_key in points:
                if d_key not in bounds:
                    bounds[d_key] = {"min": val, "max": val}
                else:
                    if val > bounds[d_key]["max"]:
                        bounds[d_key]["max"] = val
                    elif val < bounds[d_key]["min"]:
                        bounds[d_key]["min"] = val
        return bounds

    def _in_box(box: np.ndarray, points: np.ndarray) -> np.ndarray:
        """For each point, return if its in the hull."""
        hull = ConvexHull(box)
        deln = Delaunay(box[hull.vertices])
        return deln.find_simplex(points) >= 0

    bounds = _outer_bounds(b1, b2)
    dim_points = int(num_points ** (1 / 3))

    xs = np.linspace(bounds["x"]["min"], bounds["x"]["max"], dim_points)
    ys = np.linspace(bounds["y"]["min"], bounds["y"]["max"], dim_points)
    zs = np.linspace(bounds["z"]["min"], bounds["z"]["max"], dim_points)
    points = np.array([[x, y, z] for x in xs for y in ys for z in zs], copy=False)

    in_b1 = _in_box(b1, points)
    in_b2 = _in_box(b2, points)

    intersection = np.count_nonzero(in_b1 * in_b2)
    union = np.count_nonzero(in_b1 + in_b2)
    iou = intersection / union if union else 0
    return iou


def get_basis_for_3d_box(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert corners[0].sum() == 0.0

    without_first = corners[1:]
    magnitudes1 = np.sqrt((without_first * without_first).sum(1))
    v0_ind = np.argmin(magnitudes1)
    v0_mag = magnitudes1[v0_ind]
    v0 = without_first[np.argmin(magnitudes1)] / v0_mag

    orth_to_v0 = (v0.reshape(1, -1) * without_first).sum(-1) < v0_mag / 2.0
    inds_orth_to_v0 = np.where(orth_to_v0)[0]
    v1_ind = inds_orth_to_v0[np.argmin(magnitudes1[inds_orth_to_v0])]
    v1_mag = magnitudes1[v1_ind]
    v1 = without_first[v1_ind, :] / magnitudes1[v1_ind]

    orth_to_v1 = (v1.reshape(1, -1) * without_first).sum(-1) < v1_mag / 2.0
    inds_orth_to_v0_and_v1 = np.where(orth_to_v0 & orth_to_v1)[0]

    if len(inds_orth_to_v0_and_v1) != 1:
        raise RuntimeError(f"Could not find basis for {corners}")

    v2_ind = inds_orth_to_v0_and_v1[0]
    v2 = without_first[v2_ind, :] / magnitudes1[v2_ind]

    orth_mat = np.stack((v0, v1, v2), axis=1)  # Orthonormal matrix

    return orth_mat, magnitudes1[[v0_ind, v1_ind, v2_ind]]


def uniform_box_points(n):
    if n not in _UNIFORM_BOX_CACHE:
        start = 1.0 / (2 * n)
        lin_space = np.linspace(start, 1 - start, num=n).reshape(n, 1)
        mat = lin_space
        for i in range(2):
            mat = np.concatenate(
                (np.repeat(lin_space, mat.shape[0], 0), np.tile(mat, (n, 1))), axis=1,
            )
        _UNIFORM_BOX_CACHE[n] = mat

    return _UNIFORM_BOX_CACHE[n]


def iou_box_3d(b1: Sequence[Sequence[float]], b2: Sequence[Sequence[float]]) -> float:
    """Calculate the IoU between 3d bounding boxes b1 and b2."""
    import numpy as np

    b1 = np.array(b1)
    b2 = np.array(b2)

    assert b1.shape == b2.shape == (8, 3)

    b1_center = b1[:1, :]
    b1 = b1 - b1_center
    b1_orth_basis, b1_mags = get_basis_for_3d_box(corners=b1)

    b2 = (b2 - b1_center) @ b1_orth_basis
    b2_center = b2[:1, :]
    b2 = b2 - b2_center

    b2_orth_basis, b2_mags = get_basis_for_3d_box(corners=b2)

    sampled_points = b2_center.reshape(1, 3) + (
        uniform_box_points(13) @ (b2_mags.reshape(-1, 1) * np.transpose(b2_orth_basis))
    )

    prop_intersection = (
        np.logical_and(
            sampled_points > -1e-3, sampled_points <= 1e-3 + b1_mags.reshape(1, 3)
        )
        .all(-1)
        .mean()
    )

    b1_vol = np.prod(b1_mags)
    b2_vol = np.prod(b2_mags)
    intersect_vol = b2_vol * prop_intersection

    return intersect_vol / (b1_vol + b2_vol - intersect_vol)


@contextmanager
def include_object_data(controller: ai2thor.controller.Controller):
    needs_reset = len(controller.last_event.metadata["objects"]) == 0
    try:
        if needs_reset:
            controller.step("ResetObjectFilter")
            assert controller.last_event.metadata["lastActionSuccess"]
        yield None
    finally:
        if needs_reset:
            controller.step("SetObjectFilter", objectIds=[])
            assert controller.last_event.metadata["lastActionSuccess"]


class PoseMismatchError(Exception):
    pass


class ObjectInteractablePostionsCache:
    def __init__(self, max_size: int = 20000, ndigits=2):
        self._key_to_positions = lru.LRU(size=max_size)

        self.ndigits = ndigits
        self.max_size = max_size

    def _get_key(self, scene_name: str, obj: Dict[str, Any]):
        p = obj["position"]
        return (
            scene_name,
            obj["type"] if "type" in obj else obj["objectType"],
            round(p["x"], self.ndigits),
            round(p["y"], self.ndigits),
            round(p["z"], self.ndigits),
        )

    def get(
        self,
        scene_name: str,
        obj: Dict[str, Any],
        controller: ai2thor.controller.Controller,
        reachable_positions: Optional[Sequence[Dict[str, float]]] = None,
        force_cache_refresh: bool = False,
    ) -> List[Dict[str, Union[float, int, bool]]]:
        scene_name = scene_name.replace("_physics", "")
        obj_key = self._get_key(scene_name=scene_name, obj=obj)

        if force_cache_refresh or obj_key not in self._key_to_positions:
            with include_object_data(controller):
                metadata = controller.last_event.metadata

            cur_scene_name = metadata["sceneName"].replace("_physics", "")
            assert (
                scene_name == cur_scene_name
            ), f"Scene names must match when filling a cache miss ({scene_name} != {cur_scene_name})."

            obj_in_scene = next(
                (o for o in metadata["objects"] if o["name"] == obj["name"]), None,
            )
            if obj_in_scene is None:
                raise RuntimeError(
                    f"Object with name {obj['name']} must be in the scene when filling a cache miss"
                )

            desired_pos = obj["position"]
            desired_rot = obj["rotation"]

            cur_pos = obj_in_scene["position"]
            cur_rot = obj_in_scene["rotation"]

            should_teleport = (
                IThorEnvironment.position_dist(desired_pos, cur_pos) >= 1e-3
                or IThorEnvironment.rotation_dist(desired_rot, cur_rot) >= 1
            )

            object_held = obj_in_scene["isPickedUp"]
            if should_teleport:
                if object_held:
                    if not hand_in_initial_position(controller=controller):
                        raise NotImplementedError

                    controller.step("PausePhysicsAutoSim")
                    assert controller.last_event.metadata["lastActionSuccess"]

                event = controller.step(
                    "TeleportObject",
                    objectId=obj_in_scene["objectId"],
                    rotation=desired_rot,
                    **desired_pos,
                    forceAction=True,
                    allowTeleportOutOfHand=True,
                    forceKinematic=True,
                )
                assert event.metadata["lastActionSuccess"]

            metadata = controller.step(
                action="GetInteractablePoses",
                objectId=obj["objectId"],
                positions=reachable_positions,
            ).metadata
            self._key_to_positions[obj_key] = metadata["actionReturn"]

            if should_teleport:
                if object_held:
                    if hand_in_initial_position(controller=controller):
                        controller.step(
                            "PickupObject",
                            objectId=obj_in_scene["objectId"],
                            forceAction=True,
                        )
                        assert controller.last_event.metadata["lastActionSuccess"]
                        controller.step("UnpausePhysicsAutoSim")
                        assert controller.last_event.metadata["lastActionSuccess"]
                    else:
                        raise NotImplementedError
                else:
                    event = controller.step(
                        "TeleportObject",
                        objectId=obj_in_scene["objectId"],
                        rotation=cur_rot,
                        **cur_pos,
                        forceAction=True,
                    )
                    assert event.metadata["lastActionSuccess"]

        return self._key_to_positions[obj_key]
