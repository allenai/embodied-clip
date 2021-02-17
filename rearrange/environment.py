import enum
import math
import pprint
import random
import traceback
from collections import OrderedDict
from typing import Dict, Any, Tuple, Optional, Callable, List, Union, Sequence

import ai2thor
import ai2thor.controller
import ai2thor.server
import numpy as np
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.ithor_plugin.ithor_util import round_to_factor
from torch.distributions.utils import lazy_property

from datagen.datagen_constants import OBJECT_TYPES_TO_NOT_MOVE
from datagen.datagen_utils import (
    open_objs,
    get_object_ids_to_not_move_from_object_types,
    remove_objects_until_all_have_identical_meshes,
)
from rearrange.constants import (
    REQUIRED_THOR_VERSION,
    MAX_HAND_METERS,
)
from rearrange.utils import (
    BoundedFloat,
    RearrangeActionSpace,
    include_object_data,
    PoseMismatchError,
    ObjectInteractablePostionsCache,
    execute_action,
    get_pose_info,
    iou_box_3d,
)


class RearrangeMode(enum.Enum):
    """Different modes allowed in RearrangeTHOREnvironment."""

    MANIPULATE = "MANIPULATE"
    SNAP = "SNAP"


class RearrangeTaskSpec:
    """Data container encapsulating how a single rearrangement instance should
    be initialized.

    The rearrangement datasets are structured as large dictionaries of the form
    ```python
    {
        SCENE_NAME: [
            {
                DATA_DEFINING_A_SINGLE_REARRANGE_TASK
            },
            ...
        ],
        ...
    }
    ```

    This `RearrangeTaskSpec` is used to encapsulate the `DATA_DEFINING_A_SINGLE_REARRANGE_TASK`
    which allows us to use autocomplete and type checking rather than passing around raw dictionaries.

    # Attributes
    scene : A string specifying the AI2-THOR scene (e.g "FloorPlan18") in which to run the rearrange task.
    stage : A string specifying the type of instance this is data corresponds to (e.g. "train", "val", "test", etc.)
    agent_position : A Dict[str, float] specifying the "x", "y", and "z" coordinates of the agent's starting position.
    agent_rotation: A float specifying the agents starting rotation (in degrees).
    openable_data : A sequence of dictionaries specifying the degree to which certain objects in the scene should be open
        in the walkthrough and unshuffle phases. E.g. the openness of a particular cabinent might be specified by the
        dictionary:
        ```python
        {
            "name": "Cabinet_a8b4237f",
            "objectName": "Cabinet_a8b4237f",
            "objectId": "Cabinet|+01.31|+02.46|+04.36",
            "start_openness": 0.6170539671128578,
            "target_openness": 0.8788923191809455
        }
        ```
        where `start_openness` is the degree to which the cabinent is open at the start of the unshuffle phase.
    starting_poses : A sequence of dictionaries specifying the poses of all pickupable objects at the start
        of the unshuffle phase. E.g. one such dictionary might look like:
        ```python
        {
                    "name": "Bowl_803d17c0",
                    "objectName": "Bowl_803d17c0",
                    "position": {
                        "x": -0.5572903156280518,
                        "y": 0.8256161212921143,
                        "z": 6.25293493270874,
                    },
                    "rotation": {
                        "x": 359.9241943359375,
                        "y": -0.00041645264718681574,
                        "z": 0.004868899006396532,
                    },
                }
        ```
    target_poses : Similar to `starting_poses` but specifying the poses of objects during the walkthrough phase.
    runtime_sample : If `True`, then this task is meant to randomly specified at runtime. That is, the above fields
        (except for the `scene`) are to be left as `None` and the RearrangeTHOREnvironment will randomly generate
        them instead (this may be slow).
    runtime_data : A Dict[str, Any] into which the `RearrangeTHOREnvironment` may cache data for efficiency.
    metrics : Any additional metrics that might be associated with a task specification. For instance, the
        rearrangement dataset dictionaries include metrics such as `open_diff_count` which records the number
        of objects who differ in openness at the start of the walkthrough/unshuffle phases.
    """

    def __init__(
        self,
        scene: str,
        stage: Optional[str] = None,
        agent_position: Optional[Dict[str, float]] = None,
        agent_rotation: Optional[float] = None,
        openable_data: Optional[Sequence[Dict[str, Any]]] = None,
        starting_poses: Optional[Sequence[Dict[str, Any]]] = None,
        target_poses: Optional[Sequence[Dict[str, Any]]] = None,
        runtime_sample: bool = False,
        runtime_data: Optional[Dict[str, Any]] = None,
        **metrics,
    ):
        """Instantiate a `RearrangeTaskSpec` object."""
        self.scene = scene
        self.stage = stage
        self.agent_position = agent_position
        self.agent_rotation = agent_rotation
        self.openable_data = openable_data
        self.starting_poses = starting_poses
        self.target_poses = target_poses
        self.runtime_sample = runtime_sample
        self.runtime_data: Dict[
            str, Any
        ] = runtime_data if runtime_data is not None else {}
        self.metrics = metrics

    def __str__(self):
        """String representation of a `RearrangeTaskSpec` object."""
        return pprint.pformat(self.__dict__)

    @property
    def unique_id(self):
        if self.runtime_sample:
            raise NotImplementedError("Cannot create a unique id for a runtime sample.")
        return f"{self.scene}__{self.stage}__{self.metrics['index']}"


class RearrangeTHOREnvironment:
    """Custom AI2-THOR Controller for the task of object rearrangement.

    # Attributes
    mode : The current mode of rearrangement. Takes one of the values of RearrangeMode
        (RearrangeMode.SNAP or RearrangeMode.MANIPULATE).
    force_cache_reset : Whether or not we should force cache resets when using the `drop_held_object_with_snap` action.
        Setting this value to `False` results in higher FPS at the expense of possibly having `drop_held_object_with_snap`
        work/fail when it shouldn't. Setting `force_cache_reset` to `True` is recommended during validation/testing.
    obj_name_to_walkthrough_start_pose : Dictionary mapping AI2-THOR object names to their poses (positions & rotations)
        before they were shuffled (i.e. what the agent sees at the start of the walkthrough phase).
         This will be changed after every call to `reset`.
    obj_name_to_unshuffle_start_pose : Same as `obj_name_to_walkthrough_start_pose` but mapping object names to their poses (positions &
        rotations) just after they were shuffled, i.e. what the agent sees at the start of the unshuffle phase).
    current_task_spec : A `RearrangeTaskSpec` object specifying the current rearrangement task details.
    controller : A ai2thor controller used to execute all the actions.
    shuffle_called : `True` if the objects have been shuffled so that we're in the `unshuffle` phase. Otherwise `False`.
    """

    def __init__(
        self,
        mode: RearrangeMode = RearrangeMode.SNAP,
        force_cache_reset: Optional[bool] = None,
        controller_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a new rearrangement controller.

        # Parameters
        mode : See description of this class' attributes.
        controller_kwargs : Dictionary specifying any keyword arguments to be passed
            when initializing the `ai2thor.controller.Controller` (e.g. width/height).
        """
        if ai2thor.__version__ is not None:  # Allows for custom THOR installs
            for v in REQUIRED_THOR_VERSION.split(","):
                operators = ["<=", ">=", "==", "<", ">"]
                if any(op in v for op in operators):
                    op = next(op for op in operators if op in v)
                    v = f"{op}'{v.replace(op, '')}'"
                    correct_version = eval(f"'{ai2thor.__version__}'{v}")
                else:
                    correct_version = ai2thor.__version__ == v
                if not correct_version:
                    raise ValueError(f"Please use AI2-THOR v{REQUIRED_THOR_VERSION}")

        # Saving attributes
        if mode == RearrangeMode.SNAP:
            assert (
                force_cache_reset is not None
            ), "When in RearrangeMode.SNAP mode you must specify a value for `force_cache_reset`"
        else:
            force_cache_reset = force_cache_reset
        self.force_cache_reset = force_cache_reset
        self.mode = mode
        self._controller_kwargs = {} if controller_kwargs is None else controller_kwargs

        # Cache of where objects can be interacted with
        self._interactable_positions_cache = ObjectInteractablePostionsCache()

        # Object poses at start of walkthrough and unshuffle phases.
        # Reset after every call to reset and shuffle respectively.
        self.obj_name_to_walkthrough_start_pose: Optional[Dict[str, Dict]] = None
        self.obj_name_to_unshuffle_start_pose: Optional[Dict[str, Dict]] = None
        self._cached_poses: Optional[Tuple[List, List, List]] = None

        # Current task specification
        self.current_task_spec: Optional[RearrangeTaskSpec] = None

        # Caches of starting unshuffle/walkthrough object poses and other information. Reset on every call to reset
        self._sorted_and_extracted_walkthrough_start_poses: Optional[List] = None
        self._sorted_and_extracted_unshuffle_start_poses: Optional[List] = None
        self._have_warned_about_mismatch = False
        self._agent_signals_done = False  # Also reset on `shuffle()`

        # instance masks now not supported. But an Exception would be thrown if
        # `mode == RearrangeMode.MANIPULATE` and render_instance_masks is True, since masks are
        # only available on RearrangeMode.SNAP mode.
        self._render_instance_masks: bool = False
        if self.mode == RearrangeMode.MANIPULATE and self._render_instance_masks:
            raise Exception(
                "render_instance_masks is only available on RearrangeMode.SNAP mode."
            )

        # local thor controller to execute all the actions
        self.controller = self.create_controller()

        # always begin in walkthrough phase
        self.shuffle_called = False

    def create_controller(self):
        """Create the ai2thor controller."""

        assert ("width" in self._controller_kwargs) == (
            "height" in self._controller_kwargs
        ), "Either controller_kwargs must contain either both of width/height or neither."
        self._controller_kwargs["width"] = self._controller_kwargs.get("width", 300)
        self._controller_kwargs["height"] = self._controller_kwargs.get("height", 300)

        controller = ai2thor.controller.Controller(
            server_class=ai2thor.fifo_server.FifoServer,
            scene="FloorPlan17_physics",
            **self._controller_kwargs,
            # server_class=ai2thor.wsgi_server.WsgiServer,  # Possibly useful in debugging
        )
        return controller

    @property
    def held_object(self) -> Optional[Dict[str, Any]]:
        """Return the data corresponding to the object held by the agent (if
        any)."""
        with include_object_data(self.controller):
            metadata = self.controller.last_event.metadata

            held_objs = [o for o in metadata["objects"] if o["isPickedUp"]]
            assert len(held_objs) <= 1, (
                f"In scene {self.scene}: should not be able to hold more than one object."
                f"\n\nCurrently holding {held_objs}."
                f"\n\nTask spec {self.current_task_spec}."
            )
            if len(held_objs) == 1:
                return held_objs[0]
            return None

    def get_agent_location(self) -> Dict[str, Union[float, int, bool]]:
        """Returns the agent's current location.

        # Returns

        A dictionary of the form
        ```python
        {
            "x": X_POSITION_IN_SPACE, # float
            "y": Y_POSITION_IN_SPACE, # float
            "z": Z_POSITION_IN_SPACE, # float
            "rotation": AGENTS_ROTATION_ABOUT_THE_Y_AXIS_IN_DEGREES, # float or int
            "horizon": AGENTS_CAMERA_ANGLE_IN_DEGREES, # float (0 degrees is horizontal)
            "standing": WHETHER_OR_NOT_THE_AGENT_IS_STANDING, # boolean
        }
        ```
        """
        metadata = self.controller.last_event.metadata
        return {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
            "standing": metadata.get("isStanding", metadata["agent"].get("isStanding")),
        }

    @property
    def observation(self) -> Tuple[np.array, Optional[np.array]]:
        """Return the current (RGB, depth, Optional[instance masks]) frames.

        # Returns
        A tuple containing a
        * RGB frame is of shape (height)x(width)x3 with integer entries in [0:255].
        * depth frame is of shape (height)x(width) with unscaled entries representing the
            meter distance from the agent to the pixel. This will be `None` if the controller_kwargs
            passed to the initializer did not specify that depth images should be returned by AI2-THOR.
        """
        rgb = self.last_event.frame
        depth = (
            self.last_event.depth_frame
            if hasattr(self.last_event, "depth_frame")
            else None
        )
        return rgb, depth

    @lazy_property
    def walkthrough_action_space(self) -> RearrangeActionSpace:
        """Return the RearrangeActionSpace for the walkthrough phase based on
        the RearrangeMode."""

        # Walkthrough actions
        actions: Dict[Callable, Dict[str, BoundedFloat]] = {
            self.move_ahead: {},
            self.move_right: {},
            self.move_left: {},
            self.move_back: {},
            self.rotate_right: {},
            self.rotate_left: {},
            self.stand: {},
            self.crouch: {},
            self.look_up: {},
            self.look_down: {},
            self.done: {},
        }

        return RearrangeActionSpace(actions)

    @lazy_property
    def unshuffle_action_space(self) -> RearrangeActionSpace:
        """Return the RearrangeActionSpace for the unshuffle phase based on the
        RearrangeMode."""
        actions = {**self.walkthrough_action_space.actions}

        # additional shuffle allowed actions
        actions.update(
            {
                self.open_object: {
                    "x": BoundedFloat(low=0, high=1),
                    "y": BoundedFloat(low=0, high=1),
                    "openness": BoundedFloat(low=0, high=1),
                },
                self.pickup_object: {
                    "x": BoundedFloat(low=0, high=1),
                    "y": BoundedFloat(low=0, high=1),
                },
                self.push_object: {
                    "x": BoundedFloat(low=0, high=1),
                    "y": BoundedFloat(low=0, high=1),
                    "rel_x_force": BoundedFloat(low=-0.5, high=0.5),
                    "rel_y_force": BoundedFloat(low=-0.5, high=0.5),
                    "rel_z_force": BoundedFloat(low=-0.5, high=0.5),
                    "force_magnitude": BoundedFloat(low=0, high=1),
                },
                self.move_held_object: {
                    "x_meters": BoundedFloat(low=-0.5, high=0.5),
                    "y_meters": BoundedFloat(low=-0.5, high=0.5),
                    "z_meters": BoundedFloat(low=-0.5, high=0.5),
                },
                self.rotate_held_object: {
                    "x": BoundedFloat(low=-0.5, high=0.5),
                    "y": BoundedFloat(low=-0.5, high=0.5),
                    "z": BoundedFloat(low=-0.5, high=0.5),
                },
                self.drop_held_object: {},
            }
        )

        if self.mode == RearrangeMode.SNAP:
            actions.update({self.drop_held_object_with_snap: {}})

        return RearrangeActionSpace(actions)

    @property
    def action_space(self) -> RearrangeActionSpace:
        """Return the RearrangeActionSpace based on the RearrangeMode and
        whether we are in the unshuffle phase."""

        if self.shuffle_called:
            return self.unshuffle_action_space
        else:
            return self.walkthrough_action_space

    def open_object(self, x: float, y: float, openness: float) -> bool:
        """Open the object corresponding to x/y to openness.

        The action will not be successful if the specified openness would
        cause a collision or if the object at x/y is not openable.

        # Parameters
        x : (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        y : (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        # Returns
        `True` if the action was successful, otherwise `False`.
        """
        # If an object is already open, THOR doesn't support changing
        # it's openness without first closing it. So we simply try to first
        # close the object before reopening it.
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.open_object,
            thor_action="OpenObject",
            error_message=(
                "x/y/openness must be in [0:1] and we must be in the unshuffle phase."
            ),
            updated_kwarg_names={"openness": "moveMagnitude"},
            x=x,
            y=y,
            openness=openness,
        )

    def pickup_object(self, x: float, y: float) -> bool:
        """Pick up the object corresponding to x/y.

        The action will not be successful if the object at x/y is not
        pickupable.

        # Parameters
        x : (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        y : (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        # Returns
        `True` if the action was successful, otherwise `False`.
        """
        if len(self.last_event.metadata["inventoryObjects"]) != 0:
            return False
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.pickup_object,
            thor_action="PickupObject",
            error_message="x/y must be in [0:1] and we must be in the unshuffle phase.",
            x=x,
            y=y,
        )

    def push_object(
        self,
        x: float,
        y: float,
        rel_x_force: float,
        rel_y_force: float,
        rel_z_force: float,
        force_magnitude: float,
    ) -> bool:
        """Push an object along a surface.

        The action will not be successful if the object at x/y is not moveable.

        # Parameters
        x : (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        y : (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.
        rel_x_force : (float, min=-0.5, max=0.5) amount of relative force
           applied along the x axis.
        rel_y_force : (float, min=-0.5, max=0.5) amount of relative force
           applied along the y axis.
        rel_z_force : (float, min=-0.5, max=0.5) amount of relative force
           applied along the z axis.
        force_magnitude : (float, min=0, max=1) relative amount of force
           applied during this push action. Within AI2-THOR, the force is
           rescaled to be between 0 and 50 newtons, which is estimated to
           sufficiently move all pickupable objects.

        # Returns
        `True` if the action was successful, otherwise `False`.
        """

        def preprocess_kwargs(kwargs: Dict[str, Any]):
            direction = {}
            for k in ["x", "y", "z"]:
                force_key = f"rel_{k}_force"
                direction[k] = kwargs[force_key]
                del kwargs[force_key]
            kwargs["direction"] = direction
            kwargs["force_magnitude"] = 50 * kwargs["force_magnitude"]

        # TODO: is this really the definition of success we want?
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.pickup_object,
            thor_action="TouchThenApplyForce",
            error_message="Error in call to pickup object."
            " Must be in unshuffle phase (i.e., call shuffle()),"
            " x,y,force_magnitude must be in [0:1],"
            " and rel_(x/y/z)_force must be in [-0.5:0.5]",
            default_thor_kwargs=dict(handDistance=1.5),
            preprocess_kwargs_inplace=preprocess_kwargs,
            x=x,
            y=y,
            rel_x_force=rel_x_force,
            rel_y_force=rel_y_force,
            rel_z_force=rel_z_force,
            moveMagnitude=force_magnitude,
        )

    def move_ahead(self) -> bool:
        """Move the agent ahead from its facing direction by 0.25 meters."""
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_ahead,
            thor_action="MoveAhead",
        )

    def move_back(self) -> bool:
        """Move the agent back from its facing direction by 0.25 meters."""
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_back,
            thor_action="MoveBack",
        )

    def move_right(self) -> bool:
        """Move the agent right from its facing direction by 0.25 meters."""
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_right,
            thor_action="MoveRight",
        )

    def move_left(self) -> bool:
        """Move the agent left from its facing direction by 0.25 meters."""
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_left,
            thor_action="MoveLeft",
        )

    def rotate_left(self) -> bool:
        """Rotate the agent left from its facing direction."""
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.rotate_left,
            thor_action="RotateLeft",
        )

    def rotate_right(self) -> bool:
        """Rotate the agent left from its facing direction."""
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.rotate_right,
            thor_action="RotateRight",
        )

    def stand(self) -> bool:
        """Stand the agent from the crouching position."""
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.stand,
            thor_action="Stand",
        )

    def crouch(self) -> bool:
        """Crouch the agent from the standing position."""
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.crouch,
            thor_action="Crouch",
        )

    def look_up(self) -> bool:
        """Turn the agent's head and camera up by 30 degrees."""
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.look_up,
            thor_action="LookUp",
        )

    def look_down(self) -> bool:
        """Turn the agent's head and camera down by 30 degrees."""
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.look_down,
            thor_action="LookDown",
        )

    def done(self) -> bool:
        """Agent's signal that it's completed its current rearrangement phase.

        Note that we do not automatically switch from the walkthrough
        phase to the unshuffling phase, and vice-versa, that is up to
        the user. This allows users to call .poses after the agent calls
        done, and have it correspond to the current episode.
        """
        self._agent_signals_done = True
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.done,
            thor_action="Done",
        )

    def move_held_object(
        self, x_meters: float, y_meters: float, z_meters: float
    ) -> bool:
        """Move the object in the agent's hand by the specified amount.

        The maximum magnitude that the object
        can move in one time step is 0.5 meters. If the calculated magnitude is
        above 0.5, it's magnitude will be clipped to 0.5.

        The action is successful in the case that the agent is holding an
        object and moving the object by the specified amount does not bump
        into an object.

        # Parameters
        x_meters : (float, min=-0.5, max=0.5) movement meters along the x-axis.
        y_meters : (float, min=-0.5, max=0.5) movement meters along the y-axis.
        z_meters : (float, min=-0.5, max=0.5) movement meters along the z-axis.

        # Exceptions
        In walkthrough phase. This method can only be called within the
        unshuffle phase. The shuffle phase starts with controller.shuffle()
        and ends with controller.reset().
        """
        mag = math.sqrt(x_meters ** 2 + y_meters ** 2 + z_meters ** 2)

        # clips the max value at MAX_HAND_METERS.
        if MAX_HAND_METERS > mag:
            scale = MAX_HAND_METERS / mag
            x_meters *= scale
            y_meters *= scale
            z_meters *= scale

        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_held_object,
            thor_action="MoveHandDelta",
            updated_kwarg_names={"x_meters": "x", "y_meters": "y", "z_meters": "z"},
            x_meters=x_meters,
            y_meters=y_meters,
            z_meters=z_meters,
        )

    def rotate_held_object(self, x: float, y: float, z: float) -> bool:
        """Rotate the object in the agent's hand by the specified degrees.

        The rotation parameters are scaled linearly to put rotations
        between [-90:90] degrees. The action is only successful agent is holding an object.

        # Parameters
        x : (float, min=-0.5, max=0.5) rotation along the x-axis.
        y : (float, min=-0.5, max=0.5) rotation along the y-axis.
        z : (float, min=-0.5, max=0.5) rotation along the z-axis.
        """

        def rescale_xyz(kwargs: Dict[str, Any]):
            for k in ["x", "y", "z"]:
                kwargs[k] = 180 * kwargs[k]

        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.rotate_held_object,
            thor_action="RotateHand",
            preprocess_kwargs_inplace=rescale_xyz,
            x=x,
            y=y,
            z=z,
        )

    def drop_held_object(self) -> bool:
        """Drop the object in the agent's hand.

        The action is only successful agent is holding an object.
        """
        return execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.drop_held_object,
            thor_action="DropHandObject",
        )

    def drop_held_object_with_snap(self) -> bool:
        """Drop the object in the agent's hand to the target position.

        Exception is raised if shuffle has not yet been called on the current
        episode or the agent is in default mode.

        For this action to work:
            1. The agent must be within 1.5 meters from the goal object's
               position, observed during the walkthrough phase.
            2. The agent must be looking in the direction of where it was
               located in the walkthrough phase.

        Otherwise, the object will be placed in a visible receptacle or
        if this also fails, it will be simply dropped.

        # Returns

        `True` if the drop was successful, otherwise `False`.
        """
        if not self.shuffle_called:
            raise Exception("Must be in shuffle phaase.")
        if not self.mode == RearrangeMode.SNAP:
            raise Exception("Must be in RearrangeMode.SNAP mode.")

        # round positions to 2 decimals
        DEC = 2

        with include_object_data(self.controller):
            event = self.controller.last_event
            held_obj = self.held_object

            if held_obj is None:
                return False

            agent = event.metadata["agent"]
            goal_pose = self.obj_name_to_walkthrough_start_pose[held_obj["name"]]
            goal_pos = goal_pose["position"]
            goal_rot = goal_pose["rotation"]
            good_positions_to_drop_from = self._interactable_positions_cache.get(
                scene_name=self.last_event.metadata["sceneName"],
                obj={**held_obj, **{"position": goal_pos, "rotation": goal_rot},},
                controller=self.controller,
                force_cache_refresh=self.force_cache_reset,  # Forcing cache resets when not training.
            )

            def position_to_tuple(position: Dict[str, float]):
                return tuple(round(position[k], DEC) for k in ["x", "y", "z"])

            agent_xyz = position_to_tuple(agent["position"])
            agent_rot = (round(agent["rotation"]["y"] / 90) * 90) % 360
            agent_standing = int(agent["isStanding"])
            agent_horizon = round(agent["cameraHorizon"])

            for valid_agent_pos in good_positions_to_drop_from:
                # Checks if the agent is close enough to the target
                # for the object to be snapped to the target location.
                valid_xyz = position_to_tuple(valid_agent_pos)
                valid_rot = (round(valid_agent_pos["rotation"] / 90) * 90) % 360
                valid_standing = int(valid_agent_pos["standing"])
                valid_horizon = round(valid_agent_pos["horizon"])
                if (
                    valid_xyz == agent_xyz  # Position
                    and valid_rot == agent_rot  # Rotation
                    and valid_standing == agent_standing  # Standing
                    and round(valid_horizon) == agent_horizon  # Horizon
                ):
                    # Try a few locations near the target for robustness' sake
                    positions = [
                        {
                            "x": goal_pos["x"] + 0.001 * xoff,
                            "y": goal_pos["y"] + 0.001 * yoff,
                            "z": goal_pos["z"] + 0.001 * zoff,
                        }
                        for xoff in [0, -1, 1]
                        for zoff in [0, -1, 1]
                        for yoff in [0, 1, 2]
                    ]
                    self.controller.step(
                        action="TeleportObject",
                        objectId=held_obj["objectId"],
                        rotation=goal_rot,
                        positions=positions,
                        forceKinematic=True,
                        allowTeleportOutOfHand=True,
                        makeUnbreakable=True,
                    )
                    break

            if self.held_object is None:
                # If we aren't holding the object anymore, then let's check if it
                # was placed into the right location.
                if self.are_poses_equal(
                    goal_pose=get_pose_info(goal_pose),
                    cur_pose=next(
                        get_pose_info(o)
                        for o in self.last_event.metadata["objects"]
                        if o["name"] == goal_pose["name"]
                    ),
                    treat_broken_as_unequal=True,
                ):
                    return True
                else:
                    return False

            # We couldn't teleport the object to the target location, let's try placing it
            # in a visible receptacle.
            for possible_receptacle in event.metadata["objects"]:
                if possible_receptacle["visible"] and possible_receptacle["receptacle"]:
                    self.controller.step(
                        action="PlaceHeldObject",
                        objectId=possible_receptacle["objectId"],
                        rotation=goal_rot,
                        position=goal_pos,
                    )
                    if self.controller.last_event.metadata["lastActionSuccess"]:
                        break

            # We failed to place the object into a receptacle, let's just drop it.
            if not self.controller.last_event.metadata["lastActionSuccess"]:
                self.controller.step(
                    "DropHandObject", forceAction=True, autoSimulation=False
                )

            return False

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Return the AI2-THOR Event from the most recent controller action."""
        return self.controller.last_event

    @property
    def scene(self) -> str:
        """Return the current AI2-THOR scene name."""
        return self.controller.last_event.metadata["sceneName"].replace("_physics", "")

    @staticmethod
    def compare_poses(
        goal_pose: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        cur_pose: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Compare two object poses and return where they differ.

        The `goal_pose` must not have the object as broken.

        # Parameters
        goal_pose : The goal pose of the object.
        cur_pose : The current pose of the object.

        # Returns
        A dictionary with the following keys keys and values
        * "broken" - `True` if the `cur_pose` is broken in which case all below values are `None`, otherwise `False`.
        * "iou" - The IOU overlap between the two object poses (min==0, max==1) using their 3d bounding boxes. Computed
            using an approximate sampling procedure. If the `position_dist` (see below) is <0.01 and the `rotation_dist`
            is <10.0 then the IOU computation is short circuited and a value of 1 is returned.
        * "openness_diff" - `None` if the object types are not openable. Otherwise this equals the absolute difference
            between the `openness` values of the two poses.
        * "position_dist" - The euclidean distance between the positions of the center points of the two poses.
        * "rotation_dist" - The angle (in degrees) between the two poses. See the
            `IThorEnvironment.angle_between_rotations` function for more information.
        """
        if isinstance(goal_pose, Sequence):
            assert isinstance(cur_pose, Sequence)
            return [
                RearrangeTHOREnvironment.compare_poses(goal_pose=gp, cur_pose=cp)
                for gp, cp in zip(goal_pose, cur_pose)
            ]

        assert goal_pose["type"] == cur_pose["type"]
        assert not goal_pose["broken"]

        if cur_pose["broken"]:
            return {
                "broken": True,
                "iou": None,
                "openness_diff": None,
                "position_dist": None,
                "rotation_dist": None,
            }

        if goal_pose["bounding_box"] is None and cur_pose["bounding_box"] is None:
            iou = None
            position_dist = None
            rotation_dist = None
        else:
            position_dist = IThorEnvironment.position_dist(
                goal_pose["position"], cur_pose["position"]
            )
            rotation_dist = IThorEnvironment.angle_between_rotations(
                goal_pose["rotation"], cur_pose["rotation"]
            )
            if position_dist < 1e-2 and rotation_dist < 10.0:
                iou = 1.0
            else:
                try:
                    iou = iou_box_3d(
                        goal_pose["bounding_box"], cur_pose["bounding_box"]
                    )
                except Exception as _:
                    get_logger().warning(
                        "Could not compute IOU, will assume it was 0. Error during IOU computation:"
                        f"\n{traceback.format_exc()}"
                    )
                    iou = 0

        if goal_pose["openness"] is None and cur_pose["openness"] is None:
            openness_diff = None
        else:
            openness_diff = abs(goal_pose["openness"] - cur_pose["openness"])

        return {
            "broken": False,
            "iou": iou,
            "openness_diff": openness_diff,
            "position_dist": position_dist,
            "rotation_dist": rotation_dist,
        }

    @classmethod
    def pose_difference_energy(
        cls,
        goal_pose: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        cur_pose: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        min_iou: float = 0.5,
        open_tol: float = 0.2,
        pos_barrier: float = 2.0,
    ) -> Union[float, np.ndarray]:
        """Computes the energy between two poses.

        The energy (taking values in [0:1]) between two poses provides a soft and holistic measure of how
        far apart two poses are. If the energy is near 1 then the two poses are very dissimilar, if the energy
        is near 1 then the two poses are nearly equal.

        # Parameters
        goal_pose : The goal pose of the object.
        cur_pose : The current pose of the object.
        min_iou : As the IOU between the two poses increases between [0:min_iou] the contribution to the energy
            corresponding solely to the to the IOU decrease from 0.5 to 0 in a linear fashion.
        open_tol: If the object is openable, then if the absolute openness difference is less than `open_tol`
            the energy is 0. Otherwise the pose energy is 1.
        pos_barrier: If two poses are separated by a large distance, we would like to decrease the energy as
            the two poses are brought closer together. The `pos_barrier` controls when this energy decrease begins,
            namely at its default value of 2.0, the contribution of the distance to
             the energy decreases linearly from 0.5 to 0 as the distance between the two poses decreases from
             2 meters to 0 meters.
        """
        if isinstance(goal_pose, Sequence):
            assert isinstance(cur_pose, Sequence)
            return np.array(
                [
                    cls.pose_difference_energy(
                        goal_pose=p0,
                        cur_pose=p1,
                        min_iou=min_iou,
                        open_tol=open_tol,
                        pos_barrier=pos_barrier,
                    )
                    for p0, p1 in zip(goal_pose, cur_pose)
                ]
            )
        assert not goal_pose["broken"]

        pose_diff = cls.compare_poses(goal_pose=goal_pose, cur_pose=cur_pose)
        if pose_diff["broken"]:
            return 1.0

        if pose_diff["openness_diff"] is None:
            gbb = np.array(goal_pose["bounding_box"])
            cbb = np.array(cur_pose["bounding_box"])

            iou = pose_diff["iou"]
            iou_energy = max(1 - iou / min_iou, 0)

            if iou > 0:
                position_dist_energy = 0.0
            else:
                min_pairwise_dist_between_corners = np.sqrt(
                    (
                        (
                            np.tile(gbb, (1, 8)).reshape(-1, 3)
                            - np.tile(cbb, (8, 1)).reshape(-1, 3)
                        )
                        ** 2
                    ).sum(1)
                ).min()
                position_dist_energy = min(
                    min_pairwise_dist_between_corners / pos_barrier, 1.0
                )

            return 0.5 * iou_energy + 0.5 * position_dist_energy

        else:
            return 1.0 * (pose_diff["openness_diff"] > open_tol)

    @classmethod
    def are_poses_equal(
        cls,
        goal_pose: Dict[str, Any],
        cur_pose: Dict[str, Any],
        min_iou: float = 0.5,
        open_tol: float = 0.2,
        treat_broken_as_unequal: bool = False,
    ) -> bool:
        """Determine if two object poses are equal (up to allowed error).

        The `goal_pose` must not have the object as broken.

        # Parameters
        goal_pose : The goal pose of the object.
        cur_pose : The current pose of the object.
        min_iou : If the two objects are pickupable objects, they are considered equal if their IOU is `>=min_iou`.
        open_tol: If the object is openable and not pickupable, then the poses are considered equal if the absolute
            openness difference is less than `open_tol`.
        treat_broken_as_unequal : If `False` an exception will be thrown if the `cur_pose` is broken. If `True`, then
             if `cur_pose` is broken this function will always return `False`.
        """
        assert not goal_pose["broken"]

        if cur_pose["broken"]:
            if treat_broken_as_unequal:
                return False
            else:
                raise RuntimeError(
                    f"Cannot determine if poses of two objects are"
                    f" equal if one is broken object ({goal_pose} v.s. {cur_pose})."
                )

        pose_diff = cls.compare_poses(goal_pose=goal_pose, cur_pose=cur_pose)

        return (pose_diff["iou"] is None or pose_diff["iou"] > min_iou) and (
            pose_diff["openness_diff"] is None or pose_diff["openness_diff"] <= open_tol
        )

    @property
    def all_rearranged_or_broken(self):
        """Return if every object is simultaneously broken or in its correct
        pose.

        The unshuffle agent can make no more progress on its task in the
        case that that every object is either (1) in its correct
        position or (2) broken so that it can never be placed in its
        correct position. This function simply returns whether this is
        the case.
        """
        return all(
            cp["broken"] or self.are_poses_equal(goal_pose=gp, cur_pose=cp)
            for _, gp, cp in zip(*self.poses)
        )

    @property
    def poses(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (unshuffle start, walkthrough start, current) pose for every
        object in the scene.

        Can only be called during the unshuffle phase.

        # Returns
        A Tuple of containing three ordered lists of object poses `(unshuffle_start_poses, walkthrough_start_poses, current_poses)`
        such that, for `0 <= i < len(current_poses)`,
        * `unshuffle_start_poses[i]` - corresponds to the pose of the ith object at the start of the unshuffle phase.
        * `walkthrough_start_poses[i]` - corresponds to the pose of the ith object at the start of the walkthrough phase.
        * `current_poses[i]` - corresponds to the pose of the ith object in the current environment.
        During the unshuffle phase is commonly useful to compare `current_poses[i]` against `walkthrough_start_poses[i]`
        to get a sense of the agent's progress towards placing the objects into their correct locations.
        """
        # Ensure we are in the unshuffle phase.
        if not self.shuffle_called:
            raise Exception("shuffle() must be called before accessing poses")

        # Get current object information
        with include_object_data(self.controller):
            obj_name_to_current_obj = self._obj_list_to_obj_name_to_pose_dict(
                self.controller.last_event.metadata["objects"]
            )

        ordered_obj_names = list(self.obj_name_to_walkthrough_start_pose.keys())

        current_objs_list = []
        for obj_name in ordered_obj_names:
            if obj_name not in obj_name_to_current_obj:
                # obj_name_to_predicted_obj can have more objects than goal objects
                # (breaking objects can generate new ones)
                # The other way (more goal poses than predicted objs) is a problem, we will
                # assume that the disappeared objects are broken
                if not self._have_warned_about_mismatch:
                    # Don't want to warn many many times during single episode
                    self._have_warned_about_mismatch = True
                    usos = set(self.obj_name_to_unshuffle_start_pose.keys())
                    wsos = set(self.obj_name_to_walkthrough_start_pose.keys())
                    cos = set(obj_name_to_current_obj.keys())
                    get_logger().warning(
                        f"Mismatch between walkthrough start, unshuffle start, and current pose objects."
                        f"\nscene = {self.scene}, index {self.current_task_spec.metrics.get('index')}"
                        f"\nusos-wsos, wsos-usos = {usos - wsos}, {wsos - usos}"
                        f"\ncos-usos, usos-cos = {cos - usos}, {usos - cos}"
                        f"\ncos-wsos, wsos-cos = {cos - wsos}, {wsos - cos}"
                    )
                obj_name_to_current_obj[obj_name] = {
                    **self.obj_name_to_walkthrough_start_pose[obj_name],
                    "isBroken": True,
                    "broken": True,
                    "position": None,
                    "rotation": None,
                    "openness": None,
                }
            current_objs_list.append(obj_name_to_current_obj[obj_name])

        # We build a cache of object poses corresponding to the start of the walkthrough/unshuffle phases
        # as these remain the same until the `reset` function is called.
        if self._sorted_and_extracted_walkthrough_start_poses is None:
            broken_obj_names = [
                obj_name
                for obj_name in ordered_obj_names
                if self.obj_name_to_walkthrough_start_pose[obj_name]["isBroken"]
            ]
            if len(broken_obj_names) != 0:
                if not self.current_task_spec.runtime_sample:
                    # Don't worry about reporting broken objects when using
                    # a "runtime_sample" task spec as these types of things are
                    # more common.
                    get_logger().warning(
                        f"BROKEN GOAL OBJECTS!"
                        f"\nIn scene {self.scene}"
                        f"\ntask spec {self.current_task_spec}"
                        f"\nbroken objects {broken_obj_names}"
                    )

                # If we find a broken goal object, we will simply pretend as though it was not
                # broken. This means the agent can never succeed in unshuffling, this means it is
                # possible that even a perfect agent will not succeed for some tasks.
                for broken_obj_name in broken_obj_names:
                    self.obj_name_to_walkthrough_start_pose[broken_obj_name][
                        "isBroken"
                    ] = False
                    self.obj_name_to_unshuffle_start_pose[broken_obj_name][
                        "isBroken"
                    ] = False
                ordered_obj_names = list(self.obj_name_to_walkthrough_start_pose.keys())

            walkthrough_start_poses = tuple(
                self.obj_name_to_walkthrough_start_pose[k] for k in ordered_obj_names
            )
            unshuffle_start_poses = tuple(
                self.obj_name_to_unshuffle_start_pose[k] for k in ordered_obj_names
            )
            self._sorted_and_extracted_unshuffle_start_poses = get_pose_info(
                unshuffle_start_poses
            )
            self._sorted_and_extracted_walkthrough_start_poses = get_pose_info(
                walkthrough_start_poses
            )

        return (
            self._sorted_and_extracted_unshuffle_start_poses,
            self._sorted_and_extracted_walkthrough_start_poses,
            get_pose_info(current_objs_list),
        )

    def _runtime_reset(
        self, task_spec: RearrangeTaskSpec, force_axis_aligned_start: bool
    ):
        """Randomly initialize a scene at runtime.

        Rather than using a predefined collection of object states,
        randomly generate these positions at runtime. This may be useful for obtaining more
        diverse training examples.

        # Parameters
        task_spec : The RearrangeTaskSpec for this runtime sample. `task_spec.runtime_sample` should be `True`.
        force_axis_aligned_start : If `True`, this will force the agent's start rotation to be 'axis aligned', i.e.
            to equal to 0, 90, 180, or 270 degrees.
        """
        assert (
            task_spec.runtime_sample
        ), "Attempted to use a runtime reset with a task spec which has a `False` `runtime_sample` property."

        # For efficiency reasons, we do not completely reset the ai2thor scene (which
        # will reset all object states to a default configuration and restore broken
        # objects to their unbroken state) on every call to `_runtime_reset` if the scene name hasn't changed. Instead
        # we reset the ai2thor scene only every 25 calls.
        if (
            task_spec.scene != self.scene
            or self.current_task_spec.runtime_data["count"] >= 25
        ):
            count = 1
            self.controller.reset(task_spec.scene)
            remove_objects_until_all_have_identical_meshes(self.controller)
            self.controller.step(
                "InitialRandomSpawn", forceVisible=True, placeStationary=True,
            )
            md = self.controller.step("GetReachablePositions").metadata
            assert md["lastActionSuccess"]
            reachable_positions = md["actionReturn"]
        else:
            count = 1 + self.current_task_spec.runtime_data["count"]
            reachable_positions = self.current_task_spec.runtime_data[
                "reachable_positions"
            ]

        self.current_task_spec = task_spec
        self.current_task_spec.stage = "Unknown"
        self.current_task_spec.runtime_data = {
            "count": count,
            "reachable_positions": reachable_positions,
        }

        with include_object_data(self.controller):
            random.shuffle(reachable_positions)

            # set agent position
            max_teleports = min(10, len(reachable_positions))
            for teleport_count, pos in enumerate(reachable_positions):
                rot = 30 * random.randint(0, 11)
                if force_axis_aligned_start:
                    rot = round_to_factor(30 * random.randint(0, 11), 90)
                md = self.controller.step(
                    "TeleportFull",
                    rotation={"x": 0, "y": rot, "z": 0},
                    forceAction=teleport_count == max_teleports - 1,
                    **pos,
                ).metadata
                if md["lastActionSuccess"]:
                    break
            else:
                raise RuntimeError("No reachable positions?")

            assert md["lastActionSuccess"]
            self.current_task_spec.agent_position = pos
            self.current_task_spec.agent_rotation = rot
            self.current_task_spec.runtime_data["starting_objects"] = md["objects"]

    def _task_spec_reset(
        self, task_spec: RearrangeTaskSpec, force_axis_aligned_start: bool
    ):
        """Initialize a ai2thor environment from a (non-runtime sample) task
        specification (i.e. an exhaustive collection of object poses for the
        walkthrough and unshuffle phase).

        After this call, the environment will be ready for use in the walkthrough phase.

        # Parameters
        task_spec : The RearrangeTaskSpec for this task. `task_spec.runtime_sample` should be `False`.
        force_axis_aligned_start : If `True`, this will force the agent's start rotation to be 'axis aligned', i.e.
            to equal to 0, 90, 180, or 270 degrees.
        """
        assert (
            not task_spec.runtime_sample
        ), "`_task_spec_reset` requires that `task_spec.runtime_sample` is `False`."

        self.current_task_spec = task_spec

        self.controller.reset(self.current_task_spec.scene)

        if force_axis_aligned_start:
            self.current_task_spec.agent_rotation = round_to_factor(
                self.current_task_spec.agent_rotation, 90
            )

        # set agent position
        pos = self.current_task_spec.agent_position
        rot = {"x": 0, "y": self.current_task_spec.agent_rotation, "z": 0}
        self.controller.step("TeleportFull", rotation=rot, **pos, forceAction=True)

        # show object metadata
        with include_object_data(self.controller):
            # open objects
            for obj in self.current_task_spec.openable_data:
                # id is re-found due to possible floating point errors
                current_obj_info = next(
                    l_obj
                    for l_obj in self.last_event.metadata["objects"]
                    if l_obj["name"] == obj["name"]
                )
                self.controller.step(
                    action="OpenObject",
                    objectId=current_obj_info["objectId"],
                    moveMagnitude=obj["target_openness"],
                    forceAction=True,
                )

            # arrange walkthrough poses for pickupable objects
            self.controller.step(
                "SetObjectPoses", objectPoses=self.current_task_spec.target_poses
            )

    def reset(
        self, task_spec: RearrangeTaskSpec, force_axis_aligned_start: bool = False,
    ) -> None:
        """Reset the environment with respect to the new task specification.

         The environment will start in the walkthrough phase.

        # Parameters
        task_spec : The `RearrangeTaskSpec` defining environment state.
        force_axis_aligned_start : If `True`, this will force the agent's start rotation to be 'axis aligned', i.e.
            to equal to 0, 90, 180, or 270 degrees.
        """
        if task_spec.runtime_sample:
            self._runtime_reset(
                task_spec=task_spec, force_axis_aligned_start=force_axis_aligned_start
            )
        else:
            self._task_spec_reset(
                task_spec=task_spec, force_axis_aligned_start=force_axis_aligned_start,
            )

        self.shuffle_called = False
        self.obj_name_to_walkthrough_start_pose = self._obj_list_to_obj_name_to_pose_dict(
            self.last_event.metadata["objects"]
        )

        self._have_warned_about_mismatch = False
        self._sorted_and_extracted_walkthrough_start_poses = None
        self._sorted_and_extracted_unshuffle_start_poses = None
        self._agent_signals_done = False

    def _runtime_shuffle(self):
        """Randomly shuffle objects in the environment to start the unshuffle
        phase.

        Also resets the agent's position to its start position.
        """
        assert (not self.shuffle_called) and self.current_task_spec.runtime_sample

        task_spec = self.current_task_spec

        # set agent position
        pos = task_spec.agent_position
        rot = {"x": 0, "y": task_spec.agent_rotation, "z": 0}
        self.controller.step("TeleportFull", rotation=rot, **pos, forceAction=True)

        # Randomly shuffle a subset of objects.
        nobjects_to_move = random.randint(1, 5)
        pickupable = [
            o for o in task_spec.runtime_data["starting_objects"] if o["pickupable"]
        ]
        random.shuffle(pickupable)

        pickupable.sort(
            key=lambda x: 1 * (x["objectType"] in OBJECT_TYPES_TO_NOT_MOVE),
            reverse=True,
        )
        objects_to_not_move = pickupable[:-nobjects_to_move]

        object_ids_not_to_move = [o["objectId"] for o in objects_to_not_move]
        object_ids_not_to_move.extend(
            get_object_ids_to_not_move_from_object_types(
                controller=self.controller, object_types=OBJECT_TYPES_TO_NOT_MOVE,
            )
        )
        self.controller.step(
            "InitialRandomSpawn",
            excludedObjectIds=object_ids_not_to_move,
            forceVisible=True,
            placeStationary=True,
        )

        # Randomly open some subset of objects.
        num_objects_to_open = random.randint(0, 1)
        openable_objects = [
            o
            for o in self.last_event.metadata["objects"]
            if o["openable"] and not o["pickupable"]
        ]
        random.shuffle(openable_objects)
        open_objs(
            objects_to_open=openable_objects[:num_objects_to_open],
            controller=self.controller,
        )

        self.current_task_spec.runtime_data[
            "target_objects"
        ] = self.last_event.metadata["objects"]

    def _task_spec_shuffle(self, reset: bool = False):
        """Shuffle objects in the environment to start the unshuffle phase
        using the current task specification.

        Also resets the agent's position to its start position.
        """
        assert not (self.current_task_spec.runtime_sample or self.shuffle_called)

        task_spec = self.current_task_spec

        # TODO: No need to reset every time right?
        if reset:
            self.controller.reset(self.scene)

        # set agent position
        pos = task_spec.agent_position
        rot = {"x": 0, "y": task_spec.agent_rotation, "z": 0}
        self.controller.step("TeleportFull", rotation=rot, **pos, forceAction=True)

        # open objects
        with include_object_data(self.controller):
            for obj in task_spec.openable_data:
                # id is re-found due to possible floating point errors
                current_obj_info = next(
                    l_obj
                    for l_obj in self.last_event.metadata["objects"]
                    if l_obj["name"] == obj["name"]
                )

                self.controller.step(
                    action="OpenObject",
                    objectId=current_obj_info["objectId"],
                    moveMagnitude=obj["start_openness"],
                    forceAction=True,
                )

        # arrange unshuffle start poses for pickupable objects
        self.controller.step("SetObjectPoses", objectPoses=task_spec.starting_poses)

    def shuffle(self, require_reset: bool = False):
        """Shuffle objects in the environment to start the unshuffle phase."""

        assert not self.shuffle_called

        runtime_sample = self.current_task_spec.runtime_sample
        if runtime_sample:
            self._runtime_shuffle()
        else:
            self._task_spec_shuffle(reset=require_reset)

        # Save object metadata
        with include_object_data(self.controller):
            self.obj_name_to_unshuffle_start_pose = self._obj_list_to_obj_name_to_pose_dict(
                self.last_event.metadata["objects"]
            )

            if len(self.obj_name_to_unshuffle_start_pose) != len(
                self.obj_name_to_walkthrough_start_pose
            ):
                if runtime_sample or require_reset:
                    walkthrough_start_obj_names = set(
                        self.obj_name_to_walkthrough_start_pose.keys()
                    )
                    unshuffle_start_obj_names = set(
                        self.obj_name_to_unshuffle_start_pose.keys()
                    )
                    raise PoseMismatchError(
                        "Irrecoverable difference between walkthrough and unshuffle phase objects."
                        f"\ng-i, i-g = {walkthrough_start_obj_names - unshuffle_start_obj_names},"
                        f" {unshuffle_start_obj_names - walkthrough_start_obj_names}"
                    )
                else:
                    self.shuffle(require_reset=True)

        self.shuffle_called = True
        self._agent_signals_done = False

    @staticmethod
    def _obj_list_to_obj_name_to_pose_dict(
        objects: List[Dict[str, Any]]
    ) -> OrderedDict:
        """Helper function to transform a list of object data dicts into a
        dictionary."""
        objects = [
            o
            for o in objects
            if o["openable"] or o.get("objectOrientedBoundingBox") is not None
        ]
        d = OrderedDict(
            (o["name"], o) for o in sorted(objects, key=lambda x: x["name"])
        )
        assert len(d) == len(objects)
        return d

    def stop(self):
        """Terminate the current AI2-THOR session."""
        try:
            self.controller.stop()
        except Exception as _:
            pass

    def __del__(self):
        self.stop()
