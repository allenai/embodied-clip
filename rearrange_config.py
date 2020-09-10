# TODO: Implement evaluations and rotation checks

import ai2thor.controller
from collections import defaultdict
from typing import Dict, Callable, Tuple, Any, Union, List
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import random
import json
import os
import logging

REQUIRED_VERSION = '2.4.12'
DATA_DIR = './data'
MODE = 'default'  # TODO: remove added for visualization debugging
ROTATE_STEP_DEGREES = 30
MAX_HAND_METERS = 0.5
logging.basicConfig(level=logging.INFO)


class BoundedFloat:
    """Declare a bounded float placeholder variable."""

    def __init__(self, low: float, high: float):
        """High is the max float value, low is the min (both inclusive)."""
        self.types = {float, int, np.float64}
        if type(low) not in self.types or type(high) not in self.types:
            raise ValueError('Bounds must both be floats.')
        if low > high:
            raise ValueError('low must be less than high.')
        self.low = low
        self.high = high

    def sample(self) -> float:
        """Return a random float within the initialized range."""
        return random.random() * (self.high - self.low) + self.low

    def __contains__(self, n: float):
        """Assert n is within this classes bounded range."""
        if type(n) not in self.types:
            raise ValueError('n must be a float (or an int).')
        return n >= self.low and n <= self.high


class ActionSpace:
    """Control which actions with bounded variables can be executed."""

    def __init__(
            self,
            actions: Dict[Callable, Dict[str, BoundedFloat]]):
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

    def execute_random_action(self, log_choice: bool = True) -> bool:
        """Execute a random action within the specified action space.

        Return (bool) if the action is executed successfully.
        """
        action = random.choice(self.keys)
        kwargs = {
            name: bounds.sample()
            for name, bounds in self.actions[action].items()}

        # logging
        if log_choice:
            kwargs_str = str(''.join(
                f'  {k}: {v},\n' for k, v in kwargs.items()))
            kwargs_str = '\n' + kwargs_str[:-2] if kwargs_str else ''
            logging.info(f'Executing {action.__name__}(' + kwargs_str + ')')

        return action(**kwargs)

    def __contains__(self, action: Tuple[Callable, Dict[str, float]]) -> bool:
        """Return if action_fn with variables is valid in this ActionSpace."""
        action_fn, variables = action

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
        s = ''
        tab = ' ' * 2  # default tabs have like 8 spaces on shells
        for action_fn, vars in self.actions.items():
            fn_name = action_fn.__name__
            vstr = ''
            for i, (var_name, bound) in enumerate(vars.items()):
                low = bound.low
                high = bound.high
                vstr += f'{tab * 2}{var_name}: float(low={low}, high={high})'
                vstr += '\n' if i+1 == len(vars) else ',\n'
            vstr = '\n' + vstr[:-1] if vstr else ''
            s += f'{tab}{fn_name}({vstr}),\n'
        s = s[:-2] if s else ''
        return 'ActionSpace(\n' + s + '\n)'


class Helpers:
    """Static helper functions for the rearrange controller.

    These methods are NOT intended to be used by users. This helper class is
    used so that these methods would not show upon autocompletion within the
    Controller.
    """

    @staticmethod
    def l2_distance(obj1, obj2):
        """Calculate the L2 distance between object 1 and object 2."""
        p1 = obj1['position']
        p2 = obj2['position']

        a = np.array([p1['x'], p1['y'], p1['z']])
        b = np.array([p2['x'], p2['y'], p2['z']])
        return np.linalg.norm(a - b)

    @staticmethod
    def extract_obj_data(obj):
        """Return object evaluation metrics based on the env state."""
        if 'type' in obj:
            return {
                'type': obj['type'],
                'position': obj['position'],
                'rotation': obj['rotation'],
                'openness': obj['openness'],
                'broken': obj['broken'],
                'bounding_box': obj['bounding_box']
            }
        return {
            'type': obj['objectType'],
            'position': obj['position'],
            'rotation': obj['rotation'],
            'openness': obj['openPercent'] if obj['openable'] else None,
            'broken': obj['isBroken'],
            'bounding_box':
                obj['objectOrientedBoundingBox']['cornerPoints'] if
                obj['objectOrientedBoundingBox'] else None
        }

    @staticmethod
    def get_pose_info(
            objs: Union[List[Dict[str, Any]], Dict[str, Any]]
            ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Return data about each specified object.

        For each object, the return consists of its type, position, rotation,
        openness, and bounding box.
        """
        # list of objects
        if type(objs) is list:
            return [Helpers.extract_obj_data(obj) for obj in objs]
        # single object
        return Helpers.extract_obj_data(objs)

    @staticmethod
    def execute_action(
            controller: ai2thor.controller.Controller,
            action_space: ActionSpace,
            action_fn: Callable,
            thor_action: str,
            error_message: str = '',
            updated_kwarg_names: Dict[str, str] = dict(),
            default_thor_kwargs: Dict[str, Any] = dict(),
            **kwargs:  float) -> bool:
        """Execute a bounded action within the AI2-THOR controller."""
        action = action_fn, kwargs
        if action not in action_space:
            raise ValueError(error_message)

        if 'force_magnitude' in kwargs:
            # rescale for newtons
            kwargs['force_magnitude'] *= 50

        # get rid of bad variable names
        for better_kwarg, thor_kwarg in updated_kwarg_names.items():
            kwargs[thor_kwarg] = kwargs[better_kwarg]
            del kwargs[better_kwarg]

        for name, value in default_thor_kwargs.items():
            kwargs[name] = value

        event = controller.step(thor_action, **kwargs)
        return event.metadata['lastActionSuccess']

    @staticmethod
    def iou(b1: np.ndarray, b2: np.ndarray, num_points: int = 2197):
        """Calculate the IoU between 3d bounding boxes b1 and b2."""
        def _outer_bounds(points_1: np.ndarray,
                          points_2: np.ndarray) -> Dict[str, Dict[str, float]]:
            """Sample points from the outer bounds formed by points_1/2."""
            assert points_1.shape == points_2.shape
            bounds = dict()
            for i in range(len(points_1)):
                x1, y1, z1 = points_1[i]
                x2, y2, z2 = points_2[i]
                points = [(x1, 'x'), (x2, 'x'),
                        (y1, 'y'), (y2, 'y'),
                        (z1, 'z'), (z2, 'z')]
                for val, d_key in points:
                    if d_key not in bounds:
                        bounds[d_key] = {'min': val, 'max': val}
                    else:
                        if val > bounds[d_key]['max']:
                            bounds[d_key]['max'] = val
                        elif val < bounds[d_key]['min']:
                            bounds[d_key]['min'] = val
            return bounds

        def _in_box(box: np.ndarray, points: np.ndarray) -> np.ndarray:
            """For each point, return if its in the hull."""
            hull = ConvexHull(box)
            deln = Delaunay(box[hull.vertices])
            return deln.find_simplex(points) >= 0

        bounds = _outer_bounds(b1, b2)
        dim_points = int(num_points ** (1 / 3))

        xs = np.linspace(bounds['x']['min'], bounds['x']['max'], dim_points)
        ys = np.linspace(bounds['y']['min'], bounds['y']['max'], dim_points)
        zs = np.linspace(bounds['z']['min'], bounds['z']['max'], dim_points)
        points = np.array(
            [[x, y, z] for x in xs for y in ys for z in zs], copy=False)

        in_b1 = _in_box(b1, points)
        in_b2 = _in_box(b2, points)

        intersection = np.count_nonzero(in_b1 * in_b2)
        union = np.count_nonzero(in_b1 + in_b2)
        return intersection / union


class Controller:
    """Custom AI2-THOR Controller for the task of object unshuffling."""

    def __init__(
            self,
            stage: str,
            camera_pixel_width: int = 300,
            camera_pixel_height: int = 300):
        """Initialize a new rearrangement controller.

        -----
        Attributes
        :stage (str) = {'train', 'val'}.
        :camera_pixel_width (int) width of the images from the agent.
        :camera_pixel_height (int) height of the images from the agent.
        """
        if ai2thor.__version__ != REQUIRED_VERSION:
            raise ValueError(f'Please use AI2-THOR v{REQUIRED_VERSION}')

        stage = stage.lower()
        if stage not in {'train', 'val'}:
            raise ValueError("Stage must be either 'train' or 'val'.")

        if stage == 'train':
            data_path = os.path.join(DATA_DIR, 'train.json')
            # eval_data_path = os.path.join(DATA_DIR, 'evaluation', 'train.json')
        elif stage == 'val':
            data_path = os.path.join(DATA_DIR, 'val.json')
            # eval_data_path = os.path.join(DATA_DIR, 'evaluation', 'val.json')

        with open(data_path, 'r') as f:
            self.data = json.loads(f.read())

        # with open(eval_data_path, 'r') as f:
            # self.eval_data = json.loads(f.read())

        self.scenes = list(self.data.keys())
        if not self.scenes:
            raise ValueError('The scenes are not listed as keys in the json!')
        self.current_scene_idx = -1

        # assumes the same number of rearrangements per scene
        self.current_rearrangement = -1
        self.max_rearrangements = len(list(self.data.values())[0])

        # local thor controller to execute all the actions
        self.controller = ai2thor.controller.Controller(
            rotateStepDegrees=ROTATE_STEP_DEGREES,
            width=camera_pixel_width,
            height=camera_pixel_height)

        # always begin in walkthrough phase
        self.shuffle_called = False

        # for identical pickupable objects, the predicted objects are
        # re-assigned so as to minimize the distance between the predicted
        # object position and the target object position.
        self.identical_objects = {
            'FloorPlan18': {'Vase_42af4a87', 'Vase_bb6f5e2d', 'Vase_6d83d1f7'},
            'FloorPlan21': {
                'AluminumFoil_6f4f0160',
                'AluminumFoil_1236c130',
                'AluminumFoil_bc24315a'},
            'FloorPlan27': {'Ladle_16843601', 'Ladle_fe637120'},
            'FloorPlan227': {'Vase_08aa17fa', 'Vase_9c18c071'},
            'FloorPlan303': {'CD_123090e1', 'CD_b05dba56'},
            'FloorPlan307': {'CD_18d12874', 'CD_c3f7bfe8', 'CD_96b38036'},
            'FloorPlan320': {'Dumbbell_d20301f4', 'Dumbbell_f8b83754'}
        }

        # sets up the starting walkthrough
        self.reset()

    @property
    def action_space(self) -> ActionSpace:
        """Return the ActionSpace based on the current stage."""
        # walkthrough actions
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
            self.done: {}
        }

        if self.shuffle_called:
            # shuffle allowed actions
            actions.update({
                self.open_object: {
                    'x': BoundedFloat(low=0, high=1),
                    'y': BoundedFloat(low=0, high=1),
                    'openness': BoundedFloat(low=0, high=1)
                },
                self.pickup_object: {
                    'x': BoundedFloat(low=0, high=1),
                    'y': BoundedFloat(low=0, high=1),
                },
                self.push_object: {
                    'x': BoundedFloat(low=0, high=1),
                    'y': BoundedFloat(low=0, high=1),
                    'rel_x_force': BoundedFloat(low=-0.5, high=0.5),
                    'rel_y_force': BoundedFloat(low=-0.5, high=0.5),
                    'rel_z_force': BoundedFloat(low=-0.5, high=0.5),
                    'force_magnitude': BoundedFloat(low=0, high=1)
                },
                self.move_held_object: {
                    'x_meters': BoundedFloat(low=-0.5, high=0.5),
                    'y_meters': BoundedFloat(low=-0.5, high=0.5),
                    'z_meters': BoundedFloat(low=-0.5, high=0.5),
                },
                self.rotate_held_object: {
                    'x_degrees': BoundedFloat(low=-0.5, high=0.5),
                    'y_degrees': BoundedFloat(low=-0.5, high=0.5),
                    'z_degrees': BoundedFloat(low=-0.5, high=0.5),
                },
                self.drop_held_object: {}
            })

        return ActionSpace(actions)

    def open_object(self, x: float, y: float, openness: float) -> bool:
        """Open the object corresponding to x/y to openess.

        -----
        Attributes
        :x (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        :y (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        ---
        Return (bool) if the action is successful. The action will not
        be successful if the specified openness would cause a collision
        or if the object at x/y is not openable.
        """
        # openness = 0 actually means fully open under the hood! -- weird...
        openness = openness + 0.001 if openness == 0 else openness
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.open_object,
            thor_action='OpenObject',
            error_message=(
                'x/y/openness must be in [0:1] and in unshuffle phase.'),
            updated_kwarg_names={'openness': 'moveMagnitude'},
            x=x, y=y, openness=openness)

    def pickup_object(self, x: float, y: float) -> bool:
        """Pick up the object corresponding to x/y.

        -----
        Attributes
        :x (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        :y (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        ---
        Return (bool) if the action is successful. The action will not be
        successful if the object at x/y is not pickupable.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.pickup_object,
            thor_action='PickupObject',
            error_message='x/y must be in [0:1] and in unshuffle phase.',
            x=x, y=y)

    def push_object(
            self,
            x: float,
            y: float,
            rel_x_force: float,
            rel_y_force: float,
            rel_z_force: float,
            force_magnitude: float) -> bool:
        """Push an object along a surface.

        -----
        Attributes
        :x (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        :y (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.
        :rel_x_force (float, min=-0.5, max=0.5) amount of relative force
           applied along the x axis.
        :rel_y_force (float, min=-0.5, max=0.5) amount of relative force
           applied along the y axis.
        :rel_z_force (float, min=-0.5, max=0.5) amount of relative force
           applied along the z axis.
        :force_magnitude (float, min=0, max=1) relative amount of force
           applied during this push action. Within AI2-THOR, the force is
           rescaled to be between 0 and 50 newtons, which is estimated to
           sufficiently move all pickupable objects.

        ---
        Return (bool) if the action is successful. The action will not be
        successful if the object at x/y is not pickupable.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.push_object,
            thor_action='TouchThenApplyForce',
            error_message=(
                'x/y must be in [0:1] and in unshuffle phase.\n' +
                'rel_{x/y/z}_force must be in [-0.5:0.5].\n' +
                'force_magnitude must be in [0:1].'),
            default_thor_kwargs={'handDistance': 1.5},
            x=x, y=y, rel_x_force=rel_x_force, rel_y_force=rel_y_force,
            rel_z_force=rel_z_force, force_magnitude=force_magnitude)

    def move_ahead(self) -> bool:
        """Move the agent ahead from its facing direction by 0.25 meters.

        Return (bool) if the last action was successful.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_ahead,
            thor_action='MoveAhead')

    def move_back(self) -> bool:
        """Move the agent back from its facing direction by 0.25 meters.

        Return (bool) if the last action was successful.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_back,
            thor_action='MoveBack')

    def move_right(self) -> bool:
        """Move the agent right from its facing direction by 0.25 meters.

        Return (bool) if the last action was successful.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_right,
            thor_action='MoveRight')

    def move_left(self) -> bool:
        """Move the agent left from its facing direction by 0.25 meters.

        Return (bool) if the last action was successful.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_left,
            thor_action='MoveLeft')

    def rotate_left(self) -> bool:
        """Rotate the agent left from its facing direction by 30 degrees.

        Return (bool) if the last action was successful.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.rotate_left,
            thor_action='RotateLeft')

    def rotate_right(self) -> bool:
        """Rotate the agent left from its facing direction by 30 degrees.

        Return (bool) if the last action was successful.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.rotate_right,
            thor_action='RotateRight')

    def stand(self) -> bool:
        """Stand the agent from the crouching position.

        Return (bool) if the last action was successful.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.stand,
            thor_action='Stand')

    def crouch(self) -> bool:
        """Crouch the agent from the standing position.

        Return (bool) if the last action was successful.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.crouch,
            thor_action='Crouch')

    def look_up(self) -> bool:
        """Turn the agent's head and camera up by 30 degrees.

        Return (bool) if the last action was successful.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.look_up,
            thor_action='LookUp')

    def look_down(self) -> bool:
        """Turn the agent's head and camera down by 30 degrees.

        Return (bool) if the last action was successful.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.look_down,
            thor_action='LookDown')

    def done(self) -> bool:
        """Agent's signal that it's completed its current rearrangement phase.

        Note that we do not automatically switch from the walkthrough phase
        to the unshuffling phase, and vice-versa, that is up to the user.
        This allows users to call .poses after the agent calls done, and
        have it correspond to the current episode.

        Return (bool) if the action is successful. In this case, it will always
        be True.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.done,
            thor_action='Done')

    def move_held_object(
            self,
            x_meters: float,
            y_meters: float,
            z_meters: float) -> bool:
        """Move the object in the agent's hand by the specified amount.

        -----
        Clipping
        :movement magnitude too large. The maximum magnitude that the object
        can move in one time step is 0.5 meters. If the calculated magnitude is
        above 0.5, it's magnitude will be clipped to 0.5.

        -----
        Attribues
        :x_meters (float, min=-0.5, max=0.5) movement meters along the x-axis.
        :y_meters (float, min=-0.5, max=0.5) movement meters along the y-axis.
        :z_meters (float, min=-0.5, max=0.5) movement meters along the z-axis.

        -----
        Exceptions
        :in walkthrough phase. This method can only be called within the
        unshuffle phase. The shuffle phase starts with controller.shuffle()
        and ends with controller.reset().

        -----
        Return (bool) if the action is successful, which is True in the case
        that the agent is holding an object and moving the object by the
        specified amount does not bump into an object.
        """
        mag = (x_meters ** 2 + y_meters ** 2 + z_meters ** 2) ** (0.5)

        # clips the max value at MAX_HAND_METERS.
        if MAX_HAND_METERS > mag:
            x_meters /= (mag / MAX_HAND_METERS)
            y_meters /= (mag / MAX_HAND_METERS)
            z_meters /= (mag / MAX_HAND_METERS)

        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.move_held_object,
            thor_action='MoveHandDelta',
            updated_kwarg_names={
                'x_meters': 'x', 'y_meters': 'y', 'z_meters': 'z'},
            x_meters=x_meters, y_meters=y_meters, z_meters=z_meters)

    def rotate_held_object(
            self,
            x_degrees: float,
            y_degrees: float,
            z_degrees: float) -> bool:
        """Rotate the object in the agent's hand by the specified degrees.

        -----
        Attribues
        :x_degrees (float, min=-90, max=90) rotation degrees along the x-axis.
        :y_degrees (float, min=-90, max=90) rotation degrees along the y-axis.
        :z_degrees (float, min=-90, max=90) rotation degrees along the z-axis.

        -----
        Return (bool) if the action is successful, which is True in the case
        that the agent is holding an object.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.rotate_held_object,
            thor_action='RotateHand',
            updated_kwarg_names={
                'x_degrees': 'x', 'y_degrees': 'y', 'z_degrees': 'z'},
            x_degrees=x_degrees, y_degrees=y_degrees, z_degrees=z_degrees)

    def drop_held_object(self) -> bool:
        """Drop the object in the agent's hand.

        -----
        Return (bool) if the action is successful, which would would only
        occur if the agent is holding an object.
        """
        return Helpers.execute_action(
            controller=self.controller,
            action_space=self.action_space,
            action_fn=self.drop_held_object,
            thor_action='DropHandObject')

    @property
    def last_event(self):
        """Return the AI2-THOR Event from the most recent controller action."""
        return self.controller.last_event

    @property
    def poses(self):
        """Return (initial, target, predicted) pose of the scenes's objects."""
        # access cached object poses
        scene = self.scenes[self.current_scene_idx]

        if not self.shuffle_called:
            raise Exception('shuffle() must be called before accessing poses')
        predicted_objs = self.controller.last_event.metadata['objects']

        # sorts the object order
        predicted_objs = sorted(predicted_objs, key=lambda obj: obj['name'])
        initial_poses = sorted(self.initial_poses, key=lambda obj: obj['name'])
        target_poses = sorted(self.target_poses, key=lambda obj: obj['name'])

        if scene not in self.identical_objects:
            print('in 1')
            return (
                Helpers.get_pose_info(initial_poses),
                Helpers.get_pose_info(target_poses),
                Helpers.get_pose_info(predicted_objs),
            )
        else:
            print('in 2')
            identical_names = self.identical_objects[scene]
            objs = {'targets': [], 'initial': [], 'predicted': []}
            duplicate_idxs = []
            for i in range(len(predicted_objs)):
                # initial, target, and predicted names are in the same order
                # because each of them are sorted by the same names.
                # initial and target were sorted upon caching.
                name = initial_poses[i]['name']

                if name not in identical_names:
                    objs['targets'].append(
                        Helpers.get_pose_info(target_poses[i]))
                    objs['initial'].append(
                        Helpers.get_pose_info(initial_poses[i]))
                    objs['predicted'].append(
                        Helpers.get_pose_info(predicted_objs[i]))
                else:
                    duplicate_idxs.append(i)

            # stores the distances from each duplicate target to each duplicate
            # predicted object. FWIW, there's only max(3) duplicate pickupable
            # objects, so this is quite fast.
            distances = defaultdict(dict)
            for targ_i in duplicate_idxs:
                targ_obj = target_poses[targ_i]
                for pred_i in duplicate_idxs:
                    pred_obj = predicted_objs[pred_i]
                    dist = Helpers.l2_distance(pred_obj, targ_obj)
                    distances[targ_i][pred_i] = dist

            print(distances)

            # finds the one-to-one duplicate object correspondences
            pred_idxs_left = set(duplicate_idxs)
            while distances:
                min_targ_i = -1
                min_pred_i = -1
                min_dist = float('inf')

                for targ_i in distances:
                    for pred_i in pred_idxs_left:
                        dist = distances[targ_i][pred_i]
                        if dist < min_dist:
                            min_dist = dist
                            min_pred_i = pred_i
                            min_targ_i = targ_i

                print('- ' * 3)
                print(min_targ_i, min_pred_i, min_dist)
                print(distances)
                print(pred_idxs_left)

                # target idx / initial idx are in sync
                """
                objs['targets'].append(
                    Controller.get_pose_info(target_poses[min_targ_i]))
                objs['initial'].append(
                    Controller.get_pose_info(initial_poses[min_targ_i]))

                # rounded idx
                objs['predicted'].append(
                    Controller.get_pose_info(predicted_objs[min_pred_i]))
                """

                # asserts one-to-one correspondences
                pred_idxs_left.remove(min_pred_i)
                del distances[min_targ_i]

            return (objs['initial'], objs['targets'], objs['predicted'])

    def reset(self):
        """Arrange the next target data for the walkthrough phase."""
        self.current_scene_idx += 1
        self.current_scene_idx %= len(self.scenes)
        if self.current_scene_idx == 0:
            self.current_rearrangement += 1
            self.current_rearrangement %= self.max_rearrangements

        scene = self.scenes[self.current_scene_idx]
        data = self.data[scene][self.current_rearrangement]
        self.controller.reset(scene)

        # set agent position
        pos = data['agent_position']
        rot = {'x': 0, 'y': data['agent_rotation'], 'z': 0}
        self.controller.step('TeleportFull', rotation=rot, **pos)

        # open objects
        for obj in data['openable_data']:
            # id is re-found due to possible floating point errors
            id = [l_obj for l_obj in self.last_event.metadata['objects'] if
                  l_obj['name'] == obj['name']][0]['objectId']
            self.controller.step(
                action='OpenObject',
                objectId=id,
                moveMagnitude=obj['target_openness'],
                forceAction=True)

        # arrange target poses for pickupable objects
        self.controller.step(
            'SetObjectPoses', objectPoses=data['target_poses'])
        self.shuffle_called = False
        self.target_poses = self.last_event.metadata['objects']

    def shuffle(self):
        """Arranges the current starting data for the rearrangement phase."""
        self.walkthrough_phase = False
        scene = self.scenes[self.current_scene_idx]
        data = self.data[scene][self.current_rearrangement]
        self.controller.reset(scene)

        # set agent position
        pos = data['agent_position']
        rot = {'x': 0, 'y': data['agent_rotation'], 'z': 0}
        self.controller.step('TeleportFull', rotation=rot, **pos)

        # open objects
        for obj in data['openable_data']:
            self.controller.step(
                action='OpenObject',
                moveMagnitude=obj['start_openness'],
                forceAction=True)

        # arrange target poses for pickupable objects
        self.controller.step(
            'SetObjectPoses', objectPoses=data['starting_poses'])
        self.shuffle_called = True
        self.initial_poses = self.last_event.metadata['objects']

    def evaluate(self,
                 initial_poses: List[Dict[str, Any]],
                 target_poses: List[Dict[str, Any]],
                 predicted_poses: List[Dict[str, Any]]) -> float:
        """Evaluate the current episode's object poses.

        -----
        Attribues
        :initial_poses (List[Dict[str, Any]]) starting poses after shuffle.
        :target_poses (List[Dict[str, Any]]) starting poses after reset.
        :predicted_poses (List[Dict[str, Any]]) poses after the agent's
            unshuffling phase.

        -----
        Return (float) ranges between [0:1] and is calculated as follows:

        1. If any predicted object is broken, return 0.
        2. Otherwise if any non-shuffled object is out of place, return 0.
        3. Otherwise return the average number of successfully unshuffled
           objects.

        For steps 2 and 3, an object is considered in-place/unshuffled if it
        satisfies all of the following:

        1. Openness. It's openness between its target pose and predicted pose
           is off by less than 20 degrees. The openness check is only applied
           to objects that can open.
        2. Position and Rotation. The object's 3D bounding box from its target
           pose and the predicted pose must have an IoU over 0.5. The
           positional check is only relevant to object's that can move.
        """
        cumulative_reward = 0
        obj_change_count = 0

        for obj_i in range(len(initial_poses)):
            targ = target_poses[obj_i]
            init = initial_poses[obj_i]
            pred = predicted_poses[obj_i]

            # no reward for breaking a non-broken starting object
            if pred['broken']:
                return 0

            # check if the object has openness
            if targ['openness'] is not None:
                if abs(targ['openness'] - init['openness']) > 0.2:
                    # openness in the object is meant to change
                    if abs(targ['openness'] - pred['openness']) > 0.2:
                        cumulative_reward += 1
                    obj_change_count += 1
                elif abs(targ['openness'] - pred['openness']) > 0.2:
                    # scene is messed up... openness is not meant to change
                    return 0

            # iou without the agent doing anything
            expected_iou = Helpers.iou(
                targ['bounding_box'], init['bounding_box'])
            pred_iou = Helpers.iou(
                targ['bounding_box'], pred['bounding_box'])

            # check the positional change
            if expected_iou <= 0.5:
                # scene is messed up... obj not supposed to change positions
                if pred_iou > 0.5:
                    return 0
            else:
                # object position changes
                cumulative_reward += 1 if pred_iou > 0.5 else 0
                obj_change_count += 1
        return cumulative_reward / obj_change_count if obj_change_count else 0