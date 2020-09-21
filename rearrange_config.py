"""Helper file to configure scenes and evaluate object pose predictions.

For more information, see https://ai2thor.allenai.org/rearrangement.
"""

import ai2thor
import ai2thor.controller
from collections import defaultdict
from typing import Dict, Callable, Tuple, Any, Union, List, Optional
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
import random
import json
import os
import logging

REQUIRED_VERSION = '2.4.20'
DATA_DIR = './data'
ROTATE_STEP_DEGREES = 30
MAX_HAND_METERS = 0.5
logging.basicConfig(level=logging.INFO)
SIM_OBJECTS = [
    'AlarmClock', 'AluminumFoil', 'Apple', 'AppleSliced', 'ArmChair',
    'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Blinds',
    'Book', 'Boots', 'Bottle', 'Bowl', 'Box', 'Bread', 'BreadSliced',
    'ButterKnife', 'Cabine', 'Candle', 'CD', 'CellPhone', 'Chair', 'Cloth',
    'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup',
    'Curtains', 'Desk', 'DeskLamp', 'Desktop', 'DiningTable', 'DishSponge',
    'DogBed', 'Drawer', 'Dresser', 'Dumbbell', 'Egg', 'EggCracked', 'Faucet',
    'Floor', 'FloorLamp', 'Footstool', 'Fork', 'Fridge', 'GarbageBag',
    'GarbageCan', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle',
    'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'Lettuce',
    'LettuceSliced', 'LightSwitch', 'Microwave', 'Mirror', 'Mug', 'Newspaper',
    'Ottoman', 'Painting', 'Pan', 'PaperTowel', 'Pen', 'Pencil',
    'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Poster', 'Pot', 'Potato',
    'PotatoSliced', 'RemoteControl', 'RoomDecor', 'Safe', 'SaltShaker',
    'ScrubBrush', 'Shelf', 'ShelvingUnit', 'ShowerCurtain', 'ShowerDoor',
    'ShowerGlass', 'ShowerHead', 'SideTable', 'Sink', 'SinkBasin', 'SoapBar',
    'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue',
    'Stool', 'StoveBurner', 'StoveKnob', 'TableTopDecor', 'TargetCircle',
    'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster',
    'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'Tomato', 'TomatoSliced',
    'Towel', 'TowelHolder', 'TVStand', 'VacuumCleaner', 'Vase', 'Watch',
    'WateringCan', 'Window', 'WineBottle'
]


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

    def execute_random_action(self, log_choice: bool = True) -> None:
        """Execute a random action within the specified action space."""
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

        action(**kwargs)

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

        # get rid of bad variable names
        for better_kwarg, thor_kwarg in updated_kwarg_names.items():
            kwargs[thor_kwarg] = kwargs[better_kwarg]
            del kwargs[better_kwarg]

        for name, value in default_thor_kwargs.items():
            kwargs[name] = value

        event = controller.step(thor_action, **kwargs)
        return event.metadata['lastActionSuccess']

    @staticmethod
    def iou(b1: np.ndarray, b2: np.ndarray, num_points: int = 2197) -> float:
        """Calculate the IoU between 3d bounding boxes b1 and b2."""
        def _outer_bounds(points_1: np.ndarray,
                          points_2: np.ndarray) -> Dict[str, Dict[str, float]]:
            """Sample points from the outer bounds formed by points_1/2."""
            assert points_1.shape == points_2.shape
            bounds = dict()
            for i in range(points_1.shape[0]):
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
        return intersection / union if union else 0


class Environment:
    """Custom AI2-THOR Controller for the task of object unshuffling."""

    def __init__(
            self,
            stage: str,
            mode: str = 'default',
            render_depth: bool = True):
        """Initialize a new rearrangement controller.

        -----
        Attributes
        :stage (str) must be in {'train', 'val'}. (casing is ignored)
        :mode (str) must be in {'default', 'easy'}. (casing is ignored)
        :render_depth (bool) states if the depth frame should be rendered.

        """
        if ai2thor.__version__ != REQUIRED_VERSION:
            raise ValueError(f'Please use AI2-THOR v{REQUIRED_VERSION}')

        stage = stage.lower()
        if stage not in {'train', 'val'}:
            raise ValueError("stage must be either 'train' or 'val'.")

        self.mode = mode.lower()
        if self.mode not in {'default', 'easy'}:
            raise ValueError("mode must be either 'default' or 'easy'.")
        self._drop_positions: Dict[str, Any] = dict()

        # instance masks now not supported. But an Exception would be thrown if
        # mode = 'default' and render_instance_masks is True, since masks are
        # only available on easy mode.
        render_instance_masks: bool = False
        if self.mode == 'default' and render_instance_masks:
            raise Exception(
                'render_instance_masks is only available on easy mode.')

        if stage == 'train':
            data_path = os.path.join(DATA_DIR, 'train.json')
        elif stage == 'val':
            data_path = os.path.join(DATA_DIR, 'val.json')

        with open(data_path, 'r') as f:
            self._data = json.loads(f.read())

        self.scenes = list(self._data.keys())
        if not self.scenes:
            raise ValueError('The scenes are not listed as keys in the json!')
        self._current_scene_idx = -1

        # assumes the same number of rearrangements per scene
        self.current_rearrangement = -1
        self.shuffles_per_scene = len(list(self._data.values())[0])

        # local thor controller to execute all the actions
        self._render_instance_masks = render_instance_masks
        self._render_depth = render_depth
        self._controller = ai2thor.controller.Controller(
            rotateStepDegrees=ROTATE_STEP_DEGREES,
            renderDepthImage=render_depth,
            renderObjectImage=render_instance_masks,
            server_class=ai2thor.fifo_server.FifoServer)

        # always begin in walkthrough phase
        self._shuffle_called = False

        # for identical pickupable objects, the predicted objects are
        # re-assigned so as to minimize the distance between the predicted
        # object position and the goal object position.
        self._identical_objects = {
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
    def observation(self) -> Tuple[np.array, Optional[np.array]]:
        """Return the current (RGB, depth, Optional[instance masks]) frames.

        :RGB frame is 300x300x3 with integer entries in [0:255].
        :depth frame is 300x300 with unscaled entries representing the
            meter distance from the agent to the pixel.

        """
        rgb = self._last_event.frame
        depth = self._last_event.depth_frame if self._render_depth else None
        return rgb, depth

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

        if self._shuffle_called:
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
                    'x': BoundedFloat(low=-0.5, high=0.5),
                    'y': BoundedFloat(low=-0.5, high=0.5),
                    'z': BoundedFloat(low=-0.5, high=0.5),
                },
                self.drop_held_object: {}
            })

        if self.mode == 'easy' and self._shuffle_called:
            actions.update({
                self.magic_drop_held_object: {}
            })

        return ActionSpace(actions)

    def open_object(self, x: float, y: float, openness: float) -> None:
        """Open the object corresponding to x/y to openess.

        -----
        Attributes
        :x (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        :y (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        ---
        The action will not be successful if the specified openness would
        cause a collision or if the object at x/y is not openable.

        """
        if x < 0 or x > 1 or y < 0 or y > 1:
            raise ValueError('x/y must be in [0:1].')
        if openness < 0 or openness > 1:
            raise ValueError('openness must be in [0:1]')

        # openness = 0 actually means fully open under the hood.
        openness = openness + 0.001 if openness == 0 else openness

        # If an object is already open, THOR doesn't support changing
        # it's openness without first closing it. So we simply try to first
        # close the object before reopening it.
        self._controller.step('ResetObjectFilter')
        objs_1 = self._last_event.metadata['objects']
        close_event = self._controller.step('CloseObject', x=x, y=y)

        # True if the object is likely already closed, or (x, y) doesn't map to
        # an openable object.
        if not close_event.metadata['lastActionSuccess']:
            Helpers.execute_action(
                controller=self._controller,
                action_space=self.action_space,
                action_fn=self.open_object,
                thor_action='OpenObject',
                error_message=(
                    'x/y/openness must be in [0:1] and in unshuffle phase.'),
                updated_kwarg_names={'openness': 'moveMagnitude'},
                x=x, y=y, openness=openness)
        else:
            if not self._shuffle_called:
                raise Exception('Must call env.shuffle() before opening.')
            objs_2 = self._last_event.metadata['objects']

            # find which object was opened
            for i in range(len(objs_1)):
                if objs_1[i]['isOpen'] ^ objs_2[i]['isOpen']:
                    closed_object_id = objs_1[i]['objectId']
                    break
            else:
                logging.warn('Unexpected open object behavior!\n' +
                             'Please report an issue :)')

            self._controller.step(
                'OpenObject', moveMagnitude=openness,
                objectId=closed_object_id)

        # hide object metadata for next action
        self._controller.step('SetObjectFilter', objectIds=[])

    def pickup_object(self, x: float, y: float) -> None:
        """Pick up the object corresponding to x/y.

        -----
        Attributes
        :x (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        :y (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        ---
        The action will not be successful if the object at x/y is not
        pickupable.

        """
        Helpers.execute_action(
            controller=self._controller,
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
            force_magnitude: float) -> None:
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
        The action will not be successful if the object at x/y is not moveable.

        """
        if not self._shuffle_called:
            raise Exception(
                'Must be in unshuffle phase, i.e., call shuffle().')
        if x > 1 or x < 0:
            raise ValueError('x must be in [0:1]')
        if y > 1 or y < 0:
            raise ValueError('y must be in [0:1]')
        if rel_x_force > 0.5 or rel_x_force < -0.5:
            raise ValueError('rel_x_force must be in [-0.5:0.5]')
        if rel_y_force > 0.5 or rel_y_force < -0.5:
            raise ValueError('rel_y_force must be in [-0.5:0.5]')
        if rel_z_force > 0.5 or rel_z_force < -0.5:
            raise ValueError('rel_z_force must be in [-0.5:0.5]')
        if force_magnitude > 1 or force_magnitude < 0:
            raise ValueError('force_magnitude must be in [0:1]')

        self._controller.step(
            'TouchThenApplyForce', x=x, y=y, handDistance=1.5,
            direction=dict(x=rel_x_force, y=rel_y_force, z=rel_z_force),
            moveMagnitude=force_magnitude*50
        )

    def move_ahead(self) -> None:
        """Move the agent ahead from its facing direction by 0.25 meters."""
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.move_ahead,
            thor_action='MoveAhead')

    def move_back(self) -> None:
        """Move the agent back from its facing direction by 0.25 meters."""
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.move_back,
            thor_action='MoveBack')

    def move_right(self) -> None:
        """Move the agent right from its facing direction by 0.25 meters."""
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.move_right,
            thor_action='MoveRight')

    def move_left(self) -> None:
        """Move the agent left from its facing direction by 0.25 meters."""
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.move_left,
            thor_action='MoveLeft')

    def rotate_left(self) -> None:
        """Rotate the agent left from its facing direction by 30 degrees."""
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.rotate_left,
            thor_action='RotateLeft')

    def rotate_right(self) -> None:
        """Rotate the agent left from its facing direction by 30 degrees."""
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.rotate_right,
            thor_action='RotateRight')

    def stand(self) -> None:
        """Stand the agent from the crouching position."""
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.stand,
            thor_action='Stand')

    def crouch(self) -> None:
        """Crouch the agent from the standing position."""
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.crouch,
            thor_action='Crouch')

    def look_up(self) -> None:
        """Turn the agent's head and camera up by 30 degrees."""
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.look_up,
            thor_action='LookUp')

    def look_down(self) -> None:
        """Turn the agent's head and camera down by 30 degrees."""
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.look_down,
            thor_action='LookDown')

    def done(self) -> None:
        """Agent's signal that it's completed its current rearrangement phase.

        Note that we do not automatically switch from the walkthrough phase
        to the unshuffling phase, and vice-versa, that is up to the user.
        This allows users to call .poses after the agent calls done, and
        have it correspond to the current episode.
        """
        self.agent_signals_done = True
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.done,
            thor_action='Done')

    def move_held_object(
            self,
            x_meters: float,
            y_meters: float,
            z_meters: float) -> None:
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
        The action is successful in the case that the agent is holding an
        object and moving the object by the specified amount does not bump
        into an object.
        """
        mag = (x_meters ** 2 + y_meters ** 2 + z_meters ** 2) ** (0.5)

        # clips the max value at MAX_HAND_METERS.
        if MAX_HAND_METERS > mag:
            x_meters /= (mag / MAX_HAND_METERS)
            y_meters /= (mag / MAX_HAND_METERS)
            z_meters /= (mag / MAX_HAND_METERS)

        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.move_held_object,
            thor_action='MoveHandDelta',
            updated_kwarg_names={
                'x_meters': 'x', 'y_meters': 'y', 'z_meters': 'z'},
            x_meters=x_meters, y_meters=y_meters, z_meters=z_meters)

    def rotate_held_object(self, x: float, y: float, z: float) -> None:
        """Rotate the object in the agent's hand by the specified degrees.

        The rotation parameters are scaled linearly to put rotations
        between [-90:90] degrees.

        -----
        Attribues
        :x (float, min=-0.5, max=0.5) rotation along the x-axis.
        :y (float, min=-0.5, max=0.5) rotation along the y-axis.
        :z (float, min=-0.5, max=0.5) rotation along the z-axis.

        -----
        The action is only successful agent is holding an object.
        """
        if not self._shuffle_called:
            raise Exception('Must be in shuffle phase')

        if abs(x) > 0.5 or abs(y) > 0.5 or abs(z) > 0.5:
            raise ValueError('Rotations must be between [-0.5:0.5].')
        self._controller.step('RotateHand', x=x*180, y=y*180, z=z*180)

    def drop_held_object(self) -> None:
        """Drop the object in the agent's hand.

        -----
        The action is only successful agent is holding an object.
        """
        Helpers.execute_action(
            controller=self._controller,
            action_space=self.action_space,
            action_fn=self.drop_held_object,
            thor_action='DropHandObject')

    def magic_drop_held_object(self) -> None:
        """Drop the object in the agent's hand to the target position.

        Exception is raised if shuffle has not yet been called on the current
        episode or the agent is in default mode.

        For magic drop to work:
            1. The agent must be within 1.5 meters from the goal object's
               position, observed during the walkthrough phase.
            2. The agent must be looking in the direction of where it was
               located in the walkthrough phase.

        Otherwise, the normal drop in place will be applied.
        """
        if not self._shuffle_called:
            raise Exception('Must be in shuffle mode.')
        if not self.mode == 'easy':
            raise Exception('Must be in easy mode.')

        # round positions to 2 decimals
        DEC = 2

        self._controller.step('ResetObjectFilter')
        for obj in self._last_event.metadata['objects']:
            if obj['isPickedUp']:
                agent = self._last_event.metadata['agent']
                valid_agent_poses = self._drop_positions[obj['name']]
                for i in range(len(valid_agent_poses['x'])):
                    # Checks if the agent is close enough to the target
                    # for the magic drop to be applied.
                    if (
                            # position check
                            round(valid_agent_poses['x'][i], DEC) ==
                            round(agent['position']['x'][i], DEC) and
                            round(valid_agent_poses['y'][i], DEC) ==
                            round(agent['position']['y'][i], DEC) and
                            round(valid_agent_poses['z'][i], DEC) ==
                            round(agent['position']['z'][i], DEC) and

                            # rotation check (nearest 90 degree)
                            int(valid_agent_poses['rotation'][i]) ==
                            round(agent['rotation']['y'] / 90) * 90 and

                            # standing check
                            int(valid_agent_poses['standing'][i]) ==
                            int(agent['isStanding']) and

                            # horizon check
                            int(valid_agent_poses['horizon'][i]) ==
                            int(agent['cameraHorizon'])):
                        goal_pos = valid_agent_poses['obj_pos']
                        goal_rot = valid_agent_poses['obj_rot']
                        self._controller.step(
                            action='PlaceObjectAtPoint',
                            objectId=obj['objectId'],
                            rotation=goal_rot,
                            position=goal_pos)
                        break
                break
        else:
            # agent is too far away from target, just drop like normal.
            self._controller.step('DropHandObject')
        self._controller.step('SetObjectFilter', objectIds=[])

    @property
    def _last_event(self) -> ai2thor.server.Event:
        """Return the AI2-THOR Event from the most recent controller action."""
        return self._controller.last_event

    @property
    def scene(self) -> str:
        """Return the current scene name."""
        return self.scenes[self._current_scene_idx]

    @property
    def poses(self):
        """Return (initial, goal, predicted) pose of the scenes's objects."""
        # access cached object poses
        if not self._shuffle_called:
            raise Exception('shuffle() must be called before accessing poses')
        self._controller.step('ResetObjectFilter')
        predicted_objs = self._controller.last_event.metadata['objects']
        self._controller.step('SetObjectFilter', objectIds=[])

        # sorts the object order
        predicted_objs = sorted(predicted_objs, key=lambda obj: obj['name'])
        initial_poses = sorted(self._initial_poses,
                               key=lambda obj: obj['name'])
        goal_poses = sorted(self._goal_poses, key=lambda obj: obj['name'])

        if self.scene not in self._identical_objects:
            return (
                Helpers.get_pose_info(initial_poses),
                Helpers.get_pose_info(goal_poses),
                Helpers.get_pose_info(predicted_objs),
            )
        else:
            identical_names = self._identical_objects[self.scene]
            objs = {'goal': [], 'initial': [], 'predicted': []}
            duplicate_idxs = []
            for i in range(len(predicted_objs)):
                # initial, goal, and predicted names are in the same order
                # because each of them are sorted by the same names.
                name = initial_poses[i]['name']

                if name not in identical_names:
                    objs['goal'].append(
                        Helpers.get_pose_info(goal_poses[i]))
                    objs['initial'].append(
                        Helpers.get_pose_info(initial_poses[i]))
                    objs['predicted'].append(
                        Helpers.get_pose_info(predicted_objs[i]))
                else:
                    duplicate_idxs.append(i)

            # stores the distances from each duplicate goal to each duplicate
            # predicted object. FWIW, there's only max(3) duplicate pickupable
            # objects, so this is quite fast.
            distances = defaultdict(dict)
            for targ_i in duplicate_idxs:
                targ_obj = goal_poses[targ_i]
                for pred_i in duplicate_idxs:
                    pred_obj = predicted_objs[pred_i]
                    dist = Helpers.l2_distance(pred_obj, targ_obj)
                    distances[targ_i][pred_i] = dist

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

                # goal idx / initial idx are in sync
                objs['goal'].append(
                    Helpers.get_pose_info(goal_poses[min_targ_i]))
                objs['initial'].append(
                    Helpers.get_pose_info(initial_poses[min_targ_i]))

                # rounded idx
                objs['predicted'].append(
                    Helpers.get_pose_info(predicted_objs[min_pred_i]))

                # asserts one-to-one correspondences
                pred_idxs_left.remove(min_pred_i)
                del distances[min_targ_i]

            return (objs['initial'], objs['goal'], objs['predicted'])

    def reset(self,
              scene: Optional[str] = None,
              rearrangement_idx: Optional[int] = None) -> None:
        """Arrange the next goal data for the walkthrough phase.

        -----
        Attribues
        :scene_idx (string) iTHOR scene name.
        :rearrangement_idx (int, min=0, max=49) rearrangement setup instance
        from the dataset.

        """
        if scene is None:
            # iterate to the next scene
            self._current_scene_idx += 1
            self._current_scene_idx %= len(self.scenes)
            if self._current_scene_idx == 0:
                self.current_rearrangement += 1
                self.current_rearrangement %= self.shuffles_per_scene
            scene = self.scene
        else:
            # user specifies a scene
            self.current_rearrangement = (rearrangement_idx if
                                          rearrangement_idx else 0)
            self._current_scene_idx = [i for i in range(
                len(self.scenes)) if self.scenes[i] == scene][0]
            self.current_rearrangement = rearrangement_idx  # type: ignore

        data = self._data[scene][self.current_rearrangement]
        self._controller.reset(scene)

        # set agent position
        pos = data['agent_position']
        rot = {'x': 0, 'y': data['agent_rotation'], 'z': 0}
        self._controller.step('TeleportFull', rotation=rot, **pos)

        # show object metadata
        self._controller.step('ResetObjectFilter')

        # open objects
        for obj in data['openable_data']:
            # id is re-found due to possible floating point errors
            id = [l_obj for l_obj in self._last_event.metadata['objects'] if
                  l_obj['name'] == obj['name']][0]['objectId']
            self._controller.step(
                action='OpenObject',
                objectId=id,
                moveMagnitude=obj['target_openness'],
                forceAction=True)

        # arrange target poses for pickupable objects
        self._controller.step(
            'SetObjectPoses', objectPoses=data['target_poses'])
        self._shuffle_called = False
        self._goal_poses = self._last_event.metadata['objects']
        self.agent_signals_done = False
        self._object_change_n: Optional[int] = None

        # store the magic drop positions
        if self.mode == 'easy':
            for obj in self._last_event.metadata['objects']:
                if obj['pickupable']:
                    evt = self._controller.step(
                        action='PositionsFromWhichItemIsInteractable',
                        objectId=obj['objectId'])
                    valid_agent_poses = evt.metadata['actionReturn']
                    valid_agent_poses['obj_pos'] = obj['position']
                    valid_agent_poses['obj_rot'] = obj['rotation']
                    self._drop_positions[obj['name']] = valid_agent_poses

        # hide object metadata
        self._controller.step('SetObjectFilter', objectIds=[])

    def shuffle(self):
        """Arranges the current starting data for the rearrangement phase."""
        self.walkthrough_phase = False
        scene = self.scene
        data = self._data[scene][self.current_rearrangement]
        self._controller.reset(scene)

        # set agent position
        pos = data['agent_position']
        rot = {'x': 0, 'y': data['agent_rotation'], 'z': 0}
        self._controller.step('TeleportFull', rotation=rot, **pos)

        # open objects
        for obj in data['openable_data']:
            self._controller.step(
                action='OpenObject',
                moveMagnitude=obj['start_openness'],
                forceAction=True)

        # arrange initial poses for pickupable objects
        self._controller.step(
            'SetObjectPoses', objectPoses=data['starting_poses'])
        self._shuffle_called = True

        # save object metadata
        self._controller.step('ResetObjectFilter')
        self._initial_poses = self._last_event.metadata['objects']
        self._controller.step('SetObjectFilter', objectIds=[])

        self.agent_signals_done = False

    @property
    def object_change_n(self) -> int:
        """Return the number of objects changed in the current scene."""
        scene_data = self._data[self.scene][self.current_rearrangement]
        return scene_data['object_rearrangement_count']

    def evaluate(self,
                 initial_poses: List[Dict[str, Any]],
                 goal_poses: List[Dict[str, Any]],
                 predicted_poses: List[Dict[str, Any]]) -> float:
        """Evaluate the current episode's object poses.

        -----
        Attribues
        :initial_poses (List[Dict[str, Any]]) starting poses after shuffle.
        :goal_poses (List[Dict[str, Any]]) starting poses after reset.
        :predicted_poses (List[Dict[str, Any]]) poses after the agent's
            unshuffling phase.

        -----
        Return (float) ranges between [0:1] and is calculated as follows:

        1. If any predicted object is broken, return 0.
        2. Otherwise if any non-shuffled object is out of place, return 0.
        3. Otherwise return the average number of successfully unshuffled
           objects.

        For steps 2 and 3, a predicted object is considered successfully
        in-place/unshuffled if it satisfies both of the following:

        1. Openness. The openness between its goal pose and predicted pose is
           off by less than 20 percent. The openness check is only applied to
           objects that can open.
        2. Position and Rotation. The object’s 3D bounding box from its goal
           pose and the predicted pose must have an IoU over 0.5. The
           positional check is only relevant to object’s that can move.
        """
        cumulative_reward = 0
        obj_change_count = 0

        for obj_i in range(len(initial_poses)):
            targ = goal_poses[obj_i]
            init = initial_poses[obj_i]
            pred = predicted_poses[obj_i]

            # no reward for breaking a non-broken starting object
            if pred['broken']:
                return 0

            # check if the object has openness
            if targ['openness'] is not None:
                if abs(targ['openness'] - init['openness']) > 0.2:
                    # openness in the object is meant to change
                    if abs(targ['openness'] - pred['openness']) <= 0.2:
                        cumulative_reward += 1
                    obj_change_count += 1
                elif abs(targ['openness'] - pred['openness']) > 0.2:
                    # scene is messed up... openness is not meant to change
                    return 0

            # non-moveable objects do not have bounding boxes
            if init['bounding_box'] is None:
                continue

            # iou without the agent doing anything
            expected_iou = Helpers.iou(
                np.array(targ['bounding_box']), np.array(init['bounding_box']))
            pred_iou = Helpers.iou(
                np.array(targ['bounding_box']), np.array(pred['bounding_box']))

            # check the positional change
            if expected_iou > 0.5:
                # scene is messed up... obj not supposed to change positions
                if pred_iou <= 0.5:
                    return 0
            else:
                # object position changes
                cumulative_reward += 1 if pred_iou > 0.5 else 0
                obj_change_count += 1
        return (
            cumulative_reward / obj_change_count
            if obj_change_count else 0
        )

    def stop(self):
        """Terminate the current AI2-THOR session."""
        self._controller.stop()
