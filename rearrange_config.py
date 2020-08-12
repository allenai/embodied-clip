import ai2thor.controller
from collections import defaultdict
import numpy as np
import json
import os

REQUIRED_VERSION = '2.4.10'


class Controller:
    def __init__(self, stage, data_dir='./data', **kwargs):
        if ai2thor.__version__ != REQUIRED_VERSION:
            raise ValueError(f'Please use AI2-THOR v{REQUIRED_VERSION}')

        stage = stage.lower()
        if stage not in {'train', 'val'}:
            raise ValueError("Stage must be either 'train' or 'val'.")

        if stage == 'train':
            data_path = os.path.join(data_dir, 'train.json')
        elif stage == 'val':
            data_path = os.path.join(data_dir, 'val.json')

        with open(data_path, 'r') as f:
            self.data = json.loads(f.read())

        self.scenes = list(self.data.keys())
        if not self.scenes:
            raise ValueError('The scenes are not listed as keys in the json!')
        self.current_scene_idx = 0

        # assumes the same number of rearrangements per scene
        self.current_rearrangement = 0
        self.max_rearrangements = len(list(self.data.values())[0])

        # maps to the minimum required kwargs
        self.valid_rearrange_actions = {
            'PickupObject': {'x', 'y'},
            'OpenObject': {'x', 'y', 'moveMagnitude'},
            'TouchThenApplyForce': {'x', 'y'},
            'MoveHandDelta': {'x', 'y', 'z'},
            'RotateHand': {'x', 'y', 'z'},
            'MoveAhead': {},
            'MoveLeft': {},
            'MoveRight': {},
            'MoveBack': {},
            'RotateRight': {},
            'RotateLeft': {},
            'DropHandObject': {},
            'LookUp': {},
            'LookDown': {},
            'MoveHandAhead': {},
            'MoveHandBack': {},
            'MoveHandLeft': {},
            'MoveHandRight': {},
            'MoveHandUp': {},
            'MoveHandDown': {},
            'Done': {},
        }

        self.valid_walkthrough_actions = {
            'MoveAhead': {},
            'MoveRight': {},
            'MoveLeft': {},
            'MoveBack': {},
            'RotateRight': {},
            'RotateLeft': {},
            'LookUp': {},
            'LookDown': {},
            'Done': {},
        }

        # local thor controller to execute all the actions
        self.controller = ai2thor.controller.Controller(**kwargs)
        self.walkthrough_phase = True

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

    @staticmethod
    def get_pose_info(objs):
        def _extract_obj_data(obj):
            return {
                'type': obj['objectType'],
                'position': obj['position'],
                'rotation': obj['rotation'],
                'openness': obj['openPercent'] if obj['openable'] else None,
                'bounding_box':
                    obj['objectOrientedBoundingBox']['cornerPoints'] if
                    obj['objectOrientedBoundingBox'] else None
            }

        if type(objs) is list:
            # list of objects
            return [_extract_obj_data(obj) for obj in objs]
        else:
            # single object
            return _extract_obj_data(objs)

    @staticmethod
    def l2_distance(obj1, obj2):
        p1 = obj1['position']
        p2 = obj2['position']

        a = np.array([p1['x'], p1['y'], p1['z']])
        b = np.array([p2['x'], p2['y'], p2['z']])
        return np.linalg.norm(a - b)

    @property
    def last_event(self):
        return self.controller.last_event

    @property
    def poses(self):
        """Returns (initial poses, target poses, and predicted poses)
           of all objects in the scene."""
        if not self.initial_objects:
            raise Exception('shuffle() must be called before accessing poses')
        predicted_objs = self.controller.last_event.metadata['objects']

        scene = self.scenes[self.current_scene_idx]
        if scene not in self.identical_objects:
            return (
                Controller.get_pose_info(self.initial_objects),
                Controller.get_pose_info(self.target_objects),
                Controller.get_pose_info(predicted_objs),
            )
        else:
            identical_names = self.identical_objects[scene]
            objs = {'targets': [], 'initial': [], 'predicted': []}
            duplicate_idxs = []
            for i in range(len(predicted_objs)):
                # names should always remain in the same order, so using
                # initial_objects, target_objects, or predicted_objects
                # here should not matter
                name = self.initial_objects[i]['name']

                if name not in identical_names:
                    objs['targets'].append(
                        Controller.get_pose_info(self.target_objects[i]))
                    objs['initial'].append(
                        Controller.get_pose_info(self.initial_objects[i]))
                    objs['predicted'].append(
                        Controller.get_pose_info(predicted_objs[i]))
                else:
                    duplicate_idxs.append(i)

            # stores the distances from each duplicate target to each duplicate
            # predicted object. FWIW, there's only max(3) duplicate pickupable
            # objects, so this is quite fast.
            distances = defaultdict(dict)
            for targ_i in duplicate_idxs:
                targ_obj = self.target_objects[targ_i]
                for pred_i in duplicate_idxs:
                    pred_obj = predicted_objs[pred_i]
                    dist = Controller.l2_distance(pred_obj, targ_obj)
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

                # target idx
                objs['targets'].append(
                    Controller.get_pose_info(self.target_objects[min_targ_i]))
                objs['initial'].append(
                    Controller.get_pose_info(self.initial_objects[min_targ_i]))

                # rounded idx
                objs['predicted'].append(
                    Controller.get_pose_info(predicted_objs[min_pred_i]))

                # asserts one-to-one correspondences
                pred_idxs_left.remove(min_pred_i)
                del distances[min_targ_i]

            return (objs['initial'], objs['targets'], objs['predicted'])

    def reset(self):
        """Arranges the next target data for the walkthrough phase."""
        self.walkthrough_phase = True
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
            self.controller.step(
                action='OpenObject',
                moveMagnitude=obj['target_openness'],
                forceAction=True)

        # arrange target poses for pickupable objects
        event = self.controller.step(
            'SetObjectPoses', objectPoses=data['target_poses'])
        self.target_objects = event.metadata['objects']
        self.initial_objects = None

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
        event = self.controller.step(
            'SetObjectPoses', objectPoses=data['starting_poses'])
        self.initial_objects = event.metadata['objects']

    def step(self, action, **kwargs):
        """Restricted step with only the allowable actions."""
        actions = self.valid_walkthrough_actions if self.walkthrough_phase \
            else self.valid_rearrange_actions
        if action not in actions:
            actions = str(list(actions.keys()))
            raise ValueError(
                'Invalid Action! Must be in ' + actions + '.' +
                'For all actions, use controller.debug_step(action, **kwargs)')
        for req_key in actions[action]:
            if req_key not in kwargs:
                raise ValueError(f'The {action} action must specify {req_key}')
        self.controller.step(action, **kwargs)

    def debug_step(self, action, **kwargs):
        """Provides the full suite of AI2-THOR actions."""
        self.controller.step(action, **kwargs)
