# generated with ai2thor 3.5.1

import os
import cv2
import random
import argparse

import numpy as np
from ai2thor.controller import Controller

from constants import target_objects


parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str,
                    default='data/ithor_scenes',
                    help='Path output directory')
args = parser.parse_args()

os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)


controller = Controller(
    gridSize=0.25,
    makeAgentsVisible=False,
    rotateStepDegrees=90,
    renderDepthImage=True,
    renderSemanticSegmentation=True,
    renderInstanceSegmentation=True,
    quality='High',
    width=300,
    height=300,
    fieldOfView=90
)

scenes = controller.ithor_scenes(include_bathrooms=False)

for scene_name in scenes:
    print(scene_name)

    scene_id = int(scene_name.replace('FloorPlan', '').replace('_physics', ''))
    if scene_id % 100 <= 20:
        split = 'train'
    elif scene_id % 100 <= 25:
        split = 'val'
    else:
        split = 'test'

    controller.reset(scene=scene_name)
    controller.step(action="GetReachablePositions")
    locations = controller.last_event.metadata["actionReturn"][:]
    rotations = [0, 90, 180, 270]
    horizons = [45]

    data = []
    while len(data) < (100 if split == 'train' else 50):
        pos = random.sample(locations, 1)[0]
        rot, hor = None, None

        tries = 0
        while tries < 4:
            tries += 1

            rot = random.sample(rotations, 1)[0]
            hor = random.sample(horizons, 1)[0]
            e = controller.step(
                action="TeleportFull",
                position=pos,
                rotation=dict(x=0, y=rot, z=0),
                horizon=hor,
                standing=True
            )

            object_mask = np.any([v for k,v in e.class_masks.items() if k in target_objects], axis=0)
            object_frac = np.sum(object_mask) / np.prod(object_mask.shape)

            if object_frac > 0.015:
                break
        else:
            continue

        valid_moves_forward = 0
        while controller.step('MoveAhead').metadata['lastActionSuccess']:
            valid_moves_forward += 1

        data.append({
            'agent_metadata' : {
                'position' : pos,
                'rotation' : dict(x=0, y=rot, z=0),
                'horizon' : hor,
                'standing' : True
            },
            'object_metadata' : e.metadata['objects'],
            'frame' : e.frame,
            'depth_frame' : e.depth_frame,
            'semantic_frame' : e.semantic_segmentation_frame,
            'instance_frame' : e.instance_segmentation_frame,
            'object_id_to_color' : e.object_id_to_color,
            'valid_moves_forward' : valid_moves_forward
        })

    np.save(os.path.join(args.output_dir, split, f"{scene_name}.npy"), data)
