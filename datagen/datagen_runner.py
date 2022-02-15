"""A script for generating rearrangement datasets."""

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import random
import time
import warnings
from collections import defaultdict
from typing import List, Set, Dict, Optional, Any, cast

import compress_pickle
import numpy as np
import tqdm
from ai2thor.controller import Controller
from allenact.utils.misc_utils import md5_hash_str_as_int
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from setproctitle import setproctitle as ptitle

from datagen.datagen_constants import OBJECT_TYPES_TO_NOT_MOVE
from datagen.datagen_utils import (
    get_scenes,
    get_random_seeds,
    filter_pickupable,
    open_objs,
    get_object_ids_to_not_move_from_object_types,
    remove_objects_until_all_have_identical_meshes,
    check_object_opens,
)
from rearrange.constants import STARTER_DATA_DIR, THOR_COMMIT_ID
from rearrange.environment import (
    RearrangeTHOREnvironment,
    RearrangeTaskSpec,
)
from rearrange_constants import OPENNESS_THRESHOLD, IOU_THRESHOLD

mp = mp.get_context("spawn")


def generate_one_rearrangement_given_initial_conditions(
    controller: Controller,
    scene: str,
    start_kwargs: dict,
    target_kwargs: dict,
    num_objs_to_move: int,
    num_objs_to_open: int,
    object_types_to_not_move: Set[str],
    agent_pos: Dict[str, float],
    agent_rot: Dict[str, float],
    allow_putting_objects_away: bool = False,
):
    # Start position

    controller.reset(scene)
    controller.step(
        "TeleportFull", horizon=0, standing=True, rotation=agent_rot, **agent_pos
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        return None, None, None

    if not remove_objects_until_all_have_identical_meshes(controller):
        return None, None, None

    controller.step("InitialRandomSpawn", **start_kwargs)
    if not controller.last_event.metadata["lastActionSuccess"]:
        return None, None, None

    for _ in range(12):
        controller.step("Pass")

    if any(o["isBroken"] for o in controller.last_event.metadata["objects"]):
        return None, None, None

    # get initial and post random spawn object data
    objects_after_first_irs = controller.last_event.metadata["objects"]

    # of the non-movable objects randomly open some of them
    openable_objects = [
        obj
        for obj in objects_after_first_irs
        if obj["openable"] and not obj["pickupable"]
    ]
    random.shuffle(openable_objects)

    object_names_to_open = []
    for oo in openable_objects:
        if len(object_names_to_open) == num_objs_to_open:
            break
        if check_object_opens(oo, controller):
            object_names_to_open.append(oo["name"])

    if len(object_names_to_open) != num_objs_to_open:
        return None, None, None

    try:
        start_openness = open_objs(
            object_names_to_open=object_names_to_open, controller=controller
        )
    except (StopIteration, RuntimeError):
        return None, None, None

    # accounts for possibly a rare event that I cannot think of, where opening
    # a non-pickupable object moves a pickupable object.
    pickupable_objects_after_first_irs = filter_pickupable(
        objects=objects_after_first_irs,
        object_types_to_not_move=object_types_to_not_move,
    )

    # choose which objects to move
    if len(pickupable_objects_after_first_irs) < num_objs_to_move:
        return None, None, None

    random.shuffle(pickupable_objects_after_first_irs)
    if num_objs_to_move == 0:
        objects_to_not_move = pickupable_objects_after_first_irs
        moved_objs = []
    else:
        objects_to_not_move = pickupable_objects_after_first_irs[:-num_objs_to_move]
        moved_objs = pickupable_objects_after_first_irs[-num_objs_to_move:]

    moved_obj_names = {o["name"] for o in moved_objs}
    unmoved_obj_names = {o["name"] for o in objects_to_not_move}

    if allow_putting_objects_away:
        # If we're having a really hard time shuffling objects successfully, then let's
        # move some of the objects we don't care about (i.e. the ones whose position won't change)
        # into cupboards/drawers/etc so that there is more space.
        controller.step(
            "InitialRandomSpawn",
            **start_kwargs,
            excludedObjectIds=[
                o["objectId"]
                for o in controller.last_event.metadata["objects"]
                if o["name"] in moved_obj_names
            ],
        )
        if not controller.last_event.metadata["lastActionSuccess"]:
            return None, None, None

        objects_after_first_irs = controller.last_event.metadata["objects"]
        pickupable_objects_after_first_irs = filter_pickupable(
            objects=objects_after_first_irs,
            object_types_to_not_move=object_types_to_not_move,
        )

    controller.step(
        "TeleportFull", horizon=0, standing=True, rotation=agent_rot, **agent_pos
    )
    if not controller.last_event.metadata["lastActionSuccess"]:
        return None, None, None

    second_stage_success = False
    pickupable_objects_after_shuffle: Optional[List[Dict[str, Any]]] = None
    target_openness: Optional[Dict[str, float]] = None
    for retry_ind in range(2):
        object_ids_not_to_move = [
            o["objectId"]
            for o in controller.last_event.metadata["objects"]
            if o["name"] in unmoved_obj_names
        ]
        object_ids_not_to_move.extend(
            get_object_ids_to_not_move_from_object_types(
                controller=controller, object_types=object_types_to_not_move,
            )
        )
        controller.step(
            "InitialRandomSpawn",
            excludedObjectIds=object_ids_not_to_move,
            **{**target_kwargs, "randomSeed": target_kwargs["randomSeed"] + retry_ind},
        )
        if not controller.last_event.metadata["lastActionSuccess"]:
            continue

        for _ in range(12):
            # This shouldn't be necessary but we run these actions
            # to let physics settle.
            controller.step("Pass")

        # change the openness of one the same non-pickupable objects
        try:
            target_openness = open_objs(
                object_names_to_open=object_names_to_open, controller=controller
            )
        except (StopIteration, RuntimeError):
            return None, None, None

        # get initial and post random spawn object data
        pickupable_objects_after_shuffle = filter_pickupable(
            controller.last_event.metadata["objects"], object_types_to_not_move
        )

        all_teleport_success = True
        for o in pickupable_objects_after_shuffle:
            if o["name"] in moved_obj_names:
                pos = o["position"]
                positions = [
                    {
                        "x": pos["x"] + 0.001 * xoff,
                        "y": pos["y"] + 0.001 * yoff,
                        "z": pos["z"] + 0.001 * zoff,
                    }
                    for xoff in [0, -1, 1]
                    for zoff in [0, -1, 1]
                    for yoff in [0, 1, 2]
                ]
                controller.step(
                    "TeleportObject",
                    objectId=o["objectId"],
                    positions=positions,
                    rotation=o["rotation"],
                    makeUnbreakable=True,
                )
                if not controller.last_event.metadata["lastActionSuccess"]:
                    all_teleport_success = False
                    break
        if all_teleport_success:
            second_stage_success = True
            break

    for o in controller.last_event.metadata["objects"]:
        if o["isBroken"]:
            print(
                f"In scene {controller.last_event.metadata['sceneName']},"
                f" object {o['name']} broke during setup."
            )
            return None, None, None

    if not second_stage_success:
        return None, None, None

    pickupable_objects_after_first_irs.sort(key=lambda x: x["name"])
    pickupable_objects_after_shuffle.sort(key=lambda x: x["name"])

    if any(
        o0["name"] != o1["name"]
        for o0, o1 in zip(
            pickupable_objects_after_first_irs, pickupable_objects_after_shuffle
        )
    ):
        print("Pickupable object names don't match after shuffle!")
        return None, None, None

    # [opened, starting, target]
    return (
        [
            {
                "name": open_obj_name,
                "objectName": open_obj_name,
                "objectId": next(
                    o["objectId"]
                    for o in openable_objects
                    if o["name"] == open_obj_name
                ),
                "start_openness": start_openness[open_obj_name],
                "target_openness": target_openness[open_obj_name],
            }
            for open_obj_name in start_openness
        ],
        [
            {
                "name": pickupable_objects_after_first_irs[i]["name"],
                "objectName": pickupable_objects_after_first_irs[i]["name"],
                "position": pickupable_objects_after_first_irs[i]["position"],
                "rotation": pickupable_objects_after_first_irs[i]["rotation"],
            }
            for i in range(len(pickupable_objects_after_first_irs))
        ],
        [
            {
                "name": pickupable_objects_after_shuffle[i]["name"],
                "objectName": pickupable_objects_after_shuffle[i]["name"],
                "position": pickupable_objects_after_shuffle[i]["position"],
                "rotation": pickupable_objects_after_shuffle[i]["rotation"],
            }
            for i in range(len(pickupable_objects_after_shuffle))
        ],
    )


def generate_rearrangements_for_scenes(
    stage_seed: int,
    stage_scenes: List[str],
    env: RearrangeTHOREnvironment,
    object_types_to_not_move: Set[str],
    max_obj_rearrangements_per_scene: int = 5,
    scene_reuse_count: int = 50,
    obj_name_to_avoid_positions: Optional[Dict[str, np.ndarray]] = None,
    force_visible: bool = True,
    place_stationary: bool = False,
    rotation_increment: int = 30,
) -> dict:
    if 360 % rotation_increment != 0:
        raise ValueError("Rotation increment must be a factor of 360")

    if obj_name_to_avoid_positions is None:
        obj_name_to_avoid_positions = defaultdict(
            lambda: np.array([[-1000, -1000, -1000]])
        )

    controller = env.controller

    out: dict = dict()
    for scene in stage_scenes:
        print(f"Scene {scene}")

        seed = md5_hash_str_as_int(f"{stage_seed}|{scene}")
        random.seed(seed)

        out[scene] = []

        # set positions and rotations
        controller.reset(scene)

        scene_has_openable = 0 != len(
            [
                o
                for o in controller.last_event.metadata["objects"]
                if o["openable"] and not o["pickupable"]
            ]
        )
        if not scene_has_openable:
            warnings.warn(f"SCENE {scene} HAS NO OPENABLE OBJECTS")

        evt = controller.step("GetReachablePositions")
        rps: List[Dict[str, float]] = evt.metadata["actionReturn"]
        rps.sort(key=lambda d: (round(d["x"], 2), round(d["z"], 2)))
        rotations = np.arange(0, 360, rotation_increment)

        for reuse_i in range(scene_reuse_count):
            try_count = 0

            # Evenly distribute # of object rearrangements
            num_objs_to_open = scene_has_openable * (reuse_i % 2)
            num_objs_to_move = (1 - num_objs_to_open) + math.floor(
                max_obj_rearrangements_per_scene * (reuse_i / scene_reuse_count)
            )
            position_count_offset = 0

            while True:
                try_count += 1
                if try_count > 100:
                    raise RuntimeError(
                        f"Something wrong with scene {scene}, please file an issue."
                    )

                episode_seed_string = f"{scene}|ind_{reuse_i}|tries_{try_count}|counts_{position_count_offset}|seed_{stage_seed}"
                seed = md5_hash_str_as_int(episode_seed_string)
                random.seed(seed)

                # avoid agent being unable to teleport to position
                # due to object being placed there
                pos = random.choice(rps)
                rot = {"x": 0, "y": int(random.choice(rotations)), "z": 0}

                # used to make sure the positions of the objects
                # are not always the same across the same scene.
                start_kwargs = {
                    "randomSeed": random.randint(0, int(1e7) - 1),
                    "forceVisible": force_visible,
                    "placeStationary": place_stationary,
                    "excludedReceptacles": ["ToiletPaperHanger"],
                }
                target_kwargs = {
                    "randomSeed": random.randint(0, int(1e7) - 1),
                    "forceVisible": force_visible,
                    "placeStationary": place_stationary,
                    "excludedReceptacles": ["ToiletPaperHanger"],
                }

                # sometimes weird bugs arise where the pickupable
                # object count within a scene does not match
                (
                    opened_data,
                    starting_poses,
                    target_poses,
                ) = generate_one_rearrangement_given_initial_conditions(
                    controller=controller,
                    scene=scene,
                    start_kwargs=start_kwargs,
                    target_kwargs=target_kwargs,
                    num_objs_to_move=num_objs_to_move + position_count_offset,
                    num_objs_to_open=num_objs_to_open,
                    object_types_to_not_move=object_types_to_not_move,
                    agent_pos=pos,
                    agent_rot=rot,
                    allow_putting_objects_away=try_count >= 30,
                )

                if opened_data is None:
                    position_count_offset = max(position_count_offset - 1, 0)
                    print(
                        f"{episode_seed_string}: Failed during generation."  # {scene}, {pos}, {int(rot['y'])} {start_kwargs}, {target_kwargs}."
                    )
                    continue

                task_spec_dict = {
                    "agent_position": pos,
                    "agent_rotation": int(rot["y"]),
                    "object_rearrangement_count": int(num_objs_to_move)
                    + int(num_objs_to_open),
                    "openable_data": opened_data,
                    "starting_poses": starting_poses,
                    "target_poses": target_poses,
                }

                env.reset(task_spec=RearrangeTaskSpec(scene=scene, **task_spec_dict))
                env.shuffle()
                ips, gps, cps = env.poses
                pose_diffs = cast(
                    List[Dict[str, Any]], env.compare_poses(goal_pose=gps, cur_pose=cps)
                )
                reachable_positions = env.controller.step(
                    "GetReachablePositions"
                ).metadata["actionReturn"]

                failed = False
                for gp, cp, pd in zip(gps, cps, pose_diffs):
                    if pd["iou"] is not None and pd["iou"] < IOU_THRESHOLD:
                        assert gp["type"] not in object_types_to_not_move

                    if gp["broken"] or cp["broken"]:
                        failed = True
                        break

                    pose_diff_energy = env.pose_difference_energy(
                        goal_pose=gp, cur_pose=cp
                    )

                    if pose_diff_energy != 0:
                        obj_name = gp["name"]

                        # Ensure that objects to rearrange are visible from somewhere
                        interactable_poses = env.controller.step(
                            "GetInteractablePoses",
                            objectId=cp["objectId"],
                            positions=reachable_positions,
                        ).metadata["actionReturn"]
                        if interactable_poses is None or len(interactable_poses) == 0:
                            print(
                                f"{episode_seed_string}: {obj_name} is not visible despite needing to be rearranged."
                            )
                            failed = True
                            break

                        if obj_name in obj_name_to_avoid_positions:
                            if cp["pickupable"]:
                                threshold = 0.15
                                start_position = cp["position"]
                                pos_array = np.array(
                                    [[start_position[k] for k in ["x", "y", "z"]]]
                                )
                            elif cp["openness"] is not None:
                                threshold = 0.05
                                pos_array = np.array([[cp["openness"]]])
                            else:
                                continue

                            dist = np.sqrt(
                                (
                                    (obj_name_to_avoid_positions[obj_name] - pos_array)
                                    ** 2
                                ).sum(-1)
                            ).min()
                            if dist <= threshold:
                                print(
                                    f"{episode_seed_string}: {obj_name} is within the threshold ({dist} <= {threshold})."
                                )
                                failed = True
                                break
                if failed:
                    continue

                npos_diff = int(
                    sum(
                        pd["iou"] is not None and pd["iou"] < IOU_THRESHOLD
                        for pd in pose_diffs
                    )
                )
                nopen_diff = int(
                    sum(
                        pd["openness_diff"] is not None
                        and pd["openness_diff"] >= OPENNESS_THRESHOLD
                        for pd in pose_diffs
                    )
                )

                if npos_diff != num_objs_to_move:
                    position_count_offset += (npos_diff < num_objs_to_move) - (
                        npos_diff > num_objs_to_move
                    )
                    position_count_offset = max(position_count_offset, 0)

                    print(
                        f"{episode_seed_string}: Incorrect amount of objects have moved expected != actual ({num_objs_to_move} != {npos_diff})"
                    )
                    continue

                if nopen_diff != num_objs_to_open:
                    print(
                        f"{episode_seed_string}: Incorrect amount of objects have opened expected != actual ({num_objs_to_open} != {nopen_diff})"
                    )
                    continue

                task_spec_dict["position_diff_count"] = npos_diff
                task_spec_dict["open_diff_count"] = nopen_diff
                task_spec_dict["pose_diff_energy"] = float(
                    env.pose_difference_energy(goal_pose=gps, cur_pose=cps).sum()
                )

                if (npos_diff == 0 and nopen_diff == 0) or task_spec_dict[
                    "pose_diff_energy"
                ] == 0.0:
                    print(
                        f"Not enough has moved in {scene}, {pos}, {int(rot['y'])} {start_kwargs}, {target_kwargs}!"
                    )
                    continue

                if npos_diff > max_obj_rearrangements_per_scene or nopen_diff > 1:
                    print(
                        f"{episode_seed_string}: Final check failed ({npos_diff} [{max_obj_rearrangements_per_scene} max] pos. diffs,"
                        f" {nopen_diff} [1 max] opened)"
                    )
                    continue

                out[scene].append(task_spec_dict)
                print(f"{episode_seed_string} SUCCESS")
                break
    return out


def rearrangement_datagen_worker(
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    scene_to_obj_name_to_avoid_positions: Optional[
        Dict[str, Dict[str, np.ndarray]]
    ] = None,
):
    ptitle("Rearrange Datagen Worker")
    env = RearrangeTHOREnvironment(
        force_cache_reset=True, controller_kwargs={"commit_id": THOR_COMMIT_ID}
    )

    while True:
        try:
            scene, stage, seed = input_queue.get(timeout=2)
        except queue.Empty:
            break
        data = generate_rearrangements_for_scenes(
            stage_seed=seed,
            stage_scenes=[scene],
            env=env,
            object_types_to_not_move=OBJECT_TYPES_TO_NOT_MOVE,
            obj_name_to_avoid_positions=None
            if scene_to_obj_name_to_avoid_positions is None
            else scene_to_obj_name_to_avoid_positions[scene],
        )
        output_queue.put((scene, stage, data[scene]))


def get_scene_to_obj_name_to_seen_positions():
    scene_to_task_spec_dicts = compress_pickle.load(
        os.path.join(STARTER_DATA_DIR, f"train.pkl.gz")
    )
    assert len(scene_to_task_spec_dicts) == 80 and all(
        len(v) == 50 for v in scene_to_task_spec_dicts.values()
    )

    scene_to_obj_name_to_positions = {}
    for scene in tqdm.tqdm(scene_to_task_spec_dicts):
        obj_name_to_positions = defaultdict(lambda: [])
        for task_spec_dict in scene_to_task_spec_dicts[scene]:
            for od in task_spec_dict["openable_data"]:
                obj_name_to_positions[od["name"]].extend(
                    (od["start_openness"], od["target_openness"])
                )

            for sp, tp in zip(
                task_spec_dict["starting_poses"], task_spec_dict["target_poses"]
            ):
                assert sp["name"] == tp["name"]

                position_dist = IThorEnvironment.position_dist(
                    sp["position"], tp["position"]
                )
                rotation_dist = IThorEnvironment.angle_between_rotations(
                    sp["rotation"], tp["rotation"]
                )
                if position_dist >= 1e-2 or rotation_dist >= 5:
                    obj_name_to_positions[sp["name"]].append(
                        [sp["position"][k] for k in ["x", "y", "z"]]
                    )
                    obj_name_to_positions[sp["name"]].append(
                        [tp["position"][k] for k in ["x", "y", "z"]]
                    )
        scene_to_obj_name_to_positions[scene] = {
            k: np.array(v) for k, v in obj_name_to_positions.items()
        }

    return scene_to_obj_name_to_positions


def main():
    ptitle("Rearrange Datagen Manager")

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true", default=False)
    parser.add_argument("--train_unseen", "-t", action="store_true", default=False)
    args = parser.parse_args()

    nprocesses = (
        max((3 * mp.cpu_count()) // 4, 1) if platform.system() == "Linux" else 1
    )

    stage_seeds = get_random_seeds()

    scene_to_obj_name_to_avoid_positions = None
    if args.debug:
        stage_to_scenes = {"debug": ["FloorPlan428"]}
    elif args.train_unseen:
        stage_to_scenes = {"train_unseen": get_scenes("train")}
        scene_to_obj_name_to_avoid_positions = get_scene_to_obj_name_to_seen_positions()
    else:
        stage_to_scenes = {
            stage: get_scenes(stage) for stage in ("train", "val", "test")
        }

    os.makedirs(STARTER_DATA_DIR, exist_ok=True)

    stage_to_scene_to_rearrangements = {stage: {} for stage in stage_to_scenes}
    for stage in stage_to_scenes:
        path = os.path.join(STARTER_DATA_DIR, f"{stage}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                stage_to_scene_to_rearrangements[stage] = json.load(f)

    send_queue = mp.Queue()
    num_scenes_to_run = 0
    for stage in stage_to_scenes:
        for scene in stage_to_scenes[stage]:
            if scene not in stage_to_scene_to_rearrangements[stage]:
                num_scenes_to_run += 1
                send_queue.put((scene, stage, stage_seeds[stage]))

    receive_queue = mp.Queue()
    processes = []
    for i in range(nprocesses):
        p = mp.Process(
            target=rearrangement_datagen_worker,
            kwargs=dict(
                input_queue=send_queue,
                output_queue=receive_queue,
                scene_to_obj_name_to_avoid_positions=scene_to_obj_name_to_avoid_positions,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.5)

    num_received = 0
    while num_scenes_to_run > num_received:
        try:
            scene, stage, data = receive_queue.get(timeout=1)
            num_received += 1
        except queue.Empty:
            continue

        print(f"Saving {scene}")

        scene_to_rearrangements = stage_to_scene_to_rearrangements[stage]
        if scene not in scene_to_rearrangements:
            scene_to_rearrangements[scene] = []

        scene_to_rearrangements[scene].extend(data)

        with open(os.path.join(STARTER_DATA_DIR, f"{stage}.json"), "w") as f:
            json.dump(scene_to_rearrangements, f)

        compress_pickle.dump(
            obj=scene_to_rearrangements,
            path=os.path.join(STARTER_DATA_DIR, f"{stage}.pkl.gz"),
            pickler_kwargs={"protocol": 4,},  # Backwards compatible with python 3.6
        )

    for p in processes:
        try:
            p.join(timeout=1)
        except:
            pass


if __name__ == "__main__":
    main()
