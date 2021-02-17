import random
from collections import defaultdict
from typing import List, Dict, Set, Union

from ai2thor.controller import Controller

from datagen.datagen_constants import OBJECT_TYPES_THAT_CAN_HAVE_IDENTICAL_MESHES


def get_scenes(stage: str) -> List[str]:
    """Returns a list of iTHOR scene names for each stage."""
    assert stage in {"train", "train_unseen", "val", "valid", "test", "all"}

    # [1-20] for train, [21-25] for val, [26-30] for test
    if stage in ["train", "train_unseen"]:
        scene_nums = range(1, 21)
    elif stage in ["val", "valid"]:
        scene_nums = range(21, 26)
    elif stage == "test":
        scene_nums = range(26, 31)
    elif stage == "all":
        scene_nums = range(1, 31)
    else:
        raise NotImplementedError

    kitchens = [f"FloorPlan{i}" for i in scene_nums]
    living_rooms = [f"FloorPlan{200+i}" for i in scene_nums]
    bedrooms = [f"FloorPlan{300+i}" for i in scene_nums]
    bathrooms = [f"FloorPlan{400+i}" for i in scene_nums]
    return kitchens + living_rooms + bedrooms + bathrooms


def open_objs(
    objects_to_open: List[dict], controller: Controller
) -> Dict[int, Union[float, None]]:
    """Opens up the chosen pickupable objects if they're openable."""
    out: Dict[int, Union[float, None]] = defaultdict(lambda: None)
    for obj in objects_to_open:
        last_openness = obj["openness"]
        new_openness = last_openness
        while abs(last_openness - new_openness) <= 0.2:
            new_openness = random.random()

        controller.step(
            "OpenObject",
            objectId=obj["objectId"],
            moveMagnitude=new_openness,
            forceAction=True,
        )
        out[obj["name"]] = new_openness
    return out


def get_object_ids_to_not_move_from_object_types(
    controller: Controller, object_types: Set[str]
) -> List[str]:
    object_types = set(object_types)
    return [
        o["objectId"]
        for o in controller.last_event.metadata["objects"]
        if o["objectType"] in object_types
    ]


def remove_objects_until_all_have_identical_meshes(controller: Controller):
    obj_type_to_obj_list = defaultdict(lambda: [])
    for obj in controller.last_event.metadata["objects"]:
        obj_type_to_obj_list[obj["objectType"]].append(obj)

    for obj_type in OBJECT_TYPES_THAT_CAN_HAVE_IDENTICAL_MESHES:
        objs_of_type = list(
            sorted(obj_type_to_obj_list[obj_type], key=lambda x: x["name"])
        )
        random.shuffle(objs_of_type)
        objs_to_remove = objs_of_type[:-1]
        for obj_to_remove in objs_to_remove:
            obj_to_remove_name = obj_to_remove["name"]
            obj_id_to_remove = next(
                obj["objectId"]
                for obj in controller.last_event.metadata["objects"]
                if obj["name"] == obj_to_remove_name
            )
            controller.step("RemoveFromScene", objectId=obj_id_to_remove)
            if not controller.last_event.metadata["lastActionSuccess"]:
                return False
    return True
