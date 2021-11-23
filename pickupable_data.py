import os
import json
import pickle
import random


data_dir = '/home/apoorvk/nfs/clip-embodied-ai/datasets/pickupable_objects/edge_full'
output_dir = '/home/apoorvk/nfs/clip-embodied-ai/datasets/pickupable_objects'

def thor_id_to_class(thor_id):
    if '_' not in thor_id:
        return thor_id
    return thor_id[:thor_id.index('_')]


object_superset = []
for split in ['train', 'val', 'test']:
    boxes_filepath = os.path.join(data_dir, f'{split}_boxes.json')
    with open(boxes_filepath) as f:
        boxes = json.load(f)
    labels_filepath = os.path.join(data_dir, f'{split}_boxes_pickupable.json')
    with open(labels_filepath, 'r') as f:
        labels = json.load(f)
    data = []
    for image in boxes.keys():
        for o in boxes[image].keys():
            object_superset.append(thor_id_to_class(o))
object_superset = sorted(list(set(object_superset)))


for split in ['train', 'val', 'test']:
    boxes_filepath = os.path.join(data_dir, f'{split}_boxes.json')
    with open(boxes_filepath) as f:
        boxes = json.load(f)

    labels_filepath = os.path.join(data_dir, f'{split}_boxes_pickupable.json')
    with open(labels_filepath, 'r') as f:
        labels = json.load(f)

    data = [[] for i in range(111)]

    for image in boxes.keys():
        objects = set([thor_id_to_class(o) for o in boxes[image].keys()])
        pickupable_objects = set([thor_id_to_class(o) for o in labels[image]])
        for obj in objects:
            obj_id = object_superset.index(obj)
            data[obj_id].append((image, obj_id, obj in pickupable_objects))

    for i in range(111):
        positives = [d for d in data[i] if d[2] == 1]
        negatives = [d for d in data[i] if d[2] == 0][:len(positives)]
        class_data = negatives + positives
        data[i] = class_data

    data_all = []
    for i in range(111):
        for j in data[i]:
            data_all.append(j)
    random.shuffle(data_all)

    pickle.dump(
        data_all,
        open(os.path.join(output_dir, f"{split}.pkl"), 'wb')
    )
