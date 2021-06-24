import json
import os
import pickle
from collections import defaultdict

import compress_pickle

from rearrange.constants import STARTER_DATA_DIR


def combine():
    stages = ("train", "train_unseen", "val", "test")

    all_data = defaultdict(lambda: [])
    for stage in stages:
        print(stage)
        data_path = os.path.join(STARTER_DATA_DIR, f"{stage}.pkl.gz")

        if not os.path.exists(data_path):
            raise RuntimeError(f"No data at path {data_path}")

        data = compress_pickle.load(path=data_path)
        max_per_scene = 15 if "train" in stage else 10000
        count = 0
        for scene in data:
            for ind, task_spec_dict in enumerate(data[scene][:max_per_scene]):
                count += 1

                task_spec_dict["scene"] = scene
                task_spec_dict["index"] = ind
                task_spec_dict["stage"] = stage

                all_data[scene].append(task_spec_dict)

        print(count)
    all_data = dict(all_data)
    with open(os.path.join(STARTER_DATA_DIR, f"combined.json"), "w") as f:
        json.dump(all_data, f)

    compress_pickle.dump(
        obj=all_data,
        path=os.path.join(STARTER_DATA_DIR, f"combined.pkl.gz"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


if __name__ == "__main__":
    combine()
