import json
import os
from collections import defaultdict

import compress_pickle
from allenact.utils.misc_utils import partition_sequence

from rearrange.constants import STARTER_DATA_DIR


def combine(task_limit_for_train: int = 10000):
    stages = ("train", "val", "test")

    all_data = defaultdict(lambda: [])
    for stage in stages:
        print(stage)
        data_path = os.path.join(STARTER_DATA_DIR, f"{stage}.pkl.gz")

        if not os.path.exists(data_path):
            raise RuntimeError(f"No data at path {data_path}")

        data = compress_pickle.load(path=data_path)
        max_per_scene = task_limit_for_train if "train" in stage else 10000
        count = 0
        for scene in data:
            assert len(data[scene]) == 50

            for index, task_spec_dict in enumerate(data[scene]):
                task_spec_dict["scene"] = scene
                task_spec_dict["index"] = index
                task_spec_dict["stage"] = stage

            pieces_per_part = max_per_scene // 5  # 5 hardnesses
            parts = partition_sequence(data[scene], 5)
            all_together = sum([part[:pieces_per_part] for part in parts], [])

            count += len(all_together)
            all_data[scene].extend(all_together)

        print(count)
    all_data = dict(all_data)
    with open(os.path.join(STARTER_DATA_DIR, f"combined.json"), "w") as f:
        json.dump(all_data, f)

    compress_pickle.dump(
        obj=all_data,
        path=os.path.join(STARTER_DATA_DIR, f"combined.pkl.gz"),
        pickler_kwargs={"protocol": 4,},  # Backwards compatible with python 3.6
    )


if __name__ == "__main__":
    combine(10)
