## Zero-shot ObjectNav

We have [modified](https://github.com/allenai/embodied-clip/compare/allenact...zeroshot-objectnav) the [RoboTHOR ObjectNav codebase](https://github.com/allenai/embodied-clip/tree/allenact) to enable the zero-shot experiment in our paper.

To install, please follow the [instructions for RoboTHOR ObjectNav](baselines_robothor_objectnav.md), but instead clone the [`zeroshot-objectnav` branch](https://github.com/allenai/embodied-clip/tree/zeroshot-objectnav):

```bash
git clone -b zeroshot-objectnav --single-branch https://github.com/allenai/embodied-clip.git embclip-zeroshot
cd embclip-zeroshot

[ ... ]
```

### Training

```
PYTHONPATH=. python allenact/main.py -o storage/embclip-zeroshot -b projects/objectnav_baselines/experiments/robothor/clip zeroshot_objectnav_robothor_rgb_clipresnet50gru_ddppo
```

### Evaluating

We run the same experiment config in eval mode (for validation), but with the original set of 12 object types.

```bash
export CKPT_PATH=path/to/model.pt

PYTHONPATH=. python allenact/main.py -o storage/embclip-zeroshot -c $CKPT_PATH -b projects/objectnav_baselines/experiments/robothor/clip zeroshot_objectnav_robothor_rgb_clipresnet50gru_ddppo_eval --eval
```

```python
SEEN_OBJECTS = ["AlarmClock", "BaseballBat", "Bowl", "GarbageCan", "Laptop", "Mug", "SprayBottle", "Vase"]
UNSEEN_OBJECTS = ["Apple", "BasketBall", "HousePlant", "Television"]

def compute_scores(metrics_file, obj_type='Apple'):
    ''' Function for computing average success metrics per object type. '''

    metrics = json.load(open(metrics_file, 'r'))

    episodes = [ep in metrics[0]['tasks'] if ep['task_info']['object_type'] == OBJ_TYPE]

    success = [ep['success'] for ep in episodes]
    success = sum(success) / len(success)

    spl = [ep['spl'] for ep in episodes]
    spl = sum(spl) / len(spl)

    return success, spl
```

We provide the weights for the model in our paper [here](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/exp_Zeroshot-ObjectNav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000055057640.pt).
