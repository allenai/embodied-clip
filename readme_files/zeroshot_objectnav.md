## Zero-shot ObjectNav

We have [modified](https://github.com/allenai/embodied-clip/compare/allenact...zeroshot-objectnav) the [RoboTHOR ObjectNav codebase](https://github.com/allenai/embodied-clip/tree/allenact) to enable the zero-shot experiment in our paper.

To install, please follow the [instructions for RoboTHOR ObjectNav](baselines_robothor_objectnav.md), but instead clone the [`zeroshot-objectnav` branch](https://github.com/allenai/embodied-clip/tree/zeroshot-objectnav):

```bash
git clone -b zeroshot-objectnav --single-branch git@github.com:allenai/embodied-clip.git embclip-zeroshot
cd embclip-zeroshot

[ ... ]
```

### Training

```
PYTHONPATH=. python allenact/main.py -o storage/embclip-zeroshot -b projects/objectnav_baselines/experiments/robothor/clip zeroshot_objectnav_robothor_rgb_clipresnet50gru_ddppo
```

### Evaluating

We run the same experiment config in eval mode (for validation), but with the original set of 12 object types.

```
PYTHONPATH=. python allenact/main.py -o storage/embclip-zeroshot -b projects/objectnav_baselines/experiments/robothor/clip zeroshot_objectnav_robothor_rgb_clipresnet50gru_ddppo_eval --eval
```

We provide the weights for the model in our paper [here](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/exp_Zeroshot-ObjectNav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000055057640.pt).
