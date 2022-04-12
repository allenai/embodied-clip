## ImageNet vs. ObjectNav

For this experiment, we trained two additional [RoboTHOR ObjectNav](baselines_robothor_objectnav.md) agents.

- ImageNet ResNet-18
  - [Experiment Config](https://github.com/allenai/embodied-clip/blob/allenact/projects/objectnav_baselines/experiments/robothor/objectnav_robothor_rgb_resnet18gru_ddppo.py)
  - [Model Checkpoint](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/exp_Objectnav-RoboTHOR-RGB-ResNet18GRU-DDPPO__stage_00__steps_000180222019.pt)
- CLIP ResNet-50x16
  - [Experiment Config](https://github.com/allenai/embodied-clip/blob/allenact/projects/objectnav_baselines/experiments/robothor/clip/objectnav_robothor_rgb_clipresnet50x16gru_ddppo.py)
  - [Model Checkpoint](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/exp_Objectnav-RoboTHOR-RGB-ClipResNet50x16GRU-DDPPO__stage_00__steps_000160088907.pt)
