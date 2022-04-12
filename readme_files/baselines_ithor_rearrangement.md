iTHOR Rearrangement
==============================

Please refer to the README in the [rearrangement branch](https://github.com/allenai/embodied-clip/tree/rearrangement), which includes detailed instructions on installation, training, and testing.

Use the following models:
- ImageNet (1-Phase ResNet50 IL)
  - [Experiment config](https://github.com/allenai/embodied-clip/blob/rearrangement/baseline_configs/one_phase/one_phase_rgb_resnet50_dagger.py)
  - [Model weights](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/rearrangement/one-phase/exp_OnePhaseRGBImageNetResNet50Dagger_40proc_aws0__stage_00__steps_000070075580.pt)
- CLIP (1-Phase Embodied CLIP ResNet50 IL)
  - [Experiment config](https://github.com/allenai/embodied-clip/blob/rearrangement/baseline_configs/one_phase/one_phase_rgb_clipresnet50_dagger.py)
  - [Model weights](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/rearrangement/one-phase/exp_OnePhaseRGBClipResNet50Dagger_40proc__stage_00__steps_000065083050.pt)
