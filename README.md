# Embodied CLIP

Official repository for [Simple but Effective: CLIP Embeddings for Embodied AI](https://arxiv.org/abs/2111.09888).

We present competitive performance on navigation-heavy tasks in Embodied AI using frozen visual representations from [CLIP](https://github.com/openai/CLIP). This repository includes all code and pretrained models necessary to replicate the experiments in our paper:

- Baselines
  - [RoboTHOR ObjectNav](#robothor-objectnav) (Sec. 4.1)
  - [iTHOR Rearrangement](#ithor-rearrangement) (Sec. 4.2)
  - [Habitat ObjectNav](#navigation-in-habitat) (Sec. 4.3)
  - [Habitat PointNav](#navigation-in-habitat) (Sec. 4.4)
- [Probing for Navigational Primitives](#primitive-probing) (Sec. 5)
- [ImageNet Acc vs. ObjectNav Success](#imagenet-vs-objectnav) (Sec. 6)
- [Zero-shot ObjectNav in RoboTHOR](#zero-shot-objectnav) (Sec. 7)

We have included forks of other repositories as branches of this repository, as we find this is a convenient way to centralize our experiments and track changes across codebases.

---

## RoboTHOR ObjectNav

### Installation

We've included instructions for installing the full AllenAct library (modifiable) with conda for [our branch](https://github.com/allenai/embodied-clip/tree/allenact), although you can also use the [official AllenAct repo (v0.5.0)](https://github.com/allenai/allenact/tree/v0.5.0) or perhaps newer.

```bash
git clone -b allenact --single-branch git@github.com:allenai/embodied-clip.git embclip-allenact
cd embclip-allenact

export EMBCLIP_ENV_NAME=embclip-allenact
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${EMBCLIP_ENV_NAME}/pipsrc"
conda env create --file ./conda/environment-base.yml --name $EMBCLIP_ENV_NAME
conda activate $EMBCLIP_ENV_NAME

# Install the appropriate cudatoolkit
conda env update --file ./conda/environment-<CUDA_VERSION>.yml --name $EMBCLIP_ENV_NAME
# OR for cpu mode
conda env update --file ./conda/environment-cpu.yml --name $EMBCLIP_ENV_NAME

# Install RoboTHOR and CLIP plugins
conda env update --file allenact_plugins/robothor_plugin/extra_environment.yml --name $EMBCLIP_ENV_NAME
conda env update --file allenact_plugins/clip_plugin/extra_environment.yml --name $EMBCLIP_ENV_NAME

# Download RoboTHOR dataset
bash datasets/download_navigation_datasets.sh robothor-objectnav

# Download pretrained ImageNet and CLIP visual encoders
python -c "from torchvision import models; models.resnet50(pretrained=True)"
python -c "import clip; clip.load('RN50')"
```

Please refer to the [official AllenAct installation instructions](https://allenact.org/installation/installation-allenact) for more details.

### Training

```bash
# ImageNet
PYTHONPATH=. python allenact/main.py -o storage/objectnav-robothor-rgb-imagenet-rn50 -b projects/objectnav_baselines/experiments/robothor objectnav_robothor_rgb_resnet50gru_ddppo

# CLIP
PYTHONPATH=. python allenact/main.py -o storage/objectnav-robothor-rgb-clip-rn50 -b projects/objectnav_baselines/experiments/robothor/clip objectnav_robothor_rgb_clipresnet50gru_ddppo
```

### Using pretrained models

```bash
# ImageNet
curl -o pretrained_model_ckpts/objectnav-robothor-imagenet-rn50.195M.pt https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/exp_Objectnav-RoboTHOR-RGB-ResNet50GRU-DDPPO__stage_00__steps_000195242243.pt

# CLIP
curl -o pretrained_model_ckpts/objectnav-robothor-clip-rn50.130M.pt https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/exp_Objectnav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000130091717.pt
```

You can use these models with the `python allenact/main.py` arguments `-c pretrained_model_ckpts/objectnav-robothor-imagenet-rn50.195M.pt` or `-c pretrained_model_ckpts/objectnav-robothor-clip-rn50.130M.pt`.

### Evaluating 

Simply append the `--eval` argument to the above `python allenact/main.py` commands.

---

## iTHOR Rearrangement

Please refer to the README in the [rearrangement branch](https://github.com/allenai/embodied-clip/tree/rearrangement), which includes detailed instructions on installation, training, and testing.

Use the following models:
- ImageNet (1-Phase ResNet50 IL)
  - [Experiment config](https://github.com/allenai/embodied-clip/blob/rearrangement/baseline_configs/one_phase/one_phase_rgb_resnet50_dagger.py)
  - [Model weights](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/rearrangement/one-phase/exp_OnePhaseRGBImageNetResNet50Dagger_40proc_aws0__stage_00__steps_000070075580.pt)
- CLIP (1-Phase Embodied CLIP ResNet50 IL)
  - [Experiment config](https://github.com/allenai/embodied-clip/blob/rearrangement/baseline_configs/one_phase/one_phase_rgb_clipresnet50_dagger.py)
  - [Model weights](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/rearrangement/one-phase/exp_OnePhaseRGBClipResNet50Dagger_40proc__stage_00__steps_000065083050.pt)

---

## Navigation in Habitat

Please refer to the README in the [habitat branch](https://github.com/allenai/embodied-clip/tree/habitat), which has detailed instructions on installing Habitat and training/evaluating our models.

---

## ImageNet vs. ObjectNav

For this experiment, we trained two RoboTHOR ObjectNav agents in addition to [those above](#robothor-objectnav).

- ImageNet ResNet-18
  - [Experiment Config](https://github.com/allenai/embodied-clip/blob/allenact/projects/objectnav_baselines/experiments/robothor/objectnav_robothor_rgb_resnet18gru_ddppo.py)
  - [Model Checkpoint](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/exp_Objectnav-RoboTHOR-RGB-ResNet18GRU-DDPPO__stage_00__steps_000180222019.pt)
- CLIP ResNet-50x16
  - [Experiment Config](https://github.com/allenai/embodied-clip/blob/allenact/projects/objectnav_baselines/experiments/robothor/clip/objectnav_robothor_rgb_clipresnet50x16gru_ddppo.py)
  - [Model Checkpoint](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/exp_Objectnav-RoboTHOR-RGB-ClipResNet50x16GRU-DDPPO__stage_00__steps_000160088907.pt)

---

## Primitive Probing

The code present in this `main` branch can be used to replicate our experiments for probing semantic and geometric navigational primitives with linear classifiers.

### Installation

```bash
git clone --single-branch git@github.com:allenai/embodied-clip.git
cd embodied-clip
conda env create --name embclip
conda activate embclip

python -c "from torchvision import models; models.resnet50(pretrained=True)"
python -c "import clip; clip.load('RN50')"
```

### Preparing data

```bash
mkdir data

# Object Presence, Object Localization, Free Space
curl -o data/ithor_scenes.tar.gz https://prior-datasets.s3.us-east-2.amazonaws.com/embclip/ithor_scenes.tar.gz
tar xvzf data/ithor_scenes.tar.gz -C data
rm data/ithor_scenes.tar.gz

PYTHONPATH=. python generate_data/thor_image_features.py --data_dir data/ithor_scenes --output_dir data

# Reachability
curl -o data/datasets.tar.gz https://prior-datasets.s3.us-east-2.amazonaws.com/csr/datasets.tar.gz
tar xvzf data/datasets.tar.gz -C data --wildcards 'datasets/edge_full/*' --transform="s/datasets/CSR/"
rm data/datasets.tar.gz
curl -o data/CSR/edge_full/test_boxes_pickupable.json https://prior-datasets.s3.us-east-2.amazonaws.com/embclip/test_boxes_pickupable.json

PYTHONPATH=. python generate_data/reachable_metadata.py --data_dir data/CSR/edge_full --output_dir data
PYTHONPATH=. python generate_data/reachable_image_features.py --data_dir data/CSR/edge_full --output_dir data
```

### Training models

After preparing the data, you can train any of the models in our paper with the following code:

```bash
# EMB_TYPE: imagenet_avgpool, clip_avgpool, clip_attnpool
export EMB_TYPE=clip_avgpool
# PRED_TYPE: object_presence, object_localization, reachability, free_space
export PRED_TYPE=object_presence

python train.py --data-dir data --log-dir logs --embedding-type $EMB_TYPE --prediction-type $PRED_TYPE --gpus 1
```

To view training/testing logs from our runs:

```bash
tensorboard --logdir logs
```

---

## Zero-shot ObjectNav
