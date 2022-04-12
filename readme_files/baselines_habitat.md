Habitat ObjectNav & Pointnav
==============================

The [habitat branch](https://github.com/allenai/embodied-clip/tree/habitat) is a fork of [facebookresearch/habitat-lab](https://github.com/facebookresearch/habitat-lab/tree/ac4339ed7e9bafb5e45fee9c5cf68095f40edee2), modified for the Habitat ObjectNav (Sec 4.3) and PointNav (Sec 4.4) experiments in [Simple but Effective: CLIP Embeddings for Embodied AI](https://arxiv.org/abs/2111.09888). Please refer to the [README.md](https://github.com/allenai/embodied-clip/blob/habitat/README.md) for further details on Habitat. This code is sufficient to replicate the experiments in our paper.

## Installation

```bash
git clone -b habitat --single-branch git@github.com:allenai/embodied-clip.git embclip-habitat
cd embclip-habitat
```

1. You can install dependencies through the following conda environment. Please adjust the `cudatoolkit` version appropriately (must be compatible with `pytorch=1.7.1`) in `environment.yml` (line 10).

```bash
conda env create --name embclip-habitat
conda activate embclip-habitat
```

2. Download the test scenes data.

```bash
wget http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip
unzip habitat-test-scenes.zip && rm habitat-test-scenes.zip
```

3. Run the example script `python examples/example.py ` which in the end should print out number of steps agent took inside an environment (eg: `Episode finished after 2 steps.`). To verify that tests pass run `python setup.py test` which should print out a log about passed, skipped and failed tests. Finally, run `python examples/benchmark.py` to evaluate a forward only agent in a test environment downloaded in step-2.

## Downloading data

You should follow these instructions to request access to the [Matterport3D](https://github.com/facebookresearch/habitat-lab/blob/ac4339ed7e9bafb5e45fee9c5cf68095f40edee2/README.md#matterport3d) and [Gibson](https://github.com/facebookresearch/habitat-lab/blob/ac4339ed7e9bafb5e45fee9c5cf68095f40edee2/README.md#gibson) scene datasets. In particular, once you recieve access (with download links to `download_mp.py` and `gibson_habitat.zip`), run:

```bash
wget https://.../download_mp.py
# Only download the habitat task data, not the main dataset
python2 download_mp.py --task habitat -o data/scene_datasets/mp3d  # interactive script
unzip data/scene_datasets/mp3d/v1/tasks/mp3d_habitat.zip -d data/scene_datasets
rm -r data/scene_datasets/mp3d/v1
mv data/scene_datasets/README.txt data/scene_datasets/mp3d/
wget https://.../gibson_habitat.zip
unzip gibson_habitat.zip -d data/scene_datasets
rm gibson_habitat.zip
```

You should have produced data at `data/scene_datasets/mp3d` and `data/scene_datasets/gibson`.

You should also download the following task datasets:

```bash
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip
mkdir -p data/datasets/pointnav/gibson/v2
unzip pointnav_gibson_v2.zip -d data/datasets/pointnav/gibson/v2
rm pointnav_gibson_v2.zip
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip
mkdir -p data/datasets/objectnav/mp3d/v1
unzip objectnav_mp3d_v1.zip -d data/datasets/objectnav/mp3d/v1
rm objectnav_mp3d_v1.zip
```

## Training

```bash
export NUM_GPUS=8
export TASK=objectnav # {objectnav,pointnav}
export MODEL=clip     # {clip,imagenet}
GLOG_minloglevel=2 MAGNUM_LOG=quiet \
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $NUM_GPUS \
    habitat_baselines/run.py \
    --exp-config habitat_baselines/config/${TASK}/ddppo_${TASK}_rgb_${MODEL}.yaml \
    --run-type train
```

You can also train agents with depth (replace `rgb` with `rgbd` in exp-config, for ObjectNav only) and the DD-PPO baseline (exp_cfg: `habitat_baselines/config/objectnav/ddppo_objectnav_{rgb,rgbd}.yaml`), where the visual encoder is trained from scratch.

## Pre-trained models

We have provided (best validation) checkpoints for the CLIP and ImageNet agents listed in our paper.

```bash
mkdir pretrained_models
curl -o pretrained_models/objectnav-rgb-clip.175M.pth https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/habitat-objectnav-rgb-clip.175M.pth
curl -o pretrained_models/objectnav-rgb-imagenet.175M.pth https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/habitat-objectnav-rgb-imagenet.175M.pth
curl -o pretrained_models/pointnav-rgb-clip.250M.pth https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/habitat-pointnav-rgb-clip.250M.pth
curl -o pretrained_models/pointnav-rgb-imagenet.150M.pth https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/habitat-pointnav-rgb-imagenet.150M.pth
```

## Evaluation

```bash
export TASK=objectnav # {objectnav,pointnav}
export MODEL=clip     # {clip,imagenet}
python -u habitat_baselines/run.py \
    --run-type eval \
    --exp-config configs/eval/ddppo_${TASK}_rgb_${MODEL}.yaml
```
