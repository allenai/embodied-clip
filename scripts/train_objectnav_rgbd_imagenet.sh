#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

set -x
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $NUM_GPUS \
    habitat_baselines/run.py \
    --exp-config habitat_baselines/config/objectnav/ddppo_objectnav_rgbd_imagenet.yaml \
    --run-type train
