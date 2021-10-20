#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

set -x
python -u -m torch.distributed.run \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    habitat_baselines/run.py \
    --exp-config configs/challenge-2021/ddppo_objectnav.yaml \
    --run-type train
