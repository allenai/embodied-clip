Primitive Probing
==============================

The code present in the `main` branch can be used to replicate our experiments for probing semantic and geometric navigational primitives with linear classifiers.

## Installation

```bash
git clone --single-branch git@github.com:allenai/embodied-clip.git
cd embodied-clip/primitive_probing

conda env create --name embclip
conda activate embclip

python -c "from torchvision import models; models.resnet50(pretrained=True)"
python -c "import clip; clip.load('RN50')"
```

## Preparing data

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

## Training models

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
