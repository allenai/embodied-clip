import os
from glob import glob

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision import models
import clip

from constants import target_objects


data_dir = os.path.expanduser('~/nfs/clip-embodied-ai/datasets/ithor_scenes')


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if "BatchNorm" in type(module).__name__:
            module.momentum = 0.0
    model.eval()
    return model

## Load ResNet model
resnet_preprocess = T.Compose([
    T.Resize(size=224, interpolation=Image.BICUBIC),
    T.CenterCrop(size=(224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

resnet_model = models.resnet50(pretrained=True)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-2])
resnet_model = freeze_model(resnet_model)
resnet_model = resnet_model.cuda()

resnet_pool = nn.Sequential(
    nn.AdaptiveAvgPool2d(output_size=(1,1)),
    nn.Flatten()
)

## Load CLIP model
clip_model, clip_preprocess = clip.load('RN50', device=torch.device('cuda'))

clip_model = clip_model.visual
clip_model = freeze_model(clip_model)

clip_pool = clip_model.attnpool
clip_avgpool = nn.Sequential(
    nn.AdaptiveAvgPool2d(output_size=(1,1)),
    nn.Flatten()
)
clip_model.attnpool = nn.Identity()
clip_avgpool = freeze_model(clip_avgpool)


def class_mask(semantic_frame, class_color):
    if class_color is None:
        return np.zeros(semantic_frame.shape[:2], dtype=bool)
    mask = np.all(semantic_frame == class_color, axis=-1)
    return mask

def obj_presence(class_masks):
    return class_masks.sum(axis=(1,2)) > 0

def grid_bboxes(image_shape, grid_sizes):
    for i in range(grid_sizes[0]):
        for j in range(grid_sizes[1]):
            yield (
                int(i * image_shape[0] / grid_sizes[0]),
                int((i+1) * image_shape[0] / grid_sizes[0]),
                int(j * image_shape[1] / grid_sizes[1]),
                int((j+1) * image_shape[1] / grid_sizes[1]),
            )


for split in ['train', 'val', 'test']:

    features = {}
    for scene in glob(os.path.join(data_dir, split, '*.npy')):
        scene_name = os.path.splitext(os.path.basename(scene))[0]
        print(scene_name)
        features[scene_name] = []
        data = np.load(scene, allow_pickle=True)
        for point in data:
            frame = Image.fromarray(point['frame'])

            resnet_input = resnet_preprocess(frame).unsqueeze(0).cuda()
            resnet_features = resnet_model(resnet_input)
            
            resnet_features_conv = resnet_features[0].cpu()
            resnet_features_avgpool = resnet_pool(resnet_features)[0].cpu()

            clip_input = clip_preprocess(frame).unsqueeze(0).cuda()
            clip_features = clip_model(clip_input)

            clip_features_conv = clip_features.float()[0].cpu()
            clip_features_attnpool = clip_pool(clip_features).float()[0].cpu()
            clip_features_avgpool = clip_avgpool(clip_features.float())[0].cpu()

            class_masks = np.array([
                class_mask(
                    point['semantic_frame'],
                    point['object_id_to_color'].get(o, None)
                ) for o in target_objects
            ])

            object_presence = torch.tensor(obj_presence(class_masks), dtype=int, device=torch.device('cpu'))
            object_presence_grid = torch.tensor(
                [obj_presence(class_masks[:, y1:y2, x1:x2]) for (y1, y2, x1, x2) in grid_bboxes(class_masks.shape[1:3], (3, 3))],
                dtype=int,
                device=torch.device('cpu')
            )

            features[scene_name].append({
                'rn50_imagenet_conv' : resnet_features_conv,
                'rn50_imagenet_avgpool' : resnet_features_avgpool,
                'clip_conv' : clip_features_conv,
                'clip_attnpool' : clip_features_attnpool,
                'clip_avgpool' : clip_features_avgpool,
                'object_presence' : object_presence,
                'object_presence_grid' : object_presence_grid,
                'valid_moves_forward' : point['valid_moves_forward']
            })

    torch.save(features, os.path.join(data_dir, f"{split}.pt"))
