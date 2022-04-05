import os
from glob import glob
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision import models
import clip


image_list = glob('/home/apoorvk/nfs/clip-embodied-ai/datasets/reachable_objects/*.png')
output_file = '/home/apoorvk/nfs/clip-embodied-ai/datasets/reachable_objects_ft.pt'

### Loading models

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

### Image Processing

image_features = {}

for image_path in tqdm(image_list):
    frame = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    resnet_input = resnet_preprocess(frame).unsqueeze(0).cuda()
    resnet_features = resnet_model(resnet_input)

    resnet_features_conv = resnet_features[0].cpu()
    resnet_features_avgpool = resnet_pool(resnet_features)[0].cpu()

    clip_input = clip_preprocess(frame).unsqueeze(0).cuda()
    clip_features = clip_model(clip_input)

    clip_features_conv = clip_features.float()[0].cpu()
    clip_features_attnpool = clip_pool(clip_features).float()[0].cpu()
    clip_features_avgpool = clip_avgpool(clip_features.float())[0].cpu()

    image_features[image_name] = {
        'rn50_imagenet_conv' : resnet_features_conv,
        'rn50_imagenet_avgpool' : resnet_features_avgpool,
        'clip_conv' : clip_features_conv,
        'clip_attnpool' : clip_features_attnpool,
        'clip_avgpool' : clip_features_avgpool
    }

torch.save(image_features, output_file)