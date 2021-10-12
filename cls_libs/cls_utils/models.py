# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torchvision
     
class ResNet_cls(nn.Module):

    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)

    def _get_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        outs = [x]
        x = self.resnet.layer1(x)
        outs.append(x)
        x = self.resnet.layer2(x)
        outs.append(x)
        x = self.resnet.layer3(x)
        outs.append(x)
        x = self.resnet.layer4(x)
        outs.append(x)

        return tuple(outs)


    def get_features(self, x, layers):
        feats = self._get_features(x)
        if len(layers) == 1: 
            if layers[0] < 5:
                return [feats[layers[0]]]
            else:
                x=feats[-1]
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return [x]
        else:
            return [feats[i] for i in layers]


