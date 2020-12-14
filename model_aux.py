import math
import logging
import numpy as np

import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F
import math

class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, input):
        return input

class distLinear(nn.Module):
    def __init__(self, indim, outdim, weight=None, scale=5):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        if weight is not None:
            self.L.weight.data = Variable(weight)
        self.class_wise_learnable_norm = False  #See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm

        if outdim <=200:
            self.scale_factor = scale; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor * (cos_dist)

        return scores

class auxillary_classifier2(nn.Module):
    def __init__(self, feature_size=256,
                 input_features=256, in_size=32,
                 num_classes=10,n_lin=0,mlp_layers=0,  batchn=True):
        super(auxillary_classifier2, self).__init__()
        self.n_lin=n_lin
        self.in_size=in_size

        if n_lin==0:
            feature_size = input_features

        feature_size = input_features
        self.blocks = []
        for n in range(self.n_lin):
            if n==0:
                input_features = input_features
            else:
                input_features = feature_size

            if batchn:
                bn_temp = nn.BatchNorm2d(feature_size)
            else:
                bn_temp = identity()

            conv = nn.Conv2d(input_features, feature_size,
                             kernel_size=1, stride=1, padding=0, bias=False)
            self.blocks.append(nn.Sequential(conv,bn_temp))

        self.blocks = nn.ModuleList(self.blocks)
        if batchn:
            self.bn = nn.BatchNorm2d(feature_size)
        else:
            self.bn = identity()  # Identity

        if mlp_layers > 0:

            mlp_feat = feature_size * (2) * (2)
            layers = []

            for l in range(mlp_layers):
                if l==0:
                    in_feat = feature_size*4
                    mlp_feat = mlp_feat
                else:
                    in_feat = mlp_feat

                if batchn:
                    bn_temp = nn.BatchNorm1d(mlp_feat)
                else:
                    bn_temp = identity()

                layers +=[nn.Linear(in_feat,mlp_feat),
                              bn_temp,nn.ReLU(True)]
            layers += [nn.Linear(mlp_feat,num_classes)]
            self.classifier = nn.Sequential(*layers)
            self.mlp = True

        else:
            self.mlp = False
            self.classifier = nn.Linear(feature_size*2*2, num_classes)#distLinear(feature_size*4,num_classes)#nn.Linear(feature_size*2*2, num_classes)


    def forward(self, x):
        out = x
        #First reduce the size by 16
        out = F.adaptive_avg_pool2d(out,(math.ceil(self.in_size/4),math.ceil(self.in_size/4)))

        for n in range(self.n_lin):
            out = self.blocks[n](out)
            out = F.relu(out)

        out = F.adaptive_avg_pool2d(out, (2,2))
        if not self.mlp:
            out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, channels, bn=False):
        super(ResBlock, self).__init__()

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)



# Classifiers
# -----------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

class rep(nn.Module):
    def __init__(self, blocks):
        super(rep, self).__init__()
        self.blocks = blocks
    def forward(self, x, n, upto=False):
        # if upto = True we forward from the input to output of layer n
        # if upto = False we forward just through layer n

        if upto:
            for i in range(n+1):
                x = self.forward(x,i,upto=False)
            return x
        out = self.blocks[n](x)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)






class ResNet_Modular(nn.Module):

    def __init__(self, block, layers, num_classes=1000, nf=20, split_points=2, input_size=(1,32,32), **kwargs):
        super(ResNet_Modular, self).__init__()
        self.inplanes = nf*1
        self.avg_size = input_size[1]
        self.in_size = input_size[1]

        self.blocks = nn.ModuleList([])
        self.base_blocks = nn.ModuleList([])
        self.auxillary_nets = nn.ModuleList([])
        self.auxillary_size_tracker = []

        ## Initial layer
        layer = [conv3x3(3, nf * 1), nn.BatchNorm2d(nf*1), nn.ReLU(inplace=True)]

        self.base_blocks.append(nn.Sequential(*layer))
        self.auxillary_size_tracker.append((self.in_size,self.inplanes))

        self._make_layer(block, nf*1, layers[0], **kwargs)
        self._make_layer(block, nf*2, layers[1], stride=2, **kwargs)
        self._make_layer(block, nf*4, layers[2], stride=2, **kwargs)
        self._make_layer(block, nf*8, layers[3], stride=2, **kwargs)

        len_layers = len(self.base_blocks)
        split_depth = math.ceil(len(self.base_blocks) / split_points)

        for splits_id in range(split_points):
            left_idx = splits_id * split_depth
            right_idx = (splits_id + 1) * split_depth
            if right_idx > len_layers:
                right_idx = len_layers
            self.blocks.append(nn.Sequential(*self.base_blocks[left_idx:right_idx]))
            in_size, planes = self.auxillary_size_tracker[right_idx-1]
            self.auxillary_nets.append(
                auxillary_classifier2(in_size=in_size,
                                      n_lin=0, feature_size=planes,
                                      input_features=planes, mlp_layers=2,
                                      batchn=True, num_classes=num_classes)
            )
            self.auxillary_nets[-1].classifier.scale_factor=5.0*(splits_id+1)/5.0
            print(self.auxillary_nets[-1].classifier.scale_factor)

        last_hid = nf * 8  if input_size[1] in [8,16,21,32,42] else 640
    #    self.linear = nn.Linear(last_hid, num_classes)

        self.auxillary_nets[len(self.auxillary_nets)-1] = nn.Sequential(
                nn.AvgPool2d((4,4)),
                View(last_hid ),
               # distLinear(last_hid * block.expansion, num_classes, scale=5.0)
                nn.Linear(last_hid , num_classes)
            )

        self.main_cnn = rep(self.blocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            self.avg_size = int(self.avg_size/2)
            self.in_size =  int(self.in_size/2)


        self.base_blocks.append(block(self.inplanes, planes, stride, downsample))
        self.auxillary_size_tracker.append((self.in_size,planes*block.expansion))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            self.base_blocks.append(block(self.inplanes, planes))
            self.auxillary_size_tracker.append((self.in_size,planes*block.expansion))

    def forward(self, representation, n, upto=False):
        representation = self.main_cnn.forward(representation, n, upto=upto)
        outputs = self.auxillary_nets[n](representation)
        return outputs, representation

def ResNet18(nclasses, nf=20, split_points=2, input_size=(3, 32, 32)):
    return ResNet_Modular(BasicBlock, [2, 2, 2, 2], nclasses, nf,
                          split_points=split_points, input_size=input_size)