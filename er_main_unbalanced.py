"""
Classification on CIFAR10 (ResNet)
==================================
Based on pytorch example for CIFAR10
"""


import torch.optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from kymatio import Scattering2D
import torch
import argparse
import kymatio.datasets as scattering_datasets
import torch.nn as nn
from numpy.random import RandomState
import numpy as np


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
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


class Scattering2dResNet(nn.Module):
    def __init__(self, in_channels,  k=2, n=4, num_classes=10,standard=True):
        super(Scattering2dResNet, self).__init__()
        self.inplanes = 16 * k
        self.ichannels = 16 * k

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, self.ichannels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.ichannels),
            nn.ReLU(True)
        )
        self.layer1 = self._make_layer(BasicBlock, 16 * k, n)
        self.standard = True


        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.standard:
            x = x.view(x.size(0), self.K, 8, 8)

        x = self.init_conv(x)

        if self.standard:
            x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class_count=torch.ones(10).cuda()
def train(model, device, train_loader, optimizer, epoch,mask_trick=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if mask_trick:
            present = np.array([0,1,2,3,4])
            ind = np.in1d(target.cpu().numpy(),present)
            if ind.any():
               # import ipdb;ipdb.set_trace()
                output2=output[ind]; target2=target[ind]
                output1=output[~ind];target1=target[~ind]
                mask = torch.zeros_like(output)
                mask[:, present] = 1
                # import ipdb;ipdb.set_trace()
               # logits[:, present]
                # logits = logits.masked_fill(mask == 0, -1e9)


                m = torch.zeros(10).byte()
                m[present] = 1
                #  import ipdb; ipdb.set_trace()

                    #   import ipdb; ipdb.set_trace()
                w = 1. / class_count[~m]
                w = 0.5 * w / w.sum()
               # output2[:, ~m] = output2[:, ~m] * w[None, :]
                output2 = output2.masked_fill(mask == 0, -1e9)
                output1 = output1.masked_fill(mask==1,-1e9)

                loss = F.cross_entropy(output2, target2)+F.cross_entropy(output1, target1)
                loss/=2.
                for z in range(len(target)):
                    class_count[target[z]] += 1

            else:
                loss = F.cross_entropy(output, target)
        else:
            loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    """Train a simple Hybrid Resnet Scattering + CNN model on CIFAR.
    """
    parser = argparse.ArgumentParser(description='CIFAR scattering  + hybrid examples')
    parser.add_argument('--mode', type=str, default='scattering',choices=['scattering', 'standard'],
                        help='network_type')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='samples per class')
    parser.add_argument('--learning_schedule_multi', type=int, default=1,
                        help='samples per class')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for dataset subselection')
    parser.add_argument('--width', type=int, default=2,help='width factor for resnet')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")



    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    model = Scattering2dResNet(8, args.width, standard=True).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    #####cifar data
    cifar_data = datasets.CIFAR10(root='.',train=True, transform=transforms.Compose([
           # transforms.RandomHorizontalFlip(),
           # transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)
    # Extract a subset of X samples per class
    prng = RandomState(args.seed)
    random_permute = prng.permutation(np.arange(0, 5000))[0:args.num_samples]
    indx = np.concatenate([np.where(np.array(cifar_data.targets) == classe)[0][random_permute] for classe in range(0, 5)])
    indx2 = np.concatenate([np.where(np.array(cifar_data.targets) == classe)[0] for classe in range(5, 10)])
    indx = np.concatenate([indx,indx2])
   # import ipdb;ipdb.set_trace()
    cifar_data.data, cifar_data.targets = cifar_data.data[indx], list(np.array(cifar_data.targets)[indx])
    train_loader = torch.utils.data.DataLoader(cifar_data,
                                               batch_size=32, shuffle=True, num_workers=num_workers,
                                               pin_memory=pin_memory)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='.',train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)



    # Optimizer
    lr = 0.01
    M = args.learning_schedule_multi
    drops = [60*M,120*M,160*M]
    for epoch in range(0, 200*M):
        if epoch in drops or epoch==0:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0,
                                        weight_decay=0.0)
            lr*=0.2

        train(model, device, train_loader, optimizer, epoch+1)
        if epoch%1==0:
            test(model, device, test_loader)



if __name__ == '__main__':
    main()