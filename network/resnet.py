import torch
import torch.nn as nn
from torch.utils import model_zoo
import math
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

pth_dir = '../pretrainedModel/'
model_name = {
    'resnet18': 'resnet18-5c106cde.pth',
    'resnet34': 'resnet34-333f7ec4.pth',
    'resnet50': 'resnet50-19c8e357.pth',
    'resnet101': 'resnet101-5d3b4d8f.pth',
    'resnet152': 'resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=True)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=3, input_channels=1):
        super(ResNet, self).__init__()
        self.inplanes = 64
 
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=True),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # for params in self.conv1.parameters():
        #     print('\nparams : ' + str(params.shape))
        #     print('\ntype + ' + str(params.dtype))

        # if math.isnan(x[0][0][0][0]):
        #     for params in self.conv1.parameters():
        #         print('\nresnet params : ' + str(params))
        #         print('\ntype + ' + str(params.dtype))

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_pre = self.layer4(x)

        x = self.avgpool(x_pre)
        features = x.view(x.size(0), -1)
        y = self.fc(features)

        # if math.isnan(features[0][0]):
        #     print('这里')

        return y, features, x_pre


def resnet18(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2]).cuda()
    if pretrained:
        pretrained_dict = torch.load(pth_dir+model_name['resnet18'])
        delete_list = []
        for key, _ in pretrained_dict.items():
            if 'conv1' in key:
                delete_list.append(key)
            if 'fc' in key:
                delete_list.append(key)
        for key in delete_list:
            del pretrained_dict[key]
            print('del key in dict:\t', key)
        target_dict=model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in target_dict}
        print(len(pretrained_dict))
        # 2. overwrite entries in the existing state dict
        target_dict.update(pretrained_dict)
        model.load_state_dict(target_dict)
        print('using imagenet-pretrained weights')

    return model

def resnet34(pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        pretrained_dict = torch.load(pth_dir+model_name['resnet34'])
        delete_list = []
        for key, _ in pretrained_dict.items():
            if 'conv1' in key:
                delete_list.append(key)
            if 'fc' in key:
                delete_list.append(key)
        for key in delete_list:
            del pretrained_dict[key]
            print('del key in dict:\t', key)
        # del pretrained_dict['fc.weight']
        # del pretrained_dict['fc.bias']
        target_dict=model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in target_dict}
        target_dict.update(pretrained_dict)
        model.load_state_dict(target_dict)
        print('using imagenet-pretrained weights')
    return model

def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        pretrained_dict = torch.load(pth_dir+model_name['resnet50'])
        delete_list = []
        for key, _ in pretrained_dict.items():
            if 'conv1' in key:
                delete_list.append(key)
            if 'fc' in key:
                delete_list.append(key)
        for key in delete_list:
            del pretrained_dict[key]
            print('del key in dict:\t', key)
        # del pretrained_dict['fc.weight']
        # del pretrained_dict['fc.bias']
        target_dict=model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in target_dict}
        target_dict.update(pretrained_dict)
        model.load_state_dict(target_dict)
        print('using imagenet-pretrained weights')
    return model

def resnet101(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        pretrained_dict = torch.load(pth_dir+model_name['resnet101'])
        delete_list = []
        for key, _ in pretrained_dict.items():
            if 'conv1' in key:
                delete_list.append(key)
            if 'fc' in key:
                delete_list.append(key)
        for key in delete_list:
            del pretrained_dict[key]
            print('del key in dict:\t', key)
        # del pretrained_dict['fc.weight']
        # del pretrained_dict['fc.bias']
        target_dict=model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in target_dict}
        target_dict.update(pretrained_dict)
        model.load_state_dict(target_dict)
        print('using imagenet-pretrained weights')
    return model

def resnet152(pretrained=False):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        pretrained_dict = torch.load(pth_dir+model_name['resnet152'])
        delete_list = []
        for key, _ in pretrained_dict.items():
            if 'conv1' in key:
                delete_list.append(key)
            if 'fc' in key:
                delete_list.append(key)
        for key in delete_list:
            del pretrained_dict[key]
            print('del key in dict:\t', key)
        # del pretrained_dict['fc.weight']
        # del pretrained_dict['fc.bias']
        target_dict=model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in target_dict}
        target_dict.update(pretrained_dict)
        model.load_state_dict(target_dict)
        print('using imagenet-pretrained weights')
    return model

if __name__ == '__main__':
    # model = ResNet(BasicBlock, [2, 2, 2, 2])

    model = resnet18(pretrained=False)
    # for i, j in model.state_dict().items():
    #     print(i)

    # print('---\n')
    # x = torch.autograd.Variable(torch.FloatTensor(8, 12, 384, 384)).cuda()
    x = torch.autograd.Variable(torch.FloatTensor(16, 1, 338, 338))
    #model.cuda()
    output, feature, square = model(x)
    print(output.shape, feature.shape, square.shape)


    # pretrain_model = model_zoo.load_url(model_urls['resnet18'])
    # del pretrain_model['fc.weight']
    # del pretrain_model['fc.bias']
    # print(len(pretrain_model))
    # for i, j in pretrain_model.items():
    #     print(i)
    
    # print('---\n')

    # # print(model.state_dict()['bn1.num_batches_tracked'])
    # # print(type(model.state_dict()['bn1.num_batches_tracked']))
    # res_net = resnet18(pretrained=True)
    # x = torch.autograd.Variable(torch.FloatTensor(16,3,224,224))
    # print(res_net(x))
    
    