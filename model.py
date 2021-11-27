import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet50,resnet101,resnet34,vgg16_bn,vgg,vgg19_bn,vgg19,googlenet,inception_v3
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
           'resnet152_cbam']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print('1')
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        return x

class  FC_model(nn.Module):
    def __init__(self):
        super(FC_model, self).__init__()
        # model = ResNet(Bottleneck, [3, 4, 6, 3])
        # model = resnet50(pretrained=True)
        # model = resnet101(pretrained=True)
        # model = resnet34_cbam(True)
        model = vgg16_bn(True)
        # self.res = ResNet(Bottleneck, [3, 4, 6, 3])
        # pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        # now_state_dict = model.state_dict()
        # now_state_dict.update(pretrained_state_dict)
        # model.load_state_dict(now_state_dict)
        self.res = model
        self.fc1 = nn.Linear(1000,1000)
        self.fc1_1 = nn.Linear(1000, 2)

        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_2 = nn.Linear(1000, 2)

        self.fc3 = nn.Linear(1000, 1000)
        self.fc3_3 = nn.Linear(1000, 2)

        self.fc4 = nn.Linear(1000, 1000)
        self.fc4_4 = nn.Linear(1000, 2)

        self.fc5 = nn.Linear(1000, 1000)
        self.fc5_5 = nn.Linear(1000, 2)

        self.fc6 = nn.Linear(1000, 1000)
        self.fc6_6 = nn.Linear(1000, 4)

        self.fc7 = nn.Linear(1000, 1000)
        self.fc7_7 = nn.Linear(1000, 6)

        self.fc8 = nn.Linear(1000, 1000)
        self.fc8_8 = nn.Linear(1000, 20)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
        self.relu8 = nn.ReLU()

    def forward(self,x):
        x = self.res(x)
        # torch.nn.Dropout(0.5)# print("debug",x.shape)
        # out1 = self.relu1(self.fc1_1(self.fc1(x)))
        # out2 = self.relu2(self.fc2_2(self.fc2(x)))
        # out3 = self.relu3(self.fc3_3(self.fc3(x)))
        # out4 = self.relu4(self.fc4_4(self.fc4(x)))
        # out5 = self.relu5(self.fc5_5(self.fc5(x)))
        # out6 = self.relu6(self.fc6_6(self.fc6(x)))
        # out7 = self.relu7(self.fc7_7(self.fc7(x)))
        # out8 = self.relu8(self.fc8_8(self.fc8(x)))
        F.relu(x)

        # out1 = F.dropout(out1, p=0.5)
        out1 = self.relu1(self.fc1(x))
        out1 = self.relu1(self.fc1_1(out1))


        # out2 = F.dropout(out2, p=0.5)
        out2 = self.relu2(self.fc2(x))
        out2 = self.relu1(self.fc2_2(out2))


        # out3 = F.dropout(out3, p=0.5)
        out3 = self.relu3(self.fc3(x))
        out3 = self.relu1(self.fc3_3(out3))


        # out4 = F.dropout(out4, p=0.5)
        out4 = self.relu4(self.fc4(x))
        out4 = self.relu1(self.fc4_4(out4))


        # out5 = F.dropout(out5, p=0.5)
        out5 = self.relu5(self.fc5(x))
        out5 = self.relu1(self.fc5_5(out5))


        # out6 = F.dropout(out6, p=0.5)
        out6 = self.relu6(self.fc6(x))
        out6 = self.relu6(self.fc6_6(out6))


        # out7 = F.dropout(out7, p=0.5)
        out7 = self.relu7(self.fc7(x))
        out7 = self.relu6(self.fc7_7(out7))


        # out8 = F.dropout(out8, p=0.5)
        out8 = self.relu8(self.fc8(x))
        out8 = self.relu6(self.fc8_8(out8))



        # out1 = self.relu1(self.fc1_1(x))
        # out2 = self.relu2(self.fc2_2(x))
        # out3 = self.relu3(self.fc3_3(x))
        # out4 = self.relu4(self.fc4_4(x))
        # out5 = self.relu5(self.fc5_5(x))
        # out6 = self.relu6(self.fc6_6(x))
        # out7 = self.relu7(self.fc7_7(x))
        # out8 = self.relu8(self.fc8_8(x))




        return [out1, out2, out3, out4, out5, out6, out7, out8]


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def Multi_resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def Multi_Attention_Resnet50(pretrained=True, **kwargs):
    # x = torch.randn(2, 3, 244, 244)
    model = FC_model()

    pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    now_state_dict = model.state_dict()
    now_state_dict.update(pretrained_state_dict)
    model.load_state_dict(now_state_dict)
    return model

def main():
    x = torch.randn(1,3,224,224)
    # model =FC_model()

    model = FC_model()
    for name, p in model.named_parameters():
        print(name)
        print(p.requires_grad)
        print(...)
    # print(model)
    # pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
    # now_state_dict = model.state_dict()
    # now_state_dict.update(pretrained_state_dict)
    # model.load_state_dict(now_state_dict)
    out = model(x)
    # print(out)

if __name__ == '__main__':
    main()
