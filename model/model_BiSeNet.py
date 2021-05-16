'''
Code referenced from: 
https://github.com/ooooverflow/BiSeNet/blob/master/model/build_BiSeNet.py
'''

import paddle
from paddle import nn
from paddle.vision import models


__Author__ = 'Quanhao Guo'
__Date__ = '2021.05.01.16.17'


class resnet18(nn.Layer):
    def __init__(self, pretrained=True):
        super(resnet18, self).__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
        # self.pool = nn.AvgPool2d(7)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        # global average pooling to build tail
        tail = paddle.mean(feature4, 3, keepdim=True)
        tail = paddle.mean(tail, 2, keepdim=True)
        # tail = self.pool(feature4)
        return feature3, feature4, tail

class resnet101(nn.Layer):
    def __init__(self, pretrained=True):
        super(resnet101, self).__init__()
        self.features = models.resnet101(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
        # self.pool = nn.AvgPool2d(7)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        # global average pooling to build tail
        tail = paddle.mean(feature4, 3, keepdim=True)
        tail = paddle.mean(tail, 2, keepdim=True)
        # tail = self.pool(feature4)
        return feature3, feature4, tail

def build_contextpath(name='resnet18'):
    model = {
        'resnet18': resnet18(pretrained=False),
        'resnet101': resnet101(pretrained=False)
    }
    return model[name]

class ConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2D(out_channels, momentum=0.1)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Spatial_path(nn.Layer):
    def __init__(self):
        super(Spatial_path, self).__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class AttentionRefinementModule(nn.Layer):
    def __init__(self, in_channels, out_channels, size):
        super(AttentionRefinementModule, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2D(out_channels, momentum=0.1)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        # self.pool = nn.AvgPool2d(size)
    def forward(self, input):
        # global average pooling
        x = paddle.mean(input, 3, keepdim=True)
        x = paddle.mean(x, 2, keepdim=True)
        # x = self.pool(input)
        assert self.in_channels == x.shape[1], 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = paddle.multiply(input, x)
        return x


class FeatureFusionModule(nn.Layer):
    def __init__(self, n_class, size):
        super(FeatureFusionModule, self).__init__()
        # self.in_channels = input_1.channels + input_2.channels
        self.in_channels = 1024
        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=n_class, stride=1)
        self.conv1 = nn.Conv2D(n_class, n_class, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(n_class, n_class, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # self.pool = nn.AvgPool2d(size)

    def forward(self, input_1, input_2):
        x = paddle.concat([input_1, input_2], 1)
        assert self.in_channels == x.shape[1], 'in_channels of ConvBlock should be {}'.format(x.shape[1])
        feature = self.convblock(x)
        x = paddle.mean(feature, 3, keepdim=True)
        x = paddle.mean(x, 2 ,keepdim=True)
        # x = self.pool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.relu(x))
        x = paddle.multiply(feature, x)
        x = paddle.add(x, feature)
        return x
    
        
class BiSeNet(nn.Layer):
    def __init__(self, n_class, useUpsample=False, useDeconvGroup=False, addEdge=False, context_path='resnet18'):
        super(BiSeNet, self).__init__()
        
        self.addEdge = addEdge
        self.useUpsample = useUpsample
        self.useDeconvGroup = useDeconvGroup
        
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)

        # build attention refinement module
        self.attention_refinement_module1 = AttentionRefinementModule(256, 256, 14)
        self.attention_refinement_module2 = AttentionRefinementModule(512, 512, 7)

        # build feature fusion module
        self.feature_fusion_module = FeatureFusionModule(n_class, 28)

        # build final convolution
        self.conv = nn.Conv2D(in_channels=n_class, out_channels=n_class, kernel_size=1)
        if self.addEdge == True:
            self.edge = nn.Conv2D(in_channels=n_class, out_channels=n_class, kernel_size=1) # add new edge layer
        
        # upsampling
        if self.useUpsample == True:
            self.deconv1 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.deconv2 = nn.Upsample(scale_factor=4, mode='bilinear')
            self.deconv  = nn.Upsample(scale_factor=8, mode='bilinear')
        else:
            if self.useDeconvGroup == True:
                self.deconv1 = nn.Conv2DTranspose(256, 256, groups=256, kernel_size=4, stride=2, padding=1, bias_attr=False)
                self.deconv2 = nn.Conv2DTranspose(512, 512, groups=512, kernel_size=8, stride=4, padding=2, bias_attr=False)
                self.deconv  = nn.Conv2DTranspose(2, 2, groups=2, kernel_size=16, stride=8, padding=4, bias_attr=False)
            else:
                self.deconv1 = nn.Conv2DTranspose(256, 256, groups=1, kernel_size=4, stride=2, padding=1, bias_attr=False)
                self.deconv2 = nn.Conv2DTranspose(512, 512, groups=1, kernel_size=8, stride=4, padding=2, bias_attr=False)
                self.deconv  = nn.Conv2DTranspose(2, 2, groups=1, kernel_size=16, stride=8, padding=4, bias_attr=False)
        
    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)
        # output of context path
        cx1, cx2, tail = self.context_path(input)
        # print (cx1.shape, cx2.shape, tail.shape) # (1, 256, 14, 14) (1, 512, 7, 7) (1, 512, 1, 1)
        
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = paddle.multiply(cx2, tail)
        
        # upsampling
        cx1 = self.deconv1(cx1)
        cx2 = self.deconv2(cx2)
        
        # print (cx1.shape, cx2.shape) # (1, 256, 28, 28) (1, 768, 28, 28)
        cx = paddle.concat([cx1, cx2], 1)
        
        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = self.deconv(result)
        pred = self.conv(result)
    
        if self.addEdge == True:
            edge = self.edge(result)
            return pred, edge
        else:
            return pred
