import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.nn import functional as F
from RD3D.rd3d import BasicConv3d
from RD3D.resnet3d import I3DResNet
from models.resnet import resnet34
from ResNet_models_Custom import Triple_Conv, multi_scale_aspp, Classifier_Module, RCAB, BasicConv2d
from Multi_head import MHSA
import math

from SPNet.model import CIM, MFA, GCM

"""
这个文件结合SPNet论文的CMI,MFA,RFB模块,加上了DCF的CRM模块，使用了空间注意力
"""

def maxpool():
    """
    最大池化下采样层，尺寸变为原来的一半
    """
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicConv2d(nn.Module):
    """
    resnet18,34 的基础卷积块
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



###############################################################################
class ChannelAttention(nn.Module):
    """
    通道注意力模块,——适应性最大池化下采样，卷积操作
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    SpatialAttention 模块的核心思想是利用空间最大池化操作和卷积生成一个空间注意力图，之后通过 Sigmoid 激活将该图的值压缩到 [0, 1] 区间，
    表示每个空间位置的重要性。在实际使用时，通常会将这个空间注意力图与输入的特征图进行逐元素相乘，从而增强模型对重要区域的关注。
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    """
    这里是resnet50，101，152的基础残差块
    """
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def init_weight(model):
    """
    初始化卷积层，BN层的权重
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class Decoder(nn.Module):
    """
    该 Decoder 模块用于将低维的特征图进行解码，通常是从编码器的输出恢复到更高维的空间，以便进行下一步的处理，如生成、恢复或者图像分割等。
    """
    def __init__(self, in_channel, out_channel=32):
        super(Decoder, self).__init__()
        # self.reduce_conv=nn.Sequential(
        #     #nn.Conv2d(side_channel, in_channel, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(side_channel, in_channel, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(inplace=True)  ###
        # )
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)  ###
        )
        init_weight(self)

    def forward(self, x):
        # x=F.interpolate(x, size=side.size()[2:], mode='bilinear', align_corners=True)
        # side=self.reduce_conv(side)
        # x=torch.cat((x, side), 1)
        x = self.decoder(x)
        return x


class PredLayer(nn.Module):
    """
    最终返回的 x 是一个大小为 (batch_size, 1, height, width) 的张量，代表预测的单通道图像，通常用于二分类或图像分割任务。
    """
    def __init__(self, in_channel=32):
        super(PredLayer, self).__init__()
        
        # 通过前面定义的卷积层提取特征。
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # 通过最后的卷积层将特征图映射为单通道输出，
            nn.Sigmoid()  # 并使用 Sigmoid 激活函数将每个像素的值映射到 [0, 1]
        )
        init_weight(self)

    def forward(self, x, size):
        # 该函数用于上采样，指定目标输出尺寸，size 通常是一个 (height, width) 的元组
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x = self.enlayer(x)
        x = self.outlayer(x)
        return x


# 三维卷积减少通道用的
class self_Reduction3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(self_Reduction3D, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv3d(in_channel, out_channel, kernel_size=[1, 1, 1]),
            BasicConv3d(out_channel, out_channel, kernel_size=[3, 3, 3], padding=1),
            BasicConv3d(out_channel, out_channel, kernel_size=[3, 3, 3], padding=1)
        )

    def forward(self, x):
        return self.reduce(x)
    
class self_RD3D(nn.Module):
    """
    这里用来获得三维融合卷积的数据
    """
    def __init__(self,resnet):
        super(self_RD3D, self).__init__()
        self.resnet = I3DResNet(resnet)
        
        self.reductions0 = self_Reduction3D(64, 64)
        self.reductions1 = self_Reduction3D(256, 64)
        self.reductions2 = self_Reduction3D(512, 64)
        self.reductions3 = self_Reduction3D(1024, 128)
        self.reductions4 = self_Reduction3D(2048, 256)
        
        # 测试使用,下次这里还可以测试上面的Reduction3D统一减少到同一个channel，然后使用下面的来处理
        self.T_to_0 = BasicConv3d(64, 64, [2, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0])
        self.T_to_1 = BasicConv3d(64, 64, [2, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0])
        self.T_to_2 = BasicConv3d(64, 64, [2, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0])
        self.T_to_3 = BasicConv3d(128, 128, [2, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0])
        self.T_to_4 = BasicConv3d(256, 256, [2, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0])
    
    def forward(self, x):
        # 下面的是img，depth的融合图像
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x0 = self.resnet.relu(x) # 这是三维卷积的第一层
        x = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(x) # 这是三维卷积的第二层
        x2 = self.resnet.layer2(x1) # 这是三维卷积的第三层
        x3 = self.resnet.layer3(x2) # 这是三维卷积的第四层
        x4 = self.resnet.layer4(x3) # 这是三维卷积的第五层
        
        x_s0 = self.reductions0(x0)
        x_t_0 = self.T_to_0(x_s0).squeeze(2)

        x_s1 = self.reductions1(x1)
        x_t_1 = self.T_to_1(x_s1).squeeze(2)
                
        x_s2 = self.reductions2(x2)
        x_t_2 = self.T_to_2(x_s2).squeeze(2)
        
        x_s3 = self.reductions3(x3)
        x_t_3 = self.T_to_3(x_s3).squeeze(2)
        
        x_s4 = self.reductions4(x4)
        x_t_4 = self.T_to_4(x_s4).squeeze(2)
        
        return x_t_0, x_t_1, x_t_2, x_t_3, x_t_4



import torch
import torch.nn as nn

class SCA(nn.Module):
    def __init__(self, in_channels=32, out_channels=32):
        """
        初始化 SCA 模块
        :param in_channels: 输入特征的通道数，默认为 32
        :param out_channels: 输出特征的通道数，默认为 32
        """
        super(SCA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # RGB 分支
        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        # 深度分支
        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        # 跨模态特征融合
        self.cross_conv = nn.Conv2d(self.in_channels * 2, self.out_channels, 1, padding=0)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x3_r, x3_d):
        """
        前向传播
        :param x3_r: RGB 特征图，形状为 [batch_size, in_channels, H, W]
        :param x3_d: 深度特征图，形状为 [batch_size, in_channels, H, W]
        :return: 融合后的特征图，形状为 [batch_size, out_channels, H, W]
        """
        # RGB 分支
        SCA_ca = self.channel_attention_rgb(self.squeeze_rgb(x3_r))
        SCA_3_o = x3_r * SCA_ca.expand_as(x3_r)

        # 深度分支
        SCA_d_ca = self.channel_attention_depth(self.squeeze_depth(x3_d))
        SCA_3d_o = x3_d * SCA_d_ca.expand_as(x3_d)

        # 跨模态注意力
        Co_ca3 = torch.softmax(SCA_ca + SCA_d_ca, dim=1)

        # 跨模态加权
        SCA_3_co = x3_r * Co_ca3.expand_as(x3_r)
        SCA_3d_co = x3_d * Co_ca3.expand_as(x3_d)

        # 特征融合
        CR_fea3_rgb = SCA_3_o + SCA_3_co
        CR_fea3_d = SCA_3d_o + SCA_3d_co

        # 跨模态特征拼接和融合
        CR_fea3 = torch.cat([CR_fea3_rgb, CR_fea3_d], dim=1)
        CR_fea3 = self.cross_conv(CR_fea3)

        return CR_fea3


class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
    

class aggregation(nn.Module):
    def __init__(self, channel, out_channel=None):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        # 移除上采样操作
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        
        # 输出通道数可配置
        self.out_channel = out_channel if out_channel is not None else 1
        self.conv5 = nn.Conv2d(3*channel, self.out_channel, 1)

    def forward(self, x1, x2, x3):
        # 直接对 x1, x2, x3 进行逐元素相乘
        x2_1 = self.conv_upsample1(x1) * x2
        x3_1 = self.conv_upsample2(x1) * self.conv_upsample3(x2) * x3

        # 拼接操作
        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1)), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        # 生成注意力图
        x = self.conv4(x3_2)
        attention_map = self.conv5(x)  # 输出形状为 [batch_size, out_channel, H, W]

        return attention_map
    
    
import numpy as np
import scipy.stats as st
from torch.nn.parameter import Parameter
def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel
def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


class HA(nn.Module):
    # holistic attention module
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))
        return x


class DCF(nn.Module):
    def __init__(self, in_channel):
        super(DCF, self).__init__()
        self.rfb3_1 = RFB(in_channel, in_channel)
        self.rfb4_1 = RFB(in_channel, in_channel)
        self.rfb5_1 = RFB(in_channel, in_channel)
        
        self.rfb3_2 = RFB(in_channel, in_channel)
        self.rfb4_2 = RFB(in_channel, in_channel)
        self.rfb5_2 = RFB(in_channel, in_channel)
        
        self.agg1 = aggregation(in_channel, out_channel=1)
        self.agg2 = aggregation(in_channel, out_channel=in_channel)
        self.HA = HA()
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channel, in_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, rgb, depth, fusion):
        x3_1 = self.rfb3_1(fusion)
        x4_1 = self.rfb4_1(rgb)
        x5_1 = self.rfb5_1(depth)
        attention_map = self.agg1(x5_1, x4_1, x3_1)
        x3_2 = self.HA(attention_map.sigmoid(), fusion)

        x4_2 = self.conv4(x3_2)
        x5_2 = self.conv5(x4_2)
        
        
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        x5_2 = self.rfb5_2(x5_2)
        
        detection_map = self.agg2(x5_2, x4_2, x3_2)
        return detection_map


class SAINet_SPNet(nn.Module):
    """
    模型结构类
    """
    def __init__(self, channel=32, ind=50, resnet=None):
        super(SAINet_SPNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        # Backbone model
        self.layer_rgb = resnet34()
        self.layer_dep = resnet34()
        self.rgb_inplanes = 2048
        self.dep_inplanes = 2048
        self.base_width = 64

        # CAAF #
        # ###############################################
        # self.caaf_0 = CAAF(64, 64)

        # self.caaf_1 = CAAF(64, 64)

        # self.caaf_2 = CAAF(128, 64)

        # self.caaf_3 = CAAF(256, 128)

        # self.caaf_4 = CAAF(512, 256)

        # # MFIB #
        # ###############################################
        # self.mfib_layer4 = MFIB(256, 256)
        # self.mfib_layer3 = MFIB(128, 128)
        # self.mfib_layer2 = MFIB(64, 64)
        # self.mfib_layer1 = MFIB(64, 64)
        # self.mfib_layer0 = MFIB(64, 64)
        
        # CIM0 #
        ###############################################
        # self.caaf_0 = CIM(64, 64)

        # self.caaf_1 = CIM(64, 64)

        # self.caaf_2 = CIM(128, 64)

        # self.caaf_3 = CIM(256, 128)

        # self.caaf_4 = CIM(512, 256)
        
        self.caaf_0 = SCA(64, 64)

        self.caaf_1 = SCA(64, 64)

        self.caaf_2 = SCA(128, 64)

        self.caaf_3 = SCA(256, 128)

        self.caaf_4 = SCA(512, 256)

        # MFA #
        ###############################################
        self.mfib_layer4 = MFA(256)
        self.mfib_layer3 = MFA(128)
        self.mfib_layer2 = MFA(64)
        self.mfib_layer1 = MFA(64)
        self.mfib_layer0 = MFA(64)
        
        

        # 上面一行的Decoder #
        ###############################################
        # Decoder（in, out) 不会改变图像尺寸大小，最终的输出的图像通道的大小就是out
        # self.ful_gcm_4 = Decoder(256, 32)

        # # self.ful_conv_3 = nn.Sequential(BasicConv2d(1024 + 32 * 3, 256, 3, padding=1), self.relu)
        # self.ful_gcm_3 = Decoder(128 + 32, channel)

        # # self.ful_conv_2 = nn.Sequential(BasicConv2d(512 + 32 * 3, 128, 3, padding=1), self.relu)
        # self.ful_gcm_2 = Decoder(64 + 32, channel)

        # # self.ful_conv_1 = nn.Sequential(BasicConv2d(256 + 32 * 3, 128, 3, padding=1), self.relu)
        # self.ful_gcm_1 = Decoder(64 + 32, channel)

        # # self.ful_conv_0 = nn.Sequential(BasicConv2d(128 + 32 * 3, 64, 3, padding=1), self.relu)
        # self.ful_gcm_0 = Decoder(64 + 32, channel)
        # # rgb的decoder
        # self.rgb_gcm_4 = Decoder(512, 256)
        # self.rgb_gcm_3 = Decoder(256 + 256, 128)
        # self.rgb_gcm_2 = Decoder(128 + 128, 64)
        # self.rgb_gcm_1 = Decoder(64 + 64, 64)
        # self.rgb_gcm_0 = Decoder(64 + 64, 64)
        # # depth 的decoder 
        # self.dep_gcm_4 = Decoder(512, 256)
        # self.dep_gcm_3 = Decoder(256 + 256, 128)
        # self.dep_gcm_2 = Decoder(128 + 128, 64)
        # self.dep_gcm_1 = Decoder(64 + 64, 64)
        # self.dep_gcm_0 = Decoder(64 + 64, 64)
        
        # 这是使用SPNet的decoder，GCM，多尺度提取特征信息，增加感受野
        # 融合decoder
        self.ful_gcm_4 = GCM(256, 32)
        self.ful_gcm_3 = GCM(128 + 32, channel)
        self.ful_gcm_2 = GCM(64 + 32, channel)
        self.ful_gcm_1 = GCM(64 + 32, channel)
        self.ful_gcm_0 = GCM(64 + 32, channel)
        # rgb的decoder
        self.rgb_gcm_4 = GCM(512, 256)
        self.rgb_gcm_3 = GCM(256 + 256, 128)
        self.rgb_gcm_2 = GCM(128 + 128, 64)
        self.rgb_gcm_1 = GCM(64 + 64, 64)
        self.rgb_gcm_0 = GCM(64 + 64, 64)
        # depth 的decoder 
        self.dep_gcm_4 = GCM(512, 256)
        self.dep_gcm_3 = GCM(256 + 256, 128)
        self.dep_gcm_2 = GCM(128 + 128, 64)
        self.dep_gcm_1 = GCM(64 + 64, 64)
        self.dep_gcm_0 = GCM(64 + 64, 64)
        
        

        # Pred #
        ###############################################
        self.S0 = PredLayer()
        self.S1 = PredLayer()
        self.S2 = PredLayer()
        self.S3 = PredLayer()
        self.S4 = PredLayer()
        self.rgb = PredLayer(in_channel=64)
        self.dep = PredLayer(in_channel=64)
        
        # 三维卷积
        self.RD3D = self_RD3D(resnet)
        
        # 空间，通道注意力，这里测试过，没什么用
        # self.fusion4 = AdvancedFusion(256, 128)
        # self.fusion3 = AdvancedFusion(128, 64)
        # self.fusion2 = AdvancedFusion(64, 64)
        # self.fusion1 = AdvancedFusion(64, 32)
        # self.fusion0 = AdvancedFusion(32, 32)
        
        
        
    def forward(self, imgs, depths, img_depths):
        # 三维卷积数据
        x_t_0, x_t_1, x_t_2, x_t_3, x_t_4 = self.RD3D(img_depths)
        
        img_0, img_1, img_2, img_3, img_4 = self.layer_rgb(imgs)  # 这里的函数指向resnet34的函数，会分别返回五个层处理之后的结果
        dep_0, dep_1, dep_2, dep_3, dep_4 = self.layer_dep(depths)

        # CAAF #
        ###############################################
        ful_0 = self.caaf_0(img_0, dep_0)
        ful_1 = self.caaf_1(img_1, dep_1)
        ful_2 = self.caaf_2(img_2, dep_2)
        ful_3 = self.caaf_3(img_3, dep_3)
        ful_4 = self.caaf_4(img_4, dep_4)

        # RGB/D Decoder #
        ###############################################
        # rgb decoder融合
        x_rgb_42 = self.rgb_gcm_4(img_4)
        x_rgb_32 = self.rgb_gcm_3(torch.cat((img_3, self.upsample_2(x_rgb_42)), dim=1))
        x_rgb_22 = self.rgb_gcm_2(torch.cat((img_2, self.upsample_2(x_rgb_32)), dim=1))
        x_rgb_12 = self.rgb_gcm_1(torch.cat((img_1, self.upsample_2(x_rgb_22)), dim=1))
        x_rgb_02 = self.rgb_gcm_0(torch.cat((img_0, self.upsample_2(x_rgb_12)), dim=1))
        # depth decoder融合
        x_dep_42 = self.dep_gcm_4(img_4)
        x_dep_32 = self.dep_gcm_3(torch.cat((img_3, self.upsample_2(x_dep_42)), dim=1))
        x_dep_22 = self.dep_gcm_2(torch.cat((img_2, self.upsample_2(x_dep_32)), dim=1))
        x_dep_12 = self.dep_gcm_1(torch.cat((img_1, self.upsample_2(x_dep_22)), dim=1))
        x_dep_02 = self.dep_gcm_0(torch.cat((img_0, self.upsample_2(x_dep_12)), dim=1))


        # MFIB+Decoder+Pre 
        ###############################################
        x_ful_42 = self.mfib_layer4(ful_4, x_rgb_42, x_dep_42)
        x_ful_42 = x_ful_42.mul(x_t_4) + x_ful_42 # 这里加上了三维卷积的特征 0.057
        print("MFIB_4", x_ful_42.shape)
        x_ful_42 = self.upsample_2(self.ful_gcm_4(x_ful_42)) # 依次经过了decoder，上采样
        # 这里不再使用decoder，而是使用 空间 + 通道注意力，融合MFIB的结果和三维卷积的结果
        # x_ful_42 = self.upsample_2(self.fusion4(x_ful_42, x_t_4))
        

        x_ful_32 = self.mfib_layer3(ful_3, x_rgb_32, x_dep_32)
        x_ful_32 = x_ful_32.mul(x_t_3) + x_ful_32
        print("MFIB_3", x_ful_32.shape)
        x_ful_32 = self.upsample_2(self.ful_gcm_3(torch.cat((x_ful_42, x_ful_32), 1)))
        # 这里不再使用decoder，而是使用 空间 + 通道注意力，融合MFIB的结果和三维卷积的结果
        # x_ful_32 = self.upsample_2(self.fusion3(x_ful_42, x_ful_32))


        x_ful_22 = self.mfib_layer2(ful_2, x_rgb_22, x_dep_22)
        x_ful_22 = x_ful_22.mul(x_t_2) + x_ful_22
        print("MFIB_2", x_ful_22.shape)
        x_ful_22 = self.upsample_2(self.ful_gcm_2(torch.cat((x_ful_32, x_ful_22), 1)))
        # x_ful_22 = self.upsample_2(self.fusion2(x_ful_32, x_ful_22))

        x_ful_12 = self.mfib_layer1(ful_1, x_rgb_12, x_dep_12)
        x_ful_12 = x_ful_12.mul(x_t_1) + x_ful_12
        print("MFIB_1", x_ful_12.shape)
        x_ful_12 = self.upsample_2(self.ful_gcm_1(torch.cat((x_ful_22, x_ful_12), 1)))
        # x_ful_12 = self.upsample_2(self.fusion1(x_ful_22, x_ful_12))

        x_ful_02 = self.mfib_layer0(ful_0, x_rgb_02, x_dep_02)
        x_ful_02 = x_ful_02.mul(x_t_0) + x_ful_02
        print("MFIB_0", x_ful_02.shape)
        x_ful_02 = self.ful_gcm_0(torch.cat((x_ful_12, x_ful_02), 1))
        # x_ful_02 = self.fusion0(x_ful_12, x_ful_02)
        
        
        size = imgs.size()[2:]
        s0 = self.S0(x_ful_02, size)  # 上采样得到指定尺寸size的图像数据，输出通道32
        s1 = self.S1(x_ful_12, size) #通道数都是32
        s2 = self.S2(x_ful_22, size)
        s3 = self.S3(x_ful_32, size)
        s4 = self.S4(x_ful_42, size)
        rgb = self.rgb(x_rgb_02, size)  # 第一个decoder的输出，因为图像处理流是从后面往前面传的
        dep = self.dep(x_dep_02, size)

        return s0, s1, s2, s3, s4, rgb, dep


    def _make_deplayer(self, planes, blocks, stride=1, dilate=False):
        """
        该方法构建了一个由 Bottleneck 残差块组成的神经网络模块，可以用于网络中的“Deplayer”层（例如，解码器或其他深度网络层）。
        它可以通过设置步幅来实现下采样，并且支持空洞卷积来增加感受野。
        目前好像没用到
        """
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = 1
        groups = 1
        expansion = 4
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.dep_inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.dep_inplanes, planes * expansion, stride),
                norm_layer(planes * expansion),
            )

        layers = []
        layers.append(Bottleneck(self.dep_inplanes, planes, stride, downsample, groups,
                                 self.base_width, previous_dilation, norm_layer))
        self.dep_inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.dep_inplanes, planes, groups=groups,
                                     base_width=self.base_width, dilation=1,
                                     norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_agant_layer(self, inplanes, planes):
        """
        构建了一个简单的卷积层，使用 1x1 的卷积将输入通道数调整为目标通道数，然后进行批归一化和 ReLU 激活。
        """
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        """
        建了一个包含上采样操作和残差块的模块，支持通过 ConvTranspose2d 进行上采样，或者通过 Conv2d 进行通道数调整。
        该方法可以用于深度网络中的上采样层或解码器部分。
        """
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.rgbinplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.rgbinplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.rgbinplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.rgbinplanes, self.rgbinplanes))

        layers.append(block(self.rgbinplanes, planes, stride, upsample))
        self.rgbinplanes = planes

        return nn.Sequential(*layers)



