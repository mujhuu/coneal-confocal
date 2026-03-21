import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

class ConvBlock(nn.Module):
    """基础卷积块：Conv-BN-ReLU-Conv-BN-ReLU"""
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class NestedUNet(nn.Module):
    """
    UNet++ Architecture
    Reference: https://arxiv.org/abs/1807.10165
    """
    def __init__(self, n_classes=1, input_channels=1, deep_supervision=False):
        super(NestedUNet, self).__init__()
        
        self.deep_supervision = deep_supervision
        nb_filter = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ------------------- 00, 10, 20, 30, 40 (Backbone) -------------------
        self.conv0_0 = ConvBlock(input_channels, nb_filter[0])
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4])

        # ------------------- 01, 11, 21, 31 -------------------
        # input: cat(x0_0, up(x1_0)) -> 64 + 64 = 128 -> out 64
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        # ------------------- 02, 12, 22 -------------------
        # input: cat(x0_0, x0_1, up(x1_1)) -> 64*2 + 64 -> out 64
        self.conv0_2 = ConvBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = ConvBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = ConvBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        # ------------------- 03, 13 -------------------
        # input: cat(x0_0, x0_1, x0_2, up(x1_2)) -> 64*3 + 64 -> out 64
        self.conv0_3 = ConvBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = ConvBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        # ------------------- 04 -------------------
        # input: cat(x0_0, x0_1, x0_2, x0_3, up(x1_3)) -> 64*4 + 64 -> out 64
        self.conv0_4 = ConvBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # Final output layer
        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)

    def forward(self, input):
        # Backbone / Downsampling
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Level 1 Nest
        # conv0_1 takes x0_0 and up(x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        # Level 2 Nest
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        # Level 3 Nest
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # Level 4 Nest (Final output node)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # Output
        if self.deep_supervision:
            # 如果需要 Deep Supervision，可以返回列表，但此处为了兼容原代码，
            # 默认 deep_supervision=False
            output1 = self.final(x0_1)
            output2 = self.final(x0_2)
            output3 = self.final(x0_3)
            output4 = self.final(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output