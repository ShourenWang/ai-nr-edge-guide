"""
Definition of the FastDVDnet model

Copyright (C) 2019, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn
from math import floor, log2
import numpy as np
import torch
from torch import nn


class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1 ),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1 ),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.convblock(x)

class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 12
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*(3), num_in_frames*self.interm_ch, \
                      kernel_size=3, padding=1, groups=num_in_frames ),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1 ),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.convblock(x)

class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            # nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2,inplace=False),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)

class UpBlock_sade(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''
    def __init__(self, in_ch, out_ch):
        super(UpBlock_sade, self).__init__()
        self.spade = SPADE(in_ch, in_ch)
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x, segmap):
        x = self.spade(x,segmap)
        return self.convblock(x)

class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)

class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            # nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.convblock(x)

class DenBlock(nn.Module):
    def __init__(self, num_input_frames=2):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 16
        self.chs_lyr1 = 32
        self.chs_lyr2 = 64

        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock_sade(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

    def forward(self, in0, in1):
        # Input convolution block
        x0_256 = self.inc(torch.cat((in0, in1), dim=1))
        # print(x0_256.size() )
        # Downsampling
        x1_128 = self.downc0(x0_256)
        x2_64 = self.downc1(x1_128)
        # Upsampling
        x3_128 = self.upc2(x2_64)
        x4_256 = self.upc1(x3_128, x1_128)
        # Estimation
        x = self.outc(x0_256+x4_256)

        # Residual
        x = in1 + x

        return x


class Construction(nn.Module):
    """ Definition of the denosing block of FastDVDnet.
    Inputs of constructor:
        num_input_frames: int. number of input frames
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """

    def __init__(self, num_input_frames=1):
        super(Construction, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128
        self.interm_ch = 8

        self.inc = nn.Sequential(
            nn.Conv2d(num_input_frames * 3, num_input_frames * self.interm_ch, \
                      kernel_size=3, padding=1, groups=num_input_frames),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(num_input_frames * self.interm_ch, self.chs_lyr0, kernel_size=3, padding=1),
            nn.ReLU(inplace=False)
        )
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.downc2 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr2)
        self.upc1 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc0 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

    def forward(self, in0):
        '''Args:
            inX: Tensor, [N, C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        # Input convolution block
        # print(in0.size(),in1.size())
        x0_256 = self.inc( in0 )
        # print(x0_256.size() )
        # Downsampling
        x1_128 = self.downc0(x0_256)
        x2_64 = self.downc1(x1_128)
        added_32 = self.downc2(x2_64)
        # Upsampling
        added_64 = self.upc2(added_32)
        x3_128 = self.upc1(added_64)
        x4_256 = self.upc0(x3_128)
        # Estimation
        x = self.outc(x0_256+x4_256)


        return x



class edge_detect(nn.Module):
    def __init__(self, ):
        super(edge_detect, self).__init__()

        self.construction = Construction()
        self.conv_out = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        ''' Args:
            x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
            noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
        '''
        x = x * 2 - 1
        x = self.construction(x )
        edges = self.conv_out(x)

        return edges

