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

class InstanceNorm2d(nn.Module):
    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def forward(self, x):
        #x = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp


class SPADE(nn.Module):
	def __init__(self, x_nc, segmap_nc):
		super().__init__()
		self.param_free_norm = nn.InstanceNorm2d(x_nc)
		# self.param_free_norm = nn.BatchNorm2d(x_nc, affine=False)

		self.mlp_gamma = nn.Conv2d(segmap_nc, x_nc, kernel_size=3, padding=1)
		self.mlp_beta = nn.Conv2d(segmap_nc, x_nc, kernel_size=3, padding=1)

	def forward(self, x, segmap):
		# Part 1. generate parameter-free normalized activations
		normalized = self.param_free_norm(x)

		gamma = self.mlp_gamma(segmap)
		beta = self.mlp_beta(segmap)

		# apply scale and bias
		out = normalized * (gamma + 1) + beta

		return out

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1 ),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1 ),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 12
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames*(3+1), num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames ),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1 ),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
			nn.LeakyReLU(0.2,inplace=True),
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
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
		)

	def forward(self, x):
		return self.convblock(x)

class DenBlock(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

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

	def forward(self, in0, in1, noise_map):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0_256 = self.inc(torch.cat((in0, noise_map, in1, noise_map), dim=1))
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

class DenBlock_catdown(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=2):
		super(DenBlock_catdown, self).__init__()
		self.chs_lyr0 = 16
		self.chs_lyr1 = 32
		self.chs_lyr2 = 64

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1+3, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock_sade(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

	def forward(self, in0, in1, noise_map, x_pool):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0_256 = self.inc(torch.cat((in0, noise_map, in1, noise_map), dim=1))
		# Downsampling
		x1_128 = self.downc0(x0_256)
		# print(x1_128.size())
		x2_64 = self.downc1(torch.cat([x1_128,x_pool], dim=1))
		# Upsampling
		x3_128 = self.upc2(x2_64)
		x4_256 = self.upc1(x3_128, x1_128)
		# Estimation
		x = self.outc(x0_256+x4_256)

		# Residual
		x = in1 + x

		return x

class rgb2raw(nn.Module):
	def __init__(self, ):
		super(rgb2raw, self).__init__()
		self.chs_lyr0 = 16
		self.chs_lyr1 = 32
		self.chs_lyr2 = 64

		self.inc = nn.Sequential(
			nn.Conv2d(3, self.chs_lyr0, kernel_size=3, padding=1, stride=2),
		    nn.LeakyReLU(0.2))
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock_sade(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=4)

	def forward(self, in0):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0_256 = self.inc(in0)
		# Downsampling
		x1_128 = self.downc0(x0_256)
		x2_64 = self.downc1(x1_128)
		# Upsampling
		x3_128 = self.upc2(x2_64)
		x4_256 = self.upc1(x3_128, x1_128)
		# Estimation
		x = self.outc(x0_256+x4_256)

		return x

class raw2rgb(nn.Module):
	def __init__(self, ):
		super(raw2rgb, self).__init__()
		self.chs_lyr0 = 16
		self.chs_lyr1 = 32
		self.chs_lyr2 = 64

		self.inc = nn.Sequential(
			nn.Conv2d(4, self.chs_lyr0, kernel_size=3, padding=1, stride=1),
		    nn.LeakyReLU(0.2))
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock_sade(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=12)
		self.mosaic = nn.PixelShuffle(2)

	def forward(self, in0):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0_256 = self.inc(in0)
		# Downsampling
		x1_128 = self.downc0(x0_256)
		x2_64 = self.downc1(x1_128)
		# Upsampling
		x3_128 = self.upc2(x2_64)
		x4_256 = self.upc1(x3_128, x1_128)
		# Estimation
		x = self.outc(x0_256+x4_256)
		x = self.mosaic(x)

		return x

class Construction(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=2):
		super(Construction, self).__init__()
		self.chs_lyr0 = 16
		self.chs_lyr1 = 32
		self.chs_lyr2 = 64
		self.interm_ch = 8

		self.inc = nn.Sequential(
			nn.Conv2d(num_input_frames * 3, num_input_frames * self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_input_frames),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(num_input_frames * self.interm_ch, self.chs_lyr0, kernel_size=3, padding=1),
			nn.ReLU(inplace=True)
		)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock_sade(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

	def forward(self, in0, in1):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		# print(in0.size(),in1.size())
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
		x = x + in1

		return x

class FastDVDnet(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	def __init__(self, num_input_frames=2):
		super(FastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.temp1 = DenBlock_catdown(num_input_frames=2)
		self.temp2 = DenBlock(num_input_frames=2)

		self.construction = Construction()
		self.pool = nn.MaxPool2d(2)
		self.up = nn.Upsample(scale_factor=2, mode='bilinear')

	def forward(self, x, noise_map):
		''' Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))

		# First stage -- course reconstruction
		x_pool = self.temp2(self.pool(x0), self.pool(x1), self.pool(noise_map))
		x_course = self.temp1(x0, x1, noise_map, x_pool)

		# Second stage -- con struction
		x = self.construction(self.up(x_pool),x_course )

		return x_pool,x_course,x



class FastDVDnet_isp(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	def __init__(self, num_input_frames=2):
		super(FastDVDnet_isp, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.temp1 = DenBlock(num_input_frames=2)

		#
		self.rgb2raw = rgb2raw()
		self.raw2rgb = raw2rgb()
		self.construction = Construction()

	def forward(self, x, noise_map):
		''' Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))

		# First stage -- course reconstruction
		x_course = self.temp1(x0, x1, noise_map)

		# Second stage -- con struction
		raw_ = self.rgb2raw(x_course)
		rgb_ = self.raw2rgb(raw_)
		x = self.construction( rgb_, x_course) + x_course

		return raw_,x_course,x


class rgb2raw2rgb(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	def __init__(self, ):
		super(rgb2raw2rgb, self).__init__()
		self.rgb2raw = rgb2raw()
		self.raw2rgb = raw2rgb()

	def forward(self, x):
		raw_ = self.rgb2raw(x)
		rgb_ = self.raw2rgb(raw_)
		return raw_,rgb_
