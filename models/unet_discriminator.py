import numpy as np
import scipy.signal
import scipy.optimize
import torch

from torch_utils.ops import filtered_lrelu
from torch import nn
import math

def get_gaussian_kernel(kernel_size=3, sigma=2):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    return gaussian_kernel

class SPADE(nn.Module):
    def __init__(self, x_nc, segmap_nc):
        super().__init__()
        # self.param_free_norm = nn.InstanceNorm2d(x_nc)

        self.mlp_gamma = nn.Conv2d(segmap_nc, x_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(segmap_nc, x_nc, kernel_size=3, padding=1)
        self.param_free_norm = nn.Conv2d(x_nc, x_nc, kernel_size=1, padding=0)

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
    def __init__(self, in_ch, out_ch,fil_lrelu=['']):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1 ),
            nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1 ),
        )
        if len(fil_lrelu) > 1:
            self.act = filter_lrelu(int(fil_lrelu[0]),fil_lrelu[1], fil_lrelu[2])
        else:
            self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act( self.convblock(x) )

class filter_lrelu(nn.Module):
    def __init__(self, tem_res, is_torgb = True,out_channels=16):
        super(filter_lrelu, self).__init__()
        if tem_res>= 256:
            self.ksize = 5
        else:
            self.ksize = 3
        kernel2 = get_gaussian_kernel(kernel_size=self.ksize, sigma=2)
        kernel2 = torch.FloatTensor(kernel2).expand(out_channels, 1, self.ksize,self.ksize)
        self.weight2 = torch.nn.Parameter(data=kernel2, requires_grad=False)
        self.act = nn.LeakyReLU(0.2)
        self.out_channels = out_channels
        self.is_torgb = is_torgb

    def forward(self, x):
        # if self.is_torgb:
        #     x = torch.nn.functional.conv2d(x, self.weight2,padding=self.ksize//2, groups=self.out_channels )
        x = self.act(x)
        return x

# ----------------------------------------------------------------------------
class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
    def __init__(self, num_in_frames, out_ch,filter_lrelu_flag=['']):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 12
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames*(3+1), num_in_frames*self.interm_ch, \
                      kernel_size=3, padding=1, groups=num_in_frames ),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1 )
        )
        if len( filter_lrelu_flag) > 1:
            self.act = filter_lrelu(512, True,out_ch)
        else:
            self.act = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        return self.act( self.convblock(x) )

class DownBlock(nn.Module):
    ''' Downscale + (Conv2d => BN => ReLU)*2 '''
    def __init__(self, in_ch, out_ch, filter_lrelu_flag=['']):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            CvBlock(out_ch, out_ch, filter_lrelu_flag)
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
            CvBlock(in_ch, in_ch,['']),
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
        # Input convolution block
        x0_256 = self.inc(torch.cat((in0, noise_map, in1, noise_map), dim=1))
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

        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0,filter_lrelu_flag=['512',True,self.chs_lyr0])
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1,filter_lrelu_flag=['256',True,self.chs_lyr1])
        self.downc1 = DownBlock(in_ch=self.chs_lyr1+3, out_ch=self.chs_lyr2,filter_lrelu_flag=['128',False,self.chs_lyr2])
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
        # print(x0_256.size())
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
class Flatten(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        return x.flatten(self.index)

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
        )
        self.act = filter_lrelu(512, False,self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1,filter_lrelu_flag=['256',True,self.chs_lyr1])
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2,filter_lrelu_flag=['128',True,self.chs_lyr2])
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
        x0_256 = self.act( self.inc(torch.cat((in0, in1), dim=1)) )
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
        x = x

        return x, x2_64

class Discriminator(nn.Module):
    """ Definition of the FastDVDnet model.
    Inputs of forward():
        xn: input frames of dim [N, C, H, W], (C=3 RGB)
        noise_map: array with noise map of dim [N, 1, H, W]
    """
    def __init__(self, image_size=2):
        super(Discriminator, self).__init__()
        self.num_input_frames = 2
        # Define models of each denoising stage
        self.temp1 = DenBlock_catdown(num_input_frames=2)
        self.temp2 = DenBlock(num_input_frames=2)

        self.construction = Construction()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.to_logit = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.AdaptiveAvgPool2d((1,1)),
            Flatten(1),
            nn.Linear(64, 1)
        )
        self.conv_out = nn.Conv2d(3, 1, 1)

    def forward(self, x, noise_map):
            ''' Args:
                x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
                noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
            '''
            # Unpack inputs
            x0 = x*2 - 1
            x1 = x*2 - 1
            # First stage -- course reconstruction
            x_pool = self.temp2(self.pool(x0), self.pool(x1), self.pool(noise_map))
            x_pool_upx2 = self.up(x_pool)
            x_course = self.temp1(x0, x1, noise_map, x_pool)

            # Second stage -- construction
            x,temp = self.construction(x_pool_upx2, x_course )
            enc_out = self.to_logit(temp)

            dec_out = self.conv_out(x)

            return enc_out, dec_out


def cutmix_coordinates(height, width, alpha = 1.):
    lam = np.random.beta(alpha, alpha)

    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))

    return ((y0, y1), (x0, x1)), lam

def cutmix(source, target, coors, alpha = 1.):
    source, target = map(torch.clone, (source, target))
    ((y0, y1), (x0, x1)), _ = coors
    source[:, :, y0:y1, x0:x1] = target[:, :, y0:y1, x0:x1]
    return source

def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_penalty(images, outputs, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=outputs, inputs=images,
                           grad_outputs=list(map(lambda t: torch.ones(t.size()).cuda(), outputs)),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()






