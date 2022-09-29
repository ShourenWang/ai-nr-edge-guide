
import os
import subprocess
import glob
import logging
from random import choices # requires Python >= 3.6
import numpy as np
import cv2
import torch
from skimage.measure.simple_metrics import compare_psnr
from PIL import Image
# from tensorboardX import SummaryWriter
import torch.nn.functional as F
from models.nr_2d.canny_net import CannyDetector

IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types

def normalize_augment(datain, ctrl_fr_idx):
	'''Normalizes and augments an input patch of dim [N, num_frames, C. H, W] in [0., 255.] to \
		[N, num_frames*C. H, W] in  [0., 1.]. It also returns the central frame of the temporal \
		patch as a ground truth.
	'''
	img_train = datain[:,ctrl_fr_idx-2:ctrl_fr_idx+3,:,:,:]
	# convert to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]

	img_train = img_train.view(img_train.size()[0], -1, \
							   img_train.size()[-2], img_train.size()[-1])
	# print(img_train.size())
	gt_train = img_train[:,9:12,:,:]
	# print(gt_train.size())
	gt_train1 = torch.zeros_like(gt_train)
	gt_train1[:] = gt_train[:]

	# if np.random.randint(10) < 5:
	# 	gt_train = torch.clamp(gt_train, 0, 1)
	# 	img_train = torch.clamp(img_train, 0, 1)
	# 	img_train = img_train ** (2.2)
	# 	gt_train = gt_train ** (2.2)
	# std dev of each sequence

	# extract ground truth (central frame)
	# gt_train = torch.clamp(gt_train,0,1)
	# img_train = torch.clamp(img_train, 0, 1)
	return img_train, gt_train1



def get_imagenames(seq_dir, pattern=None):
	""" Get ordered list of filenames
	"""
	files = []
	for typ in IMAGETYPES:
		files.extend(glob.glob(os.path.join(seq_dir, typ)))

	# filter filenames
	if not pattern is None:
		ffiltered = []
		ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
		files = ffiltered
		del ffiltered

	# sort filenames alphabetically
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	return files

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=5):
	r""" Opens a sequence of images and expands it to even sizes if necesary
	Args:
		fpath: string, path to image sequence
		gray_mode: boolean, True indicating if images is to be open are in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
		max_num_fr: maximum number of frames to load
	Returns:
		seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	# Get ordered list of filenames
	files = get_imagenames(seq_dir)

	seq_list = []
	print("\tOpen sequence in folder: ", seq_dir)
	for fpath in files[0:max_num_fr]:

		img, expanded_h, expanded_w = open_image(fpath,\
												   gray_mode=gray_mode,\
												   expand_if_needed=expand_if_needed,\
												   expand_axis0=False)
		seq_list.append(img)
	seq = np.stack(seq_list, axis=0)
	return seq, expanded_h, expanded_w


def batch_psnr(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy().astype(np.float32)
	imgclean = imclean.data.cpu().numpy().astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
					   data_range=data_range)
	return psnr/img_cpu.shape[0]

def variable_to_cv2_image(invar, conv_rgb_to_bgr=True):
	r"""Converts a torch.autograd.Variable to an OpenCV image

	Args:
		invar: a torch.autograd.Variable
		conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
	Returns:
		a HxWxC uint8 image
	"""
	assert torch.max(invar) <= 1.0

	size4 = len(invar.size()) == 4
	if size4:
		nchannels = invar.size()[1]
	else:
		nchannels = invar.size()[0]

	if nchannels == 1:
		if size4:
			res = invar.data.cpu().numpy()[0, 0, :]
		else:
			res = invar.data.cpu().numpy()[0, :]
		res = (res*255.).clip(0, 255).astype(np.uint8)
	elif nchannels == 3:
		if size4:
			res = invar.data.cpu().numpy()[0]
		else:
			res = invar.data.cpu().numpy()
		res = res.transpose(1, 2, 0)
		res = (res*255.).clip(0, 255).astype(np.uint8)
		if conv_rgb_to_bgr:
			res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
	else:
		raise Exception('Number of color channels not supported')
	return res

def get_git_revision_short_hash():
	r"""Returns the current Git commit.
	"""
	return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def init_logger(log_dir, argdict):
	r"""Initializes a logging.Logger to save all the running parameters to a
	log file

	Args:
		log_dir: path in which to save log.txt
		argdict: dictionary of parameters to be logged
	"""
	from os.path import join

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(log_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	try:
		logger.info("Commit: {}".format(get_git_revision_short_hash()))
	except Exception as e:
		logger.error("Couldn't get commit number: {}".format(e))
	logger.info("Arguments: ")
	for k in argdict.keys():
		logger.info("\t{}: {}".format(k, argdict[k]))

	return logger

def init_logger_test(result_dir):
	r"""Initializes a logging.Logger in order to log the results after testing
	a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	from os.path import join

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def close_logger(logger):
	'''Closes the logger instance
	'''
	x = list(logger.handlers)
	for i in x:
		logger.removeHandler(i)
		i.flush()
		i.close()

def normalize(data):
	r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
	return np.float32(data/255.)

def svd_orthogonalization(lyr):
	r"""Applies regularization to the training by performing the
	orthogonalization technique described in the paper "An Analysis and Implementation of
	the FFDNet Image Denoising Method." Tassano et al. (2019).
	For each Conv layer in the model, the method replaces the matrix whose columns
	are the filters of the layer by new filters which are orthogonal to each other.
	This is achieved by setting the singular values of a SVD decomposition to 1.

	This function is to be called by the torch.nn.Module.apply() method,
	which applies svd_orthogonalization() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		weights = lyr.weight.data.clone()
		c_out, c_in, f1, f2 = weights.size()
		dtype = lyr.weight.data.type()

		# Reshape filters to columns
		# From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
		weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

		try:
			# SVD decomposition and orthogonalization
			mat_u, _, mat_v = torch.svd(weights)
			weights = torch.mm(mat_u, mat_v.t())

			lyr.weight.data = weights.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1).contiguous().type(dtype)
		except:
			pass
	else:
		pass

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary


	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = v

	return new_state_dict

def grad_true(model, layer_name):
	for name, param in model.named_parameters():
		param.requires_grad = False
	for name, param in model.named_parameters():
		if len(layer_name) == 1:
			# print(layer_name[0])
			if layer_name[0] in name:
				param.requires_grad = True
		if len(layer_name) == 2:
			if layer_name[0] in name or layer_name[1] in name:
				param.requires_grad = True
		if len(layer_name) == 3:
			if layer_name[0] in name or layer_name[1] in name or layer_name[2] in name:
				param.requires_grad = True
def grad_false(model, layer_name):
	for name, param in model.named_parameters():
		param.requires_grad = True
	for name, param in model.named_parameters():
		if len(layer_name) == 1:
			if layer_name[0] in name:
				#print(name)
				param.requires_grad = False
		if len(layer_name) == 2:
			if layer_name[0] in name or layer_name[1] in name:
				param.requires_grad = False
		if len(layer_name) == 3:
			if layer_name[0] in name or layer_name[1] in name or layer_name[2] in name:
				param.requires_grad = False


class L1_Charbonnier_loss(torch.nn.Module):
	"""L1 Charbonnierloss."""

	def __init__(self):
		super(L1_Charbonnier_loss, self).__init__()
		self.eps = 1e-6

	def forward(self, X, Y):
		# weight = torch.ones_like(X)
		# weight[:,:,2:-2,4:-4] = 2
		diff = torch.add(X, -Y)
		error = torch.sqrt(diff * diff + self.eps)
		loss = torch.mean(error)
		return loss

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image



def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def loss_tv(img_in):
	l1loss = torch.nn.L1Loss()
	img_left = img_in[:, :, 1:, :]
	img_right = img_in[:, :, :-1, :]
	img_down = img_in[:, :, :, 1:]
	img_up = img_in[:, :, :, :-1]
	loss = l1loss(img_left, img_right) + l1loss(img_up, img_down)

	return loss

# def BCEloss(D_fake, D_real, d_real_target, d_fake_target):
#     real =  F.binary_cross_entropy_with_logits(D_real,d_real_target.expand_as(D_real))
#     fake =  F.binary_cross_entropy_with_logits(D_fake,d_fake_target.expand_as(D_fake))
#     return real, fake

def BCEloss(D_fake,target):
	return F.binary_cross_entropy_with_logits(D_fake, target.expand_as(D_fake), reduction="mean")


from torch import nn
class Haar(nn.Module):
	alpha = 0.5
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		ll = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
		lh = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
		hl = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
		hh = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
		return ll,lh,hl,hh#torch.cat([ll,lh,hl,hh], axis=1)
class IHaar(nn.Module):
	alpha = 0.5

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Performs a 2d DWT Inverse reconstruction of an image using Haar Wavelets
		Arguments:
			x (torch.Tensor): input tensor of shape [b, c, h, w]
		Returns:
			out (torch.Tensor): output tensor of shape [b, c / 4, h * 2, w * 2]
		"""
		assert x.size(1) % 4 == 0, "The number of channels must be divisible by 4."
		size = [x.shape[0], x.shape[1] // 4, x.shape[2] * 2, x.shape[3] * 2]
		f = lambda i: x[:, size[1] * i : size[1] * (i + 1)]
		out = torch.zeros(size, dtype=x.dtype, device=x.device)
		out[:,:,0::2,0::2] = self.alpha * (f(0) + f(1) + f(2) + f(3))
		out[:,:,0::2,1::2] = self.alpha * (f(0) + f(1) - f(2) - f(3))
		out[:,:,1::2,0::2] = self.alpha * (f(0) - f(1) + f(2) - f(3))
		out[:,:,1::2,1::2] = self.alpha * (f(0) - f(1) - f(2) + f(3))
		return out

def nhaar(x):
	alpha = 0.5
	ll = alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
	lh = alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
	hl = alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
	hh = alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
	return ll,lh,hl,hh#torch.cat([ll,lh,hl,hh], axis=1)

def nihaar(x):
	alpha = 0.5

	size = [x.shape[0], x.shape[1] // 4, x.shape[2] * 2, x.shape[3] * 2]
	f = lambda i: x[:, size[1] * i : size[1] * (i + 1),:,:]
	out = torch.zeros(size, dtype=x.dtype, device=x.device)
	out[:,:,0::2,0::2] = alpha * (f(0) + f(1) + f(2) + f(3))
	out[:,:,0::2,1::2] = alpha * (f(0) + f(1) - f(2) - f(3))
	out[:,:,1::2,0::2] = alpha * (f(0) - f(1) + f(2) - f(3))
	out[:,:,1::2,1::2] = alpha * (f(0) - f(1) - f(2) + f(3))
	return out

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

class g_blur(nn.Module):
    def __init__(self, ksize,sigma,outchannel):
        super(g_blur, self).__init__()
        self.ksize = ksize
        self.out_channels = outchannel
        kernel2 = get_gaussian_kernel(kernel_size=self.ksize, sigma=sigma)
        kernel2 = torch.FloatTensor(kernel2).expand(outchannel, 1, self.ksize,self.ksize)
        self.weight2 = torch.nn.Parameter(data=kernel2, requires_grad=False)

    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.weight2,padding=self.ksize//2, groups=self.out_channels )
        return x

def loss_color(model):       # Color Transform
    loss_orth = torch.tensor(0., dtype = torch.float32).cuda()
    params = {}
    for name, param in model.named_parameters():
        params[name] = param
    ct = params['ct.net1.weight'].squeeze() #module.
    cti = params['cti.net1.weight'].squeeze()
    weight_squared = torch.matmul(ct, cti)
    diag = torch.eye(weight_squared.shape[0], dtype=torch.float32).cuda()
    loss = ((weight_squared - diag) **2).sum()
    loss_orth += loss
    return loss_orth

def loss_wavelet(model):
    loss_orth = torch.tensor(0., dtype = torch.float32).cuda()
    params = {}
    for name, param in model.named_parameters():
        params[name] = param
    ft = params['ft.net1.weight'].squeeze()
    fti = torch.cat([params['fti.net1.weight'],params['fti.net2.weight']],dim= 0).squeeze()
    weight_squared = torch.matmul(ft, fti)

    diag = torch.eye(weight_squared.shape[1], dtype=torch.float32).cuda()
    loss=((weight_squared - diag) **2).sum()
    loss_orth += loss
    return loss_orth

def addedge(img, edge,weight):
	img = torch.clamp(img,1e-10,1)
	imgy = 0.299 *img[:,0,:,:]+0.587 *img[:,1,:,:]+0.114 *img[:,2,:,:]
	imgy_e = imgy + weight*torch.clamp(edge[:,0,:,:], -16/255.0, 16/255.0)
	imgy_e = torch.clamp(imgy_e, 1e-10, 1)
	img[:, 0, :, :] = img[:,0,:,:] * imgy_e/imgy
	img[:, 1, :, :] = img[:, 1, :, :] * imgy_e / imgy
	img[:, 2, :, :] = img[:, 2, :, :] * imgy_e / imgy
	return img

def block_run(model,inputs,noisemap,blocksize,overlap):
	overlap_w = overlap * 2
	inc = inputs.size()[1]//2
	model = model.cuda()
	inputs =inputs.cuda()
	noisemap =noisemap.cuda()
	temp = inputs[:,:inc,:,]
	output = torch.zeros_like(temp)
	ben_i = 0
	i_flag = 0
	gau_blur3 = g_blur(3, 1, 1).cuda()
	gau_blur5 = g_blur(5, 1, inc).cuda()
	canny_operator = CannyDetector().cuda()
	mean_val = torch.mean(inputs)
	for i in range(inputs.size()[-2]*2//blocksize):
		if ben_i+ blocksize<= inputs.size()[-2] and i_flag<2:
			j_flag = 0
			ben_j = 0
			for j in range(inputs.size()[-1]*2//blocksize//8):
				if ben_j+blocksize*8 <= inputs.size()[-1] and j_flag<2:
					patch = inputs[:,:,ben_i:ben_i+blocksize,ben_j:ben_j+blocksize*8]
					pathc_map = noisemap[:,:,ben_i:ben_i+blocksize,ben_j:ben_j+blocksize*8]
					# adj_map = torch.floor ( ( torch.mean(patch) - mean_val ) / 10 ) * 10 / 200.0
					# adj_map = torch.clamp(adj_map,-1,1)
					# pathc_map = - torch.floor(adj_map * 5) / 255.0 + pathc_map
					adj_map = torch.floor( ( torch.mean(patch) - mean_val)*255.0/10.0 ) * 10
					if inc == 3:
						adj_map = torch.clamp(adj_map / 40, -1, 0.5) * 5
					else:
						adj_map = torch.clamp(adj_map / 40,-1,0.5)*5
					# print(adj_map)
					# pathc_map = -adj_map/255.0 + pathc_map
					results = model(patch,pathc_map)
					den =  results[-1]
					den = den[:,:-1,:,:]
					# print(den.size())

					den = torch.clamp(den, 0, 1)
					temp_pat = patch[:, inc:, :, :]*0.5 + den * 0.5
					# den[:,:,2:-2,2:-2] = den[:,:,2:-2,2:-2] + 0.1*(temp_pat - temp_pat )[:,:,2:-2,2:-2]
					den = torch.clamp(den, 0, 1)
					# den = den * 0.9 + patch[:,inc:,:,:] *0.1
					# sigma = 0.33
					# scale = 1
					# v = torch.median(patch)
					# lower = int(max(0, (1.0 - sigma) * v)) * scale
					# upper = int(min(1, (1.0 + sigma) * v)) * scale
					# # print(lower, upper,v)
					#
					# result = canny_operator(patch, threshold1=lower,
					# 						threshold2=upper).squeeze(1)
					# for cc in range(4):
					# 	den[:,cc,:,:] = den[:,cc,:,:] * (1-result)*0.9 + patch[:, inc+cc, :, :] *result*0.1
					# den = torch.clamp(den,0,1)
					# 方案1
					# output[:,:,ben_i:ben_i+blocksize,ben_j:ben_j+blocksize] = den[:]
					# 方案2
					# output[:,:,ben_i+overlap//2:ben_i+blocksize-overlap//2,
					# ben_j+overlap//2:ben_j+blocksize-overlap//2]=\
					# 	den[:,:,overlap//2:-overlap//2, overlap//2:-overlap//2]
					# 方案3
					# if j>0:
					# 	den[:, :, :, :overlap] = preblock[:, :, :, -overlap:] * 0.5 + den[:, :, :, :overlap] * 0.5
					# 方案4
					weights = torch.ones((1,inc,blocksize,overlap_w))
					for kk in range(overlap_w):
						weights[:,:,:,kk] = (kk+1)/(overlap_w+1)
					weights = weights.cuda()
					if j > 0:
						den[:, :, :, :overlap_w] = preblock[:, :, :, -overlap_w:] * (1-weights) + den[:, :, :, :overlap_w] * (weights)

					output[:, :, ben_i+overlap//2:ben_i+blocksize-overlap//2, \
					ben_j:ben_j + blocksize*8] = den[:,:,overlap//2:-overlap//2,:]
					preblock = den[:]
					preblock = preblock.cuda()

					ben_j = ben_j + blocksize*8 - overlap

					if ben_j + blocksize*8 >= inputs.size()[-1]:
						ben_j = inputs.size()[-1]- blocksize*8
						j_flag += 1
				else:
					pass
			ben_i = ben_i + blocksize - overlap
			if ben_i + blocksize >= inputs.size()[-2]:
				ben_i = inputs.size()[-2] - blocksize
				i_flag += 1
		else:
			pass

	return output


def make_kernel(ksize):
	kernel = np.zeros((2 * ksize + 1, 2 * ksize + 1), np.float32)
	for d in range(1, ksize + 1):
		kernel[ksize - d:ksize + d + 1, ksize - d:ksize + d + 1] += (1.0 / ((2 * d + 1) ** 2))
	kernel = kernel / kernel.sum()
	kernel = torch.FloatTensor(kernel)
	return kernel

class dim4_to_3(nn.Module):
	def __init__(self,):
		super(dim4_to_3, self).__init__()
		kernel = np.zeros((3,4,1, 1), np.float32)
		kernel[0, 0, 0, 0] = 1
		kernel[1, 1, 0, 0] = 1
		kernel[2, 2, 0, 0] = 1
		kernel = torch.FloatTensor(kernel)
		self.weight2 = torch.nn.Parameter(data=kernel, requires_grad=False)
	def forward(self, x):
		x = torch.nn.functional.conv2d(x, self.weight2,padding=0)
		return x


def NLmeans_filter(src, f, t, h):
	if len(src.size()) > 3 :
		batch, chs, H, W = src.size()
	assert batch == 1

	out = torch.zeros_like(src)
	pad_length = f+t

	kernel = make_kernel(f)
	kernel = kernel.cuda()
	h2 = h*h
	pad = nn.ReflectionPad2d(pad_length).cuda()
	src_padding_total = pad(src)
	for cc in range(chs):
		src_padding = src_padding_total[0,cc,:,:]
		for i in range(0, H):
			for j in range(0, W):
				i1 = i + f + t
				j1 = j + f + t
				W1 = src_padding[i1-f:i1+f+1, j1-f:j1+f+1] # 领域窗口W1
				# W1 = W1.cuda()
				w_max = 0
				aver = 0
				weight_sum = 0
				# 搜索窗口
				for r in range(i1-t, i1+t+1):
					for c in range(j1-t, j1+t+1):
						if (r==i1) and (c==j1):
							continue
						else:
							W2 = src_padding[r-f:r+f+1, c-f:c+f+1] # 搜索区域内的相似窗口
							# W2 = W2.cuda()
							Dist2 = torch.sum(kernel*(W1-W2)*(W1-W2))
							w = torch.exp(-Dist2/h2)
							if w > w_max:
								w_max = w
							weight_sum += w
							aver += w*src_padding[r, c]
				aver += w_max*src_padding[i1, j1] # 自身领域取最大的权重
				weight_sum += w_max
				out[0,cc, i, j] = aver/weight_sum

	return out

# 加载我们自己的数据
def convert_yuv444(yuv_file, ww, hh):
    with open(yuv_file) as f:
        rectype = np.dtype(np.uint16)
        raw = np.fromfile(f, dtype=rectype)
    raw = np.clip( raw / 4.0,0,255)
    shape2 = (ww,hh)
    rgb = np.zeros((ww, hh, 3), dtype=np.uint8)
    y = raw[ww * hh * 0: 1*ww*hh].reshape(shape2)
    u = raw[ww * hh * 1: 2*ww*hh].reshape(shape2) - 128
    v = raw[ww * hh * 2: 3*ww*hh].reshape(shape2) - 128

    rgb[:, :, 0] = np.clip( (y + 1.4075 * v),0,255)
    rgb[:, :, 1] = np.clip( (y - 0.7169 * v - 0.3455 * u),0,255)
    rgb[:, :, 2] = np.clip( (y + 1.779 * u),0,255)

    return rgb


def line_run(model,inputs,noisemap,linesize,overlap):
	inc = inputs.size()[1]//2
	model = model.cuda()
	inputs =inputs.cuda()
	noisemap =noisemap.cuda()
	temp = inputs[:,:inc,:,]
	output = torch.zeros_like(temp)
	ben_i = 0
	i_flag = 0

	mean_val = torch.mean(inputs)
	for i in range(inputs.size()[-2]*2//linesize):

		patch = inputs[:,:,ben_i:ben_i+linesize,:]
		pathc_map = noisemap[:,:,ben_i:ben_i+linesize:]

		results = model(patch,pathc_map)
		den =  results[-1]
		den = torch.clamp(den, 0, 1)[:,:inc,:,:]
		den = torch.clamp(den, 0, 1)

		weights = torch.ones((1,1,overlap, patch.size()[-1]))
		for kk in range(overlap):
			weights[:,:,kk,:] = (kk+1)/(overlap+1)
		weights = weights.cuda()
		if i > 0:
			den[:, :, :overlap, :] = preblock[:, :, -overlap:, :] * (1-weights) + den[:, :, :overlap, :] * (weights)
			output[:, :, ben_i+overlap//2:ben_i+linesize-overlap//2, :] = den[:,:,overlap//2:-overlap//2,:]
		preblock = den[:]
		preblock = preblock.cuda()

		ben_i = ben_i + linesize - overlap
		if ben_i + linesize >= inputs.size()[-2]:
			ben_i = inputs.size()[-2] - linesize
			i_flag += 1
		else:
			pass

	return output






