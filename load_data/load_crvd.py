import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import time
import torch
iso_list = [1600, 3200, 6400, 12800, 25600]
a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
b_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]
import config.config as hyper_cfg

def load_cvrd_data(shift, noisy_level, scene_ind, frame_ind, xx, yy):
	frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
	gt_name = os.path.join(hyper_cfg.data_root[1],
						   'indoor_raw_gt/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(
							   scene_ind, iso_list[noisy_level],
							   frame_list[frame_ind + shift]))
	gt_raw = cv2.imread(gt_name, -1)
	gt_raw_full = gt_raw
	gt_raw_patch = gt_raw_full[yy:yy + hyper_cfg.image_height * 2,
				   xx:xx + hyper_cfg.image_width * 2]  # 256 * 256
	gt_raw_pack = np.expand_dims(pack_gbrg_raw(gt_raw_patch), axis=0)  # 1* 128 * 128 * 4

	noisy_frame_index_for_current = np.random.randint(0, 10)

	input_name = os.path.join(hyper_cfg.data_root[1],
							  'indoor_raw_noisy/scene{}/ISO{}/frame{}_noisy{}.tiff'.format(
								  scene_ind, iso_list[noisy_level],
								  frame_list[frame_ind + shift], noisy_frame_index_for_current))
	noisy_raw = cv2.imread(input_name, -1)
	noisy_raw_full = noisy_raw
	noisy_patch = noisy_raw_full[yy:yy + hyper_cfg.image_height * 2, xx:xx + hyper_cfg.image_width * 2]
	input_pack = np.expand_dims(pack_gbrg_raw(noisy_patch), axis=0)
	return input_pack, gt_raw_pack


def load_eval_data(noisy_level, scene_ind):
	input_batch_list = []
	gt_raw_batch_list = []

	input_pack_list = []
	gt_raw_pack_list = []

	xx = 200
	yy = 200

	for shift in range(0, 7):
		# load gt raw
		frame_ind = 0
		if full_res:
			input_pack, gt_raw_pack = load_cvrd_data_full(shift, noisy_level, scene_ind, frame_ind, xx, yy)
		else:
			input_pack, gt_raw_pack = load_cvrd_data(shift, noisy_level, scene_ind, frame_ind, xx, yy)
		input_pack_list.append(input_pack)
		gt_raw_pack_list.append(gt_raw_pack)

	input_pack_frames = np.concatenate(input_pack_list, axis=3)
	gt_raw_pack_frames = np.concatenate(gt_raw_pack_list, axis=3)

	input_batch_list.append(input_pack_frames)
	gt_raw_batch_list.append(gt_raw_pack_frames)

	input_batch = np.concatenate(input_batch_list, axis=0)
	gt_raw_batch = np.concatenate(gt_raw_batch_list, axis=0)

	in_data = torch.from_numpy(input_batch.copy()).permute(0, 3, 1, 2).cuda()  # 1 * (4*25) * 128 * 128
	gt_raw_data = torch.from_numpy(gt_raw_batch.copy()).permute(0, 3, 1, 2).cuda()  # 1 * (4*25) * 128 * 128
	return in_data, gt_raw_data

def generate_file_list(scene_list):
	iso_list = [1600, 3200, 6400, 12800, 25600]
	file_num = 0
	data_name = []
	for scene_ind in scene_list:
		for iso in iso_list:
			for frame_ind in range(1,8):
				gt_name = os.path.join('ISO{}/scene{}_frame{}_gt_sRGB.png'.format(
								 iso, scene_ind, frame_ind-1))
				data_name.append(gt_name)
				file_num += 1

	random_index = np.random.permutation(file_num)
	data_random_list = []
	for i,idx in enumerate(random_index):
		data_random_list.append(data_name[idx])
	return data_random_list

def read_img(img_name, xx, yy,full_res=False):
	raw_full = cv2.imread(img_name, -1)
	if full_res:
		raw_patch = raw_full[:1080//32 * 32,:]
	else:
		raw_patch = raw_full[yy:yy + hyper_cfg.image_height * 2,\
					xx:xx + hyper_cfg.image_width * 2]  # 256 * 256
	raw_pack_data = pack_gbrg_raw(raw_patch)
	return raw_pack_data

def decode_data(data_name,trainflag,full_res):
	frame_list = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
	H = 1080
	W = 1920
	if trainflag:
		xx = np.random.randint(0, (W - hyper_cfg.image_width * 2) ) / 2 * 2
		yy = np.random.randint(0, (H - hyper_cfg.image_height * 2) ) / 2 * 2
		total_frame = hyper_cfg.frame_num
	else:
		xx = 200
		yy = 200
		total_frame = 7

	scene_ind = data_name.split('/')[1].split('_')[0]
	frame_ind = int(data_name.split('/')[1].split('_')[1][5:])
	iso_ind = data_name.split('/')[0]

	noisy_level_ind = iso_list.index(int(iso_ind[3:]))
	noisy_level = [a_list[noisy_level_ind], b_list[noisy_level_ind]]
	if full_res:
		noise_map = np.zeros((1,1080//32*16,1920//2))
	else:
		noise_map = np.zeros((1,hyper_cfg.image_height,hyper_cfg.image_width))
	noise_map[:] = noisy_level[0]/255.0

	gt_name_list = []
	noisy_name_list = []
	xx_list = []
	yy_list = []

	for shift in range(0, total_frame):
		gt_name = os.path.join(hyper_cfg.data_root[1],'indoor_raw_gt/{}/{}/frame{}_clean_and_slightly_denoised.tiff'.format(
							   scene_ind,iso_ind,frame_list[frame_ind + shift]))

		noisy_frame_index_for_current = np.random.randint(0, 10)
		noisy_name = os.path.join(hyper_cfg.data_root[1],
								  'indoor_raw_noisy/{}/{}/frame{}_noisy{}.tiff'.format(
									  scene_ind, iso_ind, frame_list[frame_ind + shift], noisy_frame_index_for_current))

		gt_name_list.append(gt_name)
		noisy_name_list.append(noisy_name)

		xx_list.append(xx)
		yy_list.append(yy)
	gt_raw_data_list = []
	noisy_data_list = []
	for ii in range(len(xx_list)):
		gt_raw  = read_img(gt_name_list[ii], int(xx_list[ii]), int(yy_list[ii]), full_res )
		noise_raw = read_img( noisy_name_list[ii], int(xx_list[ii]), int(yy_list[ii]),full_res )
		gt_raw_data_list.append(gt_raw)
		noisy_data_list.append(noise_raw)

	gt_raw_batch = np.concatenate(gt_raw_data_list, axis=2)
	noisy_raw_batch = np.concatenate(noisy_data_list, axis=2)
	gt_raw_batch = gt_raw_batch.transpose(2,0,1)
	noisy_raw_batch = noisy_raw_batch.transpose(2, 0, 1)

	return noisy_raw_batch, gt_raw_batch, noise_map


class loadImgs(Dataset):
	def __init__(self, filelist, trainflag=True,full_res=False):
		self.filelist = filelist
		self.trainflag = trainflag
		self.full_res = full_res

	def __len__(self):
		return len(self.filelist)

	def __getitem__(self, item):
		self.data_name = self.filelist[item]
		image, label, noisy_level = decode_data(self.data_name,self.trainflag,self.full_res)
		self.image = image
		self.label = label
		self.noisy_level = noisy_level
		return self.image, self.label, self.noisy_level


''' ====== compute psnr and ssim   ======= '''
from scipy.stats import poisson
from skimage.measure.simple_metrics import compare_psnr as compare_ssim
# from skimage.measure.simple_metrics import compare_psnr
import torch.nn.functional as F
import torch.nn as nn

def pack_gbrg_raw(raw):
	#pack GBRG Bayer raw to 4 channels
	black_level = 240
	white_level = 2**12-1
	im = raw.astype(np.float32)
	im = np.maximum(im - black_level, 0) / (white_level-black_level)

	im = np.expand_dims(im, axis=2)
	img_shape = im.shape
	H = img_shape[0]
	W = img_shape[1]
	# out = np.concatenate((im[1:H:2, 0:W:2, :],          # r
	# 					  im[1:H:2, 1:W:2, :],          # gr
	# 					  im[0:H:2, 1:W:2, :],          # b
	# 					  im[0:H:2, 0:W:2, :]), axis=2) # gb
	out = np.concatenate((im[1:H:2, 0:W:2, :],  # r
						  im[1:H:2, 1:W:2, :],  # gr
						  im[0:H:2, 0:W:2, :],  # gb
						  im[0:H:2, 1:W:2, :],  # b
						  ), axis=2)  # gb
	return out



def compute_sigma(input, a, b):
	sigma = np.sqrt((input - 240) * a + b)
	return sigma


def preprocess(raw):
	input_full = raw.transpose((0, 3, 1, 2))
	input_full = torch.from_numpy(input_full)
	input_full = input_full.cuda()
	return input_full

def tensor2numpy(raw):  # raw: 1 * 4 * H * W
	input_full = raw.permute((0, 2, 3, 1)) # 1 * H * W * 4
	input_full = input_full.data.cpu().numpy()
	output = np.clip(input_full,0,1)
	return output

def pack_rggb_raw_for_compute_ssim(raw):
	im = raw.astype(np.float32)
	im = np.expand_dims(im, axis=2)
	img_shape = im.shape
	H = img_shape[0]
	W = img_shape[1]
	out = np.concatenate((im[0:H:2, 0:W:2, :],
						  im[0:H:2, 1:W:2, :],
						  im[1:H:2, 1:W:2, :],
						  im[1:H:2, 0:W:2, :]), axis=2)
	return out

def compute_ssim_for_packed_raw(raw1, raw2):
	raw1_pack = pack_rggb_raw_for_compute_ssim(raw1)
	raw2_pack = pack_rggb_raw_for_compute_ssim(raw2)
	test_raw_ssim = 0
	for i in range(4):
		test_raw_ssim += compare_ssim(raw1_pack[:,:,i], raw2_pack[:,:,i], data_range=1.0)

	return test_raw_ssim/4


class PSNR(nn.Module):
	def __init__(self):
		super(PSNR, self).__init__()

	def forward(self, image, label):
		MSE = (image - label) * (image - label)
		MSE = torch.mean(MSE)
		PSNR = 10 * torch.log(1 / MSE) / torch.log(torch.Tensor([10.])).cuda()  # torch.log is log base e
		return PSNR