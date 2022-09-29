import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,5"

import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# from models.models import FastDVDnet
from models.cvpr.raw_2frame_MD_v2 import FastDVDnet
from models.raw_discriminator import *
# from load_data.load_crvd import *
# from load_data.dataset import myDataset
from utils.utils import normalize_augment,L1_Charbonnier_loss,grad_true,g_blur,grad_false
from utils.utils import BCEloss,loss_tv
from torch.utils.data import DataLoader
import numpy as np
import cv2
from utils.vggloss import VGG19_Extractor
import warnings
warnings.filterwarnings("ignore")
from models.nr_2d.canny_net import CannyDetector
from random import random
import torch.nn.functional as F
import torch.distributions as tdist
from models.nr_2d.marval_net import Debayer2x2 as Debayer5x5
import config.config as cfg
from skimage.measure import compare_psnr

iso_list  = [1600, 3200, 6400, 12800, 25600]

# Define GPU devices
device_ids = [0,1]
torch.backends.cudnn.benchmark = True  # CUDNN optimization

# Create model
G = FastDVDnet()
G = nn.DataParallel(G, device_ids=device_ids).cuda()

# Optimizer
G_optimizer = optim.Adam(G.parameters(), lr=cfg.learning_rate)
checkpoint = torch.load(cfg.model_save_root+'modelv2_best.pth')
start_epoch = checkpoint['epoch']
G.load_state_dict(checkpoint['model'])
G_optimizer.load_state_dict(checkpoint['optimizer'])
print(start_epoch)
# eval_psnr = evaluate(G, compsnr,False,True)
# eval_psnr = eval_psnr.item()
# print('evaluate full res psnr is:', eval_psnr)

def depack_rggb_raw(raw):#1,h,w,4
	H = raw.shape[1]
	W = raw.shape[2]
	output = np.zeros((H*2,W*2))
	output[1::2, 0::2] = raw[0,:,:,0]
	output[1::2, 1::2] = raw[0,:,:,1]
	output[0::2, 0::2] = raw[0,:,:,2]
	output[0::2, 1::2] = raw[0,:,:,3]

	return output

def pack_rggb_raw(raw):
	#pack RGGB Bayer raw to 4 channels
	black_level = 240
	white_level = 2**12-1
	im = raw.astype(np.float32)
	im = np.maximum(im - black_level, 0) / (white_level-black_level)

	im = np.expand_dims(im, axis=2)
	img_shape = im.shape
	H = img_shape[0]
	W = img_shape[1]
	out = np.concatenate((im[1:H:2, 0:W:2, :],  # r
						  im[1:H:2, 1:W:2, :],  # gr
						  im[0:H:2, 0:W:2, :],  # gb
						  im[0:H:2, 1:W:2, :],  # b
						  ), axis=2)  # gb
	return out

def preprocess(raw):
	input_full = raw.transpose(0, 3, 1, 2)
	input_full = torch.from_numpy(input_full).float()
	input_full = input_full.cuda()
	return input_full

def postprocess(output):
    output = output.cpu()
    output = output.detach().numpy().astype(np.float32)
    output = output.transpose( 0, 2, 3, 1)
    output = np.clip(output,0,1)
    return output

def test_big_size_raw(input_data, denoiser,noise_level, patch_h=256, patch_w=256, patch_h_overlap=64, patch_w_overlap=64):
	H = input_data.shape[1]
	W = input_data.shape[2]
	#print(input_data.shape) #1,w,h,c
	test_result = np.zeros((input_data.shape[0], H, W, 4))
	noise_map = torch.zeros((1, 1, patch_h, patch_w)).float().cuda()
	# noise_map[:] = noise_level
	# noise_map = preprocess(noise_map)
	input_data = preprocess(input_data) # b,c,h,w

	h_index = 1
	while (patch_h * h_index - patch_h_overlap * (h_index - 1)) < H:
		test_horizontal_result = np.zeros((input_data.size()[0], patch_h, W, 4))
		h_begin = patch_h * (h_index - 1) - patch_h_overlap * (h_index - 1)
		h_end = patch_h * h_index - patch_h_overlap * (h_index - 1)
		w_index = 1
		while (patch_w * w_index - patch_w_overlap * (w_index - 1)) < W:
			w_begin = patch_w * (w_index - 1) - patch_w_overlap * (w_index - 1)
			w_end = patch_w * w_index - patch_w_overlap * (w_index - 1)
			test_patch = input_data[:, :,h_begin:h_end, w_begin:w_end]
			# test_patch = preprocess(test_patch)

			with torch.no_grad():
				coeff_a = torch.from_numpy(np.array(int(noise_level) / 255.0)).float().cuda()
				# noise_map = torch.zeros((1,1,test_patch.size()[-2],test_patch.size()[-1])).float().cuda()
				# print(noise_map.size())
				_,_,output_patch = denoiser(test_patch, coeff_a.expand_as(noise_map))
			test_patch_result = postprocess(output_patch)

			if w_index == 1:
				# print(test_horizontal_result[:, :, w_begin:w_end, :].shape, test_patch_result.shape)
				test_horizontal_result[:, :, w_begin:w_end, :] = test_patch_result[:]

			else:
				for i in range(patch_w_overlap):
					test_horizontal_result[:, :, w_begin + i, :] = test_horizontal_result[:, :, w_begin + i, :] * (
								patch_w_overlap - 1 - i) / (patch_w_overlap - 1) + test_patch_result[:, :, i, :] * i / (
																			   patch_w_overlap - 1)
				test_horizontal_result[:, :, w_begin + patch_w_overlap:w_end, :] = test_patch_result[:, :,
																				   patch_w_overlap:, :]
			w_index += 1

		test_patch = input_data[:,:, h_begin:h_end, -patch_w:]
		# test_patch = preprocess(test_patch)
		with torch.no_grad():
			coeff_a = torch.from_numpy(np.array(int(noise_level) / 255.0)).float().cuda()
			# noise_map = torch.zeros((1, 1, test_patch.size()[-2], test_patch.size()[-1])).float().cuda()
			# print(noise_map.size())
			_, _, output_patch = denoiser(test_patch, coeff_a.expand_as(noise_map))
		test_patch_result = postprocess(output_patch)
		last_range = w_end - (W - patch_w)
		for i in range(last_range):
			test_horizontal_result[:, :, W - patch_w + i, :] = test_horizontal_result[:, :, W - patch_w + i, :] * (
						last_range - 1 - i) / (last_range - 1) + test_patch_result[:, :, i, :] * i / (last_range - 1)
		test_horizontal_result[:, :, w_end:, :] = test_patch_result[:, :, last_range:, :]

		if h_index == 1:
			test_result[:, h_begin:h_end, :, :] = test_horizontal_result
		else:
			for i in range(patch_h_overlap):
				test_result[:, h_begin + i, :, :] = test_result[:, h_begin + i, :, :] * (patch_h_overlap - 1 - i) / (
							patch_h_overlap - 1) + test_horizontal_result[:, i, :, :] * i / (patch_h_overlap - 1)
			test_result[:, h_begin + patch_h_overlap:h_end, :, :] = test_horizontal_result[:, patch_h_overlap:, :, :]
		h_index += 1

	test_horizontal_result = np.zeros((input_data.shape[0], patch_h, W, 4))
	w_index = 1
	while (patch_w * w_index - patch_w_overlap * (w_index - 1)) < W:
		w_begin = patch_w * (w_index - 1) - patch_w_overlap * (w_index - 1)
		w_end = patch_w * w_index - patch_w_overlap * (w_index - 1)
		test_patch = input_data[:,:, -patch_h:, w_begin:w_end]
		# test_patch = preprocess(test_patch)
		with torch.no_grad():
			coeff_a = torch.from_numpy(np.array(int(noise_level) / 255.0)).float().cuda()
			# noise_map = torch.zeros((1, 1, test_patch.size()[-2], test_patch.size()[-1])).float().cuda()
			_, _, output_patch = denoiser(test_patch, coeff_a.expand_as(noise_map))
		test_patch_result = postprocess(output_patch)
		if w_index == 1:
			test_horizontal_result[:, :, w_begin:w_end, :] = test_patch_result
		else:
			for i in range(patch_w_overlap):
				test_horizontal_result[:, :, w_begin + i, :] = test_horizontal_result[:, :, w_begin + i, :] * (
							patch_w_overlap - 1 - i) / (patch_w_overlap - 1) + test_patch_result[:, :, i, :] * i / (
																		   patch_w_overlap - 1)
			test_horizontal_result[:, :, w_begin + patch_w_overlap:w_end, :] = test_patch_result[:, :, patch_w_overlap:,
																			   :]
		w_index += 1

	test_patch = input_data[:,:, -patch_h:, -patch_w:]
	# test_patch = preprocess(test_patch)
	with torch.no_grad():
		coeff_a = torch.from_numpy(np.array(int(noise_level) / 255.0)).float().cuda()
		# noise_map = torch.zeros((1, 1, test_patch.size()[-2], test_patch.size()[-1])).float().cuda()
		# noise_map = torch.zeros((1, 1, test_patch.size()[-2], test_patch.size()[-1])).float().cuda()
		_, _, output_patch = denoiser(test_patch, coeff_a.expand_as(noise_map))
	test_patch_result = postprocess(output_patch)
	last_range = w_end - (W - patch_w)
	for i in range(last_range):
		test_horizontal_result[:, :, W - patch_w + i, :] = test_horizontal_result[:, :, W - patch_w + i, :] * (
					last_range - 1 - i) / (last_range - 1) + test_patch_result[:, :, i, :] * i / (last_range - 1)
	test_horizontal_result[:, :, w_end:, :] = test_patch_result[:, :, last_range:, :]

	last_last_range = h_end - (H - patch_h)
	for i in range(last_last_range):
		test_result[:, H - patch_w + i, :, :] = test_result[:, H - patch_w + i, :, :] * (last_last_range - 1 - i) / (
					last_last_range - 1) + test_horizontal_result[:, i, :, :] * i / (last_last_range - 1)
	test_result[:, h_end:, :, :] = test_horizontal_result[:, last_last_range:, :, :]
	test_result = np.clip(test_result,0,1)

	return test_result

iso_list = [1600, 3200, 6400, 12800, 25600]
a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]

whole_psnr = 0
cnt = 0

for iso in iso_list:
	print('processing iso={}'.format(iso))

	if not os.path.isdir(cfg.output_root + 'ISO{}'.format(iso)):
		os.makedirs(cfg.output_root + 'ISO{}'.format(iso))

	f = open('model_test_psnr_and_ssim_on_iso{}.txt'.format(iso), 'w')

	context = 'ISO{}'.format(iso) + '\n'
	f.write(context)

	scene_avg_raw_psnr = 0
	scene_avg_raw_ssim = 0
	scene_avg_srgb_psnr = 0
	scene_avg_srgb_ssim = 0


	for scene_id in range(7, 11 + 1):

		context = 'scene{}'.format(scene_id) + '\n'
		f.write(context)

		frame_avg_raw_psnr = 0
		frame_avg_raw_ssim = 0
		frame_avg_srgb_psnr = 0
		frame_avg_srgb_ssim = 0

		for i in range(1, 7 + 1):
			frame_list = []
			if i == 1:
				raw = cv2.imread('/home/data_beifen/crvd_nr/indoor_raw_noisy/scene{}/ISO{}/frame1_noisy0.tiff'.format(scene_id, iso), -1)
				input_full = np.expand_dims(pack_rggb_raw(raw), axis=0)
				frame_list.append(input_full)
				frame_list.append(input_full)
			else:
				frame_list.append(pre)
				raw = cv2.imread('/home/data_beifen/crvd_nr/indoor_raw_noisy/scene{}/ISO{}/frame{}_noisy0.tiff'.format(scene_id, iso, i),
								 -1)
				input_full_cur = np.expand_dims(pack_rggb_raw(raw), axis=0)
				frame_list.append(input_full_cur)

			input_data = np.concatenate(frame_list, axis=3)

			noisy_level_ind = iso_list.index(int(iso))
			noisy_level = a_list[noisy_level_ind]

			test_result = test_big_size_raw(input_data, G,noisy_level, patch_h=256, patch_w=256, patch_h_overlap=64,
											patch_w_overlap=64)  #  c,h,w
			pre = test_result
			if iso==25600 and i==1 and scene_id==10:
				temp  = np.clip( test_result[0,:,:,:].transpose(0,1,2),0,1)
				rgb = temp[:,:,:-1]
				rgb[:,:,2] = temp[:,:,3]
				cv2.imwrite('./results/iso25600.png',np.uint8(rgb*255)[:,:,::-1])
			test_result = depack_rggb_raw(test_result)

			test_gt = cv2.imread(
				'/home/data_beifen/crvd_nr/indoor_raw_gt/scene{}/ISO{}/frame{}_clean_and_slightly_denoised.tiff'.format(scene_id, iso, i),
				-1).astype(np.float32)

			if iso==25600 and i==1 and scene_id==10:
				test_gt1 = pack_rggb_raw(test_gt)
				rgb = test_gt1[:, :, :-1]
				rgb[:, :, 2] = test_gt1[:, :, 3]
				cv2.imwrite('./results/iso25600_gt.png', np.uint8(rgb * 255)[:, :, ::-1])

			test_gt = (test_gt - 240) / (2 ** 12 - 1 - 240)
			# test_raw_psnr = compare_psnr(test_gt, (
			# 			np.uint16(test_result * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (
			# 										 2 ** 12 - 1 - 240), data_range=1.0)
			test_raw_psnr = compare_psnr(test_gt,test_result, data_range=1.0)
			# test_raw_ssim = compute_ssim_for_packed_raw(test_gt, (
			# 			np.uint16(test_result * (2 ** 12 - 1 - 240) + 240).astype(np.float32) - 240) / (
			# 														2 ** 12 - 1 - 240))
			test_raw_ssim = 0
			#print('scene {} frame{} test raw psnr : {}, test raw ssim : {} '.format(scene_id, i, test_raw_psnr,
																					#test_raw_ssim))
			context = 'raw psnr/ssim: {}/{}'.format(test_raw_psnr, test_raw_ssim) + '\n'
			f.write(context)
			frame_avg_raw_psnr += test_raw_psnr
			frame_avg_raw_ssim += test_raw_ssim
			whole_psnr += test_raw_psnr
			cnt += 1


		frame_avg_raw_psnr = frame_avg_raw_psnr / 7
		frame_avg_raw_ssim = frame_avg_raw_ssim / 7
		frame_avg_srgb_psnr = frame_avg_srgb_psnr / 7
		frame_avg_srgb_ssim = frame_avg_srgb_ssim / 7
		print('iso: ', iso, ' scene: ', scene_id, ' psnr:', frame_avg_raw_psnr)
		context = 'frame average raw psnr:{},frame average raw ssim:{}'.format(frame_avg_raw_psnr,
																			   frame_avg_raw_ssim) + '\n'
		f.write(context)
		context = 'frame average srgb psnr:{},frame average srgb ssim:{}'.format(frame_avg_srgb_psnr,
																				 frame_avg_srgb_ssim) + '\n'
		f.write(context)

		scene_avg_raw_psnr += frame_avg_raw_psnr
		scene_avg_raw_ssim += frame_avg_raw_ssim
		scene_avg_srgb_psnr += frame_avg_srgb_psnr
		scene_avg_srgb_ssim += frame_avg_srgb_ssim

	scene_avg_raw_psnr = scene_avg_raw_psnr / 5
	scene_avg_raw_ssim = scene_avg_raw_ssim / 5
	scene_avg_srgb_psnr = scene_avg_srgb_psnr / 5
	scene_avg_srgb_ssim = scene_avg_srgb_ssim / 5

	#print('scene avg psnr:   ',scene_avg_raw_psnr)
	context = 'scene average raw psnr:{},scene frame average raw ssim:{}'.format(scene_avg_raw_psnr,
																				 scene_avg_raw_ssim) + '\n'
	f.write(context)
	context = 'scene average srgb psnr:{},scene frame average srgb ssim:{}'.format(scene_avg_srgb_psnr,
																				   scene_avg_srgb_ssim) + '\n'
	f.write(context)

print('!!! avg psnr : ',whole_psnr/cnt)






