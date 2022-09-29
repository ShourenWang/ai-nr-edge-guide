"""
Dataset related functions

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import glob
import torch
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
from load_data import unprocess

# NUMFRXSEQ = 15	# number of frames of each sequence to include in validation dataset
MYSEQPATT = '*' # pattern for name of validation sequence

class myDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, mysetdir=None, debugflag = False, NUMFRXSEQ = 5,rawflag=False):
		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(mysetdir, MYSEQPATT)))
		if debugflag:
			seqs_dirs = seqs_dirs[:1]
		self.NUMFRXSEQ =NUMFRXSEQ
		self.p = 32
		self.dis = np.ones((1,self.p*2,self.p*2))
		# for iii in range(self.p*2):
		# 	for jjj in range(self.p*2):
		# 		if iii > self.p//2 and iii < self.p //2+self.p and \
		# 				jjj > self.p // 2 and jjj < self.p // 2+self.p:
		# 			self.dis[0,iii, jjj] = 1
		# 		else:
		# 			self.dis[0,iii,jjj] = ( (iii - self.p)**2 + (jjj - self.p)**2 ) **(0.5)
		# 			self.dis[0,iii, jjj] = self.dis[0,iii,jjj] / (2*32**2)**(0.5)

		# open individual sequences and append them to the sequence list
		# [num_frames, C, H, W]
		sequences = []
		for i in range(17):
			for seq_dir in seqs_dirs:
				seq = []
				edge_seq = []
				len_p = len( glob.glob(seq_dir + '/*.png') )
				step = np.random.randint(3) + 1
				begin_idx = np.random.randint(1, len_p - (NUMFRXSEQ+1)*step-1)
				patch = self.p * 2
				img = cv2.imread(seq_dir + '/' + str(0).zfill(8) +'.png')[:, :, ::-1]
				ww, hh, _ = img.shape[:]
				ww_beg = np.random.randint(0, ww//2 - patch)
				hh_beg = np.random.randint(0, hh//2 - patch)
				for i in range(begin_idx, begin_idx+NUMFRXSEQ*step, step):
					seqpath = seq_dir + '/' + str(i).zfill(8) +'.png'
					img = cv2.imread(seqpath)[:,:,::-1]
					img = cv2.GaussianBlur(img, (3, 3), 0.1)
					img = cv2.resize(img, (0,0),fx=0.5,fy=0.5, interpolation=cv2.INTER_CUBIC)
					if np.random.randint(10) < 2:
						img = (img / 255.0) ** 2.2
						img = (img - np.random.randint(1,100) / 255.0) * (np.random.randint(10,20) / 20.0)
						img = np.clip(img,0,1) * (np.random.randint(10,20) / 20.0)
						img = img ** 1/2.2 * 255.0
					img = img[ww_beg:ww_beg+patch,hh_beg:hh_beg+patch,:]
					if rawflag:
						img = np.concatenate((np.expand_dims(img[0::2, 0::2,0], 2),
													  np.expand_dims(img[1::2, 0::2,1], 2),
													  np.expand_dims(img[0::2, 1::2,1], 2),
													  np.expand_dims(img[1::2, 1::2,2], 2)), axis=2)
					seq.append(img)
					# edge_seq.append(edge)
				seq = np.array(seq).transpose(0,3,1,2)
				# edge_seq = np.array(edge_seq).transpose(0, 3, 1, 2)
				sequences.append(seq)
				# sequences.append(edge_seq)

		self.sequences = sequences

	def __getitem__(self, index):
		sque = self.sequences[index]  # self.NUMFRXSEQ * 6 * w * h
		pre = np.random.randint(0, self.NUMFRXSEQ - 1)
		sque = sque/ 255.0
		pre_ul = np.concatenate((sque[pre, :, :self.p,:self.p],self.dis[:, :self.p,:self.p]), axis=0)
		pre_ur = np.concatenate((sque[pre, :, self.p:, :self.p],self.dis[:, self.p:, :self.p]), axis=0)
		pre_bl = np.concatenate((sque[pre, :, :self.p, self.p:],self.dis[:, :self.p, self.p:]), axis=0)
		pre_br = np.concatenate((sque[pre, :, self.p:, self.p:],self.dis[:, self.p:, self.p:]), axis=0)
		cur = np.concatenate((sque[pre+1, :, self.p//2:self.p//2*3,self.p//2:self.p//2*3],self.dis[:, self.p//2:self.p//2*3,self.p//2:self.p//2*3]), axis=0)
		# print(pre_ur.shape,pre_bl.shape)
		rgb_seq = np.concatenate((pre_ul,pre_ur,cur,pre_bl,pre_br), axis=0)
		rgb_seq = torch.from_numpy(rgb_seq).float()
		rgb_gt = sque[pre+1, :, self.p//2:self.p//2*3,self.p//2:self.p//2*3]
		rgb_gt = torch.from_numpy(rgb_gt).float()

		return rgb_seq, rgb_gt

	def __len__(self):
		return len(self.sequences)
