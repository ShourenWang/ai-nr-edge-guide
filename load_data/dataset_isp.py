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

def rgb2raw(rgbs):
    raw_patch_1 = np.concatenate((np.expand_dims(rgbs[0, 0::2, 0::2], 0),
                                  np.expand_dims(rgbs[1, 1::2, 0::2], 0),
                                  np.expand_dims(rgbs[1, 0::2, 1::2], 0),
                                  np.expand_dims(rgbs[2, 1::2, 1::2], 0)), axis=0)
    return raw_patch_1

class myDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, mysetdir=None, debugflag = False, NUMFRXSEQ = 5,rawflag=False):
		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(mysetdir, MYSEQPATT)))
		if debugflag:
			seqs_dirs = seqs_dirs[:1]
		self.NUMFRXSEQ =NUMFRXSEQ
		self.p = 16
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
		for i in range(7):
			for seq_dir in seqs_dirs:
				seq = []
				edge_seq = []
				len_p = len( glob.glob(seq_dir + '/*.png') )
				step = np.random.randint(2) + 1
				begin_idx = np.random.randint(step, len_p - (NUMFRXSEQ+1)*step-1)
				patch = self.p * 2
				img = cv2.imread(seq_dir + '/' + str(0).zfill(8) +'.png')[:, :, ::-1]
				ww, hh, _ = img.shape[:]
				# ww_beg = np.random.randint(0, ww//2 - patch)
				# hh_beg = np.random.randint(0, hh//2 - patch)
				for i in range(begin_idx, begin_idx+NUMFRXSEQ*step, step):
					ww_beg = np.random.randint(0, ww // 2 - patch)
					hh_beg = np.random.randint(0, hh // 2 - patch)

					seqpath = seq_dir + '/' + str(i).zfill(8) +'.png'
					img = cv2.imread(seqpath)[:,:,::-1]

					img = img[ww_beg:ww_beg+patch,hh_beg:hh_beg+patch,:]/255.0
					seqpath = seq_dir + '/' + str(i-step).zfill(8) +'.png'
					pre = cv2.imread(seqpath)[:,:,::-1][ww_beg:ww_beg+patch,hh_beg:hh_beg+patch,:]/255.0
					if np.random.randint(10)<5:
						scale = (np.random.randint(10,20) / 20.0)
						bias = np.random.randint(1,100) / 255.0
						pre = np.clip(pre ** (2.2) - bias,0,1 ) * scale
						img = np.clip(img ** (2.2) - bias, 0, 1) * scale

					cur = img[self.p//2: self.p//2 * 3,self.p//2: self.p//2 * 3,:]
					diffs = []
					tems = [pre, img]
					frs = [0,1,0,1,0,1,0,1]
					ics = [0,0,0,0,1,1,1,1]
					jcs = [0,0,1,1,0,0,1,1]
					for ic in range(2):
						for jc in range(2):
							diff1 = np.mean((cur - pre[ic*self.p:(ic+1)*self.p,jc*self.p:(jc+1)*self.p,:])**2 )
							diff2 = np.mean((cur - img[ic * self.p:(ic + 1) * self.p, jc * self.p:(jc + 1) * self.p, :])**2)
							diffs.append(diff1)
							diffs.append(diff2)
					ind = np.argsort(diffs, axis=0)[:3]
					x1 = tems[frs[ind[0]]][ics[ind[0]]*self.p:ics[ind[0]]*self.p+self.p,\
						 jcs[ind[0]]*self.p:jcs[ind[0]]*self.p+self.p,:]
					x0 = tems[frs[ind[1]]][ics[ind[1]] * self.p:ics[ind[1]] * self.p + self.p, \
						 jcs[ind[1]] * self.p:jcs[ind[1]] * self.p + self.p,:]
					x4 = tems[frs[ind[2]]][ics[ind[2]] * self.p:ics[ind[2]] * self.p + self.p, \
						 jcs[ind[2]] * self.p:jcs[ind[2]] * self.p + self.p,:]
					x3 = pre[self.p//2: self.p//2 * 3,self.p//2: self.p//2 * 3,:]
					# inputs = np.concatenate((x0,x1,cur,x3,x4),axis=2)
					inputs = np.concatenate((x3,cur), axis=2)

					sequences.append(inputs)

		self.sequences = sequences

	def __getitem__(self, index):
		rgb_seq = self.sequences[index].transpose(2,0,1)
		rgb_seq = torch.from_numpy(rgb_seq).float()
		rgb_gt = self.sequences[index].transpose(2,0,1)[3:6,:,:]
		rgb_gt = torch.from_numpy(rgb_gt).float()

		return rgb_seq, rgb_gt

	def __len__(self):
		return len(self.sequences)
