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
	def __init__(self, mysetdir=None, debugflag = False, NUMFRXSEQ = 25):
		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(mysetdir, MYSEQPATT)))
		self.NUMFRXSEQ = NUMFRXSEQ
		if debugflag:
			seqs_dirs = seqs_dirs[:1]

		# open individual sequences and append them to the sequence list
		# [num_frames, C, H, W]
		sequences = []
		for seq_dir in seqs_dirs:
			seq = []
			edge_seq = []
			len_p = len( glob.glob(seq_dir + '/*.png') )
			step = np.random.randint(3) + 1
			begin_idx = np.random.randint(len_p - (NUMFRXSEQ+1)*step  )
			patch = 32
			img = cv2.imread(seq_dir + '/' + str(0).zfill(8) +'.png')[:, :, ::-1]
			ww, hh, _ = img.shape[:]
			ww_beg = np.random.randint(0, ww - patch-2)
			hh_beg = np.random.randint(0, hh - patch-2)
			for i in range(begin_idx, begin_idx+NUMFRXSEQ*step, step):
				seqpath = seq_dir + '/' + str(i).zfill(8) +'.png'
				img = cv2.imread(seqpath)[:,:,::-1]
				gt_patch_1 = np.zeros((6, patch, patch), dtype=np.float32)
				gt_patch_1[0, :, :] = img[ww_beg:ww_beg+patch,hh_beg:hh_beg+patch,0]
				gt_patch_1[1, :, :] = img[ww_beg:ww_beg + patch, hh_beg:hh_beg + patch, 1]
				gt_patch_1[2, :, :] = img[ww_beg:ww_beg + patch, hh_beg:hh_beg + patch, 2]

				gt_patch_1[3, :, :] = gt_patch_1[0,:, :]
				gt_patch_1[4, :, :] = img[ww_beg+1:ww_beg+1+patch,hh_beg+1:hh_beg+1+patch,1]
				gt_patch_1[5, :, :] = gt_patch_1[2, :, :]
				seq.append(gt_patch_1)
				# edge_seq.append(edge)
			seq = np.array(seq).transpose(0,1,2,3)
			sequences.append(seq)

		self.sequences = sequences

	def __getitem__(self, index):
		sque = self.sequences[index] # self.NUMFRXSEQ * 6 * w * h
		pre = np.random.randint(0, self.NUMFRXSEQ-1)
		sque = sque[pre:pre+2,:,:,:]
		rgb_seq = torch.from_numpy(sque/255.0)
		rgb_seq = rgb_seq.reshape(12, rgb_seq.size()[-2], rgb_seq.size()[-1])
		return rgb_seq

	def __len__(self):
		return len(self.sequences)
