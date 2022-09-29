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
	def __init__(self, mysetdir=None, debugflag = False, NUMFRXSEQ = 15):
		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(mysetdir, MYSEQPATT)))
		if debugflag:
			seqs_dirs = seqs_dirs[:2]

		# open individual sequences and append them to the sequence list
		# [num_frames, C, H, W]
		sequences = []
		for seq_dir in seqs_dirs:
			seq = []
			edge_seq = []
			len_p = len( glob.glob(seq_dir + '/*.png') )
			step = np.random.randint(3) + 1
			begin_idx = np.random.randint(len_p - (NUMFRXSEQ+1)*step  )
			patch = 256
			img = cv2.imread(seq_dir + '/' + str(0).zfill(8) +'.png')[:, :, ::-1]
			ww, hh, _ = img.shape[:]
			ww_beg = np.random.randint(0, ww - patch)
			hh_beg = np.random.randint(0, hh - patch)
			for i in range(begin_idx, begin_idx+NUMFRXSEQ*step, step):
				seqpath = seq_dir + '/' + str(i).zfill(8) +'.png'
				img = cv2.imread(seqpath)[:,:,::-1]
				img = cv2.GaussianBlur(img, (3, 3), 1.5)
				img = img[ww_beg:ww_beg+patch,hh_beg:hh_beg+patch,:]
				seq.append(img)

			seq = np.array(seq).transpose(0, 3, 1, 2) / 255.0

			# seqraw = np.zeros((seq.shape[0],4,seq.shape[2]//2,seq.shape[3]//2),dtype=float)
			# if np.random.randint(10) < 5:
			# 	seq = seq ** (2.2)
			# seqraw[:,0,:,:] = seq[:,  0, 0::2, 0::2]
			# seqraw[:,1,:,:] = (seq[:, 1, 1::2, 0::2] + seq[:, 1, 0::2, 1::2]) * 0.5
			# seqraw[:,3,:,:] = (seq[:, 1, 1::2, 0::2] - seq[:, 1, 0::2, 1::2]) * 0.5
			# seqraw[:,2,:,:] = seq[:, 2, 1::2, 1::2]

			seqraw = unprocess.unprocess(torch.from_numpy(seq).float() ).numpy()
			sequences.append(seqraw)


		self.sequences = sequences


	def __getitem__(self, index):
		raw_seq = torch.from_numpy(self.sequences[index]).float()

		return raw_seq

	def __len__(self):
		return len(self.sequences)
