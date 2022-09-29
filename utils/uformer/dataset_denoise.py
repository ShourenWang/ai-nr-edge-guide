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
# NUMFRXSEQ = 15	# number of frames of each sequence to include in validation dataset
MYSEQPATT = '*' # pattern for name of validation sequence

class myDataset(Dataset):
    def __init__(self, mysetdir=None, debugflag = False, NUMFRXSEQ = 5,rawflag=False):
        # Look for subdirs with individual sequences
        seqs_dirs = sorted(glob.glob(os.path.join(mysetdir, MYSEQPATT)))
        if debugflag:
            seqs_dirs = seqs_dirs[:1]
        self.NUMFRXSEQ =NUMFRXSEQ

        sequences = []
        for i in range(1):
            for seq_dir in seqs_dirs:
                seq = []
                len_p = len( glob.glob(seq_dir + '/*.png') )
                step = np.random.randint(3) + 1
                begin_idx = np.random.randint(len_p - (NUMFRXSEQ+1)*step  )
                patch = 128
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
                seq = np.array(seq).transpose(0,3,1,2)
                sequences.append(seq)
        self.sequences = sequences

    def __getitem__(self, index):
        sque = self.sequences[index]
        pre = np.random.randint(0, self.NUMFRXSEQ - 1)
        # rgb_seq = sque[pre:pre + 2, :, :, :]
        # rgb_seq = torch.from_numpy(rgb_seq / 255.0).float()
        # rgb_seq = rgb_seq.reshape(-1, rgb_seq.size()[-2], rgb_seq.size()[-1]).float()
        rgb_gt = sque[pre + 1, :, :, :]
        rgb_gt = torch.from_numpy(rgb_gt/255.0).float()
        stdn = torch.empty((1, 1, 1)).uniform_(5/255.0, 65/255.0)
        noise = torch.zeros_like(rgb_gt)
        noise = torch.normal(mean=noise, std=stdn.expand_as(noise))
        if np.random.randint(10)<5:
            rgb_noise = torch.clamp(rgb_gt + noise,0,1)
        else:
            rgb_noise = torch.clamp(rgb_gt - noise, 0, 1)

        return rgb_noise, rgb_gt

    def __len__(self):
        return len(self.sequences)
