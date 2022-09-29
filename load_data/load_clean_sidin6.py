import torch
import torch.utils.data as torch_data
import glob
import os
import numpy as np
import rawpy
import cv2
from PIL import Image, ImageStat


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=0)
    img_shape = im.shape
    H = img_shape[1]
    W = img_shape[2]
    # print(np.mean(im[:, 0:H:2, 0:W:2])) #0.10139771
    # print(np.mean(im[:, 0:H:2, 1:W:2])) #0.16573206
    # print(np.mean(im[:, 1:H:2, 1:W:2])) #0.056788653
    # print(np.mean(im[:, 1:H:2, 0:W:2])) #0.16582495

    out = np.concatenate((im[:, 0:H:2, 0:W:2],
                          im[:, 0:H:2, 1:W:2],
                          im[:, 1:H:2, 1:W:2],
                          im[:, 1:H:2, 0:W:2]), axis=0)
    # if np.random.randint(10)< 5 :
    #     out = np.concatenate((im[:, 0:H:2, 0:W:2],
    #                           im[:, 0:H:2, 1:W:2],
    #                           im[:, 1:H:2, 1:W:2]), axis=0)
    # else:
    #     out = np.concatenate((im[:, 0:H:2, 0:W:2],
    #                           im[:, 1:H:2, 0:W:2],
    #                           im[:, 1:H:2, 1:W:2]), axis=0)


    return out


def input_2_cv(input_tensor):
    input_tensor_numpy=input_tensor.numpy()
    input_tensor_numpy_list=[]
    for tensor in input_tensor_numpy:
        tensor_cv=np.clip(tensor*255.0,0,255)
        tensor_cv=np.uint8(tensor_cv)
        input_tensor_numpy_list.append(tensor_cv)
    return input_tensor_numpy_list
    

def gt_2_cv(gt_tensor):
    gt_tensor_numpy=gt_tensor.numpy()
    gt_tensor_numpy=gt_tensor_numpy.transpose(1,2,0)
    gt_tensor_numpy=np.clip(gt_tensor_numpy*255.0,0,255)
    gt_tensor_numpy = gt_tensor_numpy[:,:,[2,1,0]]
    gt_tensor_numpy=np.uint8(gt_tensor_numpy)
    return gt_tensor_numpy



def fraction_division(ratio_1,ratio_2):
    return (ratio_1[0]*ratio_2[1])/(ratio_1[1]*ratio_2[0])


class myDataset(torch_data.Dataset):
    def __init__(self,mysetdir=None, debugflag = False, NUMFRXSEQ = 2):
        super(myDataset).__init__()

        gt_dir=mysetdir
        self.gt_dir = gt_dir
        self.phase = 'train'
        self.crop_size = 1024
        self.gt_patchs = []

        if self.phase=='train':
            sample_path_list=glob.glob(os.path.join(gt_dir,'*.ARW'))
        else:
            sample_path_list=glob.glob(os.path.join(gt_dir,'M*.ARW'))

        self.sample_id_list=sorted([os.path.basename(sample_id) for sample_id in sample_path_list])
        if debugflag:
            self.sample_id_list = self.sample_id_list[:2]

        for sample_id in self.sample_id_list:
            sample_id = sample_id.split('_')[0]

            gt_path = glob.glob(os.path.join(self.gt_dir, sample_id + '*.ARW'))[0]

            # gt_full_size_image = pack_raw(gt_raw)
            gt_raw = rawpy.imread(gt_path)
            gt_im = gt_raw.postprocess(use_camera_wb=True, half_size=True, no_auto_bright=True, output_bps=16)
            gt_im = gt_im.transpose(2, 0, 1)
            gt_full_size_image = np.float32(gt_im / 65535.0)
            shift_ = np.random.randint(-19, 19)

            # crop
            if self.crop_size > 0:
                H, W = gt_full_size_image.shape[1:3]

                if self.phase == 'train':
                    xx = np.random.randint(21, W - self.crop_size - 21)
                    yy = np.random.randint(21, H - self.crop_size - 21)
                gt_patch_1 = np.zeros((6,self.crop_size,self.crop_size),dtype=np.float32)
                gt_patch_2 = np.zeros((6,self.crop_size,self.crop_size),dtype=np.float32)

                gt_patch_1[:3,:,:] = gt_full_size_image[:, yy+shift_:yy+shift_ + self.crop_size, xx+shift_:xx+shift_ + self.crop_size]
                gt_patch_2[:3,:,:] = gt_full_size_image[:, yy:yy + self.crop_size, xx:xx + self.crop_size]

                gt_patch_1[3, :, :] = gt_patch_1[0, :, :]
                gt_patch_1[4, :, :] = gt_full_size_image[1, yy+shift_+1:yy+shift_ + self.crop_size+1, xx+shift_:xx+shift_ + self.crop_size]
                gt_patch_1[5, :, :] = gt_patch_1[2, :, :]

                gt_patch_2[3, :, :] = gt_patch_2[0, :, :]
                gt_patch_2[4, :, :] = gt_full_size_image[1, yy+1:yy+1 + self.crop_size, xx:xx + self.crop_size]
                gt_patch_2[5, :, :] = gt_patch_2[2, :, :]

            # if np.random.randint(10) < 5:
            #     gt_patch_1 = np.clip(gt_patch_1, 0, 1.0) ** (2.2)
            #     gt_patch_2 = np.clip(gt_patch_2, 0, 1.0) ** (2.2)

            # random flip and transpose
            if self.phase == 'train':
                if np.random.randint(2) == 1:
                    gt_patch_1 = np.flip(gt_patch_1, axis=1)
                    gt_patch_2 = np.flip(gt_patch_2, axis=1)
                if np.random.randint(2) == 1:
                    gt_patch_1 = np.flip(gt_patch_1, axis=2)
                    gt_patch_2 = np.flip(gt_patch_2, axis=2)
                if np.random.randint(2) == 1:
                    gt_patch_1 = np.transpose(gt_patch_1, (0, 2, 1))
                    gt_patch_2 = np.transpose(gt_patch_2, (0, 2, 1))

            gt_patch_1 = np.ascontiguousarray(gt_patch_1)
            gt_patch_2 = np.ascontiguousarray(gt_patch_2)

            self.gt_patchs.append(gt_patch_1)
            self.gt_patchs.append(gt_patch_2)


    def __getitem__(self,index):
        input_patch_1 = self.gt_patchs[2 * index]
        input_patch_2 = self.gt_patchs[2 * index+1]

        input_patch_1_torch=torch.from_numpy(input_patch_1).float()
        input_patch_2_torch=torch.from_numpy(input_patch_2).float()
        input_patch_torch = torch.cat([input_patch_1_torch, input_patch_2_torch],0)

        return input_patch_torch

    def __len__(self):
        return len(self.gt_patchs)//2

