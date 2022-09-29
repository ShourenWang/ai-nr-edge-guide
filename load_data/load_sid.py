import torch
import torch.utils.data as torch_data
import glob
import os
import numpy as np
import rawpy
import cv2
import exifread
from PIL import Image, ImageStat


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=0)
    img_shape = im.shape
    H = img_shape[1]
    W = img_shape[2]

    # out = np.concatenate((im[:,0:H:2, 0:W:2],
    #                       im[:,0:H:2, 1:W:2]/2.0+im[:,1:H:2, 1:W:2]/2.0,
    #                       im[:,1:H:2, 0:W:2]), axis=0)
    out = np.concatenate((im[:, 0:H:2, 0:W:2],
                          im[:, 0:H:2, 1:W:2],
                          im[:, 1:H:2, 1:W:2],
                          im[:, 1:H:2, 0:W:2]), axis=0)

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

def get_exposure_time(raw_path):
    with open(raw_path,'rb') as raw_f:
        tags=exifread.process_file(raw_f)
        exposure_time_ratio=tags['EXIF ExposureTime'].values[0]
    return (float(exposure_time_ratio.num),float(exposure_time_ratio.den))

def fraction_division(ratio_1,ratio_2):
    return (ratio_1[0]*ratio_2[1])/(ratio_1[1]*ratio_2[0])


class myDataset(torch_data.Dataset):
    def __init__(self,input_dir,gt_dir,crop_size=1024,phase='train'):
        super(myDataset).__init__()

        self.input_dir=input_dir
        self.gt_dir=gt_dir
        self.phase=phase
        self.crop_size=crop_size
        self.input_patchs = []
        self.gt_patchs = []
        self.noise_stds = []

        if self.phase=='train':
            sample_path_list=glob.glob(os.path.join(gt_dir,'*.ARW'))
        else:
            sample_path_list=glob.glob(os.path.join(gt_dir,'M*.ARW'))

        self.sample_id_list=sorted([os.path.basename(sample_id) for sample_id in sample_path_list])
        for sample_id in self.sample_id_list:
            sample_id = sample_id.split('_')[0]
            in_path_list = glob.glob(os.path.join(self.input_dir, sample_id + '*.ARW'))
            in_path_1, in_path_2 = np.random.choice(in_path_list, 2)
            gt_path = glob.glob(os.path.join(self.gt_dir, sample_id + '*.ARW'))[0]

            in_raw_1 = rawpy.imread(in_path_1)
            in_raw_2 = rawpy.imread(in_path_2)
            full_size_image_1 = pack_raw(in_raw_1)
            full_size_image_2 = pack_raw(in_raw_2)

            # exposure ratio
            exposure_time_1 = get_exposure_time(in_path_1)
            exposure_time_2 = get_exposure_time(in_path_2)
            exposure_time_gt = get_exposure_time(gt_path)

            exposure_ratio_1 = fraction_division(exposure_time_gt, exposure_time_1)
            exposure_ratio_2 = fraction_division(exposure_time_gt, exposure_time_2)

            input_full_size_image_1 = full_size_image_1 * exposure_ratio_1
            input_full_size_image_2 = full_size_image_2 * exposure_ratio_2

            gt_raw = rawpy.imread(gt_path)
            gt_full_size_image = pack_raw(gt_raw)

            shift_ = np.random.randint(-20, 20)

            # crop
            if self.crop_size > 0:
                H, W = input_full_size_image_1.shape[1:3]

                if self.phase == 'train':
                    xx = np.random.randint(21, W - self.crop_size - 21)
                    yy = np.random.randint(21, H - self.crop_size - 21)
                else:
                    xx = 0
                    yy = 0

                input_patch_1 = input_full_size_image_1[:, yy + shift_:yy + shift_ + self.crop_size, \
                                xx + shift_:xx + shift_ + self.crop_size]
                input_patch_2 = input_full_size_image_2[:, yy:yy + self.crop_size, xx:xx + self.crop_size]
                gt_patch = gt_full_size_image[:, yy:yy + self.crop_size, xx:xx + self.crop_size]

            else:
                input_patch_1 = input_full_size_image_1
                input_patch_2 = input_full_size_image_2
                gt_patch = gt_full_size_image

            input_patch_1 = np.clip(input_patch_1, 0, 1.0) ** (1/2.2)
            input_patch_2 = np.clip(input_patch_2, 0, 1.0) ** (1/2.2)
            gt_patch = np.clip(gt_patch, 0, 1.0) ** (1/2.2)

            # compute noise map
            patch_temp = input_patch_2[:, self.crop_size // 2:self.crop_size // 2 + 50,
                         self.crop_size // 2:self.crop_size // 2 + 50]
            patch_temp = patch_temp.transpose(1, 2, 0)
            img = Image.fromarray(np.uint8(patch_temp * 255.0))
            stat = ImageStat.Stat(img)
            stadevv = stat.stddev
            stat = np.max(np.array(stadevv)) / 255.0

            # random flip and transpose
            if self.phase == 'train':
                if np.random.randint(2) == 1:
                    input_patch_1 = np.flip(input_patch_1, axis=1)
                    input_patch_2 = np.flip(input_patch_2, axis=1)
                    gt_patch = np.flip(gt_patch, axis=1)
                if np.random.randint(2) == 1:
                    input_patch_1 = np.flip(input_patch_1, axis=2)
                    input_patch_2 = np.flip(input_patch_2, axis=2)
                    gt_patch = np.flip(gt_patch, axis=2)
                if np.random.randint(2) == 1:
                    input_patch_1 = np.transpose(input_patch_1, (0, 2, 1))
                    input_patch_2 = np.transpose(input_patch_2, (0, 2, 1))
                    gt_patch = np.transpose(gt_patch, (0, 2, 1))

            input_patch_1 = np.ascontiguousarray(input_patch_1)
            input_patch_2 = np.ascontiguousarray(input_patch_2)
            gt_patch = np.ascontiguousarray(gt_patch)
            self.input_patchs.append(input_patch_1)
            self.input_patchs.append(input_patch_2)
            self.gt_patchs.append(gt_patch)
            self.noise_stds.append(stat)

       
    def __getitem__(self,index):
        input_patch_1 = self.input_patchs[2*index]
        input_patch_2 = self.input_patchs[2 * index+1]
        gt_patch = self.gt_patchs[ index]
        stat = self.noise_stds[index]
        stat = 75/255.0

        input_patch_1_torch=torch.from_numpy(input_patch_1).float()
        input_patch_2_torch=torch.from_numpy(input_patch_2).float()
        input_patch_torch = torch.cat([input_patch_1_torch, input_patch_2_torch],0)
        gt_patch_torch=torch.from_numpy(gt_patch).float()
        noise_std = torch.from_numpy(np.array([stat])).float()

        return input_patch_torch, gt_patch_torch, noise_std

    def __len__(self):
        return len(self.gt_patchs)

def load_sid_sequence():
    noise_dir = '/home/shared1/Sony/short/'
    gt_dir = '/home/shared1/Sony/long/'
    # '10185' 10187 10198 10213 10217 10227 20107 20188 20208 20210 00073
    sample_path_list = glob.glob(os.path.join(gt_dir, '*20210*.ARW'))
    sample_id_list = sorted([os.path.basename(sample_id) for sample_id in sample_path_list])
    seq = []
    imgnames = []
    for sample_id in sample_id_list :
        sample_id = sample_id.split('_')[0]
        in_path_list = glob.glob(os.path.join(noise_dir, sample_id + '*_0.0*s.ARW'))

        if len(in_path_list) > 1:
            gt_path = glob.glob(os.path.join(gt_dir, sample_id + '*.ARW'))[0]
            for in_path_1 in in_path_list:
                in_raw_1 = rawpy.imread(in_path_1)
                full_size_image_1 = pack_raw(in_raw_1)
                exposure_time_1 = get_exposure_time(in_path_1)
                exposure_time_gt = get_exposure_time(gt_path)
                exposure_ratio_1 = fraction_division(exposure_time_gt, exposure_time_1)

                input_full_size_image_1 = full_size_image_1 * exposure_ratio_1

                # gt_raw = rawpy.imread(gt_path)
                # gt_full_size_image = pack_raw(gt_raw)
                # gt_patch = np.clip(gt_patch, 0, 1.0) ** (1 / 2.2)

                input_patch_1 = np.clip(input_full_size_image_1, 0, 1.0) ** (1 / 2.2)
                input_patch_1 = input_patch_1.transpose(2, 0, 1)
                input_patch_1 = input_patch_1.transpose(2,0, 1)

                seq.append(input_patch_1)
                imgnames.append(in_path_1.split('/')[-1])
            return seq,imgnames
        else:
            continue

