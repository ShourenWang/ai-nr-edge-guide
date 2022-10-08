import os
import torch

# generate data
data_root = ['/home/data_beifen/', '/home/data_beifen/crvd_nr/']
output_root = './results/'
model_save_root = './logs/'

image_height = 128
image_width = 128
batch_size = 16
frame_num = 7
num_workers = 4

# parameter of train
learning_rate                   = 0.0001
epochs                           = int(777)
