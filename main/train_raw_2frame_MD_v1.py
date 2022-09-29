
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
from models.cvpr.raw_2frame_MD_v1 import FastDVDnet
from models.raw_discriminator import *
from load_data.load_crvd import *
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

iso_list  = [1600, 3200, 6400, 12800, 25600]

def train(**args):
    best_psnr = 0
    r"""Performs the main training loop
    """
    # Load dataset
    train_data_name_queue = generate_file_list(['1', '2', '3', '4', '5', '6'])
    train_dataset = loadImgs(train_data_name_queue)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                                               shuffle=True, pin_memory=True)
    # Define GPU devices
    device_ids = [0,1]
    torch.backends.cudnn.benchmark = True  # CUDNN optimization

    # Create model
    G = FastDVDnet()
    G = nn.DataParallel(G, device_ids=device_ids).cuda()
    D = Discriminator((512,512)).cuda()
    vgg_model = VGG19_Extractor(output_layer_list=[1,2,3,7]).cuda()
    canny_operator = CannyDetector().cuda()
    cfa = Debayer5x5().cuda()

    # Define loss
    criterion = nn.L1Loss()
    contentloss = L1_Charbonnier_loss().cuda()
    # criterion = nn.MSELoss()
    criterion.cuda()
    compsnr = PSNR().cuda()

    # Optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=cfg.learning_rate)
    D_optimizer = optim.Adam(D.parameters(), lr=cfg.learning_rate)

    checkpoint = torch.load(cfg.model_save_root+'model_best.pth')
    start_epoch = checkpoint['epoch']
    G.load_state_dict(checkpoint['model'])
    G_optimizer.load_state_dict(checkpoint['optimizer'])
    print(start_epoch)
    eval_psnr = evaluate(G, compsnr,False,True)
    eval_psnr = eval_psnr.item()
    print('evaluate full res psnr is:', eval_psnr)
    e



    # Training
    start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        print('------------------------------------------------')
        print('Epoch                |   ', ('%06d' % epoch))
        if epoch % 27 == 0 and epoch > 0:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                                       num_workers=cfg.num_workers,
                                                       shuffle=True, pin_memory=True)
        # train
        d_loss = 0
        g_loss = 0
        eval_psnr = 0
        G_optimizer.zero_grad()
        D_optimizer.zero_grad()
        G.train()

        for iter, data in enumerate(train_loader):
            ''' load data and add noise '''
            img_trains = data[0].float().cuda() # b，w，h, c*frame
            gt_trains = data[1].float().cuda()  # b,w,h,c*frame
            noise_maps = data[2].float().cuda()  # b,2
            # print(type(noise_level),noise_level.size())
            #N, T, C, H, W = img_train.size()
            # print(len(noise_level),noise_level[0])
            '''# ################################ #
               #       train generator            #
               # ################################ # '''
            set_requires_grad(G, True)
            set_requires_grad(cfa, False)
            grad_false(G,['blur1'])

            G_optimizer.zero_grad()
            for tt in range(cfg.frame_num):
                gt_train = gt_trains[:, tt*4:tt*4+4, :, :]
                cur = img_trains[:,tt*4:tt*4+4,:,:]
                noise_map = noise_maps[:]
                if tt == 0:
                    inputn = torch.cat([cur, cur],1)
                else:
                    inputn = torch.cat([pre, cur], 1)

                x_pool, in_edge, x = G(inputn, noise_map)
                content_loss = contentloss(x, gt_train)

                gen_loss = content_loss #+ edgeloss

                gt_train_up = nn.PixelShuffle(2)(gt_train)
                x_up = nn.PixelShuffle(2)(x)
                xrgb = cfa(x_up)
                gtrgb = cfa(gt_train_up)

                out1, out2, out3, out4 = vgg_model(xrgb)
                gt1, gt2, gt3, gt4 = vgg_model(gtrgb)
                vggloss = criterion(out1, gt1) + criterion(out2, gt2) + criterion(out3, gt3) + criterion(out4, gt4)
                gen_loss += 0.001 * vggloss

                gen_loss += 0.1*criterion(xrgb, gtrgb) + 0.0000001*loss_tv(xrgb)

                gt_edge = canny_operator(gtrgb)
                x_edge = canny_operator(xrgb)

                edgeloss = BCEloss(gt_edge,x_edge)
                gen_loss = gen_loss + 1*edgeloss
                gen_loss = gen_loss / cfg.frame_num
                gen_loss.register_hook(raise_if_nan)
                gen_loss.backward()  # retain_graph=True
                G_optimizer.step()
                g_loss += gen_loss.item()
                pre = x.detach()
                G_optimizer.zero_grad()

            if epoch % 10 == 0 and iter == 0:
                tt1 = torch.cat( [x[:,:-1,:,:], gt_train[:,:-1,:,:],cur[:,:-1,:,:]],3)
                tt = in_edge.cpu().detach().numpy()[0, 0, :, :]
                tt1 = tt1.cpu().detach().numpy()[0,:,:,:].transpose(1,2,0)[:,:,::-1]
                cv2.imwrite('./results/'+str(epoch)+'rgbloss_p16_ca_sidd_edge.png',np.uint8(tt*255.0*10) )
                cv2.imwrite('./results/' + str(epoch) + 'rgbloss_p16_ca_sidd_raw.png', np.uint8(tt1 * 255.0))
            if epoch % 10 == 0 and iter == 0:
                torch.save({
                    'epoch': epoch,
                    'model': G.state_dict(),
                    'optimizer': G_optimizer.state_dict()},
                    cfg.model_save_root+'model.pth')
            if iter == 7 and epoch % 2 ==0:
                eval_psnr = evaluate(G, compsnr)
                eval_psnr = eval_psnr.item()
                if eval_psnr > best_psnr:
                    best_psnr = eval_psnr
                    torch.save({
                        'epoch': epoch,
                        'iter': iter,
                        'model': G.state_dict(),
                        'optimizer': G_optimizer.state_dict(),
                        'best_psnr': best_psnr},
                        os.path.join(cfg.model_save_root, 'model_best.pth'))

        print('   gloss and psnr: ', g_loss, eval_psnr)

    # Print elapsed time
    elapsed_time = time.time() - start_time
    print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

def evaluate(model, compsnr, trainflag=False,fullres=False):
    print('Evaluate...')
    cnt = 0
    total_psnr = 0
    model.eval()

    test_name_queue = generate_file_list(['7', '8', '9', '10', '11'])
    test_dataset = loadImgs(test_name_queue,trainflag,fullres)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1,
                                               shuffle=False, pin_memory=True)
    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            ''' load data and add noise '''
            img_trains = data[0].float().cuda()  # b，w，h, c*frame
            gt_trains = data[1].float().cuda()  # b,w,h,c*frame
            noise_maps = data[2].float().cuda()  # b,2
            frame_psnr = 0
            for tt in range(7):
                gt_train = gt_trains[:, tt * 4:tt * 4 + 4, :, :]
                cur = img_trains[:, tt * 4:tt * 4 + 4, :, :]
                noise_map = noise_maps[:]
                if tt == 0:
                    inputn = torch.cat([cur, cur], 1)
                else:
                    inputn = torch.cat([pre, cur], 1)

                x_pool, in_edge, x = model(inputn, noise_map)
                x = torch.clamp(x,0,1)

                frame_psnr += compsnr(x, gt_train)
                pre = x.detach()

                frame_psnr = frame_psnr / (7.0)
                # print('---------')
                # print('Scene: ', ('%02d' % scene_ind), 'Noisy_level: ', ('%02d' % noisy_level), 'PSNR: ', '%.8f' % frame_psnr.item())
                total_psnr += frame_psnr

                del gt_train, cur
            cnt += 1
        total_psnr = total_psnr / cnt
    print('Eval_Total_PSNR              |   ', ('%.8f' % total_psnr.item()))
    torch.cuda.empty_cache()
    return	total_psnr


train()




