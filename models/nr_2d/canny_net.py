import torch
import torch.nn as nn
import numpy as np

class CannyDetector(nn.Module):
    def __init__(self, threshold=15.0, use_cuda=True):
        super(CannyDetector, self).__init__()

        self.threshold = threshold/255.0
        self.use_cuda = use_cuda

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 1, 0, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [1, 0, 0],
                                [ 0, 0, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 1, 0],
                                [ 0, 0, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 1],
                                [ 0, 0, 0],
                                [-1, 0, 0]])


        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=filter_0.shape, padding=2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    @torch.no_grad()
    def forward(self, imgs):
        outs = torch.zeros_like(imgs)
        for cc in range(imgs.size()[1]):
            img = imgs[:,cc,:,:].unsqueeze(1)
            all_filtered = self.directional_filter(img)
            all_filtered = torch.abs(all_filtered)
            all_filtered[all_filtered<self.threshold] = 0
            all_filtered[all_filtered >= self.threshold] = 1
            all_filtered = torch.sum(all_filtered,1)
            all_filtered[all_filtered>1] = 1
            all_filtered = all_filtered.unsqueeze(1)
            outs[:,cc,:,:] = all_filtered[:,0,1:-1,1:-1]

        return outs

