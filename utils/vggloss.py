import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class VGG19_Extractor(nn.Module):
    def __init__(self, output_layer_list=[]):
        super(VGG19_Extractor, self).__init__()
        self.vgg = models.vgg19(pretrained=False)
        checkpoint = torch.load('/home/project2022/ai-based-denoising/train/vgg19-dcbb9e9d.pth')
        self.vgg.load_state_dict(checkpoint)
        # self.vgg.requires_grad_(False)
        self.vgg_features = self.vgg.features
        self.module_list = list(self.vgg_features.modules())[1:]
        self.output_layer_list = output_layer_list

        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)

    def forward(self, x):
        x = x - self.mean.repeat(x.size(0), 1, x.size(2), x.size(3))
        x = x / self.std.repeat(x.size(0), 1, x.size(2), x.size(3))
        output_list = []
        for module_idx, module in enumerate(self.module_list):
            x = module(x)
            if module_idx in self.output_layer_list:
                output_list.append(x)
            if module_idx == self.output_layer_list[-1]:
                return output_list


