import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_arch import ResNet18
from .fusion_arch import SumFusion, ConcatFusion, FiLM, GatedFusion

class AVClassifierModel(nn.Module):
    def __init__(self, args):
        super(AVClassifierModel, self).__init__()
        n_classes = args['n_classes']
        fusion = args['fusion']

        self.n_classes = n_classes
        self.audio_net = ResNet18(modality='audio')
        self.visual_net = ResNet18(modality='visual')

        if fusion == 'sum':
            self.fusion_module = SumFusion(output_dim=n_classes)
        elif fusion == 'concat':
            self.fusion_module = ConcatFusion(output_dim=n_classes)
        elif fusion == 'film':
            self.fusion_module = FiLM(output_dim=n_classes, x_film=True)
        elif fusion == 'gated':
            self.fusion_module = GatedFusion(output_dim=n_classes, x_gate=True)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion))


    def forward(self, audio, visual):

        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        a, v, out = self.fusion_module(a, v)

        return a, v, out