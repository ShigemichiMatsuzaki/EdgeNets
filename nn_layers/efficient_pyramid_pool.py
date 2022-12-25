#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

import torch
from torch import nn
import math
from torch.nn import functional as F
from nn_layers.cnn_utils import CBR, BR, Shuffle

class EfficientPyrPool(nn.Module):
    """Efficient Pyramid Pooling Module"""

    def __init__(
        self, 
        in_planes, 
        proj_planes, 
        out_planes, 
        scales=[2.0, 1.5, 1.0, 0.5, 0.1], 
        last_layer_br=True, 
        normalize=False, 
        only_feat=False
    ):
        """Initializer of EfficientPyrPool

        Parameters
        ----------
        in_planes: `int`
            The channel number of the input 
        proj_planes: `int`
            The channel number of the intermediate projection 
        out_planes: `int`
            The channel number of the output
        scales: `list`
            List of the scales for pyramid pooling
        last_layer_br: `bool`
            `True` to apply batch norm + ReLU at last. Default: `True`
        normalize: `bool`
            `True` to normalize the classification weights and the feature. Default: `False`
        only_feat: `bool`
            `True` to output the features before classification
        
        """
        super(EfficientPyrPool, self).__init__()
        self.stages = nn.ModuleList()
        scales.sort(reverse=True)

        self.projection_layer = CBR(in_planes, proj_planes, 1, 1)
        for _ in enumerate(scales):
            self.stages.append(
                nn.Conv2d(
                    proj_planes, 
                    proj_planes, 
                    kernel_size=3, 
                    stride=1, 
                    padding=1, 
                    bias=False, 
                    groups=proj_planes
                )
            )

        self.normalize = normalize
        self.only_feat = only_feat
        if self.normalize or self.only_feat:
            self.merge_layer = nn.Sequential(
                # perform one big batch normalization instead of p small ones
                BR(proj_planes * len(scales)),
                Shuffle(groups=len(scales)),
                CBR(proj_planes * len(scales), proj_planes, 3, 1, groups=proj_planes),
            )
            if not self.only_feat:
                self.last_layer = nn.Conv2d(proj_planes, out_planes, kernel_size=1, stride=1, bias=False)

            last_layer_br = False
        else:
            self.merge_layer = nn.Sequential(
                # perform one big batch normalization instead of p small ones
                BR(proj_planes * len(scales)),
                Shuffle(groups=len(scales)),
                CBR(proj_planes * len(scales), proj_planes, 3, 1, groups=proj_planes),
                nn.Conv2d(proj_planes, out_planes, kernel_size=1, stride=1, bias=False),
            )

        if last_layer_br:
            self.br = BR(out_planes)

        self.last_layer_br = last_layer_br
        self.scales = scales

    def forward(self, x):
        hs = []
        x = self.projection_layer(x)
        height, width = x.size()[2:]
        for i, stage in enumerate(self.stages):
            h_s = int(math.ceil(height * self.scales[i]))
            w_s = int(math.ceil(width * self.scales[i]))
            h_s = h_s if h_s > 5 else 5
            w_s = w_s if w_s > 5 else 5
            if self.scales[i] < 1.0:
                h = F.adaptive_avg_pool2d(x, output_size=(h_s, w_s))
                h = stage(h)
                h = F.interpolate(h, (height, width), mode='bilinear', align_corners=True)
            elif self.scales[i] > 1.0:
                h = F.interpolate(x, (h_s, w_s), mode='bilinear', align_corners=True)
                h = stage(h)
                h = F.adaptive_avg_pool2d(h, output_size=(height, width))
            else:
                h = stage(x)
            hs.append(h)

        out = torch.cat(hs, dim=1)

        # Output normalized feature
        if self.normalize and self.only_feat:
            # Normalize weight
#            self.last_layer.weight.data = F.normalize(self.last_layer.weight.data)
            # Normalize data vector
            out = self.merge_layer(out)
            out = F.normalize(out)

        # Normalize the feature, and then apply classification
        elif self.normalize:
            out = self.merge_layer(out)
            out = F.normalize(out)

            out = self.last_layer(out)

        # Output non-normalized feature
        elif self.only_feat:
            out = self.merge_layer(out)

        # Directly apply classification to the feature (original)
        else:
            out = self.merge_layer(out)

        if self.last_layer_br:
            return self.br(out)

        return out
