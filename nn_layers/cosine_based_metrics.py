"""The code is from https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py.

"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from typing import Optional


class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance: """
    def __init__(self, in_features, out_features, s=30.0, m=0.10, easy_margin=True):
        """

        Parameters
        ----------
        in_features: `int`
            size of each input sample
        out_features: `int`
            size of each output sample
        s: `float`
            norm of input feature
        m: `float`
            margin
            cos(theta + m)
        easy_margin: `bool`
            Not sure yet.

        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # out_features: C, in_features: Dimension
#        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#        nn.init.xavier_uniform_(self.weight)
        self.weight = nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, bias=False)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self, 
        input: torch.Tensor, 
        label: Optional[torch.Tensor]=None
    ):
        """Calculate the logits for ArcFace 

        Parameters
        ----------
        input: `torch.Tensor`
            Input feature map with the size (B, C, H, W)
        label: `torch.Tensor`
            Correspondin label map with the size (B, H, W)

        Returns
        -------
        logits: `torch.Tensor`
            Logits (values before taking softmax) of the ArcFace loss
        """
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        #
        ## Change this linear to convolution?
        #
#        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Normalize weight
        self.weight.weight.data = F.normalize(self.weight.weight.data, dim=1)
        cosine = self.weight(F.normalize(input, dim=1))

        # Penalized function for the value of the true label
        if label is not None:
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            #one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
#            one_hot = torch.zeros(cosine.size(), device='cuda')
#            one_hot.scatter_(1, label.view(-1, 1).long(), 1)

            num_invalid_label = torch.max(label)
            valid_max = max(self.out_features-1, num_invalid_label.item())
            one_hot = F.one_hot(label, num_classes=valid_max+1).transpose(1, 3).transpose(2, 3)
            one_hot = one_hot[:,:self.out_features,:,:]
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = torch.where(one_hot == 1, phi, cosine)
    #        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
    #        print(output[0,:,0,0])
    #        print(cosine[:,0,:,:][label==0].max(), output[:,0,:,:][label==0].min())
        else:
            output = cosine

#        print(output.mean(dim=0).mean(dim=1).mean(dim=1))
        output = self.s * output

        return output
