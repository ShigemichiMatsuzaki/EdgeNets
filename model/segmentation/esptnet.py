import torch
from torch.nn import functional as F
from .espnetv2 import ESPNetv2Segmentation
from typing import Optional


class _LabelProbEstimator(torch.nn.Module):
    """This class defines a simple architecture for estimating binary label probability of 1-d features"""

    def __init__(
        self,
        in_channels: int = 16,
        use_sigmoid: bool = False,
        spatial: bool = True
    ):
        """_summary_

        Parameters
        ----------
        in_channels : `int`, optional
            Number of input channels. Default: `16`
        use_sigmoid : `bool`, optional
            `True` to use sigmoid at last. Default: `False`
        spatial : `bool`, optional
            `True` to use 3x3 conv for classification. 
            Else, use 1x1 (point-wise) conv. Default: `True`
        """
        super().__init__()

        if spatial:
            self.conv1x1 = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)
        else:
            self.conv1x1 = torch.nn.Conv2d(
                in_channels=in_channels, out_channels=1, kernel_size=1)

        self.relu = torch.nn.ReLU()

        self.use_sigmoid = use_sigmoid

        if self.use_sigmoid:
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """Provide output

        Parameters
        ----------
        x : `torch.Tensor`
            Input

        Returns
        -------
        `torch.Tensor`
            Output
        """

        output = self.conv1x1(x)

        if self.use_sigmoid:
            output = self.sigmoid(output)

        return output


class ESPTNet(ESPNetv2Segmentation):
    # Segmentation + traversability estimation
    def __init__(
        self,
        args,
        classes: int = 21,
        dataset: str = "pascal",
        aux_layer: int = 2,
        use_sigmoid: bool = False,
        spatial: bool = True,
    ):
        """Segmentation + Traversability estimation

        Parameters
        ----------
        args : Arguments
            Parameters for model
        classes : `int`, optional
            The number of classes, Default: `21`
        dataset : `str`, optional
            Type of dataset. Not used. Default: `pascal`
        aux_layer : `int`, optional
            Layer at which the auxilirary classifier is attached.
            Default: `2`

        """
        super(ESPTNet, self).__init__(
            args, classes, dataset, aux_layer,
        )

        self.traversability_module = _LabelProbEstimator(
            in_channels=self.dim_size * 2,
            use_sigmoid=use_sigmoid,
            spatial=spatial,
        )

    def freeze_encoder(self):
        """Stop gradient for the encoder"""
        for param in self.parameters():
            param.requires_grad = False

        for param in self.traversability_module.parameters():

            param.requires_grad = True

    def get_traversability_module_params(self):
        modules_seg = [
            self.traversability_module,
        ]

        for i in range(len(modules_seg)):
            for m in modules_seg[i].named_modules():
                if (
                    isinstance(m[1], torch.nn.Conv2d)
                    or isinstance(m[1], torch.nn.BatchNorm2d)
                    or isinstance(m[1], torch.nn.PReLU)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """Forward pass

        Parameters
        ----------
        x: `torch.Tensor`
            Input RGB image
        labels: `torch.Tensor`i, optional
            GT labels to calculate ArcFace scores

        Returns
        -------
        A C-dimensional vector, C=# of classes
        """

        x_size = (x.size(2), x.size(3))  # Width and height

        #
        # First conv
        #
        enc_out_l1 = self.base_net.level1(x)  # 112
        if not self.base_net.input_reinforcement:
            del x
            x = None

        #
        # Second layer (Strided EESP)
        #
        enc_out_l2 = self.base_net.level2_0(enc_out_l1, x)  # 56

        #
        # Third layer 1 (Strided EESP)
        #
        enc_out_l3_0 = self.base_net.level3_0(
            enc_out_l2, x)  # down-sample -> 28

        #
        # EESP
        #
        for i, layer in enumerate(self.base_net.level3):
            if i == 0:
                enc_out_l3 = layer(enc_out_l3_0)
            else:
                enc_out_l3 = layer(enc_out_l3)

        #
        # Forth layer 1 (Strided EESP)
        #
        enc_out_l4_0 = self.base_net.level4_0(
            enc_out_l3, x)  # down-sample -> 14

        #
        # EESP
        #
        for i, layer in enumerate(self.base_net.level4):
            if i == 0:
                enc_out_l4 = layer(enc_out_l4_0)

            else:
                enc_out_l4 = layer(enc_out_l4)

        # *** 5th layer is for and classification and removed for segmentation ***

        # bottom-up decoding
        bu_out = self.bu_dec_l1(enc_out_l4)
        if self.aux_layer == 0:
            if self.use_cosine:
                aux_out = self.aux_decoder(bu_out)
                labels_interp = (
                    self._downsample_label(
                        labels, aux_out) if labels is not None else None
                )
                aux_logits = self.aux_classifier(aux_out, labels_interp)
            else:
                aux_logits = self.aux_decoder(bu_out,)

        # Decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l3_proj = self.merge_enc_dec_l2(enc_out_l3)
        bu_out = enc_out_l3_proj + bu_out
        bu_out = self.bu_br_l2(bu_out)
        bu_out = self.bu_dec_l2(bu_out)
        if self.aux_layer == 1:
            if self.use_cosine:
                aux_out = self.aux_decoder(bu_out)
                labels_interp = (
                    self._downsample_label(
                        labels, aux_out) if labels is not None else None
                )
                aux_logits = self.aux_classifier(aux_out, labels_interp)
            else:
                aux_logits = self.aux_decoder(bu_out,)

        # decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l2_proj = self.merge_enc_dec_l3(enc_out_l2)
        bu_out = enc_out_l2_proj + bu_out
        bu_out = self.bu_br_l3(bu_out)
        bu_out = self.bu_dec_l3(bu_out)
        if self.aux_layer == 2:
            if self.use_cosine:
                aux_out = self.aux_decoder(bu_out)
                labels_interp = (
                    self._downsample_label(
                        labels, aux_out) if labels is not None else None
                )

                aux_logits = self.aux_classifier(aux_out, labels_interp)
            else:
                aux_logits = self.aux_decoder(bu_out,)

        # decoding block
        bu_out = self.upsample(bu_out)
        enc_out_l1_proj = self.merge_enc_dec_l4(enc_out_l1)
        bu_out = enc_out_l1_proj + bu_out
        bu_out = self.bu_br_l4(bu_out)

        if self.use_cosine:
            bu_out = self.bu_dec_l4(bu_out)
            labels_interp = (
                self._downsample_label(
                    labels, bu_out) if labels is not None else None
            )
            main_logits = self.main_classifier(bu_out, labels_interp)
        else:
            main_logits = self.bu_dec_l4(bu_out,)

        # Features
        if self.use_cosine:
            main_feat = F.interpolate(
                F.normalize(bu_out), size=x_size, mode="bilinear", align_corners=True
            )
            aux_feat = F.interpolate(
                F.normalize(aux_out), size=x_size, mode="bilinear", align_corners=True
            )
        else:
            main_feat = F.interpolate(
                self.activation['output_main'], size=x_size, mode="bilinear", align_corners=True)
            aux_feat = F.interpolate(
                self.activation['output_aux'], size=x_size, mode="bilinear", align_corners=True)

        # Concatenate feature vectors
        feature = torch.cat((main_feat, aux_feat), dim=1)

        # Traversability estimation
        traversability_output = self.traversability_module(feature)

        return {
            "out": F.interpolate(
                main_logits, size=x_size, mode="bilinear", align_corners=True
            ),
            "aux": F.interpolate(
                aux_logits, size=x_size, mode="bilinear", align_corners=True
            ),
            "feat": feature,
            "main_feat": main_feat,
            "aux_feat": aux_feat,
            "trav": traversability_output,
        }
