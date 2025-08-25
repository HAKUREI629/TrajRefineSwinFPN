import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPPModule, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs

class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

class DepthwiseSeparableASPP(nn.Module):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, dilations, in_channels, channels, c1_in_channels, c1_channels):
        super(DepthwiseSeparableASPP, self).__init__()
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels,
                channels,
                1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')))
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=dict(type='ReLU'))
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=dict(type='ReLU'))
        else:
            self.c1_bottleneck = None
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * channels,
            channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=dict(type='ReLU'))
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                channels + c1_channels,
                channels,
                3,
                padding=1,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')),
            DepthwiseSeparableConvModule(
                channels,
                channels,
                3,
                padding=1,
                norm_cfg=None,
                act_cfg=dict(type='ReLU')))

    def forward(self, inputs):
        """Forward function."""
        x = inputs[0]
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=False)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[1])
            c1_output = resize(
                input=c1_output,
                size=output.shape[2:],
                mode='bilinear',
                align_corners=False)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        return output

class FRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FRM, self).__init__()
        
        # 生成注意力图的卷积层
        self.attention_conv = ConvModule(
                in_channels,
                in_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None)
        
        # 特征融合的 1x1 卷积层
        self.fusion_conv = ConvModule(
                in_channels,
                out_channels,
                1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None)
    
    def forward(self, F_j):
        # 1. 计算注意力图 A_j
        A_j = self.attention_conv(F_j)  # 生成注意力图
        A_j = F.softmax(A_j, dim=1)      # 使用 Sigmoid 归一化到 [0, 1]
        
        # 2. 上采样注意力图 A'_j
        A_j_up = F.interpolate(A_j, scale_factor=2, mode='bilinear', align_corners=False)
        
        # 3. 特征增强（逐元素相乘）
        F_hat_j = F_j * A_j_up  # 增强特征
        
        # 4. 特征融合（残差连接）
        F_fused = F_hat_j + F_j  # 跳跃连接
        
        # 5. 通过 1x1 卷积生成最终输出 R_j
        R_j = self.fusion_conv(F_fused)
        
        return R_j

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, z):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return z * y.expand_as(x)

class CatFusion(nn.Module):
    def __init__(self, 
                 out_channels,
                 level=0,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='bilinear', align_corners=False)):
        super(CatFusion, self).__init__()
        self.level = level
        self.upsample_cfg = upsample_cfg
        # self.mask_convs1 = nn.ModuleList()
        # self.mask_convs2 = nn.ModuleList()
        # self.gate_mask = nn.ModuleList()
        self.directional_conv = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        ])

        self.mask_convs1 = DepthwiseSeparableConvModule(
            out_channels * 2,
            out_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg if not no_norm_on_lateral else None,
            act_cfg=act_cfg,
            inplace=False)
        self.mask_convs2 = ConvModule(
            out_channels * 3,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg if not no_norm_on_lateral else None,
            act_cfg=act_cfg,
            inplace=False)
        self.gate_mask = nn.Sequential(
            ConvModule(
            out_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg if not no_norm_on_lateral else None,
            act_cfg=act_cfg,
            inplace=False)
            )
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x_s , x_l):

        prev_shape = x_l.shape[2:]
        x_s_up =  F.interpolate(x_s, size=prev_shape, **self.upsample_cfg)
        x = torch.cat((x_l, x_s_up), dim=1)
        feat1 = self.mask_convs1(x)
        dir_feats = [conv(feat1) for conv in self.directional_conv]
        dir_feats = torch.cat(dir_feats, dim=1)

        feat2 = self.mask_convs2(dir_feats)
        feat2 = F.softmax(feat2, dim=1) 
        x_s_up = x_s_up * feat2
        # combined = torch.cat([feat1, feat2], dim=1)
        # gate = self.gate_mask(combined)
        fusion = x_l + self.gamma * x_s_up

        return fusion

@HEADS.register_module()
class FPNASPPHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, upsample_cfg=dict(mode='bilinear', align_corners=False), **kwargs):
        super(FPNASPPHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.num_outs = len(feature_strides)
        self.upsample_cfg = upsample_cfg

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.cat_convs = nn.ModuleList()

        #self.aspp0 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp1 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp2 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp3 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)

        self.down1 = ConvModule(
                        self.channels * 2,
                        self.channels * 2,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)
        self.down2 = ConvModule(
                        self.channels * 2,
                        self.channels * 2,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)
        self.down3 = ConvModule(
                        self.channels * 2,
                        self.channels * 2,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)

        for i in range(0, self.num_outs):
            l_conv = FRM(
                self.in_channels[i],
                self.channels * 2,
                )
            cat_conv = CatFusion(self.channels * 2)
            fpn_conv = ConvModule(
                self.channels * 2,
                self.channels * 2,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.cat_convs.append(cat_conv)

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.channels * 2 if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs):

        inputs = self._transform_inputs(inputs)
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # laterals.append(self.aspp0([inputs[-1], inputs[0]]))

        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # laterals[i - 1] += F.interpolate(laterals[i],
                #                                  **self.upsample_cfg)
                laterals[i - 1] = self.cat_convs[i - 1](laterals[i], laterals[i - 1])
            else:
                # prev_shape = laterals[i - 1].shape[2:]
                # laterals[i - 1] += F.interpolate(
                #     laterals[i], size=prev_shape, **self.upsample_cfg)
                laterals[i - 1] = self.cat_convs[i - 1](laterals[i], laterals[i - 1])
        
        # laterals[used_backbone_levels] = self.aspp0([laterals[used_backbone_levels], laterals[0]])
        laterals[used_backbone_levels] = self.aspp1([laterals[used_backbone_levels], laterals[0]])
        laterals[used_backbone_levels - 1] = self.aspp2([laterals[used_backbone_levels - 1], laterals[0]])
        laterals[used_backbone_levels - 2] = self.aspp3([laterals[used_backbone_levels - 2], laterals[0]])
        
        # down1 = self.down1(laterals[0])
        # laterals[used_backbone_levels - 2] = laterals[used_backbone_levels - 2] + down1
        # down2 = self.down2(down1)
        # laterals[used_backbone_levels - 1] = laterals[used_backbone_levels - 1] + down2
        # down3 = self.down3(down2)
        # laterals[used_backbone_levels] = laterals[used_backbone_levels] + down3

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels + 1)
        ]

        output = self.scale_heads[0](outs[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](outs[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = self.cls_seg(output)
        return output
