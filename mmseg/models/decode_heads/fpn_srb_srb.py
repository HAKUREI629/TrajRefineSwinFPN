import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

class SpatialRefinementBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialRefinementBlock, self).__init__()
        # 通道压缩
        self.reduce_high = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.reduce_low = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 上采样
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 偏移与权重计算
        self.offset_conv = nn.Conv2d(in_channels * 2, 2, kernel_size=3, padding=1)
        self.weight_conv = nn.Conv2d(in_channels * 2, 1, kernel_size=3, padding=1)

        # 输出卷积
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, high_feat, low_feat):
        # 通道压缩
        high_feat_reduced = self.reduce_high(high_feat)
        low_feat_reduced = self.reduce_low(low_feat)

        # 上采样
        high_feat_upsampled = self.deconv(high_feat_reduced)

        # 拼接特征
        concat_feat = torch.cat([high_feat_upsampled, low_feat_reduced], dim=1)

        # 偏移计算
        offset = self.offset_conv(concat_feat)
        weight = torch.sigmoid(self.weight_conv(concat_feat))

        # 双线性插值进行可微分采样
        B, C, H, W = high_feat_upsampled.size()
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(high_feat.device)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1) + offset

        grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (W - 1) - 1
        grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (H - 1) - 1
        grid = grid.permute(0, 2, 3, 1)

        refined_feat = F.grid_sample(high_feat_upsampled, grid, align_corners=True)

        # 权重修正
        fused_feat = self.final_conv(weight * refined_feat + low_feat)
        return fused_feat

class ChannelRefinementBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelRefinementBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_downsample = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.conv_fuse = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, high_feat, low_feat):
        # 低层特征通道权重
        alpha = self.channel_fc(self.global_avg_pool(low_feat))

        # 高层特征通道加权
        refined_high_feat = high_feat * alpha

        # 特征融合
        low_feat_down = self.conv_downsample(low_feat)
        fused_feat = self.conv_fuse(refined_high_feat + low_feat_down)
        return fused_feat

@HEADS.register_module()
class FPNSRBCRBHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, upsample_cfg=dict(mode='bilinear', align_corners=False), **kwargs):
        super(FPNSRBCRBHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.num_outs = len(feature_strides)
        self.upsample_cfg = upsample_cfg

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.cat_convs = nn.ModuleList()

        # self.aspp0 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
    

        for i in range(0, self.num_outs):
            l_conv = ConvModule(
                self.in_channels[i],
                self.channels * 2,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
        
        self.srb2 = SpatialRefinementBlock(self.channels * 2)
        self.srb3 = SpatialRefinementBlock(self.channels * 2)
        self.srb4 = SpatialRefinementBlock(self.channels * 2)

        self.crb2 = ChannelRefinementBlock(self.channels * 2)
        self.crb3 = ChannelRefinementBlock(self.channels * 2)
        self.crb4 = ChannelRefinementBlock(self.channels * 2)

        self.scale_heads = nn.ModuleList()
        self.scaleup_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            scaleup_head = []
            for k in range(1):
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
                    scaleup_head.append(
                        nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
            self.scaleup_heads.append(nn.Sequential(*scaleup_head))

    def forward(self, inputs):

        inputs = self._transform_inputs(inputs)
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # laterals.append(self.aspp0([inputs[-1], inputs[0]]))

        used_backbone_levels = len(laterals) - 1
        laterals[used_backbone_levels - 1] = self.srb4(laterals[used_backbone_levels], laterals[used_backbone_levels - 1])
        laterals[used_backbone_levels - 2] = self.srb3(laterals[used_backbone_levels - 1], laterals[used_backbone_levels - 2])
        laterals[used_backbone_levels - 3] = self.srb2(laterals[used_backbone_levels - 2], laterals[used_backbone_levels - 3])
        
        laterals[used_backbone_levels - 2] = self.crb4(laterals[used_backbone_levels - 2], laterals[used_backbone_levels - 3])
        laterals[used_backbone_levels - 1] = self.crb3(laterals[used_backbone_levels - 1], laterals[used_backbone_levels - 2])
        laterals[used_backbone_levels] = self.crb2(laterals[used_backbone_levels], laterals[used_backbone_levels - 1])

        outs = laterals

        # output = self.scale_heads[0](outs[0])
        # for i in range(1, len(self.feature_strides)):
        #     # non inplace
        #     output = output + resize(
        #         self.scale_heads[i](outs[i]),
        #         size=output.shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        
        output = self.scale_heads[3](outs[3])
        output = self.scaleup_heads[3](output)
        for i in range(len(self.feature_strides) - 2, -1, -1):
            # non inplace
            output = output + self.scale_heads[i](outs[i])
            output = self.scaleup_heads[i](output)

        output = self.cls_seg(output)
        return output