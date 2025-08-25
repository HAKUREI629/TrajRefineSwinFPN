import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
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

class SpatialRefinementBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=1, 
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='bilinear', align_corners=False)):
        """
        SRB 模块初始化
        Args:
            in_channels (int): 输入特征图的通道数
            reduction_ratio (int): 通道压缩比率，默认为2
        """
        super(SpatialRefinementBlock, self).__init__()
        reduced_channels = in_channels // reduction_ratio

        # 通道压缩 (1x1卷积)
        self.reduce_high = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.reduce_low = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)

        # 上采样模块 (3x3反卷积)
        self.deconv = nn.ConvTranspose2d(reduced_channels, reduced_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 采样点偏移与全局权重学习 (3x3卷积)
        self.offset_conv = nn.Conv2d(reduced_channels * 2, 2, kernel_size=3, padding=1)  # 偏移映射 (dx, dy)
        self.weight_conv = nn.Conv2d(reduced_channels * 2, 1, kernel_size=3, padding=1)  # 全局权重映射

        # 最终融合卷积
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, F_high, F_low):
        """
        前向传播
        Args:
            F_high (Tensor): 上层特征图 (较低分辨率)
            F_low (Tensor): 下层特征图 (较高分辨率)
        Returns:
            Tensor: 精细化后的特征图
        """
        # 通道压缩
        Fh_reduced = self.reduce_high(F_high)
        Fl_reduced = self.reduce_low(F_low)

        # 上采样高层特征
        Fh_upsampled = self.deconv(Fh_reduced)

        # 拼接特征图
        concat_features = torch.cat([Fh_upsampled, Fl_reduced], dim=1)

        # 偏移与全局权重学习
        offset = self.offset_conv(concat_features)  # (B, 2, H, W)
        weight = torch.sigmoid(self.weight_conv(concat_features))  # (B, 1, H, W)

        # 使用grid_sample进行差异化双线性插值
        B, _, H, W = offset.size()
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(offset.device)  # (2, H, W)

        # 偏移归一化
        offset_norm = offset / torch.tensor([W, H]).view(1, 2, 1, 1).to(offset.device)
        sampling_grid = torch.stack((grid[0] / W, grid[1] / H), dim=-1)  # (H, W, 2)
        sampling_grid = sampling_grid + offset_norm.permute(0, 2, 3, 1)  # (B, H, W, 2)

        # 应用grid_sample
        Fh_sampled = F.grid_sample(Fh_upsampled, 2 * sampling_grid - 1, align_corners=True)

        # 全局权重细化
        Fh_refined = Fh_sampled * weight

        # 最终特征融合
        out = self.final_conv(Fh_refined + F.interpolate(F_high, size=F_low.shape[2:], mode='bilinear', align_corners=True) + F_low)
        return out


class SpatialRefinementBlockV1(nn.Module):
    def __init__(self, in_channels, reduction_ratio=1, 
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='bilinear', align_corners=False)):
        """
        SRB 模块初始化
        Args:
            in_channels (int): 输入特征图的通道数
            reduction_ratio (int): 通道压缩比率，默认为2
        """
        super(SpatialRefinementBlockV1, self).__init__()
        reduced_channels = in_channels // reduction_ratio

        # 通道压缩 (1x1卷积)
        self.reduce_high = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.reduce_low = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)

        # 上采样模块 (3x3反卷积)
        self.deconv = nn.ConvTranspose2d(reduced_channels, reduced_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 采样点偏移与全局权重学习 (3x3卷积)
        self.offset_conv = nn.Conv2d(reduced_channels * 2, 2, kernel_size=3, padding=1)  # 偏移映射 (dx, dy)
        self.weight_conv = nn.Conv2d(reduced_channels * 3, 1, kernel_size=3, padding=1)  # 全局权重映射

        # 最终融合卷积
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.directional_conv = nn.ModuleList([
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=2, dilation=2)
        ])

        self.mask_convs1 = DepthwiseSeparableConvModule(
            reduced_channels * 2,
            reduced_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg if not no_norm_on_lateral else None,
            act_cfg=act_cfg,
            inplace=False)


    def forward(self, F_high, F_low):
        """
        前向传播
        Args:
            F_high (Tensor): 上层特征图 (较低分辨率)
            F_low (Tensor): 下层特征图 (较高分辨率)
        Returns:
            Tensor: 精细化后的特征图
        """
        # 通道压缩
        Fh_reduced = F_high
        Fl_reduced = F_low

        # 上采样高层特征
        Fh_upsampled = self.deconv(Fh_reduced)

        # 拼接特征图
        concat_features = torch.cat([Fh_upsampled, Fl_reduced], dim=1)

        # 偏移与全局权重学习
        offset = self.offset_conv(concat_features)  # (B, 2, H, W)

        feat1 = self.mask_convs1(concat_features)
        dir_feats = [conv(feat1) for conv in self.directional_conv]
        dir_feats = torch.cat(dir_feats, dim=1)
        weight = torch.sigmoid(self.weight_conv(dir_feats))  # (B, 1, H, W)

        # 使用grid_sample进行差异化双线性插值
        B, _, H, W = offset.size()
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=0).float().to(offset.device)  # (2, H, W)

        # 偏移归一化
        offset_norm = offset / torch.tensor([W, H]).view(1, 2, 1, 1).to(offset.device)
        sampling_grid = torch.stack((grid[0] / W, grid[1] / H), dim=-1)  # (H, W, 2)
        sampling_grid = sampling_grid + offset_norm.permute(0, 2, 3, 1)  # (B, H, W, 2)

        # 应用grid_sample
        Fh_sampled = F.grid_sample(Fh_upsampled, 2 * sampling_grid - 1, align_corners=True)

        # 全局权重细化
        Fh_refined = Fh_sampled * weight

        # 最终特征融合
        out = self.final_conv(Fh_refined + F_low)
        return out

class SpatialRefinementBlockV2(nn.Module):
    def __init__(self, in_channels, reduction_ratio=1, 
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='bilinear', align_corners=False)):
        """
        SRB 模块初始化
        Args:
            in_channels (int): 输入特征图的通道数
            reduction_ratio (int): 通道压缩比率，默认为2
        """
        super(SpatialRefinementBlockV2, self).__init__()
        reduced_channels = in_channels // reduction_ratio

        # 通道压缩 (1x1卷积)
        self.reduce_high = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.reduce_low = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)

        # 上采样模块 (3x3反卷积)
        self.deconv = nn.ConvTranspose2d(reduced_channels, reduced_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 采样点偏移与全局权重学习 (3x3卷积)
        self.offset_conv = nn.Conv2d(reduced_channels * 2, 2 * 3 * 3, kernel_size=3, padding=1)  # 偏移映射 (dx, dy)
        self.weight_conv = nn.Conv2d(reduced_channels * 3, 1, kernel_size=3, padding=1)  # 全局权重映射

        self.deform_conv = DeformConv2d(reduced_channels, reduced_channels, kernel_size=3, padding=1)

        # 最终融合卷积
        self.final_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.directional_conv = nn.ModuleList([
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, padding=2, dilation=2)
        ])

        self.mask_convs1 = DepthwiseSeparableConvModule(
            reduced_channels * 2,
            reduced_channels,
            3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg if not no_norm_on_lateral else None,
            act_cfg=act_cfg,
            inplace=False)


    def forward(self, F_high, F_low):
        """
        前向传播
        Args:
            F_high (Tensor): 上层特征图 (较低分辨率)
            F_low (Tensor): 下层特征图 (较高分辨率)
        Returns:
            Tensor: 精细化后的特征图
        """
        # 通道压缩
        Fh_reduced = F_high
        Fl_reduced = F_low

        # 上采样高层特征
        Fh_upsampled = self.deconv(Fh_reduced)

        # 拼接特征图
        concat_features = torch.cat([Fh_upsampled, Fl_reduced], dim=1)

        # 偏移与全局权重学习
        offset = self.offset_conv(concat_features)  # (B, 2, H, W)

        feat1 = self.mask_convs1(concat_features)
        dir_feats = [conv(feat1) for conv in self.directional_conv]
        dir_feats = torch.cat(dir_feats, dim=1)
        weight = torch.sigmoid(self.weight_conv(dir_feats))  # (B, 1, H, W)

        # 使用grid_sample进行差异化双线性插值
        # B, _, H, W = offset.size()
        # grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        # grid = torch.stack((grid_x, grid_y), dim=0).float().to(offset.device)  # (2, H, W)

        # # 偏移归一化
        # offset_norm = offset / torch.tensor([W, H]).view(1, 2, 1, 1).to(offset.device)
        # sampling_grid = torch.stack((grid[0] / W, grid[1] / H), dim=-1)  # (H, W, 2)
        # sampling_grid = sampling_grid + offset_norm.permute(0, 2, 3, 1)  # (B, H, W, 2)

        # # 应用grid_sample
        # Fh_sampled = F.grid_sample(Fh_upsampled, 2 * sampling_grid - 1, align_corners=True)

        Fh_sampled = self.deform_conv(Fh_upsampled, offset)

        # 全局权重细化
        Fh_refined = Fh_sampled * weight

        # 最终特征融合
        out = self.final_conv(Fh_refined + F_low)
        return out


@HEADS.register_module()
class FPNASPPHeadv5(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, upsample_cfg=dict(mode='bilinear', align_corners=False), **kwargs):
        super(FPNASPPHeadv5, self).__init__(
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
        self.aspp1 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp2 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp3 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)


        for i in range(0, self.num_outs):
            l_conv = ConvModule(
                self.in_channels[i],
                self.channels * 2,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            cat_conv = SpatialRefinementBlock(self.channels * 2)
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

@HEADS.register_module()
class FPNASPPHeadv5gai(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, upsample_cfg=dict(mode='bilinear', align_corners=False), **kwargs):
        super(FPNASPPHeadv5gai, self).__init__(
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
        self.aspp1 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp2 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp3 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)


        for i in range(0, self.num_outs):
            l_conv = ConvModule(
                self.in_channels[i],
                self.channels * 2,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            cat_conv = SpatialRefinementBlockV1(self.channels * 2)
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

# final used model !!!!!!!!!!!
@HEADS.register_module()
class FPNASPPHeadv5gai1(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, upsample_cfg=dict(mode='bilinear', align_corners=False), **kwargs):
        super(FPNASPPHeadv5gai1, self).__init__(
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
        self.aspp1 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp2 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp3 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)


        for i in range(0, self.num_outs):
            l_conv = ConvModule(
                self.in_channels[i],
                self.channels * 2,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            cat_conv = SpatialRefinementBlockV2(self.channels * 2)
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

        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = self.cat_convs[i - 1](laterals[i], laterals[i - 1])
            else:
                laterals[i - 1] = self.cat_convs[i - 1](laterals[i], laterals[i - 1])
        
        laterals[used_backbone_levels] = self.aspp1([laterals[used_backbone_levels], laterals[0]])
        laterals[used_backbone_levels - 1] = self.aspp2([laterals[used_backbone_levels - 1], laterals[0]])
        laterals[used_backbone_levels - 2] = self.aspp3([laterals[used_backbone_levels - 2], laterals[0]])
        
        np.save('./tools/laterals.npy', laterals[0].detach().cpu().numpy())
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels + 1)
        ]
        

        output = self.scale_heads[3](outs[3])
        output = self.scaleup_heads[3](output)
        for i in range(len(self.feature_strides) - 2, -1, -1):
            # non inplace
            output = output + self.scale_heads[i](outs[i])
            output = self.scaleup_heads[i](output)
        np.save('./tools/laterals_final.npy', output.detach().cpu().numpy())
        output = self.cls_seg(output)
        return output

@HEADS.register_module()
class FPNASPPHeadv5gai1_noUp(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, upsample_cfg=dict(mode='bilinear', align_corners=False), **kwargs):
        super(FPNASPPHeadv5gai1_noUp, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.num_outs = len(feature_strides)
        self.upsample_cfg = upsample_cfg

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        # self.cat_convs = nn.ModuleList()

        # self.aspp0 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp1 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp2 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp3 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)


        for i in range(0, self.num_outs):
            l_conv = ConvModule(
                self.in_channels[i],
                self.channels * 2,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            # cat_conv = SpatialRefinementBlockV2(self.channels * 2)
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
            # self.cat_convs.append(cat_conv)

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

        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] =  laterals[i - 1] + F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
        
        laterals[used_backbone_levels] = self.aspp1([laterals[used_backbone_levels], laterals[0]])
        laterals[used_backbone_levels - 1] = self.aspp2([laterals[used_backbone_levels - 1], laterals[0]])
        laterals[used_backbone_levels - 2] = self.aspp3([laterals[used_backbone_levels - 2], laterals[0]])
        

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels + 1)
        ]
        
        output = self.scale_heads[3](outs[3])
        output = self.scaleup_heads[3](output)
        for i in range(len(self.feature_strides) - 2, -1, -1):
            # non inplace
            output = output + self.scale_heads[i](outs[i])
            output = self.scaleup_heads[i](output)
        np.save('./tools/laterals_noUp.npy', output.detach().cpu().numpy())
        output = self.cls_seg(output)
        return output

@HEADS.register_module()
class FPNASPPHeadv5gai1_noASPP(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, upsample_cfg=dict(mode='bilinear', align_corners=False), **kwargs):
        super(FPNASPPHeadv5gai1_noASPP, self).__init__(
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
        # self.aspp1 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        # self.aspp2 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        # self.aspp3 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)


        for i in range(0, self.num_outs):
            l_conv = ConvModule(
                self.in_channels[i],
                self.channels * 2,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            cat_conv = SpatialRefinementBlockV2(self.channels * 2)
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

        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = self.cat_convs[i - 1](laterals[i], laterals[i - 1])
            else:
                laterals[i - 1] = self.cat_convs[i - 1](laterals[i], laterals[i - 1])
        
        

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels + 1)
        ]
        
        output = self.scale_heads[3](outs[3])
        output = self.scaleup_heads[3](output)
        for i in range(len(self.feature_strides) - 2, -1, -1):
            # non inplace
            output = output + self.scale_heads[i](outs[i])
            output = self.scaleup_heads[i](output)
        np.save('./tools/laterals_noASPP.npy', output.detach().cpu().numpy())
        output = self.cls_seg(output)
        return output

@HEADS.register_module()
class FPNASPPHeadv5gai1_noTop(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, upsample_cfg=dict(mode='bilinear', align_corners=False), **kwargs):
        super(FPNASPPHeadv5gai1_noTop, self).__init__(
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
        self.aspp1 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp2 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)
        self.aspp3 = DepthwiseSeparableASPP(dilations=(1, 6, 12, 18), in_channels=self.channels * 2, channels=self.channels * 2, c1_in_channels=self.channels * 2, c1_channels=48)


        for i in range(0, self.num_outs):
            l_conv = ConvModule(
                self.in_channels[i],
                self.channels * 2,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            cat_conv = SpatialRefinementBlockV2(self.channels * 2)
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

        used_backbone_levels = len(laterals) - 1
        for i in range(used_backbone_levels, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = self.cat_convs[i - 1](laterals[i], laterals[i - 1])
            else:
                laterals[i - 1] = self.cat_convs[i - 1](laterals[i], laterals[i - 1])
        
        laterals[used_backbone_levels] = self.aspp1([laterals[used_backbone_levels], laterals[0]])
        laterals[used_backbone_levels - 1] = self.aspp2([laterals[used_backbone_levels - 1], laterals[0]])
        laterals[used_backbone_levels - 2] = self.aspp3([laterals[used_backbone_levels - 2], laterals[0]])
        

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