# Written by Haram Choi
import math
import torch
import torch.nn as nn
from einops import rearrange

from torchvision.transforms import functional as TF
from timm.models.layers import to_2tuple, to_ntuple, _assert, trunc_normal_, DropPath, Mlp
#from timm.models.helpers import named_apply
#from timm.models.vision_transformer import get_init_weights_vit
#from timm.models.efficientnet_builder import _init_weight_goog
from timm.models.fx_features import register_notrace_function

from typing import Optional
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead

def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww (xaxis matrix & yaxis matrix)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        head_dim (int, optional): Number of channels per head (dim // num_heads if not set)
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    Inputs:
        x: [B*wh*ww, WH*WW, D]
    Returns:
        x:  [B*wh*ww, WH*WW, D]
    """

    def __init__(self, dim, num_heads, window_size, head_dim=None, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # WH, WW
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        
        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))
        
        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(win_h, win_w))
        
        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        
    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [B*wh*ww, WH*WW, D]
            mask: (0/-inf) mask with shape of (wh*ww, WH*WW, WH*WW) or None
        Returns:
            x: [B*wh*ww, WH*WW, D]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4) # [qkv(3), B*wh*ww, nheads, WH*WW, dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        
        # scaled cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) # [B*wh*ww, nheads, WH*WW, WH*WW]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        
        attn = attn + self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1) # [B*wh*ww, WH*WW, nheads*dim_per_head]=[B*wh*ww, WH*WW, D]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class NGramContext(nn.Module):
    '''
    Args:
        dim (int): Number of input channels.
        window_size (int or tuple[int]): The height and width of the window.
        ngram (int): How much windows(or patches) to see.
        ngram_num_heads (int):
        padding_mode (str, optional): How to pad.  Default: seq_refl_win_pad
                                                   Options: ['seq_refl_win_pad', 'zero_pad']
    Inputs:
        x: [B, ph, pw D] or [B, C, H, W]
    Returns:
        context: [B, wh, ww, 1, 1, D] or [B, C, ph, pw]
    '''
    def __init__(self, dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad'):
        super(NGramContext, self).__init__()
        _assert(padding_mode in ['seq_refl_win_pad', 'zero_pad'], "padding mode should be 'seq_refl_win_pad' or 'zero_pad'!!")
        
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.ngram = ngram
        self.padding_mode = padding_mode
        
        self.unigram_embed = nn.Conv2d(dim, dim//2,
                                       kernel_size=(self.window_size[0], self.window_size[1]), 
                                       stride=self.window_size, padding=0, groups=dim//2)
        self.ngram_attn = WindowAttention(dim=dim//2, num_heads=ngram_num_heads, window_size=ngram)
        self.avg_pool = nn.AvgPool2d(ngram)
        self.merge = nn.Conv2d(dim, dim, 1, 1, 0)
        
    def seq_refl_win_pad(self, x, back=False):
        if self.ngram == 1: return x
        x = TF.pad(x, (0,0,self.ngram-1,self.ngram-1)) if not back else TF.pad(x, (self.ngram-1,self.ngram-1,0,0))
        if self.padding_mode == 'zero_pad':
            return x
        if not back:
            (start_h, start_w), (end_h, end_w) = to_2tuple(-2*self.ngram+1), to_2tuple(-self.ngram)
            # pad lower
            x[:,:,-(self.ngram-1):,:] = x[:,:,start_h:end_h,:]
            # pad right
            x[:,:,:,-(self.ngram-1):] = x[:,:,:,start_w:end_w]
        else:
            (start_h, start_w), (end_h, end_w) = to_2tuple(self.ngram), to_2tuple(2*self.ngram-1)
            # pad upper
            x[:,:,:self.ngram-1,:] = x[:,:,start_h:end_h,:]
            # pad left
            x[:,:,:,:self.ngram-1] = x[:,:,:,start_w:end_w]
            
        return x
    
    def sliding_window_attention(self, unigram):
        slide = unigram.unfold(3, self.ngram, 1).unfold(2, self.ngram, 1)
        slide = rearrange(slide, 'b c h w ww hh -> b (h hh) (w ww) c') # [B, 2(wh+ngram-2), 2(ww+ngram-2), D/2]
        slide, num_windows = window_partition(slide, self.ngram) # [B*wh*ww, ngram, ngram, D/2], (wh, ww)
        slide = slide.view(-1, self.ngram*self.ngram, self.dim//2) # [B*wh*ww, ngram*ngram, D/2]
        
        context = self.ngram_attn(slide).view(-1, self.ngram, self.ngram, self.dim//2) # [B*wh*ww, ngram, ngram, D/2]
        context = window_unpartition(context, num_windows) # [B, wh*ngram, ww*ngram, D/2]
        context = rearrange(context, 'b h w d -> b d h w') # [B, D/2, wh*ngram, ww*ngram]
        context = self.avg_pool(context) # [B, D/2, wh, ww]
        return context
        
    def forward(self, x):
        B, ph, pw, D = x.size()
        x = rearrange(x, 'b ph pw d -> b d ph pw') # [B, D, ph, pw]
        unigram = self.unigram_embed(x) # [B, D/2, wh, ww]
        
        unigram_forward_pad = self.seq_refl_win_pad(unigram, False) # [B, D/2, wh+ngram-1, ww+ngram-1]
        unigram_backward_pad = self.seq_refl_win_pad(unigram, True) # [B, D/2, wh+ngram-1, ww+ngram-1]
        
        context_forward = self.sliding_window_attention(unigram_forward_pad) # [B, D/2, wh, ww]
        context_backward = self.sliding_window_attention(unigram_backward_pad) # [B, D/2, wh, ww]
        
        context_bidirect = torch.cat([context_forward, context_backward], dim=1) # [B, D, wh, ww]
        context_bidirect = self.merge(context_bidirect) # [B, D, wh, ww]
        context_bidirect = rearrange(context_bidirect, 'b d h w -> b h w d') # [B, wh, ww, D]
        
        return context_bidirect.unsqueeze(-2).unsqueeze(-2).contiguous() # [B, wh, ww, 1, 1, D]

class NGramWindowPartition(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        window_size (int): The height and width of the window.
        ngram (int): How much windows to see as context.
        ngram_num_heads (int):
        shift_size (int, optional): Shift size for SW-MSA.  Default: 0
    Inputs:
        x: [B, ph, pw, D]
    Returns:
        [B*wh*ww, WH, WW, D], (wh, ww)
    """
    def __init__(self, dim, window_size, ngram, ngram_num_heads, shift_size=0):
        super(NGramWindowPartition, self).__init__()
        self.window_size = window_size
        self.ngram = ngram
        self.shift_size = shift_size
        
        self.ngram_context = NGramContext(dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad')
    
    def forward(self, x):
        B, ph, pw, D = x.size()
        wh, ww = ph//self.window_size, pw//self.window_size # number of windows (height, width)
        _assert(0 not in [wh, ww], "feature map size should be larger than window size!")
        
        context = self.ngram_context(x) # [B, wh, ww, 1, 1, D]
        
        windows = rearrange(x, 'b (h wh) (w ww) c -> b h w wh ww c', 
                            wh=self.window_size, ww=self.window_size).contiguous() # [B, wh, ww, WH, WW, D]. semi window partitioning
        windows+=context # [B, wh, ww, WH, WW, D]. inject context
        
        # Cyclic Shift
        if self.shift_size>0:
            x = rearrange(windows, 'b h w wh ww c -> b (h wh) (w ww) c').contiguous() # [B, ph, pw, D]. re-patchfying
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) # [B, ph, pw, D]. cyclic shift
            windows = rearrange(shifted_x, 'b (h wh) (w ww) c -> b h w wh ww c', 
                                wh=self.window_size, ww=self.window_size).contiguous() # [B*wh*ww, WH, WW, D]. re-semi window partitioning
            
        windows = rearrange(windows, 'b h w wh ww c -> (b h w) wh ww c').contiguous() # [B*wh*ww, WH, WW, D]. window partitioning
        
        return windows, (wh, ww)

def window_partition(x, window_size: int):
    """
    Args:
        x: [B, ph, pw, D]
        window_size (int): The height and width of the window.
    Returns:
        [B*wh*ww, WH, WW, D], (wh, ww)
    """
    B, ph, pw, D = x.size()
    wh, ww = ph//window_size, pw//window_size # number of windows (height, width)
    if 0 in [wh, ww]:
        # if feature map size is smaller than window size, do not partition
        return x, (wh, ww)
    windows = rearrange(x, 'b (h wh) (w ww) c -> (b h w) wh ww c', wh=window_size, ww=window_size).contiguous()
    return windows.contiguous(), (wh, ww)
    
@register_notrace_function  # reason: int argument is a Proxy
def window_unpartition(windows, num_windows):
    """
    Args:
        windows: [B*wh*ww, WH, WW, D]
        num_windows (tuple[int]): The height and width of the window.
    Returns:
        x: [B, ph, pw, D]
    """
    x = rearrange(windows, '(p h w) wh ww c -> p (h wh) (w ww) c', h=num_windows[0], w=num_windows[1])
    return x.contiguous()

class NSTB(nn.Module): # N-Gram Swin Transformer Block
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        training_num_patches: Number of height and width of the patch only when training.
        ngram (int): ngram: (int): How much windows to see as context.
        num_heads (int): Number of attention heads.
        window_size (int): The size of the window.
        shift_size (int): Shift size for SW-MSA.
        head_dim (int, optional): Number of channels per head (dim // num_heads if not set).  Default: None
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    Inputs:
        [B, ph*pw, D], (ph, pw)
    Returns:
        [B, ph*pw, D], (ph, pw)
    """

    def __init__(
        self, dim, training_num_patches, ngram, num_heads, window_size, shift_size,
        head_dim=None, mlp_ratio=2., qkv_bias=True, 
        drop=0., attn_drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super().__init__()
        self.dim = dim
        self.training_num_patches = training_num_patches
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        _assert(0 <= self.shift_size < self.window_size, "shift_size must in 0~window_size")
        
        self.ngram_window_partition = NGramWindowPartition(dim, window_size, ngram, num_heads, shift_size=shift_size)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, head_dim=head_dim, window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        # window mask
        attn_mask = self.make_mask(training_num_patches) if shift_size>0 else None
        self.register_buffer("attn_mask", attn_mask)
        
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        
    def make_mask(self, num_patches):
        ph, pw = num_patches
        img_mask = torch.zeros((1, ph, pw, 1)) # [1, ph, pw, 1]
        cnt = 0
        for h in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)):
            for w in (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None)):
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows, (wh,ww) = window_partition(img_mask, self.window_size)  # [wh*ww*1, WH, WW, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [wh*ww, WH*WW]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # [wh*ww, WH, WW]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0)) # [wh*ww, WH, WW]
        return attn_mask
    
    def _attention(self, x, num_patches):
        # window partition - (cyclic shift) - cosine attention - window unpartition - (reverse shift)
        ph, pw = num_patches
        B, p, D = x.size()
        _assert(p == ph * pw, f"size is wrong!")
        
        x = x.view(B, ph, pw, D) # [B, ph, pw, D], Unembedding
        
        # N-Gram Window Partition (-> cyclic shift)
        x_windows, (wh,ww) = self.ngram_window_partition(x) # [B*wh*ww, WH, WW, D], (wh, ww)
        
        x_windows = x_windows.view(-1, self.window_size * self.window_size, D)  # [B*wh*ww, WH*WW, D], Re-embedding
        
        # W-MSA/SW-MSA
        if self.training_num_patches==num_patches:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [B*wh*ww, WH*WW, D]
        else:
            eval_attn_mask = self.make_mask(num_patches).to(x.device) if self.shift_size>0 else None
            attn_windows = self.attn(x_windows, mask=eval_attn_mask)  # [B*wh*ww, WH*WW, D]
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, D) # [B*wh*ww, WH, WW, D], Unembedding
        
        # Window Unpartition
        shifted_x = window_unpartition(attn_windows, (wh,ww))  # [B, ph, pw, D]
        
        # Reverse Cyclic Shift
        reversed_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x # [B, ph, pw, D]
        reversed_x = reversed_x.view(B, ph*pw, D) # [B, ph*pw, D], Re-embedding
        
        return reversed_x

    def forward(self, x, num_patches):
        x_ = x
        # (S)W Attention -> Layer-Norm -> Drop-Path -> Skip-Connection
        x = x + self.drop_path(self.norm1(self._attention(x, num_patches))) # [B, ph*pw, D]
        # FFN -> Layer-Norm -> Drop-Path -> Skip-Connection
        x = x + self.drop_path(self.norm2(self.ffn(x))) # [B, ph*pw, 4D] -> [B, ph*pw, D]
        return x_, x, num_patches

class DecoderLayer(nn.Module):
    """ A N-Gram Context Swin Transformer Decoder Layer for one stage.
    Args:
        dim (int): Number of input dimension (channels).
        training_num_patches (tuple[int]): Number of height and width of the patch only when training.
        ngram (int): ngram: (int): How much windows to see as context.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): The size of the window.
        head_dim (int, optional): Channels per head (dim // num_heads if not set).  Default: None
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.  Default: 2.0
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        last (bool, optional): The final layer of decoder stages or not.  Default: False
    """
    
    def __init__(
        self, dim, training_num_patches, ngram, depth, num_heads,
        window_size, head_dim=None, mlp_ratio=2., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        
        super(DecoderLayer, self).__init__()
        self.dim = dim
        
        self.blocks = nn.Sequential(*[
            NSTB(
                dim=dim, training_num_patches=training_num_patches, ngram=ngram,
                num_heads=num_heads, window_size=window_size, shift_size=0 if (i%2==0) else window_size//2, 
                head_dim=head_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, 
                act_layer=act_layer, norm_layer=norm_layer)
            for i in range(depth)])
    
    def forward(self, x, num_patches, out_size=None):
        x_ = 0 # for within stage residual connection
        for i, blk in enumerate(self.blocks):
            # Within Stage Residual Connection
            x_, x, num_patches = blk(x+x_, num_patches)
            
        return x, out_size
    
    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        # NSTB blocks: ngram_window_partition, (S)W-MSA, norm1, FFN, norm2
        for blk in self.blocks:
            flops += blk.flops((H,W))
        return flops

class Reconstruction(nn.Sequential):
    def __init__(self, dim, out_chans=3):
        super(Reconstruction, self).__init__()
        
        self.upscale = upscale = 1
        self.dim = dim
        self.out_chans = out_chans
            
        self.add_module('before_shuffle', nn.Conv2d(dim, out_chans*(upscale**2), 3, 1, 1))
        self.add_module('shuffler', nn.PixelShuffle(upscale)) # [B, dim/(upscale^2), upscale*H, upscale*W)                
        self.add_module('to_origin', nn.Conv2d(out_chans, out_chans, 3, 1, 1))
            
@HEADS.register_module()
class NGswinHead(BaseDecodeHead):
    """
    Args:
        training_img_size (int or tuple[int]): Input image size.  Default 64
        ngrams tuple[int]: How much windows to see as context in each encoder and decoder.  Default: (2,2,2,2)
        in_chans (int): Number of input image channels.  Default: 3
        embed_dim (int): Patch embedding dimension. Dimension of all encoder layers.  Default: 64
        depths (tuple[int]): Depth of each encoder stage. i.e., number of NSTBs.  Default: (6,4,4)
        num_heads (tuple[int]): Number of attention heads in each encoder layer.  Default: (6,4,4)
        head_dim (int or tuple[int]): Channels per head of each encoder layer (dim // num_heads if not set).  Default: None
        dec_dim (int): Dimension of decoder stage.  Default: 64
        dec_depths (int): Depth of a decoder stage.  Default: 6
        dec_num_heads (int): Number of attention heads in a decoder layer.  Default: 6
        dec_head_dim (int): Channels per head of decoder layer (dim // num_heads if not set).  Default: None
        target_mode (str): light_x2, light_x3, light_x4.  Default: light_x2
        window_size (int): The size of the window.  Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.  Default: 2.0
        qkv_bias (bool): If True, add a learnable bias to query, key, value.  Default: True
        drop_rate (float): Dropout rate.  Default: 0.0
        attn_drop_rate (float): Attention dropout rate.  Default: 0.0
        drop_path_rate (float): Stochastic depth rate.  Default: 0.0
        act_layer (nn.Module, optional): Activation layer.  Default: nn.GELU
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        weight_init (str): The way to initialize weights of layers.  Default: ''
    """
    # def __init__(
    #     self, training_img_size=64, ngrams=(2,2,2,2),
    #     in_chans=3, embed_dim=64, depths=(6,4,4), num_heads=(6,4,4), head_dim=None,
    #     dec_dim=64, dec_depths=6, dec_num_heads=6, dec_head_dim=None,
    #     target_mode='light_x1', window_size=8, mlp_ratio=2., qkv_bias=True, img_norm=True,
    #     drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
    #     act_layer=nn.GELU, norm_layer=nn.LayerNorm, weight_init='', **kwargs):
    def __init__(self,
                 pretrain_img_size=512,
                 ngrams=(2,2,2,2),
                 depths=(6,4,4),
                 dec_dim=64,
                 dec_depths=6, 
                 dec_num_heads=6, 
                 dec_head_dim=None,
                 window_size=8,
                 mlp_ratio=2., 
                 qkv_bias=True, 
                 drop_rate=0.0, 
                 attn_drop_rate=0.0, 
                 drop_path_rate=0.0,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        
        super(NGswinHead, self).__init__(**kwargs)
        self.training_img_size = to_2tuple(pretrain_img_size)
        self.ngrams = ngrams
        self.depths = depths
        dec_depths = (dec_depths,)
        dec_num_heads = (dec_num_heads,)
        self.dec_dim=dec_dim
        self.dec_depths=dec_depths
        self.dec_num_heads=dec_num_heads
        self.window_size = window_size
        self.num_encoder_stages = len(depths)
        self.num_decoder_stages = len(dec_depths)
        self.scale = 1

        
        # head_dim = to_ntuple(self.num_encoder_stages)(head_dim) # default None
        dec_head_dim = to_ntuple(self.num_decoder_stages)(dec_head_dim) # default None
        mlp_ratio = to_ntuple(self.num_encoder_stages+self.num_decoder_stages)(mlp_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)+sum(dec_depths))]  # stochastic depth decay rule
        
        # Main Branch
        # self.shallow_extract = ShallowExtractor(in_chans, embed_dim)
        # self.inter_pool = InterPool(embed_dim)
        # for i in range(self.num_encoder_stages):
        #     self.add_module(f'encoder_layer{i+1}', EncoderLayer(
        #                                             dim=embed_dim,
        #                                             training_num_patches=(self.training_img_size[0]//2**i,
        #                                                                   self.training_img_size[1]//2**i),
        #                                             ngram=ngrams[i],
        #                                             depth=depths[i],
        #                                             num_heads=num_heads[i],
        #                                             window_size=window_size,
        #                                             head_dim=head_dim[i],
        #                                             mlp_ratio=mlp_ratio[i],
        #                                             qkv_bias=qkv_bias,
        #                                             drop=drop_rate,
        #                                             attn_drop=attn_drop_rate,
        #                                             drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
        #                                             act_layer=act_layer,
        #                                             norm_layer=norm_layer,
        #                                             downsample=PatchMerging if (i+1)!=self.num_encoder_stages else None,
        #                                             downsample_dim=embed_dim if (i+1)!=self.num_encoder_stages else None,
        #                                             num_cas=i+1))
        # self.bottleneck = SCDPBottleneck(num_encoder_stages=self.num_encoder_stages, 
        #                                  enc_dim=embed_dim, dec_dim=dec_dim, 
        #                                  act_layer=act_layer, norm_layer=norm_layer)
        for i in range(self.num_decoder_stages):
            self.add_module(f'decoder_layer{i+1}', DecoderLayer(
                                                    dim=dec_dim,
                                                    training_num_patches=(self.training_img_size[0]//2**(1-i),
                                                                          self.training_img_size[1]//2**(1-i)),
                                                    ngram=ngrams[self.num_encoder_stages+i],
                                                    depth=dec_depths[i],
                                                    num_heads=dec_num_heads[i],
                                                    window_size=window_size,
                                                    head_dim=dec_head_dim[i],
                                                    mlp_ratio=mlp_ratio[self.num_encoder_stages+i],
                                                    qkv_bias=qkv_bias,
                                                    drop=drop_rate,
                                                    attn_drop=attn_drop_rate,
                                                    drop_path=dpr[sum(depths)+sum(dec_depths[:i]):sum(depths)+sum(dec_depths[:i+1])],
                                                    act_layer=act_layer,
                                                    norm_layer=norm_layer))
        self.norm = norm_layer(dec_dim)
        
        # Reconstruction
        self.to_target = Reconstruction(dec_dim, self.num_classes)
        
        # self.apply(self._init_weights)
        
#         if weight_init != 'skip':
#             self.init_weights(weight_init)

            
    # def _init_weights(self, m):
    #     # Swin V2 manner
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd
    
    def forward_decoder(self, x, num_patches, out_size, e1_):
        x, num_pathces = self.decoder_layer1(x+e1_, num_patches) # [B, HW, D], (H, W) enc-dec skip connection
        out = self.norm(x)
        return out
    
    def forward_reconstruct(self, x, img_size):
        x = rearrange(x, 'p (h w) c -> p c h w', h=img_size[0]) # for pixel-shffule and cnn
        out = self.to_target(x) # reconstruct to target image size
        return out
    
    def forward(self, inputs):
        # _,_,H_ori,W_ori = x.size()
        # x = self.forward_size_norm(x)
        # B, C, H, W = x.size()
        input = self._transform_inputs(inputs)
        x, num_patches, shallow, e1_ = input[0], input[1], input[2], input[3]
        dec_out = self.forward_decoder(x, num_patches, (self.training_img_size[0],self.training_img_size[1]), e1_)
        dec_out += shallow # global skip connection
        out = self.forward_reconstruct(dec_out, (self.training_img_size[0],self.training_img_size[1]))
        
        # out = denormalize(out, self.mean, self.std) if self.img_norm else out
        out = out[:, :, :self.training_img_size[0]*self.scale, :self.training_img_size[1]*self.scale]
        return out
