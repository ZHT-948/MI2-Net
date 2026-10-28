import time
import math
from functools import partial
from typing import Optional, Callable
from networks.layers import general_conv3d_prenorm, fusion_prenorm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.amp import autocast


from einops import rearrange, repeat
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import factory_kwargs

from mamba_ssm import Mamba
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass


# an alternative for mamba_ssm (in which causal_conv1d is needed)

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass



basic_dims = 8

class GMamba(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba_D = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.mamba_H = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.mamba_W = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale_D = nn.Parameter(torch.ones(1))
        self.skip_scale_H = nn.Parameter(torch.ones(1))
        self.skip_scale_W = nn.Parameter(torch.ones(1))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.temb_proj = torch.nn.Linear(512,
                                         input_dim)
        self.fusion = general_conv3d_prenorm(input_dim, output_dim, pad_type='zeros')
        # self.fusion2 = general_conv3d_prenorm(input_dim, output_dim, pad_type='reflect')

    def forward(self, x):

        B, C, D, H, W = x.shape
        assert C == self.input_dim


        x_D = x.permute(0, 1, 2, 3, 4) # B, C, D, H, W
        img_dims_D = x_D.shape[2:]
        n_tokens_D = x_D.shape[2:].numel()
        x_flat_D = x_D.reshape(B, C, n_tokens_D).transpose(-1, -2)
        x_norm_D = self.norm(x_flat_D)
        x_mamba_D = self.mamba_D(x_norm_D) + x_flat_D * self.skip_scale_D
        x_mamba_D = self.norm(x_mamba_D)

        x_H = x.permute(0, 1, 3, 2, 4) # B, C, H, D, W
        img_dims_H = x_H.shape[2:]
        n_tokens_H = x_H.shape[2:].numel()
        x_flat_H = x_H.reshape(B, C, n_tokens_H).transpose(-1, -2)
        x_norm_H = self.norm(x_flat_H)
        x_mamba_H = self.mamba_H(x_norm_H) + x_flat_H * self.skip_scale_H
        x_mamba_H = self.norm(x_mamba_H)

        x_W = x.permute(0, 1, 4, 2, 3)  # B, C, W, D, H
        img_dims_W = x_W.shape[2:]
        n_tokens_W = x_W.shape[2:].numel()
        x_flat_W = x_W.reshape(B, C, n_tokens_W).transpose(-1, -2)
        x_norm_W = self.norm(x_flat_W)
        x_mamba_W = self.mamba_W(x_norm_W) + x_flat_W * self.skip_scale_W
        x_mamba_W = self.norm(x_mamba_W)

        x_mamba_D = self.proj(x_mamba_D)
        x_mamba_H = self.proj(x_mamba_H)
        x_mamba_W = self.proj(x_mamba_W)

        x_mamba_D = x_mamba_D.transpose(-1, -2).reshape(B, self.output_dim, *img_dims_D)

        x_mamba_H = x_mamba_H.transpose(-1, -2).reshape(B, self.output_dim, *img_dims_H)
        x_mamba_H_inverse = x_mamba_H.permute(0, 1, 3, 2, 4)

        x_mamba_W = x_mamba_W.transpose(-1, -2).reshape(B, self.output_dim, *img_dims_W)
        x_mamba_W_inverse = x_mamba_W.permute(0, 1, 3, 4, 2)

        out = self.fusion(x_mamba_D + x_mamba_H_inverse + x_mamba_W_inverse)
        out = out + x
        # out = self.fusion2(out)

        return out

class LMamba(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba_D = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.mamba_H = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.mamba_W = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale_D = nn.Parameter(torch.ones(1))
        self.skip_scale_H = nn.Parameter(torch.ones(1))
        self.skip_scale_W = nn.Parameter(torch.ones(1))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.temb_proj = torch.nn.Linear(512,
                                         input_dim)
        self.fusion = general_conv3d_prenorm(input_dim, output_dim, pad_type='zeros')
        # self.fusion2 = general_conv3d_prenorm(input_dim, output_dim, pad_type='reflect')

    def forward(self, x):

        B, C, D, H, W = x.shape
        assert C == self.input_dim


        x_D = x # B, C, D, H, W
        x_flat_D = x_D.permute(0, 2, 1, 3, 4).reshape(B * D, C, H * W).transpose(-1, -2)
        x_norm_D = self.norm(x_flat_D)
        x_mamba_D = self.mamba_D(x_norm_D) + x_flat_D * self.skip_scale_D
        x_mamba_D = self.norm(x_mamba_D)

        x_H = x.permute(0, 1, 3, 2, 4) # B, C, H, D, W
        x_flat_H = x_H.permute(0, 2, 1, 3, 4).reshape(B * H, C, D * W).transpose(-1, -2)
        x_norm_H = self.norm(x_flat_H)
        x_mamba_H = self.mamba_H(x_norm_H) + x_flat_H * self.skip_scale_H
        x_mamba_H = self.norm(x_mamba_H)

        x_W = x.permute(0, 1, 4, 2, 3)  # B, C, W, D, H
        x_flat_W = x_W.permute(0, 2, 1, 3, 4).reshape(B * W, C, D * H).transpose(-1, -2)
        x_norm_W = self.norm(x_flat_W)
        x_mamba_W = self.mamba_W(x_norm_W) + x_flat_W * self.skip_scale_W
        x_mamba_W = self.norm(x_mamba_W)

        x_mamba_D = self.proj(x_mamba_D)
        x_mamba_H = self.proj(x_mamba_H)
        x_mamba_W = self.proj(x_mamba_W)

        x_mamba_D = x_mamba_D.transpose(-1, -2).reshape(B, D, self.output_dim, H, W).permute(0, 2, 1, 3, 4)

        x_mamba_H = x_mamba_H.transpose(-1, -2).reshape(B, H, self.output_dim, D, W).permute(0, 2, 3, 1, 4)

        x_mamba_W = x_mamba_W.transpose(-1, -2).reshape(B, W, self.output_dim, D, H).permute(0, 2, 3, 4, 1)

        out = self.fusion(x_mamba_D + x_mamba_H + x_mamba_W)
        out = out + x
        # out = self.fusion2(out)

        return out

class ParallelGLMamba(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()

        self.gmamba = GMamba(dim, dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.lmamba = LMamba(dim, dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.fusion_conv = nn.Sequential(
            nn.Conv3d(dim * 2, dim, kernel_size=1),
            nn.InstanceNorm3d(dim),
            nn.SiLU()
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        x_g = self.gmamba(x)
        x_l = self.lmamba(x)

        w = torch.sigmoid(self.alpha)

        x_cat = torch.cat([w * x_g, (1 - w) * x_l], dim=1)

        x_fused = self.fusion_conv(x_cat)

        return x + x_fused


class ChannelMamba(nn.Module):

    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.channel_mamba = Mamba(
            d_model=2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.sigmoid = nn.Sigmoid()
        self.excite = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape

        y_avg = self.avg_pool(x).view(B, C, 1)
        y_max = self.max_pool(x).view(B, C, 1)

        y = torch.cat([y_avg, y_max], dim=2)

        y_mamba = self.channel_mamba(y)  # [B, C, 2]

        weights = y_mamba.mean(dim=2)  # [B, C]
        weights = self.sigmoid(weights).view(B, C, 1, 1, 1)

        return x * weights


class CGMamba(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.reduce = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.spatial_mamba = GMamba(out_channels, out_channels)

        self.channel_mamba_gate = ChannelMamba(out_channels)

        self.norm = nn.LayerNorm(out_channels)
        self.out_proj = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C_in, D, H, W] (Concatenated features)
        x = self.reduce(x)  # [B, C_out, D, H, W]

        x_spatial = self.spatial_mamba(x)


        x_channel = self.channel_mamba_gate(x)

        out = x + x_spatial + x_channel
        return self.out_proj(out)


class Mamba_Context_Encoder(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.proj = nn.Linear(dim, dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        B, C, D, H, W = x.shape

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # [B, L, C]
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + x_flat  # Residual
        x_out = self.proj(self.activation(x_mamba))

        # Reshape back
        x_out = x_out.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return x_out


class Mamba_Impute_Maskmodal(nn.Module):
    def __init__(self, dim, num_modalities=4):
        super(Mamba_Impute_Maskmodal, self).__init__()
        self.dim = dim
        self.num_modalities = num_modalities

        self.context_encoder = Mamba_Context_Encoder(dim)

        self.generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(dim, dim, kernel_size=1),
                nn.InstanceNorm3d(dim),
                nn.SiLU(),  # LeakyReLU -> SiLU
                nn.Conv3d(dim, dim, kernel_size=1),
                # nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])

        self.refine = nn.Sequential(
            general_conv3d_prenorm(dim, dim, pad_type='zeros'),
            nn.Conv3d(dim, dim, kernel_size=1)
        )

    def forward(self, x, mask):
        B, K, C, D, H, W = x.size()
        m = mask.view(B, K, 1, 1, 1, 1).type_as(x)

        sum_feats = (x * m).sum(dim=1)
        count_feats = m.sum(dim=1).view(B, 1, 1, 1, 1).clamp(min=1.0)
        avg_feats = sum_feats / count_feats

        context = self.context_encoder(avg_feats)

        hallucinated_list = []
        for k in range(K):
            delta = self.generators[k](context)
            h_feat = context * (1 + delta)
            h_feat = self.refine(h_feat)
            hallucinated_list.append(h_feat)

        hallucinated_stack = torch.stack(hallucinated_list, dim=1)

        out_stack = x * m + hallucinated_stack * (1 - m)
        # Flatten [B, K*C, D, H, W]
        out = out_stack.view(B, -1, D, H, W)

        return out, hallucinated_stack





class GLMamba_Encoder(nn.Module):
    def __init__(self):
        super(GLMamba_Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                               padding_mode='zeros', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='zeros')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='zeros')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type='zeros')
        self.e2_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='zeros')
        self.e2_c3 = ParallelGLMamba(dim=basic_dims * 2, d_state=basic_dims * 2)

        self.e3_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='zeros')
        self.e3_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='zeros')
        self.e3_c3 = ParallelGLMamba(dim=basic_dims * 4, d_state=basic_dims * 4)

        self.e4_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='zeros')
        self.e4_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='zeros')
        self.e4_c3 = ParallelGLMamba(dim=basic_dims * 8, d_state=basic_dims * 8)

        self.e5_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='zeros')
        self.e5_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='zeros')
        self.e5_c3 = ParallelGLMamba(dim=basic_dims * 16, d_state=basic_dims * 16)

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='zeros')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='zeros')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='zeros')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='zeros')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        # pred = self.softmax(logits)

        return logits

class GLMamba_Decoder_fuse1(nn.Module):
    def __init__(self, num_cls=4):
        super(GLMamba_Decoder_fuse1, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='zeros')

        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='zeros')

        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='zeros')

        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='zeros')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims*16, num_cls=3)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims*8, num_cls=3)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims*4, num_cls=3)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims*2, num_cls=3)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims*1, num_cls=3)


    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.seg_d4(de_x5)
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.seg_d3(de_x4)
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.seg_d2(de_x3)
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.seg_d1(de_x2)
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        # pred = self.softmax(logits)

        return logits, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))

class GLMamba_Decoder_fuse2(nn.Module):
    def __init__(self, num_cls=4):
        super(GLMamba_Decoder_fuse2, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='zeros')

        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='zeros')

        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='zeros')

        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='zeros')


        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear')
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear')

        self.RFM5 = fusion_prenorm(in_channel=basic_dims*16, num_cls=3)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims*8, num_cls=3)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims*4, num_cls=3)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims*2, num_cls=3)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims*1, num_cls=3)


    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.seg_d4(de_x5)
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.seg_d3(de_x4)
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.seg_d2(de_x3)
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.seg_d1(de_x2)
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        # pred = self.softmax(logits)

        return logits, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))


class MI2Net_MyoPS(nn.Module):
    def __init__(self, num_cls=4):
        super(MI2Net_MyoPS, self).__init__()

        self.share_encoder = GLMamba_Encoder()


        self.sep_decoder = Decoder_sep(num_cls=num_cls)

        self.fuse_GMamba1 = GMamba(input_dim=basic_dims * 48, output_dim=basic_dims * 48)
        self.fuse_LMamba1 = LMamba(input_dim=basic_dims * 48, output_dim=basic_dims * 48)

        self.fuse_GMamba2 = GMamba(input_dim=basic_dims * 48, output_dim=basic_dims * 48)
        self.fuse_LMamba2 = LMamba(input_dim=basic_dims * 48, output_dim=basic_dims * 48)


        self.fusion = CGMamba(in_channels=basic_dims * 96, out_channels=basic_dims * 48)


        self.GLMamba_fuse_decoder1 = GLMamba_Decoder_fuse1(num_cls=num_cls)
        self.GLMamba_fuse_decoder2 = GLMamba_Decoder_fuse2(num_cls=num_cls)

        self.masker_x1 = Mamba_Impute_Maskmodal(dim=basic_dims)
        self.masker_x2 = Mamba_Impute_Maskmodal(dim=basic_dims * 2)
        self.masker_x3 = Mamba_Impute_Maskmodal(dim=basic_dims * 4)
        self.masker_x4 = Mamba_Impute_Maskmodal(dim=basic_dims * 8)
        self.masker_x5 = Mamba_Impute_Maskmodal(dim=basic_dims * 16)
        self.is_training = False

        self.dropout = nn.Dropout3d(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x, mask):

        bssFP_x1, bssFP_x2, bssFP_x3, bssFP_x4, bssFP_x5 = self.share_encoder(x[:, 0:1, :, :, :])
        LGE_x1, LGE_x2, LGE_x3, LGE_x4, LGE_x5 = self.share_encoder(x[:, 1:2, :, :, :])
        T2_x1, T2_x2, T2_x3, T2_x4, T2_x5 = self.share_encoder(x[:, 2:3, :, :, :])
        # t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.share_encoder(x[:, 3:4, :, :, :])



        if self.is_training:
            bssFP_pred = self.sep_decoder(bssFP_x1, bssFP_x2, bssFP_x3, bssFP_x4, bssFP_x5)

            LGE_pred = self.sep_decoder(LGE_x1, LGE_x2, LGE_x3, LGE_x4, LGE_x5)

            T2_pred = self.sep_decoder(T2_x1, T2_x2, T2_x3, T2_x4, T2_x5)

            # t2_pred = self.GLMamba_sep_decoder(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)

        stack_x3 = torch.stack((bssFP_x3, LGE_x3, T2_x3), dim=1)
        stack_x4 = torch.stack((bssFP_x4, LGE_x4, T2_x4), dim=1)
        stack_x5 = torch.stack((bssFP_x5, LGE_x5, T2_x5), dim=1)

        x1, rec_1 = self.masker_x1(torch.stack((bssFP_x1, LGE_x1, T2_x1), dim=1), mask)
        x2, rec_2 = self.masker_x2(torch.stack((bssFP_x2, LGE_x2, T2_x2), dim=1), mask)
        x3, rec_3 = self.masker_x3(torch.stack((bssFP_x3, LGE_x3, T2_x3), dim=1), mask)
        x4, rec_4 = self.masker_x4(torch.stack((bssFP_x4, LGE_x4, T2_x4), dim=1), mask)
        x5, rec_5 = self.masker_x5(torch.stack((bssFP_x5, LGE_x5, T2_x5), dim=1), mask)


        fused_Gmamba = self.fuse_GMamba1(x5)
        fused_GLmamba = self.fuse_LMamba1(fused_Gmamba)

        fused_Lmamba = self.fuse_LMamba2(x5)
        fused_LGmamba = self.fuse_GMamba2(fused_Lmamba)

        fuse = self.fusion(torch.cat([fused_GLmamba, fused_LGmamba], dim=1)) + x5

        # fused_mamba_post = fused_Lmamba + fused_Gmamba

        if self.is_training:
            fuse_1 = self.dropout(fuse)
            fuse_2 = self.dropout(fuse)
        else:
            fuse_1 = fuse
            fuse_2 = fuse

        fuse_pred1, preds1 = self.GLMamba_fuse_decoder1(x1, x2, x3, x4, fuse_1)
        fuse_pred2, preds2 = self.GLMamba_fuse_decoder2(x1, x2, x3, x4, fuse_2)



        if self.is_training:
            rec_features_list = [rec_3, rec_4, rec_5]
            gt_features_list = [stack_x3, stack_x4, stack_x5]

            return (fuse_pred1, fuse_pred2), (bssFP_pred, LGE_pred, T2_pred),  (preds1, preds2), (
                rec_features_list, gt_features_list)
        return fuse_pred1, fuse_pred2

