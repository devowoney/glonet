import torch
from torch import nn
from functools import partial
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from NO import GF_Block, Fblock, Mlp, AFNO
from NN import *

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.lls = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.lls:
            y += layer(x)
        return y


class mspace(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, h, w, ker=[3, 5, 7, 11], groups=8):
        super(mspace, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid // 2, channel_hid, ker=ker, groups=groups)]
        for i in range(1, N_T - 1):
            enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, ker=ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid // 2, channel_hid, ker=ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid // 2, channel_hid, ker=ker, groups=groups)]
        for i in range(1, N_T - 1):
            dec_layers.append(
                Inception(2 * channel_hid, channel_hid // 2, channel_hid, ker=ker, groups=groups))
        dec_layers.append(Inception(2 * channel_hid, channel_hid // 2, channel_in, ker=ker, groups=groups))
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(channel_hid)

        self.enc = nn.Sequential(*enc_layers)
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]
        self.h = h
        self.w = w
        self.blocks = nn.ModuleList([Fblock(
            dim=channel_hid,
            mlp_ratio=4,
            drop=0.,
            drop_path=dpr[i],
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            h = self.h,
            w = self.w)
            for i in range(12)
        ])
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        bias = x
        x = x.reshape(B, T * C, H, W)

        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z)
            if i < self.N_T - 1:
                skips.append(z)
        
        B, D, H, W = z.shape
        N = H * W
        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(B, N, D)
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z).permute(0, 2, 1).contiguous()

        z = z.reshape(B, D, H, W)

        z = self.dec[0](z)
        for i in range(1, self.N_T):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        y = z.reshape(B, T, C, H, W)
        return y + bias

class tmp(nn.Module):
    def __init__(self, dim, ni=3, n_heads=8, patch_size=[8,8]):
        super(tmp, self).__init__()
        T, C, H, W = dim
        self.n_heads=n_heads
        self.NN = SS(in_channels=C, out_channels=C)
        self.NO = GF_Block(
            img_size=[H,W],
            patch_size=patch_size,
            in_channels=C,
            out_channels=C,
            input_frames=T,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            uniform_drop=False,
            drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=None,
            dropcls=0.,
            n_heads=self.n_heads
        )
        self.up = nn.ConvTranspose2d(C, C, kernel_size=3, stride=1, padding=1)
        self.down = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1)
        self.ni = ni

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        gf_features = self.NO(x_raw)
        lc_features = self.NN(x_raw)

        for _ in range(self.ni):
            gf_features_up = self.up(gf_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = gf_features_up + lc_features

            gf_features = self.NO(combined_features)
            lc_features = self.NN(combined_features)

            gf_features_down = self.down(gf_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = gf_features_down + lc_features

            gf_features = self.NO(combined_features)
            lc_features = self.NN(combined_features)

        return gf_features + lc_features




