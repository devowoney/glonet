import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from blocks import *
from NN import *
class Glonet(nn.Module):
    def __init__(self, dim, dT=64, dS=64, NT=2, NS=8, ker=[3,5,7], groups=8):
        super(Glonet, self).__init__()
        T, C, H, W = dim; self.Hn = int(H / 2 ** (NT / 2)); self.Wn = int(W / 2 ** (NT / 2))
        self.space = mspace(T*dT, dS, NS, self.Hn, self.Wn, ker, groups) # .to('cuda:0')
        self.dynamics = tmp(dim=dim, n_heads=4, patch_size=[16,16]) # .to('cuda:0')
        self.maps = Encoder(C, dT, NT) # .to('cuda:0')
        self.mapsback = Decoder(dT, C, NT) # .to('cuda:0')
    def forward(self, x):
        B, T, C, H, W = x.shape
        # x = x.to('cuda:0')
        s_f = self.dynamics(x)
        s_e, s_k_f = self.maps(s_f.view(-1, C, H, W))
        t_e = self.space(s_e.view(B, T, *s_e.shape[1:]))
        forecast = self.mapsback(t_e.view(B*T, *t_e.shape[2:]), s_k_f)
        return forecast.view(B, T, C, H, W)[:, 0]
