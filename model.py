
import time
import pywt
import cv2
import torch
import pywt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numbers

from torch.autograd import Function
from pdb import set_trace as stx
from functools import partial

from einops import rearrange


class DWTFunction(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.float()
        w_ll = w_ll.float()
        w_lh = w_lh.float()
        w_hl = w_hl.float()
        w_hh = w_hh.float()
        
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]

        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            b, c, h, w = ctx.shape
            dx = dx.view(b, 4, -1, h//2, w//2)

            dx = dx.transpose(1, 2).reshape(b, -1, h//2, w//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(c, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=c)

        return dx, None, None, None, None


class IDWTFunction(Function):
    @staticmethod
    def forward(ctx, x, filters):
        x = x.float()
        filters = filters.float()
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        b, _, h, w = x.shape
        x = x.view(b, 4, -1, h, w).transpose(1, 2)
        c = x.shape[1]
        x = x.reshape(b, -1, h, w)
        filters = filters.repeat(c, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=c)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            _, c, _, _ = ctx.shape
            c = c // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(c, -1, -1, -1), stride=2, groups=c)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT2D(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer("filters", filters)
        self.filters = self.filters.to(dtype=torch.float16)

    def forward(self, x):
        x = x.float()
        return IDWTFunction.apply(x, self.filters)


class DWT2D(nn.Module):
    def __init__(self, wave):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer("w_ll", w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_lh", w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hl", w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer("w_hh", w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float16)
        self.w_lh = self.w_lh.to(dtype=torch.float16)
        self.w_hl = self.w_hl.to(dtype=torch.float16)
        self.w_hh = self.w_hh.to(dtype=torch.float16)

    def forward(self, x):
        x = x.float()
        return DWTFunction.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, group, num_hw):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.num_hw = num_hw // 2
        self.temperature0 = nn.Parameter(torch.ones(dim, self.num_hw, self.num_hw))
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature3 = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim * 3, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_g = nn.Parameter(torch.randn(dim, self.num_hw, 1))
        self.k_g = nn.Parameter(torch.randn(dim, self.num_hw, 1))
        self.dwt = DWT2D(wave="haar")
        self.i_dwt = IDWT2D(wave="haar")

    def forward(self, x):
        b, c, h, w = x.shape
        
        x = self.dwt(x)
        l, q, k, v = x.chunk(4, dim=1)
        qkv = torch.cat([q, k, v], 1)
        b, c, h, w = l.shape
        qkv = self.qkv(qkv)
        qkv = self.qkv_dwconv(qkv)
        q0, k0, v0 = qkv.chunk(3, dim=1)
        
        k = k0.transpose(-2, -1) @ self.q_g
        q = q0 @ self.k_g
        
        att = k @ q.transpose(-2, -1) * self.temperature0
        att = att.softmax(dim=-1)

        att = rearrange(att, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v0, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        l = rearrange(l, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        att = torch.nn.functional.normalize(att, dim=-1)
        v = torch.nn.functional.normalize(v, dim=-1)
        l = torch.nn.functional.normalize(l, dim=-1)

        attn = (att @ v.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)

        v = (attn @ v)   
        attn2 = v * self.temperature2
        attn2 = attn2.softmax(dim=-1)

        attn3 = (attn2 @ l.transpose(-2, -1)) * self.temperature3
        attn3 = attn3.softmax(dim=-1)

        out = attn3 @ l

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        v = rearrange(v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        
        out = torch.cat((out, q0, k0, v), 1)
        
        out = self.i_dwt(out)

        out = self.project_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, group, num_hw):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, group, num_hw)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class AFF(nn.Module):
    def __init__(self, out_channel, bias):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, kernel_size=1, bias=bias),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class ChannelRepair(nn.Module):
    def __init__(self, in_channels, out_channels, group):
        super(ChannelRepair, self).__init__()
        self.group = group
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels * group,
                      kernel_size=3, stride=1, padding=1, groups=group),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * group, out_channels=out_channels * group,
                      kernel_size=3, stride=1, padding=1, groups=group),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * group, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        res = self.conv4(x)
        out = res + out
        return out

##---------- Restormer -----------------------
class DAF(nn.Module):
    def __init__(self,
                 inp_channels=4,
                 out_channels=4,
                 dim=32,
                 num_blocks=[2, 2, 2, 2],
                 num_hw=[256, 128, 64, 32],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 heads_=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'
                 ):
        super(DAF, self).__init__()

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type, group=3, num_hw=num_hw[0]) for i in range(num_blocks[0])])

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, group=3, num_hw=num_hw[1]) for i in range(num_blocks[1])])

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 16), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, group=3, num_hw=num_hw[2]) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 16), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, group=3, num_hw=num_hw[2]) for i in range(num_blocks[2])])

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 4), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, group=3, num_hw=num_hw[1]) for i in range(num_blocks[1])])

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, group=3, num_hw=num_hw[0]) for i in range(num_blocks[0])])
        self.refinement1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, group=1, num_hw=num_hw[0]) for i in range(num_refinement_blocks)])
        self.refinement2 = nn.Sequential(*[
            TransformerBlock(dim=int(out_channels) * 3, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, group=1, num_hw=num_hw[0]) for i in range(num_refinement_blocks)])
        self.aff = nn.ModuleList([
            AFF(32, bias),
            AFF(128, bias)
        ])
        self.output = nn.ModuleList([
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Conv2d(int(dim), out_channels * 3, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.Conv2d(int(out_channels), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        ])
        self.cr = nn.ModuleList([
            ChannelRepair(12, 32, 3),
        ])
        self.dwt = DWT2D(wave="haar")
        self.i_dwt = IDWT2D(wave="haar")

    def forward(self, cloud1, cloud2, cloud3):    
        l = torch.cat((cloud1, cloud2, cloud3), 1)
        l0 = self.cr[0](l)
        enc1 = self.encoder_level1(l0)   

        enc2 = self.dwt(enc1)       
        enc2 = self.encoder_level2(enc2)   
        
        enc3 = self.dwt(enc2)  
        
        enc3 = self.encoder_level3(enc3) 
        LL, LH, HL, HH = enc3.chunk(4, dim=1)
        enc3 = torch.cat([HH, HL, LH, LL], 1)
        dec3 = self.decoder_level3(enc3) 

        HH, HL, LH, LL = dec3.chunk(4, dim=1)
        enc3 = torch.cat([LL, LH, HL, HH], 1)
        dec2 = self.i_dwt(dec3)   
        dec2_skip = self.aff[1](dec2, enc2)
        dec2 = self.decoder_level2(dec2_skip) 

        dec1 = self.i_dwt(dec2)    
        dec1_skip = self.aff[0](dec1, enc1)  
        dec1 = self.decoder_level1(dec1_skip)   
        
        ref = self.refinement1(dec1)     
        ref = self.output[1](ref) + l    
        
        ref = self.refinement2(ref)     
        output = self.output[0](ref)       
    

        return output
