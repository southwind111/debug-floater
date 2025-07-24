#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss_origin(network_output, gt, mask):
    return torch.abs((network_output - gt)).mean()

def l1_loss(network_output, gt, mask):
    if mask is not None:
        # network_output *= mask #! 这里可以保存一下掩码查看一下质量，可能是掩码边界质量不行导致的噪声
        # gt *= mask
        l1 = torch.abs((network_output - gt))
        masked_l1 = l1 * mask
        valid_pixels = mask.sum()
        if valid_pixels == 0:
            return torch.tensor(0.0, device=l1.device)
        
        loss = masked_l1.sum() / valid_pixels / 3
    else:
        loss = torch.abs((network_output - gt)).mean()
        
    return loss

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, target_mask, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # return _ssim(img1, img2, target_mask, window, window_size, channel, size_average)
    return _ssim_origin(img1, img2, target_mask, window, window_size, channel, size_average)

def _ssim_origin(img1, img2, target_mask, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

#! render、gt、mask
def _ssim(img1, img2, target_mask, window, window_size, channel, size_average=True):
    # if target_mask is not None:
    #     img1 = img1.unsqueeze(0) * target_mask
    #     img2 = img2.unsqueeze(0) * target_mask
    # else:
    #     img1 = img1.unsqueeze(0)
    #     img2 = img2.unsqueeze(0)
    
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)
        
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq)
    simga2_sq = (F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq)
    sigma12 = (F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)- mu1_mu2)

    C1 = 0.01**2
    C2 = 0.03**2

    #! 每个像素每个通道都有一个ssim值，ssim_map是一个四维张量
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + simga2_sq + C2))

    ssim_map = ssim_map.squeeze(0)  #! 去掉批次维度

    if target_mask is not None:
        masked_ssim = ssim_map * target_mask
        valid_pixels = target_mask.sum()
        if valid_pixels == 0:
            return torch.tensor(0.0, device=img1.device)
        ssim_value = masked_ssim.sum() / valid_pixels / 3
    else:
        ssim_value = ssim_map.mean()

    return ssim_value


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
