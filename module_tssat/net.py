# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:35:21 2020

@author: ZJU
"""

import torch
import os
import torch.nn as nn
import torch.nn.functional as F

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def TSSAT(cf, sf, patch_size=5, stride=1):  # cf,sf  Batch_size x C x H x W
    b, c, h, w = sf.size()  # 2 x 256 x 64 x 64
    kh, kw = patch_size, patch_size
    sh, sw = stride, stride

    # Create convolutional filters by style features
    sf_unfold = sf.unfold(2, kh, sh).unfold(3, kw, sw)
    patches = sf_unfold.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(b, -1, c, kh, kw)
    patches_norm = torch.norm(patches.reshape(*patches.shape[:2], -1), dim=2).reshape(b, -1, 1, 1, 1)
    patches_norm = patches / patches_norm
    # patches size is 2 x 3844 x 256 x 3 x 3

    cf = adaptive_instance_normalization(cf, sf)
    for i in range(b):
        cf_temp = cf[i].unsqueeze(0)  # [1 x 256 x 64 x 64]
        patches_norm_temp = patches_norm[i]  # [3844, 256, 3, 3]
        patches_temp = patches[i]

        _, _, ch, cw = cf.size()
        for c_i in range(0, ch, patch_size):
            ###################################################
            if (c_i + patch_size) > ch:
                break
            elif (c_i + 2*patch_size) > ch:
                ckh = ch - c_i
            else:
                ckh = patch_size
            ###################################################

            for c_j in range(0, cw, patch_size):
                ###################################################
                if (c_j + patch_size) > cw:
                    break
                elif (c_j + 2 * patch_size) > cw:
                    ckw = cw - c_j
                else:
                    ckw = patch_size
                ###################################################

                temp = cf_temp[:, :, c_i:c_i + ckh, c_j:c_j + ckw]
                conv_out = F.conv2d(temp, patches_norm_temp, stride=patch_size)
                index = conv_out.argmax(dim=1).squeeze()
                style_temp = patches_temp[index].unsqueeze(0)
                stylized_part = adaptive_instance_normalization(temp, style_temp)

                if c_j == 0:
                    p = stylized_part
                else:
                    p = torch.cat([p, stylized_part], 3)

            if c_i == 0:
                q = p
            else:
                q = torch.cat([q, p], 2)

        if i == 0:
            out = q
        else:
            out = torch.cat([out, q], 0)

    return out


def exponent(input, device):
    b, h, w = input.size()
    x = (torch.ones(b, h, w) + 0.02).to(device)
    y = torch.pow(x, input)
    out = y / torch.sum(y, dim=-1, keepdim=True)
    return out



class SANet(nn.Module):
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes = in_planes)
        self.sanet5_1 = SANet(in_planes = in_planes)
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1, content5_1, style5_1):
        self.upsample5_1 = nn.Upsample(size=(content4_1.size()[2], content4_1.size()[3]), mode='nearest')
        return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))

class Net(nn.Module):
    def __init__(self, encoder, decoder, start_iter):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        #transform
        self.transform = Transform(in_planes = 512)
        self.decoder = decoder
        self.sm = nn.Softmax(dim=-1)

        if(start_iter > 0):
            self.transform.load_state_dict(torch.load('transformer_iter_' + str(start_iter) + '.pth'))
            self.decoder.load_state_dict(torch.load('decoder_iter_' + str(start_iter) + '.pth'))
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def attention_map(self, input1, input2, eps=1e-5):
        F = mean_variance_norm(input1)
        G = mean_variance_norm(input2)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)

        n = h * w
        T = torch.zeros((b, n, n - 1))
        # REMOVE DIAGONAL
        for i in range(b):
            T[i] = S[i].flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
        T = self.sm(T / 100)

        return T

    def attention_loss(self, content, stylization):
        attention_map1 = self.attention_map(content, content)
        attention_map2 = self.attention_map(stylization, stylization)
        return self.mse_loss(attention_map1, attention_map2)

    def calc_content_loss(self, input, target, norm = False):
        if not norm:
          return self.mse_loss(input, target)
        else:
          return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_local_style_loss(self, cf, sf, patch_size=5, stride=1):
        b, c, h, w = sf.size()  # 2 x 256 x 64 x 64
        kh, kw = patch_size, patch_size
        sh, sw = stride, stride

        # Create convolutional filters by style features
        sf_unfold = sf.unfold(2, kh, sh).unfold(3, kw, sw)
        patches = sf_unfold.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(b, -1, c, kh, kw)
        patches_norm = torch.norm(patches.reshape(*patches.shape[:2], -1), dim=2).reshape(b, -1, 1, 1, 1)
        patches_norm = patches / patches_norm
        # patches size is 2 x 3844 x 256 x 3 x 3

        loss = 0
        for i in range(b):
            cf_temp = cf[i].unsqueeze(0)  # [1 x 256 x 64 x 64]
            patches_norm_temp = patches_norm[i]  # [3844, 256, 3, 3]
            patches_temp = patches[i]

            _, _, ch, cw = cf.size()
            for c_i in range(0, ch, patch_size):
                ###################################################
                if (c_i + patch_size) > ch:
                    break
                elif (c_i + 2 * patch_size) > ch:
                    ckh = ch - c_i
                else:
                    ckh = patch_size
                ###################################################

                for c_j in range(0, cw, patch_size):
                    ###################################################
                    if (c_j + patch_size) > cw:
                        break
                    elif (c_j + 2 * patch_size) > cw:
                        ckw = cw - c_j
                    else:
                        ckw = patch_size
                    ###################################################

                    temp = cf_temp[:, :, c_i:c_i + ckh, c_j:c_j + ckw]
                    conv_out = F.conv2d(temp, patches_norm_temp, stride=patch_size)
                    index = conv_out.argmax(dim=1).squeeze()
                    style_temp = patches_temp[index].unsqueeze(0)

                    input_mean, input_std = calc_mean_std(temp)
                    target_mean, target_std = calc_mean_std(style_temp)
                    loss += self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

        return loss
    
    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)

        stylized = TSSAT(content_feats[3], style_feats[3])
        g_t = self.decoder(stylized)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[3], content_feats[3]) + self.calc_content_loss(g_t_feats[4], content_feats[4])   # True; True
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        """LOCAL STYLE LOSSES"""
        loss_s_local = self.calc_local_style_loss(g_t_feats[3], style_feats[3])

        """ATTENTION LOSSES"""
        loss_attention1 = self.attention_loss(content_feats[3], g_t_feats[3])
        loss_attention2 = self.attention_loss(content_feats[4], g_t_feats[4])
        loss_attention = loss_attention1 + loss_attention2

        """IDENTITY LOSSES"""
        Icc = self.decoder(TSSAT(content_feats[3], content_feats[3]))
        Iss = self.decoder(TSSAT(style_feats[3], style_feats[3]))
        l_identity1 = self.calc_content_loss(Icc, content) + self.calc_content_loss(Iss, style)
        Fcc = self.encode_with_intermediate(Icc)
        Fss = self.encode_with_intermediate(Iss)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 5):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])

        return g_t, loss_c, loss_s, loss_s_local, loss_attention, l_identity1, l_identity2