import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from comfy.utils import ProgressBar
from torchvision import transforms

from . import net


class MODEL_TSSAT:
    def __init__(self, dec_path: str, vgg_path: str, device) -> None:
        self.decoder = net.decoder
        vgg = net.vgg

        self.decoder.load_state_dict(torch.load(dec_path))
        vgg.load_state_dict(torch.load(vgg_path))

        self.enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1

        self.enc_1.eval().to(device)
        self.enc_2.eval().to(device)
        self.enc_3.eval().to(device)
        self.enc_4.eval().to(device)
        self.decoder.eval().to(device)

    def encode(self, item):
        out = self.enc_1(item)
        out = self.enc_2(out)
        out = self.enc_3(out)
        out = self.enc_4(out)
        return out


def inference_tssat(
    src_img: torch.Tensor,
    style_img: torch.Tensor,
    device,
    size: int,
    do_crop: False,
    max_steps: int,
    model: MODEL_TSSAT,
):
    content_tf = test_transform(size, do_crop)
    content = content_tf(src_img).to(device)  # [1, C, H, W]
    style = content_tf(style_img).to(device)  # [1, C, H, W]

    for _ in range(max_steps):
        Content4_1 = model.encode(content)
        Style4_1 = model.encode(style)
        content = model.decoder(TSSAT(Content4_1, Style4_1))

    return content


def forward(item, enc_1, enc_2, enc_3, enc_4):
    out = enc_1(item)
    out = enc_2(out)
    out = enc_3(out)
    out = enc_4(out)
    return out


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform = transforms.Compose(transform_list)
    # [B, C, H, W]
    return transform


def TSSAT(cf, sf, patch_size=5, stride=1):  # cf,sf  Batch_size x C x H x W
    b, c, h, w = sf.size()  # 2 x 256 x 64 x 64
    # print(cf.size())
    kh, kw = patch_size, patch_size
    sh, sw = stride, stride

    # Create convolutional filters by style features
    sf_unfold = sf.unfold(2, kh, sh).unfold(3, kw, sw)
    patches = sf_unfold.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(b, -1, c, kh, kw)
    patches_norm = torch.norm(patches.reshape(*patches.shape[:2], -1), dim=2).reshape(
        b, -1, 1, 1, 1
    )
    patches_norm = patches / patches_norm
    # patches size is 2 x 3844 x 256 x 3 x 3

    cf = adaptive_instance_normalization(cf, sf)

    for i in range(b):
        cf_temp = cf[i].unsqueeze(0)  # [1 x 256 x 64 x 64]
        patches_norm_temp = patches_norm[i]  # [3844, 256, 3, 3]
        patches_temp = patches[i]

        _, _, ch, cw = cf.size()
        pbar = ProgressBar(ch)
        for c_i in tqdm.tqdm(range(0, ch, patch_size), desc="Optimizing"):
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

                temp = cf_temp[:, :, c_i : c_i + ckh, c_j : c_j + ckw]
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

            pbar.update_absolute(c_i, ch)

        if i == 0:
            out = q
        else:
            out = torch.cat([out, q], 0)

    return out


def adaptive_instance_normalization(content_feat, style_feat):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.contiguous().view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
