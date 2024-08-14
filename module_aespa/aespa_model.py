import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf_f
import random
import torch.nn.functional as F

from .aespanet_models import (
    AdaptiveMultiAttn_Transformer_v2,
    VGGDecoder,
    VGGEncoder,
)


def inference_aespa(
    src_img: torch.Tensor,
    style_img: torch.Tensor,
    device,
    size: int,
    do_crop: bool,
    encoder: VGGEncoder,
    decoder: VGGDecoder,
    transformer: AdaptiveMultiAttn_Transformer_v2,
):
    content_tf = test_transform(
        size, do_crop, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    content = content_tf(src_img).to(device)  # [B, C, H, W]
    style = content_tf(style_img).to(device)

    gray_content = tf_f.rgb_to_grayscale(content).repeat(1, 3, 1, 1).to(device)
    gray_style = tf_f.rgb_to_grayscale(style).repeat(1, 3, 1, 1).to(device)

    style_weight = (
        adaptive_gram_weight(style, 1, 8, encoder)
        + adaptive_gram_weight(style, 2, 8, encoder)
        + adaptive_gram_weight(style, 3, 8, encoder)
    ) / 3

    gray_style_weight = (
        adaptive_gram_weight(gray_style, 1, 8, encoder)
        + adaptive_gram_weight(gray_style, 2, 8, encoder)
        + adaptive_gram_weight(gray_style, 3, 8, encoder)
    ) / 3

    style_adaptive_alpha = (
        style_weight.unsqueeze(1).cuda() + gray_style_weight.unsqueeze(1).cuda()
    ) / 2

    content_skips = {}
    style_skips = {}

    _ = encoder.encode(content, content_skips) # content_feat
    _ = encoder.encode(style, style_skips) # style_feat

    if gray_content is not None:
        gray_content_skips = {}
        _ = encoder.encode(gray_content, gray_content_skips) # gray_content_feat
    if gray_style is not None:
        gray_style_skips = {}
        _ = encoder.encode(gray_style, gray_style_skips) # gray_style_feat

    (
        local_transformed_feature,
        _,
        _,
        _,
        _,
    ) = transformer.forward(
        content_skips["conv4_1"],
        style_skips["conv4_1"],
        content_skips["conv5_1"],
        style_skips["conv5_1"],
        adaptive_get_keys(content_skips, 4, 4, target_feat=content_skips["conv4_1"]),
        adaptive_get_keys(style_skips, 1, 4, target_feat=style_skips["conv4_1"]),
        adaptive_get_keys(content_skips, 5, 5, target_feat=content_skips["conv5_1"]),
        adaptive_get_keys(style_skips, 1, 5, target_feat=style_skips["conv5_1"]),
    )

    if gray_content is not None:
        global_transformed_feat = feature_wct_simple(
            gray_content_skips["conv4_1"], gray_style_skips["conv4_1"]
        )
    else:
        global_transformed_feat = feature_wct_simple(
            content_skips["conv4_1"], style_skips["conv4_1"]
        )

    transformed_feature = (
        global_transformed_feat * (1 - style_adaptive_alpha.unsqueeze(-1).unsqueeze(-1))
        + style_adaptive_alpha.unsqueeze(-1).unsqueeze(-1) * local_transformed_feature
    )

    stylized_image = decoder.decode(transformed_feature, content_skips, style_skips)

    output = un_normalize_batch(stylized_image)

    return output


def adaptive_gram_weight(image, level, ratio, encoder):
    if level == 0:
        encoded_features = image
    else:
        encoded_features = encoder.get_features(image, level)  # B x C x W x H
    global_gram = gram_matrix(encoded_features)

    B, C, w, h = encoded_features.size()
    target_w, target_h = w // ratio, h // ratio
    
    patches = extract_image_patches(encoded_features, target_w, target_h)
    _, patches_num, _, _, _ = patches.size()
    cos = torch.nn.CosineSimilarity(eps=1e-6)

    intra_gram_statistic = []
    inter_gram_statistic = []
    comb = torch.combinations(torch.arange(patches_num), r=2)
    if patches_num >= 10:
        sampling_num = int(comb.size(0) * 0.05)
    else:
        sampling_num = comb.size(0)
    for idx in range(B):
        if patches_num < 2:
            continue
        cos_gram = []

        for patch in range(0, patches_num):
            cos_gram.append(
                cos(global_gram, gram_matrix(patches[idx][patch].unsqueeze(0)))
                .mean()
                .item()
            )

        intra_gram_statistic.append(torch.tensor(cos_gram))

        cos_gram = []
        for idxes in random.choices(list(comb), k=sampling_num):
            cos_gram.append(
                cos(
                    gram_matrix(patches[idx][idxes[0]].unsqueeze(0)),
                    gram_matrix(patches[idx][idxes[1]].unsqueeze(0)),
                )
                .mean()
                .item()
            )

        inter_gram_statistic.append(torch.tensor(cos_gram))

    intra_gram_statistic = torch.stack(intra_gram_statistic).mean(dim=1)
    inter_gram_statistic = torch.stack(inter_gram_statistic).mean(dim=1)
    results = (intra_gram_statistic + inter_gram_statistic) / 2

    ##For boosting value
    results = 1 / (1 + torch.exp(-10 * (results - 0.6)))

    return results


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def extract_image_patches(x, kernel, stride=1):
    b, c, h, w = x.shape

    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.contiguous().view(b, c, -1, kernel, kernel)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()

    # return patches.view(b, number_of_patches, c, h, w)
    return patches.view(b, -1, c, kernel, kernel)


def adaptive_get_keys(feat_skips, start_layer_idx, last_layer_idx, target_feat):
    B, C, th, tw = target_feat.shape
    results = []
    target_conv_layer = "conv" + str(last_layer_idx) + "_1"
    _, _, h, w = feat_skips[target_conv_layer].shape
    for i in range(start_layer_idx, last_layer_idx + 1):
        target_conv_layer = "conv" + str(i) + "_1"
        if i == last_layer_idx:
            results.append(mean_variance_norm(feat_skips[target_conv_layer]))
        else:
            results.append(
                mean_variance_norm(F.interpolate(feat_skips[target_conv_layer], (h, w)))
            )

    return F.interpolate(torch.cat(results, dim=1), (th, tw))


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def feature_wct_simple(content_feat, style_feat, alpha=1):
    target_feature = Bw_wct_core(content_feat, style_feat)

    target_feature = target_feature.view_as(content_feat)
    target_feature = alpha * target_feature + (1 - alpha) * content_feat
    return target_feature


def Bw_wct_core(content_feat, style_feat, weight=1, registers=None, device="cpu"):
    N, C, H, W = content_feat.size()
    cont_min = content_feat.min().item()
    cont_max = content_feat.max().item()

    whiten_cF, _, _ = SwitchWhiten2d(content_feat)
    _, wm_s, s_mean = SwitchWhiten2d(style_feat)

    targetFeature = torch.bmm(torch.inverse(wm_s), whiten_cF)
    targetFeature = targetFeature.view(N, C, H, W)
    targetFeature = targetFeature + s_mean.unsqueeze(2).expand_as(targetFeature)
    targetFeature.clamp_(cont_min, cont_max)

    return targetFeature


def SwitchWhiten2d(x):
    N, C, H, W = x.size()

    in_data = x.view(N, C, -1)

    eye = in_data.data.new().resize_(C, C)
    eye = torch.nn.init.eye_(eye).view(1, C, C).expand(N, C, C)

    # calculate other statistics
    mean_in = in_data.mean(-1, keepdim=True)
    x_in = in_data - mean_in
    # (N x g) x C x C
    cov_in = torch.bmm(x_in, torch.transpose(x_in, 1, 2)).div(H * W)

    mean = mean_in
    cov = cov_in + 1e-5 * eye

    # perform whitening using Newton's iteration
    Ng, c, _ = cov.size()
    P = torch.eye(c).to(cov).expand(Ng, c, c)

    rTr = (cov * P).sum((1, 2), keepdim=True).reciprocal_()
    cov_N = cov * rTr
    for k in range(5):
        P = torch.baddbmm(1.5, P, -0.5, torch.matrix_power(P, 3), cov_N)

    wm = P.mul_(rTr.sqrt())
    x_hat = torch.bmm(wm, in_data - mean)

    return x_hat, wm, mean


def test_transform(size, crop, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    transform_list = []

    if size != 0:
        # Ensure size is even
        size = (size // 2) * 2

        # Limit size to 1024
        if size > 1024:
            size = 1024

        if crop:
            transform_list.extend(
                [transforms.Resize(size), transforms.CenterCrop(size)]
            )
        else:
            transform_list.append(transforms.Resize((size, size)))

    transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)


def un_normalize_batch(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
