import torch
import torch.nn.functional as F
from torchvision import transforms

from . import net
from .function import (
    adaptive_instance_normalization,
    adaptive_mean_normalization,
    adaptive_std_normalization,
    coral,
    exact_feature_distribution_matching,
    histogram_matching,
)


def inference_efdm(
    src_img: torch.Tensor,
    style_img: torch.Tensor,
    device,
    decoder_path: str,
    vgg_path: str,
    alpha: float,
    size: int,
    style_type: str,
    do_crop=False,
    preserve_color=False,
    style_interpolation_weights: list[int] | None = None,
):
    if style_img.shape[0] > 1:
        if style_interpolation_weights is None:
            style_interpolation_weights = [
                1 / style_img.shape[0] for _ in range(style_img.shape[0])
            ]
        else:
            style_interpolation_weights = [
                w / sum(style_interpolation_weights)
                for w in style_interpolation_weights
            ]
    else:
        style_interpolation_weights = None

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(decoder_path))
    vgg.load_state_dict(torch.load(vgg_path))

    vgg = torch.nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(size, do_crop)

    content = content_tf(src_img).to(device)
    style = content_tf(style_img).to(device)
    style = F.interpolate(
        style,
        size=(content.shape[2], content.shape[3]),
        mode="bilinear",
        align_corners=False,
    )
    content = content.expand_as(style)

    print(f"Resized: {content.shape=}")
    print(f"Resized: {style.shape=}")

    if preserve_color:
        tmp_content = content.squeeze()
        tmp_styles = [coral(stl, tmp_content).unsqueeze(0) for stl in style]
        style = torch.cat(tmp_styles, dim=0)
        style = style.to(device)
    
    if device.type != "cpu":
        torch.cuda.empty_cache()

    output = style_transfer(
        vgg,
        decoder,
        content,
        style,
        device,
        alpha,
        style_interpolation_weights,
        style_type=style_type,
    )
    
    if device.type != "cpu":
        torch.cuda.empty_cache()

    # [N, C, H, W]
    return output


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform = transforms.Compose(transform_list)
    # [B, C, H, W]
    return transform


def style_transfer(
    vgg,
    decoder,
    content,
    style,
    device,
    alpha=1.0,
    interpolation_weights=None,
    style_type="adain",
):
    assert 0.0 <= alpha <= 1.0
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        if style_type == "adain":
            base_feat = adaptive_instance_normalization(content_f, style_f)
        elif style_type == "adamean":
            base_feat = adaptive_mean_normalization(content_f, style_f)
        elif style_type == "adastd":
            base_feat = adaptive_std_normalization(content_f, style_f)
        elif style_type == "efdm":
            base_feat = exact_feature_distribution_matching(content_f, style_f)
        elif style_type == "hm":
            feat = histogram_matching(content_f, style_f)
        else:
            raise NotImplementedError
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i : i + 1]
        content_f = content_f[0:1]
    else:
        if style_type == "adain":
            feat = adaptive_instance_normalization(content_f, style_f)
        elif style_type == "adamean":
            feat = adaptive_mean_normalization(content_f, style_f)
        elif style_type == "adastd":
            feat = adaptive_std_normalization(content_f, style_f)
        elif style_type == "efdm":
            feat = exact_feature_distribution_matching(content_f, style_f)
        elif style_type == "hm":
            feat = histogram_matching(content_f, style_f)
        else:
            raise NotImplementedError
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)
