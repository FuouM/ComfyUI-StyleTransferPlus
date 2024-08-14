import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf_f

from .aespanet_models import (
    AdaptiveMultiAttn_Transformer_v2,
    VGGDecoder,
    VGGEncoder,
)
from .utils import adaptive_get_keys, adaptive_gram_weight, feature_wct_simple


class MODEL_AESPA:
    def __init__(
        self, vgg_51_path: str, dec_path: str, transformer_path: str, device
    ) -> None:
        pretrained_vgg = torch.load(vgg_51_path)
        self.encoder = VGGEncoder(pretrained_vgg)
        self.decoder = VGGDecoder()
        self.transformer = AdaptiveMultiAttn_Transformer_v2(
            in_planes=512,
            out_planes=512,
            query_planes=512,
            key_planes=512 + 256 + 128 + 64,
        )

        self.decoder.load_state_dict(torch.load(dec_path)["state_dict"])
        self.transformer.load_state_dict(torch.load(transformer_path)["state_dict"])

        self.encoder.eval().to(device)
        self.decoder.eval().to(device)
        self.transformer.eval().to(device)

        self.device = device

    def forward(self, content: torch.Tensor, style: torch.Tensor):
        gray_content = tf_f.rgb_to_grayscale(content).repeat(1, 3, 1, 1).to(self.device)
        gray_style = tf_f.rgb_to_grayscale(style).repeat(1, 3, 1, 1).to(self.device)

        style_weight = (
            adaptive_gram_weight(style, 1, 8, self.encoder)
            + adaptive_gram_weight(style, 2, 8, self.encoder)
            + adaptive_gram_weight(style, 3, 8, self.encoder)
        ) / 3

        gray_style_weight = (
            adaptive_gram_weight(gray_style, 1, 8, self.encoder)
            + adaptive_gram_weight(gray_style, 2, 8, self.encoder)
            + adaptive_gram_weight(gray_style, 3, 8, self.encoder)
        ) / 3

        style_adaptive_alpha = (
            style_weight.unsqueeze(1).to(self.device)
            + gray_style_weight.unsqueeze(1).to(self.device)
        ) / 2

        content_skips = {}
        style_skips = {}

        _ = self.encoder.encode(content, content_skips)  # content_feat
        _ = self.encoder.encode(style, style_skips)  # style_feat

        if gray_content is not None:
            gray_content_skips = {}
            _ = self.encoder.encode(
                gray_content, gray_content_skips
            )  # gray_content_feat
        if gray_style is not None:
            gray_style_skips = {}
            _ = self.encoder.encode(gray_style, gray_style_skips)  # gray_style_feat

        (
            local_transformed_feature,
            _,
            _,
            _,
            _,
        ) = self.transformer.forward(
            content_skips["conv4_1"],
            style_skips["conv4_1"],
            content_skips["conv5_1"],
            style_skips["conv5_1"],
            adaptive_get_keys(
                content_skips, 4, 4, target_feat=content_skips["conv4_1"]
            ),
            adaptive_get_keys(style_skips, 1, 4, target_feat=style_skips["conv4_1"]),
            adaptive_get_keys(
                content_skips, 5, 5, target_feat=content_skips["conv5_1"]
            ),
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
            global_transformed_feat
            * (1 - style_adaptive_alpha.unsqueeze(-1).unsqueeze(-1))
            + style_adaptive_alpha.unsqueeze(-1).unsqueeze(-1)
            * local_transformed_feature
        )

        stylized_image = self.decoder.decode(
            transformed_feature, content_skips, style_skips
        )

        return stylized_image


def inference_aespa(
    src_img: torch.Tensor,
    style_img: torch.Tensor,
    device,
    size: int,
    do_crop: bool,
    model: MODEL_AESPA,
):
    content_tf = test_transform(
        size, do_crop, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    content = content_tf(src_img).to(device)  # [B, C, H, W]
    style = content_tf(style_img).to(device)

    stylized_image = model.forward(content, style)

    output = un_normalize_batch(stylized_image)

    return output


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
