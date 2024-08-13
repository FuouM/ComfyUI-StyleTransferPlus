import numpy as np
import torch
import torchvision.transforms as transforms

from .model import video_Style_transfer


def inference_unist(
    src: torch.Tensor,
    src_style: torch.Tensor,
    device,
    size: int,
    do_crop: bool,
    network: video_Style_transfer,
    content_type="image",
):
    content_tf = test_transform(size, do_crop)
    content = content_tf(src).unsqueeze(0).to(device)  # [1, B, C, H, W]

    style = content_tf(src_style).to(device)  # [B, C, H, W]

    if content_type == "video":
        style = match_shape(style, content[0])

    print(f"{content.shape=}")
    print(f"{style.shape=}")

    y_hat = network.forward(
        content, style, content_type, tab="inference"
    )  # [B, C, H, W]
    print(f"{y_hat.shape=}")
    output = un_normalize_batch(y_hat)  # [B, C, H, W]

    return output


def test_transform(size, crop):
    transform_list = []

    if crop and size != 0:
        transform_list.append(transforms.Resize(size))
        transform_list.append(transforms.CenterCrop(size))
    elif size != 0:
        transform_list.append(transforms.Resize((size, size)))

    transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = transforms.Compose(transform_list)
    # [B, C, H, W]
    return transform


def un_normalize_batch(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def match_shape(a: torch.Tensor, b: torch.Tensor):
    B_a, C, H, W = a.shape
    B_b, _, _, _ = b.shape

    if B_a == 1 and B_b > 1:
        # Expand a along the batch dimension to match the shape of b
        expanded_a = a.expand(B_b, -1, -1, -1)
    elif B_a > 1 and B_a < B_b:
        # Repeat a along the batch dimension to match the shape of b
        repeated_a = a.repeat(B_b // B_a, 1, 1, 1)
        if B_b % B_a != 0:
            remaining = B_b % B_a
            repeated_a = torch.cat([repeated_a, a[-remaining:]], dim=0)
        expanded_a = repeated_a
    elif B_a > B_b:
        # Trim a along the batch dimension to match the shape of b
        trimmed_a = a[:B_b]
        expanded_a = trimmed_a
    else:
        # If B_a == B_b, no change is needed
        expanded_a = a

    return expanded_a
