import torch

from . import net_microAST
import torch.nn.functional as F
from torchvision import transforms


def microast_inference(
    src_img: torch.Tensor,
    style_img: torch.Tensor,
    size: int,
    do_crop: bool,
    alpha: float,
    device,
    network
):
    content_tf = test_transform(size, do_crop)

    content = content_tf(src_img).to(device)
    style = content_tf(style_img).to(device)
    style = F.interpolate(
        style,
        size=(content.shape[2], content.shape[3]),
        mode="bilinear",
        align_corners=False,
    )

    # print(f"Resized: {content.shape=}")
    # print(f"Resized: {style.shape=}")

    torch.cuda.synchronize()
    output = network(content, style, alpha)
    torch.cuda.synchronize()

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
