import torch
import torch.nn.functional as F

from .stylize import produce_stylization
from .vgg import Vgg16Pretrained


def resize(src: torch.Tensor, target_size: int, scale_long=True):
    # [B, C, H, W]
    src_h, src_w = src.shape[2:]
    if scale_long:
        factor = target_size / max(src_h, src_w)
    else:
        factor = target_size / min(src_h, src_w)

    new_h = int(src_h * factor)
    new_w = int(src_w * factor)

    new_x = F.interpolate(src, (new_h, new_w), mode="bilinear", align_corners=True)

    return new_x


def inference_neural_neighbor(
    src_img: torch.Tensor,
    style_img: torch.Tensor,
    device,
    size=512,
    scale_long=True,
    flip=False,
    content_loss=False,
    colorize=True,
    alpha=0.75,
    max_iter=200,
):
    # [B, H, W, C]
    content_im_orig = resize(
        src_img.permute(0, 3, 1, 2).contiguous(), size, scale_long
    ).to(device)
    style_im_orig = resize(
        style_img.permute(0, 3, 1, 2).contiguous(), size, scale_long
    ).to(device)
    content_weight = 1 - alpha

    max_scls = 4 if size == 512 else 5

    cnn = Vgg16Pretrained().to(device)

    def phi(x, y, z):
        return cnn.forward(x, inds=y, concat=z)

    torch.cuda.synchronize()
    output = produce_stylization(
        content_im_orig,
        style_im_orig,
        phi,
        max_iter=max_iter,
        lr=2e-3,
        content_weight=content_weight,
        max_scls=max_scls,
        flip_aug=flip,
        content_loss=content_loss,
        dont_colorize=not colorize,
        device=device,
    )
    torch.cuda.synchronize()

    output = torch.clip(output, 0, 1).permute(0, 2, 3, 1)

    return output
