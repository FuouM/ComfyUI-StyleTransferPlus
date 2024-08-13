from pathlib import Path

import torch

from .module_extra.function import coral

base_dir = Path(__file__).resolve().parent


class CoralColorTransfer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
                "style_img": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_img",)
    FUNCTION = "todo"
    CATEGORY = "StyleTransferPlus/Extra"

    def todo(
        self,
        src_img: torch.Tensor,
        style_img: torch.Tensor,
    ):
        print(f"{src_img.shape=}")
        print(f"{style_img.shape=}")

        device = torch.device("cpu")

        src_img = src_img.permute(0, 3, 1, 2).to(device)  # [B, C, H, W]
        style_img = style_img.permute(0, 3, 1, 2).to(device)

        result = []
        if src_img.shape[0] == 1 and style_img.shape[0] == 1:
            # Case 1: Single source image, single style image
            result = [coral(src_img[0], style_img[0]).unsqueeze(0)]
        elif src_img.shape[0] == 1:
            # Case 2: Single source image, multiple style images
            num_frames = style_img.shape[0]
            for i in range(num_frames):
                transferred = coral(src_img[0], style_img[i])
                result.append(transferred.unsqueeze(0))
        elif style_img.shape[0] == 1:
            # Case 3: Multiple source images, single style image
            num_frames = src_img.shape[0]
            for i in range(num_frames):
                transferred = coral(src_img[i], style_img[0])
                result.append(transferred.unsqueeze(0))
        else:
            # Case 4: Multiple source images, multiple style images
            num_frames = min(src_img.shape[0], style_img.shape[0])
            for i in range(num_frames):
                transferred = coral(src_img[i], style_img[i])
                result.append(transferred.unsqueeze(0))

        res_tensor = torch.cat(result, dim=0).permute(0, 2, 3, 1)

        return (res_tensor,)
