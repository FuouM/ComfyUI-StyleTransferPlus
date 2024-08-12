"""
@author: Fuou Marinas
@title: ComfyUI-StyleTransferPlus
@nickname: StyleTransferPlus
@description: A collection of style trasnfer nodes.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
from comfy.utils import ProgressBar

from .module_cast.cast_model import inference_ucast

from .constants import (
    CAST_DEFAULT,
    CAST_NET_AE_PATH,
    CAST_NET_DEC_B_PATH,
    CAST_TYPES,
    CAST_VGG_PATH,
)

from .module_neural_neighbor.neural_neighbor_model import inference_neural_neighbor


base_dir = Path(__file__).resolve().parent


class NeuralNeighbor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
                "style_img": ("IMAGE",),
                "size": ([512, 1024], {"default": 512}),
                "scale_long": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "flip": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "content_loss": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "colorize": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "content_weight": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.01, "max": 1.00, "round": 0.01},
                ),
                "max_iter": (
                    "INT",
                    {"default": 200, "min": 1, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_img",)
    FUNCTION = "todo"
    CATEGORY = "StyleTransferPlus"

    def todo(
        self,
        src_img: torch.Tensor,
        style_img: torch.Tensor,
        size: int,
        scale_long: bool,
        flip: bool,
        content_loss: bool,
        colorize: bool,
        content_weight: float,
        max_iter: int,
    ):
        print(f"{src_img.shape=}")
        print(f"{style_img.shape=}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params = {
            "style_img": style_img,
            "device": device,
            "size": size,
            "scale_long": scale_long,
            "flip": flip,
            "content_loss": content_loss,
            "colorize": colorize,
            "alpha": content_weight,
            "max_iter": max_iter,
        }

        num_frames = src_img.size(0)
        result: list[torch.Tensor] = []
        with torch.inference_mode(mode=False):
            for i in range(num_frames):
                params["src_img"] = src_img[i].unsqueeze(0)
                res_tensor = inference_neural_neighbor(**params)
                result.append(res_tensor)

        return (torch.cat(result, dim=0),)


class CAST:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
                "style_img": ("IMAGE",),
                "model_arch": (CAST_TYPES, {"default": CAST_DEFAULT}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("res_img",)
    FUNCTION = "todo"
    CATEGORY = "StyleTransferPlus"

    def todo(
        self,
        src_img: torch.Tensor,
        style_img: torch.Tensor,
        model_arch: str,
    ):
        print(f"{src_img.shape=}")
        print(f"{style_img.shape=}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params = {
            "style_img": style_img,
            "device": device,
            "ckpt_ae": f"{base_dir}/models/{model_arch}_model/{CAST_NET_AE_PATH}",
            "ckpt_decB": f"{base_dir}/models/{model_arch}_model/{CAST_NET_DEC_B_PATH}",
            "vgg_path": f"{base_dir}/{CAST_VGG_PATH}",
        }

        num_frames = src_img.size(0)
        pbar = ProgressBar(num_frames)

        result: list[torch.Tensor] = []
        with torch.inference_mode(mode=False):
            for i in range(num_frames):
                params["src_img"] = src_img[i].unsqueeze(0)
                res_tensor = inference_ucast(**params)
                result.append(res_tensor.permute(0, 2, 3, 1))
                pbar.update_absolute(i, num_frames)

        return (torch.cat(result, dim=0),)


# class RealViFormerSR:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "src_video": ("IMAGE",),
#                 "interval": (
#                     "INT",
#                     {"default": 50, "min": 0, "step": 1},
#                 ),
#             },
#         }

#     RETURN_TYPES = ("IMAGE",)
#     RETURN_NAMES = ("res_video",)
#     FUNCTION = "todo"
#     CATEGORY = "FM_nodes"

#     def todo(self, src_video: torch.Tensor, interval: int):
#         src_video = src_video.permute(0, 3, 1, 2)
#         print(f"{src_video.shape=}")

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = RealViformer(
#             num_feat=48,
#             num_blocks=[2, 3, 4, 1],
#             spynet_path=None,
#             heads=[1, 2, 4],
#             ffn_expansion_factor=2.66,
#             merge_head=2,
#             bias=False,
#             LayerNorm_type="BiasFree",
#             ch_compress=True,
#             squeeze_factor=[4, 4, 4],
#             masked=True,
#         )
#         model.load_state_dict(
#             torch.load(f"{base_dir}/{REALVIFORMER_MODEL_PATH}")["params"], strict=False
#         )
#         model.eval()
#         model = model.to(device)

#         if src_video.shape[0] <= interval:
#             out_tensor = inference_realviformer(
#                 src_video.unsqueeze(0).to(device), model
#             )
#             out_tensor = out_tensor.squeeze(dim=0).permute(0, 2, 3, 1)
#             return (out_tensor,)

#         num_imgs = src_video.shape[0]
#         outputs: list[torch.Tensor] = []
#         pbar = ProgressBar(num_imgs)

#         for idx in tqdm.tqdm(range(0, num_imgs, interval)):
#             interval = min(interval, num_imgs - idx)
#             imgs = src_video[idx : idx + interval]
#             imgs = imgs.unsqueeze(0).to(device)  # [b, n, c, h, w]
#             outputs.append(inference_realviformer(imgs, model).squeeze(dim=0))
#             pbar.update_absolute(idx + interval, num_imgs)

#         out_tensor = torch.cat(outputs, dim=0).permute(0, 2, 3, 1)
#         return (out_tensor,)


# class ProPIH_Harmonizer:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "composite": ("IMAGE",),
#                 "background": ("IMAGE",),
#             },
#             "optional": {"foreground_mask": ("IMAGE",), "foreground_MASK": ("MASK",)},
#         }

#     RETURN_TYPES = (
#         "IMAGE",
#         "IMAGE",
#         "IMAGE",
#         "IMAGE",
#     )
#     RETURN_NAMES = (
#         "out_0",
#         "out_1",
#         "out_2",
#         "out_3",
#     )
#     FUNCTION = "todo"
#     CATEGORY = "FM_nodes"

#     def todo(
#         self,
#         composite: torch.Tensor,
#         background: torch.Tensor,
#         foreground_mask: torch.Tensor | None = None,
#         foreground_MASK: torch.Tensor | None = None,
#     ):
#         if foreground_mask is None and foreground_MASK is None:
#             raise ValueError("Please provide one mask image")

#         if foreground_MASK is not None:
#             mask = foreground_MASK.unsqueeze(dim=0)
#         else:
#             mask = img_to_mask(foreground_mask.permute(0, 3, 1, 2))

#         composite = composite.permute(0, 3, 1, 2)
#         background = background.permute(0, 3, 1, 2)

#         propih = VGG19HRNetModel(
#             vgg_path=f"{base_dir}/{PROPIH_VGG_MODEL_PATH}",
#             g_path=f"{base_dir}/{PROPIH_G_MODEL_PATH}",
#         )
#         outputs = propih.forward(comp=composite, style=background, mask=mask)
#         final_outputs = []
#         for ts in outputs:
#             final_outputs.append(ts.permute(0, 2, 3, 1))

#         return (
#             final_outputs[0],
#             final_outputs[1],
#             final_outputs[2],
#             final_outputs[3],
#         )


# class CoLIE_LowLight_Enhance:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "src_img": ("IMAGE",),
#                 "down_res": (
#                     "INT",
#                     {"default": 256, "min": 1, "step": 1},
#                 ),
#                 "epochs": (
#                     "INT",
#                     {"default": 100, "min": 1, "step": 1},
#                 ),
#                 "cxt_window": (
#                     "INT",
#                     {"default": 1, "min": 1, "step": 1},
#                 ),
#                 "loss_mean": (
#                     "FLOAT",
#                     {"default": 0.3, "min": 0.01},
#                 ),
#                 "alpha": (
#                     "FLOAT",
#                     {"default": 1.0, "min": 0.01},
#                 ),
#                 "beta": (
#                     "FLOAT",
#                     {"default": 20.0, "min": 0.01},
#                 ),
#                 "gamma": (
#                     "FLOAT",
#                     {"default": 8.0, "min": 0.01},
#                 ),
#                 "delta": (
#                     "FLOAT",
#                     {"default": 5.0, "min": 0.01},
#                 ),
#             },
#         }

#     RETURN_TYPES = ("IMAGE",)
#     RETURN_NAMES = ("res_img",)
#     FUNCTION = "todo"
#     CATEGORY = "FM_nodes"

#     def todo(
#         self,
#         src_img: torch.Tensor,
#         down_res: int,
#         epochs: int,
#         cxt_window: int,
#         loss_mean: float,
#         alpha: float,
#         beta: float,
#         gamma: float,
#         delta: float,
#     ):
#         result: list[torch.Tensor] = []
#         num_frames = src_img.size(0)
#         pbar = ProgressBar(num_frames)
#         for i in range(num_frames):
#             image = src_img[i].unsqueeze(0)
#             res_tensor = run_colie(
#                 image,
#                 CoLIE_Config(
#                     down_res=down_res,
#                     epochs=epochs,
#                     cxt_window=cxt_window,
#                     loss_mean=loss_mean,
#                     alpha=alpha,
#                     beta=beta,
#                     gamma=gamma,
#                     delta=delta,
#                 ),
#             )
#             result.append(res_tensor)
#             pbar.update_absolute(i, num_frames)

#         return (torch.cat(result, dim=0),)
