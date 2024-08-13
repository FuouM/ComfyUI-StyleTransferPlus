"""
@author: Fuou Marinas
@title: ComfyUI-StyleTransferPlus
@nickname: StyleTransferPlus
@description: A collection of style transfer nodes.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
from comfy.utils import ProgressBar

from .constants import (
    CAST_DEFAULT,
    CAST_NET_AE_PATH,
    CAST_NET_DEC_B_PATH,
    CAST_TYPES,
    CAST_VGG_PATH,
    EFDM_DEFAULT,
    EFDM_PATH,
    EFDM_STYLE_TYPES,
    MICROAST_CONTENT_ENCODER_PATH,
    MICROAST_DECODER_PATH,
    MICROAST_MODULATOR_PATH,
    MICROAST_STYLE_ENCODER_PATH,
)
from .module_cast import net as net_cast
from .module_cast.cast_model import inference_ucast, load_a_ckpt
from .module_efdm import net as net_efdm
from .module_efdm.efdm_model import inference_efdm
from .module_microast import net_microAST
from .module_microast.microast_model import microast_inference
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

        vgg = net_cast.vgg
        vgg.load_state_dict(torch.load(f"{base_dir}/{CAST_VGG_PATH}"))
        vgg = torch.nn.Sequential(*list(vgg.children())[:31])

        netAE = net_cast.ADAIN_Encoder(vgg)
        netDec_B = net_cast.Decoder()

        netAE.load_state_dict(
            load_a_ckpt(f"{base_dir}/models/{model_arch}_model/{CAST_NET_AE_PATH}")
        )
        netDec_B.load_state_dict(
            load_a_ckpt(f"{base_dir}/models/{model_arch}_model/{CAST_NET_DEC_B_PATH}")
        )

        netAE = netAE.to(device).eval()
        netDec_B = netDec_B.to(device).eval()

        params = {
            "style_img": style_img,
            "device": device,
            "netAE": netAE,
            "netDec_B": netDec_B,
        }

        num_frames = src_img.size(0)
        pbar = ProgressBar(num_frames)

        result: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(num_frames):
                params["src_img"] = src_img[i].unsqueeze(0)
                res_tensor = inference_ucast(**params)
                result.append(res_tensor.permute(0, 2, 3, 1))
                pbar.update_absolute(i, num_frames)

        return (torch.cat(result, dim=0),)


class EFDM:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
                "style_img": ("IMAGE",),
                "style_interp_weights": ("STRING", {"default": ""}),
                "model_arch": (EFDM_STYLE_TYPES, {"default": EFDM_DEFAULT}),
                "style_strength": (
                    "FLOAT",
                    {"default": 1.00, "min": 0.00, "max": 1.00, "round": 0.01},
                ),
                "do_crop": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "preserve_color": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "size": (
                    "INT",
                    {"default": 512, "min": 1, "step": 1},
                ),
                "use_cpu": (
                    "BOOLEAN",
                    {"default": False},
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
        style_interp_weights: str,
        model_arch: str,
        style_strength: float,
        do_crop: bool,
        preserve_color: bool,
        size: int,
        use_cpu: bool,
    ):
        print(f"{src_img.shape=}")
        print(f"{style_img.shape=}")
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not use_cpu else "cpu"
        )

        decoder = net_efdm.decoder
        vgg = net_efdm.vgg

        decoder.eval()
        vgg.eval()

        decoder.load_state_dict(torch.load(f"{base_dir}/{EFDM_PATH}"))
        vgg.load_state_dict(torch.load(f"{base_dir}/{CAST_VGG_PATH}"))

        vgg = torch.nn.Sequential(*list(vgg.children())[:31])

        vgg.to(device)
        decoder.to(device)

        params = {
            "style_img": style_img.permute(0, 3, 1, 2),
            "device": device,
            "decoder": decoder,
            "vgg": vgg,
            "alpha": style_strength,
            "size": size,
            "style_type": model_arch,
            "do_crop": do_crop,
            "preserve_color": preserve_color,
            "style_interpolation_weights": deserialize_floats(style_interp_weights)
            or None,
        }

        print(f"{params['style_interpolation_weights']=}")

        num_frames = src_img.size(0)
        pbar = ProgressBar(num_frames)

        result: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(num_frames):
                params["src_img"] = src_img[i].unsqueeze(0).permute(0, 3, 1, 2)
                res_tensor = inference_efdm(**params)
                result.append(res_tensor.permute(0, 2, 3, 1))
                pbar.update_absolute(i, num_frames)

        return (torch.cat(result, dim=0),)


class MicroAST:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
                "style_img": ("IMAGE",),
                "do_crop": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "size": (
                    "INT",
                    {"default": 512, "min": 1, "step": 1},
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
        do_crop: bool,
        size: int,
    ):
        print(f"{src_img.shape=}")
        print(f"{style_img.shape=}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        content_encoder = net_microAST.Encoder()
        style_encoder = net_microAST.Encoder()
        modulator = net_microAST.Modulator()
        decoder = net_microAST.Decoder()

        content_encoder.eval()
        style_encoder.eval()
        modulator.eval()
        decoder.eval()

        content_encoder.load_state_dict(
            torch.load(f"{base_dir}/{MICROAST_CONTENT_ENCODER_PATH}")
        )
        style_encoder.load_state_dict(
            torch.load(f"{base_dir}/{MICROAST_STYLE_ENCODER_PATH}")
        )
        modulator.load_state_dict(torch.load(f"{base_dir}/{MICROAST_MODULATOR_PATH}"))
        decoder.load_state_dict(torch.load(f"{base_dir}/{MICROAST_DECODER_PATH}"))

        network = net_microAST.TestNet(
            content_encoder, style_encoder, modulator, decoder
        )
        network.to(device)

        params = {
            "style_img": style_img.permute(0, 3, 1, 2),
            "device": device,
            "alpha": 1.0,
            "size": size,
            "do_crop": do_crop,
            "network": network,
        }

        num_frames = src_img.size(0)
        pbar = ProgressBar(num_frames)

        result: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(num_frames):
                params["src_img"] = src_img[i].unsqueeze(0).permute(0, 3, 1, 2)
                res_tensor = microast_inference(**params)
                result.append(res_tensor.permute(0, 2, 3, 1))
                pbar.update_absolute(i, num_frames)

        return (torch.cat(result, dim=0),)


def serialize_floats(lst: list[float]):
    return ",".join(map(str, lst))


def deserialize_floats(floats_str: str):
    return [float(w) for w in floats_str.split(",") if w.strip()]
