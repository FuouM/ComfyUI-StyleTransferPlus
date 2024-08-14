"""
@author: Fuou Marinas
@title: ComfyUI-StyleTransferPlus
@nickname: StyleTransferPlus
@description: A collection of style transfer nodes.
"""

from pathlib import Path

import torch
import tqdm
from comfy.utils import ProgressBar

from .constants import (
    AESFA_PATH,
    AESPA_DEC_PATH,
    AESPA_TRANSFORMER_PATH,
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
    TSSAT_DEC_PATH,
    UNIST_DEC_PATH,
    UNIST_ENC_PATH,
    UNIST_PATH,
    VGG_NORM_CONV51_PATH,
)

# AesFA
from .module_aesfa.aesfa_model import (
    MODEL_AESFA,
    inference_aesfa,
    inference_aesfa_style_blend,
)

# AesPA
from .module_aespa.aespa_model import MODEL_AESPA, inference_aespa

# CAST
from .module_cast.cast_model import MODEL_CAST, inference_ucast

# EFDM
from .module_efdm.efdm_model import MODEL_EFDM, inference_efdm

# MicroAST
from .module_microast.microast_model import MODEL_MICROAST, inference_microast

# Neural Neighbor
from .module_neural_neighbor.neural_neighbor_model import inference_neural_neighbor

# TSSAT
from .module_tssat.tssat_model import MODEL_TSSAT, inference_tssat

# UniST
from .module_unist.unist_model import inference_unist, video_Style_transfer

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

        model = MODEL_CAST(
            vgg_path=f"{base_dir}/{CAST_VGG_PATH}",
            net_ae_path=f"{base_dir}/models/{model_arch}_model/{CAST_NET_AE_PATH}",
            net_dec_b_path=f"{base_dir}/models/{model_arch}_model/{CAST_NET_DEC_B_PATH}",
            device=device,
        )

        params = {
            "style_img": style_img,
            "device": device,
            "netAE": model.netAE,
            "netDec_B": model.netDec_B,
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

        model = MODEL_EFDM(
            dec_path=f"{base_dir}/{EFDM_PATH}",
            vgg_path=f"{base_dir}/{CAST_VGG_PATH}",
            device=device,
        )

        params = {
            "style_img": style_img.permute(0, 3, 1, 2),
            "device": device,
            "decoder": model.decoder,
            "vgg": model.vgg,
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

        model = MODEL_MICROAST(
            content_dec_path=f"{base_dir}/{MICROAST_CONTENT_ENCODER_PATH}",
            style_enc_path=f"{base_dir}/{MICROAST_STYLE_ENCODER_PATH}",
            modulator_path=f"{base_dir}/{MICROAST_MODULATOR_PATH}",
            decoder_path=f"{base_dir}/{MICROAST_DECODER_PATH}",
            device=device,
        )

        params = {
            "style_img": style_img.permute(0, 3, 1, 2),
            "device": device,
            "alpha": 1.0,
            "size": size,
            "do_crop": do_crop,
            "network": model.network,
        }

        num_frames = src_img.size(0)
        pbar = ProgressBar(num_frames)

        result: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(num_frames):
                params["src_img"] = src_img[i].unsqueeze(0).permute(0, 3, 1, 2)
                res_tensor = inference_microast(**params)
                result.append(res_tensor.permute(0, 2, 3, 1))
                pbar.update_absolute(i, num_frames)

        return (torch.cat(result, dim=0),)


class UniST:
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
    RETURN_NAMES = ("out_img",)
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

        network = video_Style_transfer(
            ckpt_path=f"{base_dir}/{UNIST_PATH}",
            encoder_path=f"{base_dir}/{UNIST_ENC_PATH}",
            decoder_path=f"{base_dir}/{UNIST_DEC_PATH}",
        )

        network.eval()
        network.to(device)

        params = {
            "src_style": style_img.permute(0, 3, 1, 2),
            "device": device,
            "size": size,
            "do_crop": do_crop,
            "network": network,
            "content_type": "image",
        }

        num_frames = src_img.size(0)
        pbar = ProgressBar(num_frames)

        result: list[torch.Tensor] = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(num_frames), desc="Generating"):
                params["src"] = src_img[i].unsqueeze(0).permute(0, 3, 1, 2)
                res_tensor = inference_unist(**params)
                result.append(res_tensor.permute(0, 2, 3, 1))
                pbar.update_absolute(i, num_frames)

        return (torch.cat(result, dim=0),)


class UniST_Video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_video": ("IMAGE",),
                "style_video": ("IMAGE",),
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
    RETURN_NAMES = ("out_img",)
    FUNCTION = "todo"
    CATEGORY = "StyleTransferPlus"

    def todo(
        self,
        src_video: torch.Tensor,
        style_video: torch.Tensor,
        do_crop: bool,
        size: int,
    ):
        print(f"{src_video.shape=}")
        print(f"{style_video.shape=}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        network = video_Style_transfer(
            ckpt_path=f"{base_dir}/{UNIST_PATH}",
            encoder_path=f"{base_dir}/{UNIST_ENC_PATH}",
            decoder_path=f"{base_dir}/{UNIST_DEC_PATH}",
        )

        network.eval()
        network.to(device)

        num_frames = src_video.size(0)
        batch_size = 3
        result = []

        params = {
            "src_style": style_video.permute(0, 3, 1, 2),
            "device": device,
            "size": size,
            "do_crop": do_crop,
            "network": network,
            "content_type": "video",
        }

        pbar = ProgressBar(num_frames)

        with torch.no_grad():
            for i in tqdm.tqdm(range(0, num_frames, batch_size), "Generating"):
                batch_end = min(i + batch_size, num_frames)
                src_batch = src_video[i:batch_end].permute(0, 3, 1, 2)

                params["src"] = src_batch

                res_tensor = inference_unist(**params).permute(0, 2, 3, 1)
                result.append(res_tensor)

                pbar.update_absolute(batch_end, num_frames)

        final_result = torch.cat(result, dim=0)
        return (final_result,)


class AesPA:
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
                    {"default": 512, "min": 1, "max": 1024, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("out_img",)
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

        model = MODEL_AESPA(
            vgg_51_path=f"{base_dir}/{VGG_NORM_CONV51_PATH}",
            dec_path=f"{base_dir}/{AESPA_DEC_PATH}",
            transformer_path=f"{base_dir}/{AESPA_TRANSFORMER_PATH}",
            device=device,
        )

        params = {
            "style_img": style_img.permute(0, 3, 1, 2),
            "device": device,
            "size": size,
            "do_crop": do_crop,
            "model": model,
        }

        num_frames = src_img.size(0)
        pbar = ProgressBar(num_frames)

        result: list[torch.Tensor] = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(num_frames), desc="Generating"):
                params["src_img"] = src_img[i].unsqueeze(0).permute(0, 3, 1, 2)
                res_tensor = inference_aespa(**params)
                result.append(res_tensor.permute(0, 2, 3, 1))
                pbar.update_absolute(i, num_frames)

        return (torch.cat(result, dim=0),)


class TSSAT:
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
                "max_iter": (
                    "INT",
                    {"default": 1, "min": 1, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("out_img",)
    FUNCTION = "todo"
    CATEGORY = "StyleTransferPlus"

    def todo(
        self,
        src_img: torch.Tensor,
        style_img: torch.Tensor,
        do_crop: bool,
        size: int,
        max_iter: int,
    ):
        print(f"{src_img.shape=}")
        print(f"{style_img.shape=}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MODEL_TSSAT(
            dec_path=f"{base_dir}/{TSSAT_DEC_PATH}",
            vgg_path=f"{base_dir}/{CAST_VGG_PATH}",
            device=device,
        )

        params = {
            "style_img": style_img.permute(0, 3, 1, 2),
            "device": device,
            "size": size,
            "do_crop": do_crop,
            "max_steps": max_iter,
            "model": model,
        }

        num_frames = src_img.size(0)

        result: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(num_frames):
                params["src_img"] = src_img[i].unsqueeze(0).permute(0, 3, 1, 2)
                res_tensor = inference_tssat(**params)
                result.append(res_tensor.permute(0, 2, 3, 1))

        return (torch.cat(result, dim=0),)


class AesFA:
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
    RETURN_NAMES = ("out_img",)
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

        model = MODEL_AESFA(
            checkpoint_path=f"{base_dir}/{AESFA_PATH}",
            device=device,
        )

        params = {
            "style_img": style_img.permute(0, 3, 1, 2),
            "device": device,
            "size": size,
            "do_crop": do_crop,
            "model": model,
        }

        num_frames = src_img.size(0)

        result: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(num_frames):
                params["src_img"] = src_img[i].unsqueeze(0).permute(0, 3, 1, 2)
                res_tensor = inference_aesfa(**params)
                result.append(res_tensor.permute(0, 2, 3, 1))

        return (torch.cat(result, dim=0),)


class AesFAStyleBlend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "src_img": ("IMAGE",),
                "style_hi": ("IMAGE",),
                "style_lo": ("IMAGE",),
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
    RETURN_NAMES = ("out_img",)
    FUNCTION = "todo"
    CATEGORY = "StyleTransferPlus"

    def todo(
        self,
        src_img: torch.Tensor,
        style_hi: torch.Tensor,
        style_lo: torch.Tensor,
        do_crop: bool,
        size: int,
    ):
        print(f"{src_img.shape=}")
        print(f"{style_hi.shape=}")
        print(f"{style_lo.shape=}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MODEL_AESFA(
            checkpoint_path=f"{base_dir}/{AESFA_PATH}",
            device=device,
        )

        params = {
            "style_hi": style_hi.permute(0, 3, 1, 2),
            "style_lo": style_lo.permute(0, 3, 1, 2),
            "device": device,
            "size": size,
            "do_crop": do_crop,
            "model": model,
        }

        num_frames = src_img.size(0)

        result: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(num_frames):
                params["src_img"] = src_img[i].unsqueeze(0).permute(0, 3, 1, 2)
                res_tensor = inference_aesfa_style_blend(**params)
                result.append(res_tensor.permute(0, 2, 3, 1))

        return (torch.cat(result, dim=0),)


def serialize_floats(lst: list[float]):
    return ",".join(map(str, lst))


def deserialize_floats(floats_str: str):
    return [float(w) for w in floats_str.split(",") if w.strip()]
