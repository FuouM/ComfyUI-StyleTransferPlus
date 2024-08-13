import torch

from . import net


def load_a_ckpt(ckpt_path: str):
    state_dict = torch.load(ckpt_path)
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    return state_dict


def inference_ucast(
    src_img: torch.Tensor,
    style_img: torch.Tensor,
    netAE,
    netDec_B,
    device,
):
    real_A = src_img.permute(0, 3, 1, 2).to(device)
    real_B = style_img.permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        real_A_feat = netAE.forward(real_A, real_B)
        fake_B = netDec_B.forward(real_A_feat)

    if device.type != "cpu":
        torch.cuda.empty_cache()

    return fake_B
