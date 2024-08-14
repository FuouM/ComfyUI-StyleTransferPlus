import torch

from . import net


class MODEL_CAST:
    def __init__(
        self, vgg_path: str, net_ae_path: str, net_dec_b_path: str, device
    ) -> None:
        vgg = net.vgg
        vgg.load_state_dict(torch.load(vgg_path))
        vgg = torch.nn.Sequential(*list(vgg.children())[:31])

        self.netAE = net.ADAIN_Encoder(vgg)
        self.netDec_B = net.Decoder()

        self.netAE.load_state_dict(load_a_ckpt(net_ae_path))
        self.netDec_B.load_state_dict(load_a_ckpt(net_dec_b_path))

        self.netAE.to(device).eval()
        self.netDec_B.to(device).eval()


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


def load_a_ckpt(ckpt_path: str):
    state_dict = torch.load(ckpt_path)
    if hasattr(state_dict, "_metadata"):
        del state_dict._metadata
    return state_dict
