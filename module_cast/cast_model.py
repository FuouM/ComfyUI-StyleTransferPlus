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
    ckpt_ae: str,
    ckpt_decB: str,
    vgg_path,
    device,
):
    vgg = net.vgg
    vgg.load_state_dict(torch.load(vgg_path))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])

    netAE = net.ADAIN_Encoder(vgg)
    netDec_B = net.Decoder()

    netAE.load_state_dict(load_a_ckpt(ckpt_ae))
    netDec_B.load_state_dict(load_a_ckpt(ckpt_decB))

    netAE = netAE.to(device).eval()
    netDec_B = netDec_B.to(device).eval()

    real_A = src_img.permute(0, 3, 1, 2).to(device)
    real_B = style_img.permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        real_A_feat = netAE.forward(real_A, real_B)
        fake_B = netDec_B.forward(real_A_feat)
    
    if device.type != "cpu":
        torch.cuda.empty_cache()

    return fake_B
