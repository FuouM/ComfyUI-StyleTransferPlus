import torch
import torch.nn.functional as F
from torchvision import transforms

from . import net_microAST


class MODEL_MICROAST:
    def __init__(
        self,
        content_dec_path: str,
        style_enc_path: str,
        modulator_path: str,
        decoder_path: str,
        device,
    ) -> None:
        content_encoder = net_microAST.Encoder()
        style_encoder = net_microAST.Encoder()
        modulator = net_microAST.Modulator()
        decoder = net_microAST.Decoder()

        content_encoder.eval()
        style_encoder.eval()
        modulator.eval()
        decoder.eval()

        content_encoder.load_state_dict(torch.load(content_dec_path))
        style_encoder.load_state_dict(torch.load(style_enc_path))
        modulator.load_state_dict(torch.load(modulator_path))
        decoder.load_state_dict(torch.load(decoder_path))

        self.network = net_microAST.TestNet(
            content_encoder, style_encoder, modulator, decoder
        )
        self.network.to(device)


def inference_microast(
    src_img: torch.Tensor,
    style_img: torch.Tensor,
    size: int,
    do_crop: bool,
    alpha: float,
    device,
    network,
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
