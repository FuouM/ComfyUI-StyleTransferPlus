import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .models import Transformer


class video_Style_transfer(nn.Module):
    def __init__(self, ckpt_path: str, encoder_path: str, decoder_path: str):
        super(video_Style_transfer, self).__init__()
        self.ckpt_path = ckpt_path
        self.model = Transformer(encoder_path, decoder_path)
        self.load_weights()

    def load_weights(self):
        # print("Loading model from checkpoint")
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        self.model.load_state_dict(get_keys(ckpt, "model"), strict=False)

    def forward(
        self,
        content_frames,
        style_images,
        content_type="image",
        id_loss="transfer",
        tab=None,
    ):
        transfer_result = self.model.forward(
            content_frames, style_images, content_type, id_loss, tab
        )
        return transfer_result


def inference_unist(
    src: torch.Tensor,
    src_style: torch.Tensor,
    device,
    size: int,
    do_crop: bool,
    network: video_Style_transfer,
    content_type="image",
):
    content_tf = test_transform(size, do_crop)
    content = content_tf(src).unsqueeze(0).to(device)  # [1, B, C, H, W]

    style = content_tf(src_style).to(device)  # [B, C, H, W]

    if content_type == "video":
        style = match_shape(style, content[0])

    y_hat = network.forward(content, style, content_type, tab="inference")
    output = un_normalize_batch(y_hat)  # [B, C, H, W]

    return output


def test_transform(size, crop, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    transform_list = []

    if crop and size != 0:
        transform_list.append(transforms.Resize(size))
        transform_list.append(transforms.CenterCrop(size))
    elif size != 0:
        transform_list.append(transforms.Resize((size, size)))

    transform_list.append(transforms.Normalize(mean, std))

    transform = transforms.Compose(transform_list)
    # [B, C, H, W]
    return transform


def un_normalize_batch(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def match_shape(a: torch.Tensor, b: torch.Tensor):
    B_a, C, H, W = a.shape
    B_b = b.shape[0]

    if B_a == B_b:
        return a
    elif B_a == 1:
        return a.expand(B_b, -1, -1, -1)
    elif B_a < B_b:
        repeat_factor = B_b // B_a
        remainder = B_b % B_a
        return torch.cat([a.repeat(repeat_factor, 1, 1, 1), a[:remainder]], dim=0)
    else:  # B_a > B_b
        return a[:B_b]


def get_keys(d, name):
    if "state_dict" in d:
        d = d["state_dict"]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name)] == name}
    return d_filt
