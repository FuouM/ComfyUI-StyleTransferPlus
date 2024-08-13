import torch
from torch import nn

from .models import Transformer


def get_keys(d, name):
    if "state_dict" in d:
        d = d["state_dict"]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name)] == name}
    return d_filt


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
