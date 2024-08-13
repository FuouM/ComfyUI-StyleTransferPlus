import math

import torch
import torch.nn as nn
from einops import rearrange

from .transformer_layers import DecoderLayer_style, EncoderLayer, EncoderLayer_cross


class Encoder_vgg(nn.Module):
    def __init__(self):
        super(Encoder_vgg, self).__init__()
        # vgg
        # 256 x 256
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0)
        self.reflecPad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        # 256 x 256

        self.conv2 = nn.Conv2d(3, 64, 3, 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        # 256 x 256

        self.reflecPad3 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu3 = nn.ReLU(inplace=True)
        # 256 x 256

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 128 x 128

        self.reflecPad4 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu4 = nn.ReLU(inplace=True)
        # 128 x 128

        self.reflecPad5 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu5 = nn.ReLU(inplace=True)
        # 128 x 128 x 128

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 64 x 64

        self.reflecPad6 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv6 = nn.Conv2d(128, 256, 3, 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        # 64 x 64

        self.reflecPad7 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu7 = nn.ReLU(inplace=True)
        # 64 x 64

        self.reflecPad8 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv8 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu8 = nn.ReLU(inplace=True)
        # 64 x 64

        self.reflecPad9 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv9 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu9 = nn.ReLU(inplace=True)
        # 64 x 64

        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 32 x 32

        self.reflecPad10 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv10 = nn.Conv2d(256, 512, 3, 1, 0)
        self.relu10 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.reflecPad1(x)
        x = self.relu2(self.conv2(x))
        x = self.reflecPad3(x)
        x = self.relu3(self.conv3(x))

        x = self.maxPool(x)

        x = self.reflecPad4(x)
        x = self.relu4(self.conv4(x))
        x = self.reflecPad5(x)
        x = self.relu5(self.conv5(x))

        x = self.maxPool2(x)

        x = self.reflecPad6(x)
        x = self.relu6(self.conv6(x))
        x = self.reflecPad7(x)
        x = self.relu7(self.conv7(x))
        x = self.reflecPad8(x)
        x = self.relu8(self.conv8(x))
        x = self.reflecPad9(x)
        x = self.relu9(self.conv9(x))

        x = self.maxPool3(x)

        x = self.reflecPad10(x)
        x = self.relu10(self.conv10(x))

        return x


class Decoder_vgg(nn.Module):
    def __init__(self):
        super(Decoder_vgg, self).__init__()
        # decoder
        self.reflecPad11 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv11 = nn.Conv2d(512, 256, 3, 1, 0)
        self.relu11 = nn.ReLU(inplace=True)
        # 32 x 32

        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        # 64 x 64

        self.reflecPad12 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv12 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu12 = nn.ReLU(inplace=True)
        # 64 x 64

        self.reflecPad13 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv13 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu13 = nn.ReLU(inplace=True)
        # 64 x 64

        self.reflecPad14 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv14 = nn.Conv2d(256, 256, 3, 1, 0)
        self.relu14 = nn.ReLU(inplace=True)
        # 64 x 64

        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(256, 128, 3, 1, 0)
        self.relu15 = nn.ReLU(inplace=True)
        # 64 x 64

        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        # 128 x 128

        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(128, 128, 3, 1, 0)
        self.relu16 = nn.ReLU(inplace=True)
        # 128 x 128

        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(128, 64, 3, 1, 0)
        self.relu17 = nn.ReLU(inplace=True)
        # 128 x 128

        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        # 256 x 256

        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(64, 64, 3, 1, 0)
        self.relu18 = nn.ReLU(inplace=True)
        # 256 x 256

        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x):
        # 128x128x128
        out = self.reflecPad11(x)
        out = self.conv11(out)
        out = self.relu11(out)

        out = self.unpool(out)

        out = self.reflecPad12(out)
        out = self.conv12(out)
        out = self.relu12(out)
        out = self.reflecPad13(out)
        out = self.conv13(out)
        out = self.relu13(out)
        out = self.reflecPad14(out)
        out = self.conv14(out)
        out = self.relu14(out)
        out = self.reflecPad15(out)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool2(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.unpool3(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        return out


class Encoder_video(nn.Module):
    """For Content Enc."""

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_output):
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)
        return enc_output


class Encoder_image(nn.Module):
    """For Style Enc."""

    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_output):
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output)
        return enc_output


class Encoder_cross(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer_cross(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, dec_output, enc_output):
        # print(f"{dec_output.shape=}")
        # print(f"{enc_output.shape=}")
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(dec_output, enc_output)
        return enc_output


class Decoder_style_tranfer(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout=0.1):
        super().__init__()

        self.layer_stack = nn.ModuleList(
            [
                DecoderLayer_style(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, dec_output, enc_output):
        # print(f"{dec_output.shape=}")
        # print(f"{enc_output.shape=}")
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_output)

        _, l, _ = dec_output.size()
        dec_output = rearrange(
            dec_output, "b (m1 m2) c -> b c m1 m2", m1=int(math.sqrt(l))
        )

        return dec_output


class Transformer(nn.Module):
    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        d_model=512,
        d_inner=4096,
        n_layers=3,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder_frames = Encoder_video(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=2,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.encoder_style = Encoder_image(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=1,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.image_cross_video = Encoder_cross(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=1,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.video_cross_image = Encoder_cross(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=1,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.style_decoder = Decoder_style_tranfer(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.encoder = Encoder_vgg()

        self.decoder = Decoder_vgg()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        print("Loading pretrained AutoEncoder")
        encoder_ckpt = torch.load(encoder_path, map_location="cpu")
        decoder_ckpt = torch.load(decoder_path, map_location="cpu")
        self.encoder.load_state_dict(encoder_ckpt, strict=True)
        self.decoder.load_state_dict(decoder_ckpt, strict=True)

    def forward(
        self,
        content,
        style_images,
        content_type="image",
        id_loss="transfer",
        tab=None,
    ):
        if id_loss == "transfer":
            b, t, c, h, w = content.size()
            content = content.view(b * t, c, h, w)
            enc_content = self.encoder(content)
            enc_content = rearrange(enc_content, "bt c h w ->bt (h w) c")
            enc_content = self.encoder_frames(enc_content)
            if content_type == "image":
                content_self_attn = self.image_cross_video(enc_content, enc_content)
            elif content_type == "video":
                content_self_attn = self.video_cross_image(enc_content, enc_content)
            enc_style = self.encoder.forward(style_images)
            enc_style = rearrange(enc_style, "b c h w ->b (h w) c")
            enc_style_image = self.encoder_style.forward(enc_style)
            dec_output = self.style_decoder.forward(content_self_attn, enc_style_image)
            style_result = self.decoder.forward(dec_output)
            return style_result
