import torch
import torchvision.transforms as transforms

from .model import AesFA_test


class MODEL_AESFA:
    def __init__(self, checkpoint_path: str, device) -> None:
        self.aesfa_model = AesFA_test(
            alpha_in=0.5,  # input ratio of low-frequency channel
            alpha_out=0.5,  # output ratio of low-frequency channel
            style_kernel=3,  # size of style kernel
            input_nc=3,  # of input image channel
            nf=64,  # of feature map channel after Encoder first layer
            output_nc=3,  # of output image channel
            freq_ratio=[1, 1],  # [high, low] ratio at the last layer
        )

        dict_model = torch.load(checkpoint_path)

        self.aesfa_model.netE.load_state_dict(dict_model["netE"])
        self.aesfa_model.netS.load_state_dict(dict_model["netS"])
        self.aesfa_model.netG.load_state_dict(dict_model["netG"])

        self.aesfa_model.to(device)


def inference_aesfa(
    src_img: torch.Tensor,
    style_img: torch.Tensor,
    device,
    size: int,
    do_crop: bool,
    model: MODEL_AESFA,
):
    content_tf = test_transform(
        size, do_crop, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    content = content_tf(src_img).to(device)
    style = content_tf(style_img).to(device)

    stylized_image = model.aesfa_model.forward(content, style)
    output = un_normalize_batch(stylized_image)

    return output


def inference_aesfa_style_blend(
    src_img: torch.Tensor,
    style_hi: torch.Tensor,
    style_lo: torch.Tensor,
    device,
    size: int,
    do_crop: bool,
    model: MODEL_AESFA,
):
    content_tf = test_transform(
        size, do_crop, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    content = content_tf(src_img).to(device)
    style_high = content_tf(style_hi).to(device)
    style_low = content_tf(style_lo).to(device)

    stylized_image = model.aesfa_model.style_blending(content, style_high, style_low)
    output = un_normalize_batch(stylized_image)

    return output


def test_transform(size, crop, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    transform_list = []

    if crop and size != 0:
        transform_list.append(transforms.Resize(size))
        transform_list.append(transforms.CenterCrop(size))
    elif size != 0:
        transform_list.append(transforms.Resize((size, size)))

    transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)


def un_normalize_batch(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
