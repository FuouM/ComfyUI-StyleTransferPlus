import torch
from torch import nn

from . import networks


class AesFA_test(nn.Module):
    def __init__(
        self, alpha_in, alpha_out, style_kernel, input_nc, nf, output_nc, freq_ratio
    ):
        super(AesFA_test, self).__init__()

        self.netE = networks.define_network(
            "Encoder",
            alpha_in,
            alpha_out,
            style_kernel,
            input_nc,
            nf,
            output_nc,
            freq_ratio,
        )
        self.netS = networks.define_network(
            "Encoder",
            alpha_in,
            alpha_out,
            style_kernel,
            input_nc,
            nf,
            output_nc,
            freq_ratio,
        )
        self.netG = networks.define_network(
            "Generator",
            alpha_in,
            alpha_out,
            style_kernel,
            input_nc,
            nf,
            output_nc,
            freq_ratio,
        )

    def forward(self, real_A, real_B):
        with torch.no_grad():
            content_A = self.netE.forward_test(real_A, "content")
            style_B = self.netS.forward_test(real_B, "style")
            # if freq:
            #     trs_AtoB, trs_AtoB_high, trs_AtoB_low = self.netG(content_A, style_B)
            #     end = time.time()
            #     during = end - start
            #     return trs_AtoB, trs_AtoB_high, trs_AtoB_low, during
            # else:
            trs_AtoB = self.netG.forward_test(content_A, style_B)
            return trs_AtoB

    def style_blending(self, real_A, real_B_1, real_B_2):
        with torch.no_grad():
            content_A = self.netE.forward_test(real_A, "content")
            style_B1_h = self.netS.forward_test(real_B_1, "style")[0]
            style_B2_l = self.netS.forward_test(real_B_2, "style")[1]
            style_B = style_B1_h, style_B2_l

            trs_AtoB = self.netG.forward_test(content_A, style_B)

        return trs_AtoB
