import torch.nn as nn
from module.sttr import STTR
from module.pcwnet import PCWNet_uncertainty


class ELFNet(nn.Module):
    def __init__(self, args):
        super(ELFNet, self).__init__()
        self.args = args
        self.pcwnet = PCWNet_uncertainty(args.maxdisp)
        self.sttr = STTR(args)

    def moe_nig(self, u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
        la = la1 + la2
        u = (la1 * u1 + u2 * la2) / la
        alpha = alpha1 + alpha2 + 0.5
        beta = beta1 + beta2 + 0.5 * \
            (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
        return u, la, alpha, beta

    def compute_uncertainty(self, u, la, alpha, beta):
        aleatoric = beta / (alpha - 1)
        epistemic = beta / (alpha - 1) / la
        return aleatoric, epistemic

    def forward(self, inputs):
        output = self.sttr(inputs)

        (disp_finetune, la, alpha, beta) = self.pcwnet(inputs)
        output['disp_pcw'] = disp_finetune
        output['la_pcw'] = la
        output['alpha_pcw'] = alpha
        output['beta_pcw'] = beta

        output['disp_combine'], output['la_combine'], output['alpha_combine'], output['beta_combine'] = \
            self.moe_nig(disp_finetune, la, alpha, beta,
                         output['disp_pred'], output['la'], output['alpha'], output['beta'])

        output['aleatoric'], output['epistemic'] = \
            self.compute_uncertainty(
                output['disp_combine'], output['la_combine'], output['alpha_combine'], output['beta_combine'])
        return output
