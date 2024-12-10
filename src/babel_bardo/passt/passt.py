import pathlib

import torch.nn as nn
import torch

from hear21passt.base import get_basic_model, get_model_passt


class PaSSTMTG(nn.Module):
    def __init__(self, n_classes=183, sigmoid=True):
        super(PaSSTMTG, self).__init__()

        self.sigmoid = sigmoid
        self.passt = get_basic_model(mode="logits")
        self.passt.net =  get_model_passt(arch="passt_s_swa_p16_128_ap476", n_classes=n_classes)

    def forward(self, x):
        passt_logit = self.passt(x)

        if self.sigmoid:
            return nn.Sigmoid()(passt_logit)

        return passt_logit

def get_passt(n_classes=183, sigmoid=True):
    passt = PaSSTMTG(n_classes, sigmoid)

    state_dict = pathlib.Path(__file__).parent.joinpath('passt_epoch_1_acc_0.975.pth').resolve()
    state_dict = torch.load(state_dict)

    passt.load_state_dict(state_dict)

    return passt