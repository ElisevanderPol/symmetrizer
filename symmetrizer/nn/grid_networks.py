import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from symmetrizer.nn.modules import BasisConv2d, GlobalAveragePool, \
    BasisLinear, GlobalMaxPool
from symmetrizer.ops import c2g
from symmetrizer.groups import P4, P4Intermediate, P4toOutput, P4toInvariant


class BasisGridNetwork(torch.nn.Module):
    """
    """
    def __init__(self, input_size, hidden_sizes=[512], channels=[16, 32],
                 filters=[8,5], strides=[1, 1], paddings=[0, 0],
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        in_group = P4()

        out_1 = int(hidden_sizes[0] / np.sqrt(4))

        layers = []

        for l, channel in enumerate(channels):
            c = int(channel/np.sqrt(4))
            f = (filters[l], filters[l])
            s = strides[l]
            p = paddings[l]
            if l == 0:
                first_layer = True
            else:
                first_layer = False

            conv = BasisConv2d(input_size, c, filter_size=f, group=in_group,
                                gain_type=gain_type,
                                basis=basis,
                                first_layer=first_layer, padding=p,
                                stride=s)
            layers.append(conv)
            input_size = c
        self.convs = torch.nn.ModuleList(layers)

        self.pool = GlobalMaxPool()
        between_group = P4Intermediate()
        self.fc1 = BasisLinear(input_size, out_1, between_group,
                               gain_type=gain_type,
                               basis=basis, bias_init=True)
        out_group = P4toOutput()
        inv_group = P4toInvariant()
        self.fc4 = BasisLinear(out_1, 1, out_group, gain_type=gain_type,
                               basis=basis, bias_init=True)
        self.fc5 = BasisLinear(out_1, 1, inv_group,
                                gain_type=gain_type,
                                basis=basis, bias_init=True)

    def forward(self, state):
        """
        """
        outputs = []
        for i, c in enumerate(self.convs):
            outputs.append(state[0])
            state = F.relu(c(state))
        outputs.append(state[0])
        conv_output = state
        pool = c2g(self.pool(conv_output), 4).squeeze(-1).squeeze(-1)
        outputs.append(pool)
        fc_output = F.relu(self.fc1(pool))
        policy = self.fc4(fc_output)
        value = self.fc5(fc_output)
        return policy, value


class BasisGridLayer(torch.nn.Module):
    """
    """
    def __init__(self, input_size, output_size, filter_size=(3, 3),
                 gain_type='he', basis="equivariant", out="equivariant"):
        super().__init__()
        in_group = P4()

        self.fc1 = BasisConv2d(input_size, output_size,
                                   filter_size=filter_size, group=in_group,
                                   gain_type=gain_type,
                                   basis=basis,
                                   first_layer=True)
        self.pool = GlobalAveragePool()

    def forward(self, state):
        """
        """
        return c2g(self.pool(self.fc1(state)), 4)

