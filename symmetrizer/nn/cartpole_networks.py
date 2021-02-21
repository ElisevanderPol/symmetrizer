import torch

from symmetrizer.groups import MatrixRepresentation
from symmetrizer.nn.modules import BasisLinear
from symmetrizer.ops import *


class BasisCartpoleNetworkWrapper(torch.nn.Module):
    """
    Wrapper for cartpole basis network
    """
    def __init__(self, input_size, hidden_sizes, basis="equivariant",
                 gain_type="xavier"):
        super().__init__()
        in_group = get_cartpole_state_group_representations()
        out_group = get_cartpole_action_group_representations()

        repr_in = MatrixRepresentation(in_group, out_group)
        repr_out = MatrixRepresentation(out_group, out_group)


        self.network = BasisCartpoleNetwork(repr_in, repr_out, input_size,
                                            hidden_sizes, basis=basis)

    def forward(self, state):
        """
        """
        return self.network(state)


class SingleBasisCartpoleLayer(torch.nn.Module):
    """
    Single layer for cartpole symmetries
    """
    def __init__(self, input_size, output_size, basis="equivariant",
                 gain_type="xavier", **kwargs):
        super().__init__()
        in_group = get_cartpole_state_group_representations()
        out_group = get_cartpole_action_group_representations()

        repr_in = MatrixRepresentation(in_group, out_group)


        self.fc1 = BasisLinear(input_size, output_size, group=repr_in,
                               basis=basis, gain_type=gain_type,
                               bias_init=False)

    def forward(self, state):
        """
        """
        return self.fc1(state.unsqueeze(1))


class BasisCartpoleLayer(torch.nn.Module):
    """
    """
    def __init__(self, input_size, output_size, basis="equivariant",
                 out="equivariant", gain_type="xavier"):
        """
        Wrapper for single layer with cartpole symmetries, allows
        invariance/equivariance switch. Equivariance is for regular layers,
        invariance is needed for the value network output
        """
        super().__init__()
        if out == "equivariant":
            out_group = get_cartpole_action_group_representations()
            repr_out = MatrixRepresentation(out_group, out_group)
        elif out == "invariant":
            in_group = get_cartpole_action_group_representations()
            out_group = get_cartpole_invariants()
            repr_out = MatrixRepresentation(in_group, out_group)

        self.fc1 = BasisLinear(input_size, output_size, group=repr_out,
                               basis=basis, gain_type=gain_type,
                               bias_init=False)

    def forward(self, state):
        """
        """
        return self.fc1(state.unsqueeze(1))

class BasisCartpoleNetwork(torch.nn.Module):
    """
    """
    def __init__(self, repr_in, repr_out, input_size, hidden_sizes,
                 basis="equivariant", gain_type="xavier"):
        """
        Construct basis network for cartpole
        """
        super().__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        in_out_list = zip(hidden_sizes[:-1], hidden_sizes[1:])
        input_layer = BasisLinear(input_size, hidden_sizes[0], group=repr_in,
                                  basis=basis, gain_type=gain_type,
                                  bias_init=False)

        hidden_layers = [BasisLinear(n_in, n_out, group=repr_out,
                                     gain_type=gain_type, basis=basis,
                                     bias_init=False)
                         for n_in, n_out in in_out_list]

        sequence = list()
        sequence.extend([input_layer, torch.nn.ReLU()])
        for layer in hidden_layers:
            sequence.extend([layer, torch.nn.ReLU()])

        self.model = torch.nn.Sequential(*sequence)


    def forward(self, state):
        """
        """
        out = self.model(state.unsqueeze(1))
        return out

