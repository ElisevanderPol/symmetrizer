import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from symmetrizer.groups.groups import MatrixRepresentation
from symmetrizer.ops.ops import get_basis, get_coeffs, get_invariant_basis, \
    compute_gain


class BasisLayer(torch.nn.Module):
    """
    Linear layer forward pass
    """
    def forward(self, x):
        """
        Normal forward pass, using weights formed by the basis
        and corresponding coefficients
        """
        self.W = torch.sum(self.basis*self.coeffs, 0)

        x = x[:, None, None, :, :]
        self.W = self.W[None, :, :, :, :]
        wx = self.W * x
        out = torch.sum(wx, [-2, -1])
        if self.has_bias:
            self.b = torch.sum(self.basis_bias*self.coeffs_bias, 0)
            return out + self.b
        else:
            return out


class BasisConvolutionalLayer(torch.nn.Module):
    """
    """
    def forward(self, x):
        """
        Normal convolutional forward pass, using weights formed by the basis
        and corresponding coefficients
        """
        self.W = torch.sum(self.basis*self.coeffs, 0)

        self.W = torch.reshape(self.W, [self.channels_out*self.repr_size_out,
                                        self.channels_in*self.repr_size_in,
                                        self.fx, self.fy])
        out = F.conv2d(x, self.W, stride=self.stride, padding=self.padding,
                       bias=None)
        if self.has_bias:
            self.b = torch.sum(self.basis_bias*self.coeffs_bias, 0)
            self.b = torch.reshape(self.b, [1, self.channels_out*self.repr_size_out, 1, 1])
            return out + self.b
        else:
            return out



class BasisLinear(BasisLayer):
    """
    Group-equivariant linear layer
    """
    def __init__(self, channels_in, channels_out, group, bias=True,
                 n_samples=4096, basis="equivariant", gain_type="xavier",
                 bias_init=False):
        """
        """
        super().__init__()

        self.group = group
        self.space = basis
        self.repr_size_in = group.repr_size_in
        self.repr_size_out = group.repr_size_out
        self.channels_in = channels_in
        self.channels_out = channels_out

        ### Getting Basis ###
        size = (n_samples, self.repr_size_out, self.repr_size_in)
        new_size = [1, self.repr_size_out, 1, self.repr_size_in]
        basis, self.rank = get_basis(size, group, new_size, space=self.space)
        self.register_buffer("basis", basis)

        gain = compute_gain(gain_type, self.rank, self.channels_in,
                            self.channels_out, self.repr_size_in,
                            self.repr_size_out)

        ### Getting Coefficients ###
        size = [self.rank, self.channels_out, 1, self.channels_in, 1]
        self.coeffs = get_coeffs(size, gain)

        ### Getting bias basis and coefficients ###
        self.has_bias = False
        if bias:
            self.has_bias = True
            if not bias_init:
                gain = 1
            size = [n_samples, self.repr_size_out, 1]
            new_size = [1, self.repr_size_out]
            basis_bias, self.rank_bias = get_invariant_basis(size, group,
                                                             new_size,
                                                             space=self.space)

            self.register_buffer("basis_bias", basis_bias)

            size = [self.rank_bias, self.channels_out, 1]
            self.coeffs_bias = get_coeffs(size, gain=gain)


    def __repr__(self):
        string = f"{self.space} Linear({self.repr_size_in}, "
        string += f"{self.channels_in}, {self.repr_size_out}, "
        string += f"{self.channels_out}), bias={self.has_bias})"
        return string


class BasisConv2d(BasisConvolutionalLayer):
    """
    Convolutional layer for groups
    """
    def __init__(self, channels_in, channels_out, group, filter_size=(3,3),
                 bias=True, n_samples=4096, gain_type="he",
                 basis="equivariant", stride=1, padding=0, first_layer=False):
        super().__init__()
        self.group = group
        self.space = basis
        self.stride = stride
        self.padding = padding


        if first_layer:
            self.repr_size_in = 1
        else:
            self.repr_size_in = group.num_elements
        self.repr_size_out = group.num_elements
        self.fx, self.fy = filter_size

        self.channels_in = channels_in
        self.channels_out = channels_out

        size = [n_samples, self.repr_size_out, self.repr_size_in, self.fx, self.fy]
        new_size = [1, self.repr_size_out, 1, self.repr_size_in, self.fx, self.fy]
        basis, self.rank = get_basis(size, group, new_size, space=self.space)
        self.register_buffer("basis", basis)

        gain = compute_gain(gain_type, self.rank, self.channels_in,
                            self.channels_out, self.repr_size_in,
                            self.repr_size_out)

        ### Getting Coefficients ###
        size = [self.rank, self.channels_out, 1, self.channels_in, 1, 1, 1]
        self.coeffs = get_coeffs(size, gain)

        self.has_bias = False
        if bias:
            self.has_bias = True
            size = [n_samples, self.repr_size_out, 1]
            new_size = [1, self.repr_size_out]
            basis_bias, self.rank_bias = get_invariant_basis(size, group,
                                                             new_size,
                                                             space=self.space)
            self.register_buffer("basis_bias", basis_bias)
            size = [self.rank_bias, self.channels_out, 1]
            self.coeffs_bias = get_coeffs(size, gain=gain)


    def __repr__(self):
        repr_str = f"{self.space} Conv2d("
        repr_str += f"{self.repr_size_in}, {self.channels_in}, "
        repr_str += f"{self.repr_size_out}, {self.channels_out}, "
        repr_str += f"kernel_size=({self.fx}, {self.fy}), "
        repr_str += f"stride={self.stride}, "
        repr_str += f"padding={self.padding},"
        repr_str += f"bias={self.has_bias})"
        return repr_str


class GlobalMaxPool(nn.Module):
    """
    Max pooling in an equivariant network
    """
    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, x):
        """
        """
        mx = torch.max(torch.max(x, dim=-1, keepdim=True)[0], dim=-2,
                        keepdim=True)[0]
        return mx


class GlobalAveragePool(nn.Module):
    """
    Average pooling in an equivariant network
    """

    def __init__(self):
        """
        """
        super().__init__()

    def forward(self, x):
        """
        """
        avg = torch.mean(x, dim=[-2, -1], keepdim=True)
        return avg
