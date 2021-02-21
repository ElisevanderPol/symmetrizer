import numpy as np
from symmetrizer.ops import *

class Group:
    """
    Abstract group class
    """
    def __init__(self):
        """
        Set group parameters
        """
        raise NotImplementedError

    def _input_transformation(self, weights, transformation):
        """
        Specify input transformation
        """
        raise NotImplementedError

    def _output_transformation(self, weights, transformation):
        """
        Specify output transformation
        """
        raise NotImplementedError


class MatrixRepresentation(Group):
    """
    Representing group elements as matrices
    """
    def __init__(self, input_matrices, output_matrices):
        """
        """
        self.repr_size_in = input_matrices[0].shape[1]
        self.repr_size_out = output_matrices[0].shape[1]
        self._input_matrices = input_matrices
        self._output_matrices = output_matrices

        self.parameters = range(len(input_matrices))

    def _input_transformation(self, weights, params):
        """
        Input transformation comes from the input group
        W F_g z
        """
        weights = np.matmul(weights, self._input_matrices[params])
        return weights

    def _output_transformation(self, weights, params):
        """
        Output transformation from the output group
        P_g W z
        """
        weights = np.matmul(self._output_matrices[params], weights)
        return weights


class P4(Group):
    """
    Group of 90 degree rotations
    """
    def __init__(self):
        """
        """
        self.parameters = [i for i in range(4)]
        self.num_elements = 4
        self.__name__ = "P4 Group"

    def _input_transformation(self, weights, angle):
        """
        """
        weights = np.rot90(weights, k=angle, axes=(3, 4))
        weights = np.roll(weights, angle, axis=2)
        return weights

    def _output_transformation(self, weights, angle):
        """
        """
        weights = np.roll(weights, angle, axis=1)
        return weights


class P4Intermediate(Group):
    """
    P4 group representation
    in=intermediate permutations
    out=intermediate permutations
    """
    def __init__(self):
        """
        """
        self.parameters = [0, 1, 2, 3]
        self.num_elements = len(self.parameters)
        self.repr_size_out = 4
        self.repr_size_in = 4
        self.__name__ = "P4 Permutations"
        self.in_permutations = get_grid_rolls()
        self.out_permutations = get_grid_rolls()


    def _input_transformation(self, weights, g):
        permute = self.in_permutations[g]
        weights = np.matmul(weights, permute)
        return weights

    def _output_transformation(self, weights, g):
        """
        """
        permute = self.out_permutations[g]
        weights = np.matmul(permute, weights)
        return weights


class P4toOutput(Group):
    """
    P4 group representation
    in=intermediate permutations
    out=action permutations
    """
    def __init__(self):
        """
        """
        self.parameters = [0, 1, 2, 3]
        self.num_elements = len(self.parameters)
        self.repr_size_out = 5
        self.repr_size_in = 4
        self.__name__ = "P4 Horizontal"
        self.in_permutations = get_grid_rolls()
        self.out_permutations = get_grid_actions()

    def _input_transformation(self, weights, g):
        permute = self.in_permutations[g]
        weights = np.matmul(weights, permute)
        return weights


    def _output_transformation(self, weights, g):
        """
        """
        permute = self.out_permutations[g]
        weights = np.matmul(permute, weights)
        return weights


class P4toInvariant(Group):
    """
    P4 group representation
    in=intermediate permutations
    out=invariant representation
    """
    def __init__(self):
        """
        """
        self.parameters = [0, 1, 2, 3]
        self.num_elements = len(self.parameters)
        self.repr_size_out = 1
        self.repr_size_in = 4
        self.__name__ = "P4 inv"
        self.in_permutations = get_grid_rolls()
        self.out_permutations = [np.eye(1), np.eye(1), np.eye(1), np.eye(1)]

    def _input_transformation(self, weights, flip):
        permute = self.in_permutations[flip]
        weights = np.matmul(weights, permute)
        return weights


    def _output_transformation(self, weights, flip):
        """
        """
        permute = self.out_permutations[flip]
        weights = np.matmul(permute, weights)
        return weights

