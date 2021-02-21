import numpy as np
import torch
import torch.nn.functional as F

from .ops import GroupRepresentations


def get_cartpole_state_group_representations():
    """
    Representation of the group symmetry on the state: a multiplication of all
    state variables by -1
    """
    representations = [torch.FloatTensor(np.eye(4)),
                       torch.FloatTensor(-1*np.eye(4))]
    return GroupRepresentations(representations, "CartPoleStateGroupRepr")


def get_cartpole_action_group_representations():
    """
    Representation of the group symmetry on the policy: a permutation of the
    actions
    """
    representations = [torch.FloatTensor(np.eye(2)),
                       torch.FloatTensor(np.array([[0, 1], [1, 0]]))]
    return GroupRepresentations(representations, "CartPoleActionGroupRepr")


def get_cartpole_invariants():
    """
    Function to enable easy construction of invariant layers (for value
    networks)
    """
    representations = [torch.FloatTensor(np.eye(1)),
                       torch.FloatTensor(np.eye(1))]
    return GroupRepresentations(representations, "CartPoleInvariantGroupRepr")

