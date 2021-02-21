import numpy as np
import torch

from .ops import GroupRepresentations



def get_grid_rolls():
    perm = np.zeros((4, 4))
    permutes = [[3, 0, 1, 2],
                [2, 3, 0, 1],
                [1, 2, 3, 0]]
    perms = [perm.copy() for p in permutes]
    for i, per in enumerate(perms):
        p = permutes[i]
        for j in range(4):
            per[j][p[j]] = 1

    representations = [torch.FloatTensor(np.eye(4)),
                       torch.FloatTensor(perms[0]),
                       torch.FloatTensor(perms[1]),
                       torch.FloatTensor(perms[2])]
    return GroupRepresentations(representations, "GridRolls")

def get_grid_actions():
    perm = np.zeros((5, 5))
    permutes = [[0, 4, 1, 2, 3],
                [0, 3, 4, 1, 2],
                [0, 2, 3, 4, 1]]
    perms = [perm.copy() for p in permutes]
    for i, per in enumerate(perms):
        p = permutes[i]
        for j in range(5):
            per[j][p[j]] = 1

    representations = [torch.FloatTensor(np.eye(5)),
                       torch.FloatTensor(perms[0]),
                       torch.FloatTensor(perms[1]),
                       torch.FloatTensor(perms[2])]
    return GroupRepresentations(representations, "GridActions")
