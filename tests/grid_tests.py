import numpy as np
import gym
import gridworld

import torch
import torch.nn.functional as F

from symmetrizer.nn.grid_networks import BasisGridLayer, BasisGridNetwork
from symmetrizer.ops import g2c, c2g, get_grid_rolls, get_grid_actions
from symmetrizer.groups.ops import closed_group


def test_prebuilt_network():
    """
    Test if the Grid policy network is equivariant
    """
    env = gym.make("GridEnv-v1")
    frame = env.reset()
    x = np.expand_dims(env.reset(), axis=0)
    network = BasisGridNetwork(1)
    out_group = get_grid_actions()

    t_x = torch.Tensor(x)
    t_r_x1 = torch.Tensor(np.rot90(x.copy(), k=1, axes=(2, 3)).copy())
    t_r_x2 = torch.Tensor(np.rot90(x.copy(), k=2, axes=(2, 3)).copy())
    t_r_x3 = torch.Tensor(np.rot90(x.copy(), k=3, axes=(2, 3)).copy())

    pi, val = network(t_x)
    pi2, val2 = network(t_r_x1)
    pi3, val3 = network(t_r_x2)
    pi4, val4 = network(t_r_x3)

    out_tpx = [pi.squeeze(), pi2.squeeze(), pi3.squeeze(), pi4.squeeze()]
    out_vals = [val, val2, val3, val4]

    assert_permutations(out_group, out_tpx)
    assert_similar(out_vals)
    return True


def test_singlelayer():
    """
    """
    env = gym.make("GridEnv-v1")
    frame = env.reset()
    x = np.expand_dims(env.reset(), axis=0)

    network = BasisGridLayer(1, 1)
    out_group = get_grid_rolls()

    t_x = torch.Tensor(x)
    t_r_x1 = torch.Tensor(np.rot90(x.copy(), k=1, axes=(2, 3)).copy())
    t_r_x2 = torch.Tensor(np.rot90(x.copy(), k=2, axes=(2, 3)).copy())
    t_r_x3 = torch.Tensor(np.rot90(x.copy(), k=3, axes=(2, 3)).copy())

    pi = network(t_x)
    pi2 = network(t_r_x1)
    pi3 = network(t_r_x2)
    pi4 = network(t_r_x3)

    out_tpx = [pi.squeeze(), pi2.squeeze(), pi3.squeeze(), pi4.squeeze()]
    assert_permutations(out_group, out_tpx)
    return True


def test_group():
    """
    Test if the group is closed
    """
    group_repr = get_grid_rolls()
    is_closed = closed_group(group_repr.representations)
    assert(is_closed)
    group_repr = get_grid_actions()
    is_closed = closed_group(group_repr.representations)
    assert(is_closed)
    return True


def assert_permutations(permutations, out_x):
    """
    """
    x = out_x[0]
    for i in range(len(permutations)):
        p_i = permutations[i].float()
        x_i = out_x[i]
        pi_x = torch.matmul(p_i, x)
        assert(torch.allclose(pi_x, x_i))
    return True


def assert_similar(out_x):
    """
    Assert to test if invariance holds
    """
    x = out_x[0]
    for i in range(len(out_x)):
        x_i = out_x[i]
        assert(torch.allclose(x, x_i))
    return True

if __name__ == "__main__":
    print("="*25)
    print("Running Grid World equivariance/invariance tests...")
    print("="*25)
    print("Testing if group is closed..")
    success = test_group()
    print("Test passed:", success)
    print("\nTesting single conv layer...")
    success = test_singlelayer()
    print("Test passed:", success)
    print("\nTesting prebuilt policy and value network...")
    success = test_prebuilt_network()
    print("Test passed:", success)
    print("="*25)
