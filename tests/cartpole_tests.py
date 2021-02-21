import numpy as np
import gym

import torch
import torch.nn.functional as F

from symmetrizer.ops import *
from symmetrizer.nn.cartpole_networks import BasisCartpoleNetworkWrapper, \
    BasisCartpoleLayer, SingleBasisCartpoleLayer
from symmetrizer.groups.ops import closed_group



def test_prebuilt_policy():
    """
    Test if the CartPole policy network is equivariant
    """
    env = gym.make("CartPole-v1")
    x = env.reset()

    network = BasisCartpoleNetworkWrapper(1, [64, 1])

    representations = get_cartpole_state_group_representations()
    out_group = get_cartpole_action_group_representations()

    p_m2 = representations[1].detach().cpu().numpy()

    t_x = torch.Tensor(x).unsqueeze(0)
    t_p_x1 = torch.Tensor(np.matmul(x, p_m2)).unsqueeze(0)

    out_x = network(t_x)
    out_p_x1 = network(t_p_x1)

    out_tpx = [out_x, out_p_x1]

    assert_permutations(out_group, out_tpx)
    return True


def test_value_network():
    """
    Test if the CartPole value network is invariant
    """
    network = BasisCartpoleNetworkWrapper(1, [64, 64])
    layer = BasisCartpoleLayer(64, 1, out="invariant")

    representations = get_cartpole_state_group_representations()

    p_m2 = representations[1].detach().cpu().numpy()

    env = gym.make("CartPole-v1")
    x = env.reset()

    t_x = torch.Tensor(x).unsqueeze(0)
    t_p_x1 = torch.Tensor(np.matmul(x, p_m2)).unsqueeze(0)

    out_x = layer(F.relu(network(t_x)))
    out_p_x1 = layer(F.relu(network(t_p_x1)))

    out_tpx = [out_x, out_p_x1]

    assert_similar(out_tpx)
    return True



def test_group():
    """
    Test if the group is closed
    """
    group_repr = get_cartpole_state_group_representations()
    is_closed = closed_group(group_repr.representations)
    assert(is_closed)
    group_repr = get_cartpole_action_group_representations()
    is_closed = closed_group(group_repr.representations)
    assert(is_closed)
    return True


def test_single_layer():
    """
    Test if a single CartPole layer is equivariant
    """
    network = SingleBasisCartpoleLayer(1, 1)

    representations = get_cartpole_state_group_representations()
    out_group = get_cartpole_action_group_representations()

    p_m2 = representations[1].detach().cpu().numpy()

    env = gym.make("CartPole-v1")
    x = env.reset()

    t_x = torch.Tensor(x).unsqueeze(0)
    t_p_x1 = torch.Tensor(np.matmul(p_m2, x)).unsqueeze(0)

    out_x = network(t_x)
    out_p_x1 = network(t_p_x1)

    out_tpx = [out_x, out_p_x1]

    assert_permutations(out_group, out_tpx)
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


def assert_permutations(permutations, out_x):
    """
    Assert to test if equivariance holds wrt permutations
    """
    x = out_x[0]
    for i in range(len(permutations)):
        p_i = torch.FloatTensor(permutations[i])
        x_i = out_x[i]
        pi_x = torch.matmul(x, p_i)
        assert(torch.allclose(pi_x, x_i))
    return True


if __name__ == "__main__":
    print("="*25)
    print("Running CartPole equivariance/invariance tests...")
    print("="*25)
    print("Testing if group is closed..")
    success = test_group()
    print("Test passed:", success)
    print("\nTesting single linear layer...")
    success = test_single_layer()
    print("Test passed:", success)
    print("\nTesting prebuilt multilayer policy...")
    success = test_prebuilt_policy()
    print("Test passed:", success)
    print("\nTesting prebuilt value network...")
    success = test_value_network()
    print("Test passed:", success)
    print("="*25)
