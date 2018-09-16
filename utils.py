import numpy as np
import torch
import argparse
import sys
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython=True)
def to_power_two_matrix(matrix):
    """Transform matrix to a power 2 matrix. Maximum value: 65566"""
    power_matrix = np.zeros(
        shape=(1, matrix.shape[0], matrix.shape[1], 16), dtype=np.float32
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] == 0:
                power_matrix[0][i][j][0] = 1.0
            else:
                power = int(np.log(matrix[i][j]) / np.log(2))
                power_matrix[0][i][j][power] = 1.0

    return power_matrix


def selection_action(
    eps_threshold, valid_movements, dqn_net, state, size_board, device
):
    sample = np.random.rand(1)

    if sample > eps_threshold:
        with torch.no_grad():
            output = dqn_net(torch.from_numpy(state).double().to(device), 1, size_board)
            output_np = output.cpu().detach().numpy()
            ordered = np.flip(np.argsort(output_np), axis=1)[0]
            for x in ordered:
                if valid_movements[x] != 0:
                    return x

    else:
        return np.random.choice(np.nonzero(valid_movements)[0])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--pretrain", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--size_board", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.00025)
    parser.add_argument("--ep_update_target", type=int, default=10)
    parser.add_argument("--decay_rate", type=float, default=0.00005)

    args = parser.parse_args()

    return args


def plot_info(
    total_steps_per_episode, total_rewards_per_episode, total_loss_per_episode, episodes
):

    plt.plot(range(episodes), total_steps_per_episode)
    plt.ylabel("Duração dos episódios")
    plt.xlabel("Episódios")
    plt.show()

    plt.plot(range(episodes), total_rewards_per_episode)
    plt.ylabel("Reward")
    plt.xlabel("Episódios")
    plt.show()

    plt.plot(range(episodes), total_loss_per_episode)
    plt.ylabel("Loss")
    plt.xlabel("Episódios")
    plt.show()
