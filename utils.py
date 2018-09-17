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
    parser.add_argument("--capacity", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--size_board", type=int, default=4)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=0.00025)
    parser.add_argument("--ep_update_target", type=int, default=10)
    parser.add_argument("--decay_rate", type=float, default=0.00005)
    parser.add_argument("--interval_mean", type=int, default=5)

    args = parser.parse_args()

    return args


def get_mean_interval(array, interval_mean):
    interval_mean_list = []
    for x in range(interval_mean):
        interval_mean_list.append(0)

    for i in range(len(array)):
        if i + interval_mean == len(array):
            break
        else:
            interval_mean_list.append(np.mean(array[i: interval_mean + i]))

    return interval_mean_list


def plot_info(
    total_steps_per_episode,
    total_rewards_per_episode,
    total_loss_per_episode,
    total_score_per_episode,
    interval_mean,
    episodes,
):

    interval_steps = get_mean_interval(total_steps_per_episode, interval_mean)
    plt.plot(range(episodes), total_steps_per_episode)
    plt.plot(range(episodes), interval_steps)
    plt.ylabel("Duração dos episódios")
    plt.xlabel("Episodes")
    plt.show()

    interval_rewards = get_mean_interval(total_rewards_per_episode, interval_mean)
    plt.plot(range(episodes), total_rewards_per_episode)
    plt.plot(range(episodes), interval_rewards)
    plt.ylabel("Reward")
    plt.xlabel("Episodes")
    plt.show()

    interval_score = get_mean_interval(total_score_per_episode, interval_mean)
    plt.plot(range(episodes), total_score_per_episode)
    plt.plot(range(episodes), interval_score)
    plt.ylabel("Score")
    plt.xlabel("Episodes")
    plt.show()

    interval_loss = get_mean_interval(total_loss_per_episode, interval_mean)
    plt.plot(range(episodes), total_loss_per_episode)
    plt.plot(range(episodes), interval_loss)
    plt.ylabel("Loss")
    plt.xlabel("Episodes")
    plt.show()
