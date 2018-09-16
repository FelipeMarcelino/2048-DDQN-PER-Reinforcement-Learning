import argparse
import time
from copy import deepcopy
from env import Game2048Env
from utils import parse_args
from train import pre_train, train
from numba import int64
from memory import Memory
from model import CNN_2048_MODEL


def main():
    # Arguments
    args = parse_args()

    seed = args.seed
    pre_train_len = args.pretrain
    size_board = args.size_board
    batch_size = args.batch_size
    episodes = args.num_episodes
    ep_update_target = args.ep_update_target
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    explore_start = 1.
    explore_stop = 0.01
    gamma = 0.95

    # Create enviroment
    env = Game2048Env(size_board, seed)

    # Create memory replay
    memory = Memory(size_board, pre_train_len)

    # Create model
    c_in_1 = c_in_2 = size_board * size_board
    c_out_1 = c_out_2 = 128
    dqn_net = CNN_2048_MODEL(c_in_1, c_in_2, c_out_1, c_out_2)
    target_net = deepcopy(dqn_net)

    start = time.time()
    # Pretrain
    pre_train(env, pre_train_len, memory)
    print("Execution pre-train (in seconds):", time.time() - start)

    start = time.time()
    # Train
    train(
        dqn_net,
        target_net,
        env,
        memory,
        batch_size,
        size_board,
        episodes,
        ep_update_target,
        decay_rate,
        explore_start,
        explore_stop,
        learning_rate,
        gamma,
    )
    print("Execution train (in seconds)", time.time() - start)


if __name__ == "__main__":
    main()
