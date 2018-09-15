import argparse
from env import Game2048Env
from utils import parse_args
from train import pre_train
from numba import int64
from memory import Memory


def main():
    # Arguments
    args = parse_args()

    seed = args.seed
    pre_train_len = args.pretrain
    size_board = args.size_board

    env = Game2048Env(size_board, seed)

    memory = Memory(size_board, pre_train_len)
    pre_train(env, pre_train_len, memory)

    # b_idx, memory_batch, weights = memory.sample(5)


if __name__ == "__main__":
    main()
