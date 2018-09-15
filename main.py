import argparse
from env import Game2048Env
from utils import parse_args
from train import pre_train
from numba import int64
from memory import SumTree


def main():
    # Arguments
    args = parse_args()

    seed = args.seed
    pre_train_len = args.pretrain
    size_board = args.size_board

    env = Game2048Env(size_board, seed)

    # pre_train(env, pre_train_len)

    sum_tree = SumTree(10, size_board)


if __name__ == "__main__":
    main()
