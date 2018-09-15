import argparse
from env import Game2048Env
from utils import parse_args
from train import pre_train
from numba import int64


def main():
    # Arguments
    args = parse_args()

    seed = args.seed
    pre_train_len = args.pretrain

    env = Game2048Env(4, seed)

    pre_train(env, pre_train_len)


if __name__ == "__main__":
    main()
