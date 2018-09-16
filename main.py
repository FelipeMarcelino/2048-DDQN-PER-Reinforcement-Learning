import argparse
from env import Game2048Env
from utils import parse_args
from train import pre_train
from numba import int64
from memory import Memory
from model import CNN_2048_MODEL


def main():
    # Arguments
    args = parse_args()

    seed = args.seed
    pre_train_len = args.pretrain
    size_board = args.size_board

    # Create enviroment
    env = Game2048Env(size_board, seed)

    # Create memory replay
    memory = Memory(size_board, pre_train_len)

    # Create model
    c_in_1 = c_in_2 = size_board * size_board
    c_out_1 = c_out_2 = 128
    model = CNN_2048_MODEL(c_in_1, c_in_2, c_out_1, c_out_2)

    # Pretrain
    pre_train(env, pre_train_len, memory, model, size_board)


if __name__ == "__main__":
    main()
