import numpy as np


class Game2048:
    def __init__(self, size_board, seed):
        self.__size_board = size_board
        self.__seed = seed
        self.__board = self.__init_board()

    def __init_board(self):
        return np.zeros((self.__size_board, self.__size_board))

    def get_board(self):
        return self.__board
