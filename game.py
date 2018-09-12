import numpy as np
import random
from copy import deepcopy


class Game2048:
    """Class responsible for game engine."""

    def __init__(self, size_board, seed=None):

        self.__size_board = size_board
        self.__seed = seed
        self.__board = self.__init_board()
        self.__total_score = 0
        self.__merged = 0
        self.__scores_move = 0
        self.__temp_board = np.zeros((size_board, size_board))
        self.__valid_movements = []

        if self.__seed:
            random.seed(self.__seed)

        self.__add_two_or_four()
        self.__add_two_or_four()

    def __init_board(self):
        return np.zeros((self.__size_board, self.__size_board))

    def __get_empty_spaces_index(self):
        # Return empty index
        return np.where(self.__board == 0)

    def __add_two_or_four(self):
        """Add number two or four in board"""

        # Get avaliable index
        indexes = self.__get_empty_spaces_index()

        # Select one indec
        index = np.random.choice(range(len(indexes[0])))

        # Select two or four
        sample = np.random.rand(1)

        if sample > 0.9:
            self.__board[indexes[0][index]][indexes[1][index]] = 4
        else:
            self.__board[indexes[0][index]][indexes[1][index]] = 2

    def __merge(self, array, reverse):
        # Merges two cells in one by sum

        # Get cells that are not zero
        array = array[array != 0]

        temp_array = np.zeros(self.__size_board)
        if reverse:
            count_index = self.__size_board - 1
        else:
            count_index = 0
        i = 0
        while True:
            # Verify limits of board
            if i >= (len(array) - 1):
                if i == (len(array) - 1):
                    temp_array[count_index] = array[i]
                return temp_array

            # Verify if cells side by side is equal
            if (array[i] == array[i + 1]) and array[i] != 0:
                temp_array[count_index] = array[i] + array[i + 1]
                self.__scores_move += temp_array[count_index]
                self.__merged += 1
                i = i + 2
                if reverse:
                    count_index -= 1
                else:
                    count_index += 1
            else:
                if array[i] != 0:
                    temp_array[count_index] = array[i]
                    if reverse:
                        count_index -= 1
                    else:
                        count_index += 1
                i = i + 1

    def __up(self):
        self.__temp_board = np.zeros((self.__size_board, self.__size_board))
        for column in range(self.__size_board):
            self.__temp_board[:, column] = self.__merge(
                self.__board[:, column].copy(), False
            )

    def __down(self):
        self.__temp_board = np.zeros((self.__size_board, self.__size_board))
        for column in range(self.__size_board):
            self.__temp_board[:, column] = self.__merge(
                self.__board[:, column].copy(), True
            )

    def __right(self):
        self.__temp_board = np.zeros((self.__size_board, self.__size_board))
        for line in range(self.__size_board):
            self.__temp_board[line, :] = self.__merge(
                self.__board[line, :].copy(), True
            )

    def __left(self):
        self.__temp_board = np.zeros((self.__size_board, self.__size_board))
        for line in range(self.__size_board):
            self.__temp_board[line, :] = self.__merge(
                self.__board[line, :].copy(), False
            )

    def __check_available_moves(self):
        """Verify valid movements"""
        self.__valid_movements = []
        for i in range(4):
            self.make_move(i)
            if np.array_equal(self.__board, self.__temp_board) is False:
                self.__valid_movements.append(i)

        return self.__valid_movements

    def make_move(self, move):
        """Select one move"""
        self.__merged = 0
        self.__scores_move = 0
        if move == 0:
            self.__up()
        if move == 1:
            self.__down()
        if move == 2:
            self.__right()
        if move == 3:
            self.__left()

    def confirm_move(self):
        """Execute movement and get info about merges and scores"""
        self.__board = deepcopy(self.__temp_board)
        self.__total_score += self.__scores_move
        returned_move_scores = self.__scores_move
        returned_merged = self.__merged
        self.__valid_movements = self.__check_available_moves()

        if len(self.__valid_movements) != 0:
            self.__add_two_or_four()

        return returned_move_scores, returned_merged, self.__valid_movements

    def get_board(self):
        """Return the board of the game."""
        return self.__board

    def get_total_score(self):
        """Return total score for this episode"""
        return self.__total_score

    def reset(self):
        """Reset game board and total score"""
        self.__board = self.__init_board()
        self.__total_score = 0
        self.__add_two_or_four()
        self.__add_two_or_four()
        return self.get_board()
