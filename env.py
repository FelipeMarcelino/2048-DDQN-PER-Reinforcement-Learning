import gym
import numpy as np
import sys
import math
from six import StringIO
from game import Game2048
from gym import spaces
from gym.utils import seeding


class InvalidMove(Exception):
    pass


class Game2048Env(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, size_board, seed=None):
        self.__size_board = size_board
        self.__game = Game2048(size_board, seed)

        # Numbers of possible movements
        self.action_space = spaces.Discrete(4)

        # Numbers of observations
        self.observation_space = spaces.Box(
            0, 2 ** 16, (size_board * size_board,), dtype=np.int
        )

        # Reward range
        self.reward_range = (0., np.inf)

        # Initialise seed
        self.np_random, seed = seeding.np_random(seed)

        # Legends
        self.__actions_legends = {0: "UP", 1: "DOWN", 2: "RIGHT", 3: "LEFT"}

        # Old max
        self.__old_max = 0

        # Debug
        self.__last_action = None
        self.__last_scores_move = None

        print("Environment initialised...")

    def __reward_calculation(self, merged, reward):
        max_board = self.__game.get_board().max()
        if max_board > self.__old_max:
            self.__old_max = max_board
            reward += math.log(self.__old_max, 2) * 0.1

        reward += merged

    def reset(self):
        """Reset the game"""
        self.__game.reset()
        print("Game reset...")
        valid_movements = np.ones(4)
        return (self.__game.get_board(), valid_movements)

    def step(self, action):
        print("The enviroment will take a action:", self.__actions_legends[action])
        done = False
        reward = 0
        try:
            self.__last_action = self.__actions_legends[action]

            self.__game.make_move(action)
            returned_move_scores, returned_merged, valid_movements = (
                self.__game.confirm_move()
            )

            self.__reward_calculation(returned_merged, reward)

            if len(np.nonzero(valid_movements)[0]) == 0:
                done = True

            self.__last_scores_move = returned_move_scores

            info = dict()
            info["valid_movements"] = valid_movements
            info["total_score"] = self.__game.get_total_score()
            info["last_action"] = self.__actions_legends[action]
            info["scores_move"] = returned_move_scores
            return self.__game.get_board(), reward, done, info

        except InvalidMove as e:
            print("Invalid move")
            done = False
            reward = 0

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout
        info_render = "Score: {}\n".format(self.__game.get_total_score())
        info_render += "Highest: {}\n".format(self.__game.get_board().max())
        npa = np.array(self.__game.get_board())
        grid = npa.reshape((self.__size_board, self.__size_board))
        info_render += "{}\n".format(grid)
        info_render += "Last action: {}\n".format(self.__last_action)
        info_render += "Last scores move: {}".format(self.__last_scores_move)
        info_render += "\n"
        outfile.write(info_render)
        return outfile

    def get_actions_legends(self):
        return self.__actions_legends
