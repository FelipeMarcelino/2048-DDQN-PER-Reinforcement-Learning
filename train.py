import torch
import torch.nn as nn
from utils import to_power_two_matrix, selection_action


def optimize_model():
    pass


def pre_train(env, pre_train_len):
    board, valid_movements = env.reset()
    state = to_power_two_matrix(board)

    # Only random choice
    eps_threshold = 1

    for i in range(pre_train_len):

        # Random action
        action = selection_action(eps_threshold, valid_movements, None, state)

        # Get the rewards
        new_board, reward, done, info = env.step(action)

        # If doesnt't have any movement more
        if done:

            # We finished the episode
            next_state = np.zeros(state.shape)

            experience = state, action, reward, next_state, done
            # memory.store(experience)

            # Start a new episode
            board, valid_movements = env.reset()

        else:
            # Get the next state
            next_state = to_power_two_matrix(new_board)

            # Add experience to memory
            experience = state, action, reward, next_state, done
            # memory.store(experience)

            # Our state is now the next_state
            state = next_state

            # Valid movements
            valid_movements = info["valid_movements"]


def train(model):
    pass
