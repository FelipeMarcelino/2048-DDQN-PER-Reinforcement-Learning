import torch
import torch.nn as nn
import numpy as np
import torch
import sys
from copy import deepcopy
from utils import to_power_two_matrix, selection_action, to_power_two_matrix, plot_info
from memory import Transition
from torch import optim


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def optimize_model(
    dqn_net,
    target_net,
    memory,
    learning_rate,
    batch_size,
    size_board,
    gamma,
    optmizer,
    device,
):

    # Sample batch
    tree_indexes, memory_batch, batch_ISWeights = memory.sample(batch_size)
    # print(batch_ISWeights)

    samples = Transition(*zip(*memory_batch))

    states_batch = samples.state
    actions_batch = samples.action
    rewards_batch = samples.reward
    next_states_batch = samples.next_state
    dones_batch = samples.done

    target_qs_batch = []

    torch_next_states_batch = torch.from_numpy(np.asarray(next_states_batch)).to(device)

    # Get Q values for next state
    q_next_state = dqn_net(torch_next_states_batch, batch_size, size_board)

    # REMOVER detach depois e testar !!!!!!!!!!!
    q_target_next_state = (
        target_net(torch_next_states_batch, batch_size, size_board).cpu().detach()
    )

    for i in range(0, len(memory_batch)):
        terminal = dones_batch[i]

        # Get max action value index
        action = np.argmax(q_next_state[i].cpu().detach().numpy())

        # If we are in terminal state, only equals reward
        if terminal:
            target_qs_batch.append(rewards_batch[i])
        else:
            target = rewards_batch[i] + gamma * q_target_next_state[i][action]
            target_qs_batch.append(target)

    targets_batch = np.array([each for each in target_qs_batch])

    torch_states_batch = torch.from_numpy(np.asarray(states_batch)).to(device)

    output = dqn_net(torch_states_batch, batch_size, size_board)

    torch_actions_batch = torch.from_numpy(np.asarray(actions_batch))
    torch_actions_batch = torch_actions_batch.unsqueeze(0)
    torch_actions_batch = torch_actions_batch.view(batch_size, 1)

    # Q is our predicted Q value
    q_values = output.gather(1, torch_actions_batch.to(device))

    # Absolute error for update tree
    absolute_errors = (
        torch.abs(
            q_values - torch.from_numpy(targets_batch).view(batch_size, 1).to(device)
        )
        .cpu()
        .detach()
        .numpy()
    )

    torch_batch_ISWeights = torch.from_numpy(batch_ISWeights).double().to(device)

    # Mean squared error
    diff_target = q_values - torch.from_numpy(targets_batch).double().view(
        batch_size, 1
    ).to(device)
    squared_diff = diff_target ** 2
    weighted_squared_diff = squared_diff * torch_batch_ISWeights

    # Loss
    loss = torch.mean(weighted_squared_diff)

    # Optimization
    optmizer.zero_grad()

    loss.backward()
    optmizer.step()

    # Squeze absolute errors
    absolute_errors = np.squeeze(absolute_errors, 1)

    # Memory tree update
    memory.batch_update(tree_indexes, absolute_errors)

    return loss.cpu().detach().numpy()


def pre_train(env, pre_train_len, memory):
    print("Starting pretrain...")
    board, valid_movements = env.reset()
    state = to_power_two_matrix(board)

    # Only random choice
    eps_threshold = 1

    for i in range(pre_train_len):
        # env.render()

        # Random action
        action = selection_action(
            eps_threshold, valid_movements, None, None, None, None
        )

        # Get the rewards
        new_board, reward, done, info = env.step(action)

        # If doesnt't have any movement more
        if done:
            # print("Finished...")
            # env.render()

            # We finished the episode
            next_state = np.zeros(state.shape)

            # experience = state, action, reward, next_state, done
            memory.store(state, action, reward, next_state, done)

            # Start a new episode
            board, valid_movements = env.reset()

        else:
            # Get the next state
            next_state = to_power_two_matrix(new_board)
            # print(next_state)

            # Add experience to memory
            memory.store(state, action, reward, next_state, done)

            # Our state is now the next_state
            state = next_state

            # Valid movements
            valid_movements = info["valid_movements"]


def train(
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
    interval_mean,
):

    # Using GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn_net.to(device)
    target_net.to(device)

    print("Starting training...")
    decay_step = 0

    total_steps_per_episode = []
    total_rewards_per_episode = []
    total_loss_per_episode = []
    total_score_per_episode = []

    best_board = None
    best_score = 0
    best_steps = 0
    best_ep = -1

    # Optmizer
    optmizer = optim.RMSprop(dqn_net.parameters(), lr=learning_rate)

    for ep in range(episodes):
        # Set step to 0
        step = 0

        # Rewards of the episode
        episode_rewards = []
        board, valid_movements = env.reset()
        state = to_power_two_matrix(board)
        done = False
        loss_ep = []

        while True:
            step += 1

            # Increase decay step to choose net output instead random action
            decay_step += 1

            # Make a action
            eps_threshold = explore_stop + (explore_start - explore_stop) * np.exp(
                -decay_rate * decay_step
            )
            action = selection_action(
                eps_threshold, valid_movements, dqn_net, state, size_board, device
            )
            new_board, reward, done, info = env.step(action)

            # Add episode reward inside list
            episode_rewards.append(reward)

            if done:
                total_steps_per_episode.append(step)

                next_state = np.zeros((1, size_board, size_board, 16))

                total_reward = np.sum(episode_rewards)

                total_rewards_per_episode.append(total_reward)

                memory.store(state, action, reward, next_state, done)

                loss_total_ep = np.sum(loss_ep) / step
                total_loss_per_episode.append(loss_total_ep)

                total_score_per_episode.append(info["total_score"])

                print("Episode:", ep)
                print("Total Reward:", total_reward)
                print("Eps_threshold:", eps_threshold)
                print("Loss ep:", loss_total_ep)
                env.render()
                print("---------------------------")

                if info["total_score"] > best_score:
                    best_score = info["total_score"]
                    best_ep = ep
                    best_board = deepcopy(new_board)
                    best_score = step

            else:
                next_state = to_power_two_matrix(new_board)

                memory.store(state, action, reward, next_state, done)

                state = deepcopy(next_state)

                # Valid movements
                valid_movements = info["valid_movements"]

                # Change board
                board = deepcopy(new_board)

            # Learning part
            loss = optimize_model(
                dqn_net,
                target_net,
                memory,
                learning_rate,
                batch_size,
                size_board,
                gamma,
                optmizer,
                device,
            )

            loss_ep.append(loss)

            if done:
                break

        # Update target net
        if ep % ep_update_target == 0:
            print("Update target_net")
            target_net = deepcopy(dqn_net)

    plot_info(
        total_steps_per_episode,
        total_rewards_per_episode,
        total_loss_per_episode,
        total_score_per_episode,
        interval_mean,
        episodes,
    )
