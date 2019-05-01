import numpy as np
import time

# Initial Q-space
Q1 = np.zeros([16, 4])

# Updated Q-space
Q2 = np.zeros([16, 4])

# Define Rewards table
# [ 0,  1,  2,  3]
# [ 4,  5,  6,  7]
# [ 8,  9, 10, 11]
# [12, 13, 14, 15]
# R1 = np.asarray([[0,   0,   0, 100,
#                   0, -10,   0, -10,
#                   0,   0,  20, -10,
#                   0, -10, -10, -10]])

states_and_rewards = np.asarray([[0, 0], [1, 0], [2, 0], [3, 100],
                                 [4, 0], [5, -10], [6, 0], [7, -10],
                                 [8, 0], [9, 0], [10, 20], [11, -10],
                                 [12, 0], [13, -10], [14, -10], [15, -10]])

# Keep track of amount of loops, for discount factor
amount_of_steps = 0

non_terminal_states = [0, 1, 2, 4, 6, 8, 9, 10, 12]
# action order: up, right, down, left.
# up = 0. right = 1. down = 2. left = 3.

# Discount
GAMMA = 0.9


def return_stateprobabilities(state, action):
    state_probabilities = [[1, state]]
    if state == 0:
        if action == 1:
            state_probabilities = [[0.95, 1], [0.05, 3]]
        if action == 2:
            state_probabilities = [[0.95, 4], [0.05, 12]]
    elif state == 1:
        if action == 1:
            state_probabilities = [[0.95, 2], [0.05, 3]]
        if action == 2:
            state_probabilities = [[1, 5]]
        if action == 3:
            state_probabilities = [[1, 0]]
    elif state == 2:
        if action == 1:
            state_probabilities = [[1, 3]]
        if action == 2:
            state_probabilities = [[0.95, 6], [0.05, 14]]
        if action == 3:
            state_probabilities = [[0.95, 1], [0.05, 0]]
    elif state == 4:
        if action == 0:
            state_probabilities = [[1, 0]]
        if action == 1:
            state_probabilities = [[1, 5]]
        if action == 2:
            state_probabilities = [[0.95, 8], [0.05, 12]]
    elif state == 6:
        if action == 0:
            state_probabilities = [[1, 2]]
        if action == 1:
            state_probabilities = [[1, 7]]
        if action == 2:
            state_probabilities = [[0.95, 10], [0.05, 14]]
        if action == 3:
            state_probabilities = [[1, 5]]
    elif state == 8:
        if action == 0:
            state_probabilities = [[0.95, 4], [0.05, 0]]
        if action == 1:
            state_probabilities = [[0.95, 9], [0.05, 11]]
        if action == 2:
            state_probabilities = [[1, 12]]
    elif state == 9:
        if action == 0:
            state_probabilities = [[1, 5]]
        if action == 1:
            state_probabilities = [[0.95, 10], [0.05, 11]]
        if action == 2:
            state_probabilities = [[1, 13]]
        if action == 3:
            state_probabilities = [[1, 8]]
    elif state == 10:
        if action == 0:
            state_probabilities = [[0.95, 6], [0.05, 2]]
        if action == 1:
            state_probabilities = [[1, 11]]
        if action == 2:
            state_probabilities = [[1, 14]]
        if action == 3:
            state_probabilities = [[0.95, 9], [0.05, 8]]
    elif state == 12:
        if action == 0:
            state_probabilities = [[0.95, 8], [0.05, 0]]
        if action == 1:
            state_probabilities = [[1, 13]]
    return state_probabilities


G = 0
states_and_returns = []

for state_reward in np.flip(states_and_rewards, axis=0):
    if state_reward[0] in non_terminal_states:
        states_and_returns.append([state_reward[0], G])
        G = state_reward[1] + GAMMA * G # now update this to consider probability
        print('yo')
    else:
        states_and_returns.append([state_reward[0], 0])
        print('yo')

print(np.flip(states_and_returns, axis=0))
