import numpy as np
import time
import math
import random


def qlearning_softmax(env, gamma, temperature, alpha, number_of_games, V_pi):  # theta): ####)
    t = time.perf_counter()
    episode_count = 0
    RMSE = []

    Q, V, policy = env.initialize()

    for i in range(number_of_games):  # loop for each episode
        episode_count += 1
        state = env.initialize_state()
        while state in env.non_terminal_states:  # in other words: loop for each episode)
            action = env.softmax_action_selection(Q, state, temperature)
            R, next_state = env.take(state, action)
            Q[state, action] = Q[state, action] + alpha * (
                        R + gamma * Q[next_state, np.argmax(Q[next_state])] - Q[state, action])
            state = next_state

        V = np.zeros([16, 1])
        for state in range(16):
            V[state] = np.amax(Q[state])

        # print(np.reshape(V, [4,4])) # Watch the algorithm converge

        RMSE.append(math.sqrt(((V - V_pi) ** 2).mean(axis=None)))

    elapsed_time = time.perf_counter() - t
    return Q, episode_count, elapsed_time, RMSE  # cumulative_reward