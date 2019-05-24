import numpy as np
import time
import math
import random


def doubleqlearning(env, gamma, param, alpha, number_of_games, V_pi):
    """
    :param env: Environment object
    :param gamma: Step size parameter
    :param param: Epsilon for greedy exploration (input as a positive number)
                  Temperature for softmax (input as a negative number)
    :param alpha: learning rate
    :param number_of_games: desired number of episodes
    :param V_pi: Optimal policy learned through value iteration
    :return: A Q-table created by taking the average of two Q-tables, number of episodes, elapsed time, and a list of
             RMSE measurements.
    """
    episode_count = 0
    RMSE = [] 
    times = []
    t = time.perf_counter()

    Q1, V, policy = env.initialize()
    Q2, V2, policy2 = env.initialize()

    for i in range(number_of_games):
        episode_count += 1 
        state = env.initialize_state()
        while state in env.non_terminal_states:
            if param > 0:
                action = env.e_greedy_action_selection(Q1+Q2, state, param)    
            else:
                action = env.softmax_action_selection(Q1+Q2, state, -param)

            R, next_state = env.take(state, action)
            if random.random() < 0.5: 
                Q1[state,action] = Q1[state,action] + alpha*(R + gamma*Q2[next_state, np.argmax(Q1[next_state])] - Q1[state,action])
            else:
                Q2[state,action] = Q2[state,action] + alpha*(R + gamma*Q1[next_state, np.argmax(Q2[next_state])] - Q2[state,action])
            
            state = next_state

        V = np.zeros([16, 1])
        Q_avg = (Q1+Q2)/2
        for state in range(16):
            V[state] = np.amax(Q_avg[state])  

        RMSE.append(math.sqrt(((V - V_pi) ** 2).mean(axis=None)))
        times.append(time.perf_counter() - t)

    elapsed_time = time.perf_counter() - t
    return Q_avg, episode_count, elapsed_time, RMSE, times
