import numpy as np
import time
import math


def qlearning(env, gamma, param, alpha, number_of_games, V_pi):
    """
    :param env: Environment object
    :param gamma: Step size parameter
    :param param: Epsilon for greedy exploration (input as a positive number)
                  Temperature for softmax (input as a negative number)
    :param alpha: learning rate
    :param number_of_games: desired number of episodes
    :param V_pi: Optimal policy learned through value iteration
    :return: A Q-table, number of episodes, elapsed time, and a list of
             RMSE measurements.
    """
    episode_count = 0
    RMSE = [] 
    times = []
    t = time.perf_counter()

    Q, V, policy = env.initialize()

    for i in range(number_of_games):      # loop for each episode
        episode_count += 1 
        state = env.initialize_state()
        while state in env.non_terminal_states: # in other words: loop for each episode)
            if param > 0:
                action = env.e_greedy_action_selection(Q, state, param)
            else:
                action = env.softmax_action_selection(Q, state, -param)

            R, next_state = env.take(state, action)
            Q[state,action] = Q[state,action] + alpha*(R + gamma*Q[next_state, np.argmax(Q[next_state])] - Q[state,action])
            state = next_state
            


        V = np.zeros([16, 1])
        for state in range(16):
            V[state] = np.amax(Q[state])

        

        # print(np.reshape(V, [4,4])) # Watch the algorithm converge

        RMSE.append(math.sqrt(((V - V_pi) ** 2).mean(axis=None)))
        times.append(time.perf_counter() - t)

    elapsed_time = time.perf_counter() - t  # redundant, can just take last value of times
    return Q, episode_count, elapsed_time, RMSE, times


def qlearning_watkins(env, gamma, param, alpha, labda, number_of_games, V_pi):
    """
    :param env: Environment object
    :param gamma: Step size parameter
    :param param: Epsilon for greedy exploration (input as a positive number)
                  Temperature for softmax (input as a negative number)
    :param alpha: learning rate
    :param number_of_games: desired number of episodes
    :param V_pi: Optimal policy learned through value iteration
    :return: A Q-table, number of episodes, elapsed time, and a list of
             RMSE measurements.
    """
    episode_count = 0
    RMSE = []
    times = []
    t = time.perf_counter()

    Q, V, policy = env.initialize()
    e = np.zeros([16,4])

    for i in range(number_of_games):  # loop for each episode
        episode_count += 1
        state = env.initialize_state()
        action = env.initialize_action()

        while state in env.non_terminal_states:  # for each step of the episode)
            R, next_state = env.take(state, action)
            if param > 0:
                next_action = env.e_greedy_action_selection(Q, next_state, param)
            else:
                next_action = env.softmax_action_selection(Q, next_state, -param)

            a_star = np.argmax(Q[next_state])
            delta = R + gamma * Q[next_state, a_star] - Q[state, action]
            e[state, action] += 1

            for state in env.non_terminal_states:
                for action in range(0,3):
                    Q[state, action] += alpha * delta * e[state, action]
                    if next_action == a_star:
                        e[state, action] *= gamma * labda
                    else:
                        e[state, action] = 0

            state = next_state
            action = next_action

        V = np.zeros([16, 1])
        for state in range(16):
            V[state] = np.amax(Q[state])

        # print(np.reshape(V, [4,4])) # Watch the algorithm converge
        RMSE.append(math.sqrt(((V - V_pi) ** 2).mean(axis=None)))
        times.append(time.perf_counter() - t)

    elapsed_time = time.perf_counter() - t  # redundant, can just take last value of times
    return Q, episode_count, elapsed_time, RMSE, times
