import numpy as np
import time
import math

def sarsa(env, gamma, epsilon, alpha, number_of_episodes, V_pi):
    t = time.perf_counter()
    episode_count = 0
    RMSE = []

    Q, V, policy = env.initialize()  # don't need policy and V

    for i in range(number_of_episodes):      #while not done / loop for each episode / while not_converged:
        episode_count += 1

        state = env.initialize_state()
        action = env.e_greedy_action_selection(Q, state, epsilon)
        while state in env.non_terminal_states: # in other words: loop for each episode)
            R, next_state = env.take(state, action)
            next_action = env.e_greedy_action_selection(Q, next_state, epsilon)

            Q[state,action] = Q[state,action] + alpha*(R + gamma*Q[next_state, next_action] - Q[state,action])
            state = next_state
            action = next_action

        V = np.zeros([16, 1])
        for state in range(16):
            V[state] = np.amax(Q[state])

        # print(np.reshape(V, [4,4])) # Watch the algorithm converge

        RMSE.append(math.sqrt(((V - V_pi) ** 2).mean(axis=None)))

    elapsed_time = time.perf_counter() - t
    return Q, episode_count, elapsed_time, RMSE