import numpy as np
import time

def sarsa(env, gamma, epsilon, alpha, number_of_episodes):
    t = time.perf_counter()
    episode_count = 0
    #cumulative_reward = 0

    Q, V, policy = env.initialize(epsilon)  # don't need policy and V
    print(number_of_episodes)
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
    elapsed_time = time.perf_counter() - t
    return Q, episode_count, elapsed_time