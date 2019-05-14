import numpy as np
import time
import random


def qlearning(env, gamma, epsilon, alpha): # theta): ####)    
    t = time.perf_counter()
    episode_count = 0
    #cumulative_reward = 0
    Q, V, policy = env.initialize(epsilon)   # maybe initialization of Q-table should not be an env. function
    for i in range(999):      #while not done / loop for each episode / while not_converged:
        episode_count += 1
        state = env.initialize_state
        while state in env.non_terminal_states: # in other words: loop for each epsidoe)
            action = env.e_greedy_action_selection(Q, state, epsilon) 
            R, next_state = env.take(state, action) # currently not taking slipping chance into account
            Q[state,action] = update(Q, state, action, reward, next_state, alpha, gamma)
            state = next_state
    elapsed_time = time.perf_counter() - t
    return V, policy, episode_count, elapsed_time #cumulative_reward
  
def update(Q, state, action, reward, next_state, alpha, gamma):
    Q_updated = Q[state,action] + alpha*(R + gamma*Q[next_state, np.argmax(Q[next_state])] - Q[state,action])
    # update V and policy
       
