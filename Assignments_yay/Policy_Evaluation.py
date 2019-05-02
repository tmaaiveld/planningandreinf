import numpy as np
import time


def policy_evaluation(env, gamma):
    t = time.perf_counter()
    V, policy = env.initialize()
    loop_counter = 0

    while True:
        loop_counter += 1
        v = np.copy(V)
        V = evaluate_policy(env, V, policy, gamma)

        #  Check for convergence
        if np.array_equal(v, V):
            elapsed_time = time.perf_counter() - t
            return V, None, loop_counter, elapsed_time


def evaluate_policy(env, V, policy, gamma):
    for state in env.non_terminal_states:
        new_state_value = 0
        for action in range(4):
            for possible_outcome in env.transition_function(state, action):
                trans_prob = possible_outcome[0]
                next_state = possible_outcome[1]
                new_state_value += policy[state, action] * trans_prob * (env.R[next_state] + gamma * V[next_state])
        V[state] = new_state_value
    return V
