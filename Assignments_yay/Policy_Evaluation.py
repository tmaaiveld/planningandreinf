import numpy as np


def policy_evaluation(env, gamma):
    #  Initialize a random policy
    V, policy = env.initialize()

    while True:
        env.amount_of_steps += 1
        v = np.copy(V)
        evaluate_policy(env, V, policy, gamma)

        #  Check for Convergence
        if np.array_equal(v, V):

            break
    return V, env.amount_of_steps


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
