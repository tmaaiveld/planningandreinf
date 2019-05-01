import numpy as np


def policy_evaluation(env, gamma):
    #  Initialize a random policy
    V, policy = env.initialize()

    while True:
        env.amount_of_steps += 1
        v = np.copy(V)
        for state in env.non_terminal_states:
                V[state] = evaluate_policy(env, state, policy, gamma)

        #  Check for Convergence
        if np.array_equal(v, env.V):
            print('______-___'*3)
            print("The algorithm has converged.")
            print('______-___' * 3)
            final_Q_values = np.amax(env.V, axis=1)
            break
    return final_Q_values, env.amount_of_steps 

                
def evaluate_policy(env, state, policy, gamma):
    new_state_value = 0
    for action in range(4):
            for possible_outcome in env.transition_function(state, action):
                trans_prob = possible_outcome[0]
                next_state = possible_outcome[1]
                new_state_value += policy[state, action] * trans_prob * (env.R1[next_state] + gamma * env.V[next_state])
    return new_state_value
