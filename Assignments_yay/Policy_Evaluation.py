import numpy as np


def policy_evaluation(env, gamma):
    #  Initialize a random policy
    policy = np.ones([16, 4]) * 0.25  # terminal states won't be picked
    
    while True:
        env.amount_of_steps += 1
        v = np.copy(env.V) 
        for i in env.non_terminal_states:
                env.V[i] = evaluate_policy(env, i, policy, gamma)

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
    for j in range(4):
            for possible_outcome in env.transition_function(state, j):
                trans_prob = possible_outcome[0]
                next_state = possible_outcome[1]
                new_state_value += policy[state, j] * trans_prob * (env.R1[next_state] + gamma * env.V[next_state])
    return new_state_value
