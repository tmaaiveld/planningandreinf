import numpy as np
import time

def policy_evaluation(env): 
        
    #  Initialize a random policy
    policy = np.ones([16, 4]) * 0.25 # terminal states won't be picked anyway
    
    while True:
        env.amount_of_steps +=1 
        v = np.copy(env.V) 
        for i in env.non_terminal_states:
                env.V[i] = evaluate_policy(env, i, policy)

         #Check for Convergence
        if np.array_equal(v, env.V):
            print('______-___'*3)
            final_Q_values = np.amax(env.V, axis=1)
            break
    return final_Q_values, env.amount_of_steps 

                
def evaluate_policy(env, state, policy):
        new_state_value = 0 
        for j in range(4):
                for possible_outcome in env.return_stateprobabilities(state, j):
                    trans_prob = possible_outcome[0]              
                    next_state = possible_outcome[1]
                    new_state_value += policy[state, j] * trans_prob * (env.R1[next_state] + env.GAMMA * env.V[next_state])

        return new_state_value

       