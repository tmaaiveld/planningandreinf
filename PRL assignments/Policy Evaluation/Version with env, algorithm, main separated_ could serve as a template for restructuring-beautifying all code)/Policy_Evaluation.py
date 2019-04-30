import numpy as np
import time

def policy_evaluation(env): #list_action_prob):  
    # action_prob = list_action_prob
    
    #  Initialize a random policy
    policy = np.ones([16, 4]) * 0.25 # terminal states won't be picked anyway
    
    while True:
        env.amount_of_steps +=1 
        #Update our known data
        v = np.copy(env.V) 
        for i in env.non_terminal_states:
                env.V[i] = evaluate_policy(env, i, policy)

         #This if statement checks for converging, and prints tables.
        
        if np.array_equal(v, env.V):
            # print("Completed {} cycle(s)".format(env.amount_of_steps))
            # print("The tables are identical, the policy has converged.")
            #print(env.Q1)
            print('______-___'*3)
            final_Q_values = np.amax(env.V, axis=1)
            break
    #print("Completed {} cycle(s)".format(env.amount_of_steps))
    return final_Q_values, env.amount_of_steps 

                
def evaluate_policy(env, state, policy):
        env.V[state] = 0
        #i = state
        for j in range(4):
                # possible_outcome = env.return_stateprobabilities(state, j)
                # trans_prob = possible_outcome[0]              
                # next_state = possible_outcome[1]
                # env.V[state] += policy[state, j] * trans_prob * (env.R1[next_state] + env.GAMMA*env.V[next_state])

 
                for possible_outcome in env.return_stateprobabilities(state, j):
                    trans_prob = possible_outcome[0]              
                    next_state = possible_outcome[1]
                    env.V[state] += policy[state, j] * trans_prob * (env.R1[next_state] + env.GAMMA*env.V[next_state])
                     
                # trans_prob = env.return_stateprobabilities(i,j)
                # term = policy[j] * trans_prob[0][0] * (env.R1[0,trans_prob[0][1]] + env.GAMMA * (np.max(env.Q1[trans_prob[0][1]])))
                # state_value_update += term    


        #print('Updatd state value of state ' + str(i) + ' is' + str(state_value_update))
        #return state_value_update
        return env.V[state]

       