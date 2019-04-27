import numpy as np
import time

def policy_evaluation(env, list_action_prob):  
    action_prob = list_action_prob
    
    while True:
        env.amount_of_steps +=1 
        #Update our known data
        env.Q1 = np.copy(env.Q2) 
        for i in env.non_terminal_states:
                env.Q2[i,0] = evaluate_policy(env, i, action_prob)

         #This if statement checks for converging, and prints tables.
        
        if np.array_equal(env.Q1,env.Q2):
            # print("Completed {} cycle(s)".format(env.amount_of_steps))
            # print("The tables are identical, the policy has converged.")
            #print(env.Q1)
            print('______-___'*3)
            final_Q_values = np.amax(env.Q1, axis=1)
            break
    #print("Completed {} cycle(s)".format(env.amount_of_steps))
    return final_Q_values, env.amount_of_steps 

                
def evaluate_policy(env, state, action_prob):
        state_value_update = 0
        i = state
        for j in range(4):
                trans_prob = env.return_stateprobabilities(i,j)
                term = action_prob[j] * trans_prob[0][0] * (env.R1[0,trans_prob[0][1]] + env.GAMMA * (np.max(env.Q1[trans_prob[0][1]])))
                state_value_update += term    
        #print('Updatd state value of state ' + str(i) + ' is' + str(state_value_update))
        return state_value_update

       