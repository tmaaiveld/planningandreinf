from environment import Gridworld 
from Policy_Evaluation import policy_evaluation, evaluate_policy 

def printing_values(values):
    print("\n The values of all 16 states are shown below")
    print(values[0:4])
    print(values[4:8])
    print(values[8:12])
    print(values[12:16]) 

def printing_number_of_cycles(cycles):
    print('Completed ' + str(cycles) + ' cycle(s)')

'''
def run_algorithm(algorithm_index, environment):
    global final_state_values 
    global cycles_to_convergence  
    if algorithm_index == 1:
        final_state_values, cycles_to_convergence = policy_evaluation(iceworld)  
    return final_state_values, cycles_to_convergence'''
         

gamma = 0.9
iceworld = Gridworld(gamma)

#algorithm_index = input('Please select an algorithm. \n 1 -> Policy Evaluation on a Random Policy \n 2 -> Value Iteration \n 3 -> Howard\'s Policy Iteration \n')

#final_state_values, cycles_to_convergence = run_algorithm(algorithm_index, iceworld)
 
#####__________  Policy Evaluation (Must Have 2) __________#####
 
final_state_values, cycles_to_convergence = policy_evaluation(iceworld) #, action_prob)
printing_values(final_state_values)
printing_number_of_cycles(cycles_to_convergence)

#####__________  Howard (Must Have 4) __________#####

#state_values_final = howard_policy_iteration(iceworld)