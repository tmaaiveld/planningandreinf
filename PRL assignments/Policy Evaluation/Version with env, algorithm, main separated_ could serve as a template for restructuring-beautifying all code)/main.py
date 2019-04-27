from environment import Gridworld 
from Policy_Evaluation import policy_evaluation, evaluate_policy 
from Howard_musthave4 import howard_policy_iteration

def printing_values(values):
    print("\n The values of all 16 states are shown below")
    print(values[0:4])
    print(values[4:8])
    print(values[8:12])
    print(values[12:16]) 

def printing_number_of_cycles(cycles):
    print('Completed ' + str(cycles) + ' cycle(s)')
 

gamma = 0.9
iceworld = Gridworld(gamma)

#####__________  Policy Evaluation (Must Have 2) __________#####

action_prob = [0.25, 0.25, 0.25, 0.25] # Initialize with random policy 
# final_state_values, cycles_to_convergence = policy_evaluation(iceworld, action_prob)
# printing_values(final_state_values)
# printing_number_of_cycles(cycles_to_convergence)

#####__________  Howard (Must Have 4) __________#####

state_values_final = howard_policy_iteration(iceworld)