import numpy as np

from environment import Gridworld
from Policy_Evaluation import policy_evaluation, evaluate_policy
from Howards_Policy_Iteration import how_pol

#  Constants
THETA = 0.0001

def print_values(values):
    print("\n The values of all 16 states are shown below")
    print(np.reshape(values, [4,4]))

def print_number_of_cycles(cycles):
    print('Completed ' + str(cycles) + ' cycle(s)')

def print_moves(policy):
    solution_matrix = np.chararray([16])
    i = 0
    for row in policy:
        if row[0] == 0.25:
            solution_matrix[i] = "T"
        elif row[0] == 1:
            solution_matrix[i] = "U"
        elif row[1] == 1:
            solution_matrix[i] = "R"
        elif row[2] == 1:
            solution_matrix[i] = "D"
        elif row[3] == 1:
            solution_matrix[i] = "L"
        i += 1
    return np.reshape(solution_matrix, [4,4])


'''
def run_algorithm(algorithm_index, environment):
    global final_state_values 
    global cycles_to_convergence  
    if algorithm_index == 1:
        final_state_values, cycles_to_convergence = policy_evaluation(iceworld)  
    return final_state_values, cycles_to_convergence
'''

gammas = np.arange(0.9,1,0.01)
print(gammas)
iceworld = Gridworld()



#####__________  Policy Evaluation (Must Have 2) __________#####
 
final_state_values, cycles_to_convergence = policy_evaluation(iceworld) #, action_prob)
print_values(final_state_values)
print_number_of_cycles(cycles_to_convergence)

#####__________  Howard (Must Have 4) __________#####



state_values_final = howard_policy_iteration(iceworld, THETA)