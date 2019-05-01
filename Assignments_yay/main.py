import numpy as np

from environment import Gridworld
from Policy_Evaluation import policy_evaluation, evaluate_policy
from Howards_Policy_Iteration import how_pol

#  Constants
NUMBER_OF_ALGORITHMS = 3
GAMMA_INCREMENT = 0.01
THETA = 0.0001







def print_results(V, policy, cycles,time):
    print("\n A table with the final policy is shown below.")
    print(print_moves(policy))

    print("\nThe values of all 16 states are shown below.")
    print(np.reshape(V, [4, 4]))

    print_number_of_cycles(cycles)

    print('[ time to convergence here? ]')


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


def print_number_of_cycles(cycles):
    print('Completed ' + str(cycles) + ' cycle(s)')


# this function is now redundant, print_results does the same. However, it needs some more arguments, so some adaptation
# is necessary.
def print_values(values):
    print("\n The values of all 16 states are shown below")
    print(np.reshape(values, [4, 4]))


# def initialize():
#     """
# This function creates two empty 4-D arrays, built for storing the results for each algorithm.
#
# Currently under construction....
#     """
#     global V_tabs
#     global policy_tabs
#     V_tabs = np.zeros((4, 4, int(0.1/GAMMA_INCREMENT), NUMBER_OF_ALGORITHMS))
#     policy_tabs = np.zeros((4, 4, int(0.1/GAMMA_INCREMENT), NUMBER_OF_ALGORITHMS))
#     convergence_times = array([NUMBER_OF_ALGORITHMS])


'''
def run_algorithm(algorithm_index, environment):
    global final_state_values 
    global cycles_to_convergence  
    if algorithm_index == 1:
        final_state_values, cycles_to_convergence = policy_evaluation(iceworld)  
    return final_state_values, cycles_to_convergence
'''

'''
Standard form of hyperparam loop:
for gamma in gammas:
    do foo()
    V_tabs.append(...)
    policy_tabs.append(...)
    convergence_time.append(...)
'''

#  To implement: run a for loop to test different parameters for gamma.
# gammas = np.arange(0.9,1,0.01)
# print(gammas)

#  Temporary:
gamma = 0.9
iceworld = Gridworld()


#####__________  Policy Evaluation (Must Have 2) __________#####

final_state_values, cycles_to_convergence = policy_evaluation(iceworld, gamma) #, action_prob)
print_values(final_state_values)
print_number_of_cycles(cycles_to_convergence)


#####__________  Howard (Must Have 4) __________#####

V, policy = how_pol(iceworld, gamma, THETA)
print_results(V,policy,10, 10)
