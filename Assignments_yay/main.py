import numpy as np

from environment import Gridworld
from Policy_Evaluation import policy_evaluation, evaluate_policy
from Howards_Policy_Iteration import howards_policy_iteration
from Simple_Policy_Iteration import simple_policy_iteration
from Value_Iteration import value_iteration
from HiddenPrints import HiddenPrints

#  Constants
NUMBER_OF_ALGORITHMS = 3
GAMMA_INCREMENT = 0.01
THETA = 0.0001


def print_header(title):
    print("\n" + "-" * (len(title) + 6))
    print(" " * 3 + title)
    print("-" * (len(title) + 6))

def print_results(V, optimal_policy, cycles, time):
    """
    :param V: A 4x4 array of V values for the implemented policy.
    :param policy: An optimal policy converged upon by the algorithm.
    :param cycles: The amount of cycles before convergence.
    :param time: Time elapsed while the algorithm was running.
    """
    if optimal_policy is not None:
        print("\n A table with the final policy is shown below.")
        print(print_moves(optimal_policy))

    print("\nThe values of all 16 states are shown below.")
    print(np.reshape(V, [4, 4]))

    print_number_of_cycles(cycles)
    # print('\n[ time to convergence here? ]')


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
        final_state_values, cycles_to_convergence = policy_evaluation(ice_world)  
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



#  Initialize parameters and environment
gammas = np.arange(0.9, 1, 0.01)
# gamma = 0.9
ice_world = Gridworld()

#  Initialize arrays for results
V_tabs = []
policy_tabs = []

for gamma in gammas:
    #  Policy Evaluation (MH-2)
    print_header("Random Policy Evaluation")
    with HiddenPrints():
        V, cycles = policy_evaluation(ice_world, gamma)
    print_results(V, None, cycles, 10)
    
    #  Value Iteration (MH-3)
    print_header("Value Iteration")
    #with HiddenPrints():
    V, policy, cycles = value_iteration(ice_world, gamma, THETA)
    print_results(V, policy, cycles, 10)
    
    #  Howard's Policy Iteration (MH-4)
    print_header("Howard's Policy Iteration")
    with HiddenPrints():
        V, policy = howards_policy_iteration(ice_world, gamma, THETA)
    print_results(V, policy, 10, 10)
    
    #  Simple Policy Iteration (5a.)
    print_header("Simple Policy Iteration")
    with HiddenPrints():
        V, policy = simple_policy_iteration(ice_world, gamma, THETA)
    print_results(V, policy, 10, 10)

