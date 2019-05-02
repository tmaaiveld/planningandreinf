import numpy as np

from environment import Gridworld
from Policy_Evaluation import policy_evaluation
from Howards_Policy_Iteration import howards_policy_iteration
from Simple_Policy_Iteration import simple_policy_iteration
from Value_Iteration import value_iteration
from Assignments_yay.HiddenPrints import HiddenPrints

#  Constants
ALGORITHMS = ["Random Policy Evaluation", "Value Iteration", "Howard's Policy Iteration", "Simple Policy Iteration"]
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


# # this function is now redundant, print_results does the same. However, it needs some more arguments, so some adaptation
# # is necessary.
# def print_values(values):
#     print("\n The values of all 16 states are shown below")
#     print(np.reshape(values, [4, 4]))


def append_results(algorithm, V, policy, cycles, convergence_time):
    """
    There's probably a much easier way to do this with Pandas or even NumPy, but this works fine.
    """
    V_tabs[ALGORITHMS.index(algorithm)].append(np.reshape(V,[4,4]))
    policy_tabs[ALGORITHMS.index(algorithm)].append(policy)
    cycle_counts[ALGORITHMS.index(algorithm)].append(cycles)
    convergence_times[ALGORITHMS.index(algorithm)].append(convergence_time)


#  Initialize parameters and environment
# gammas = np.arange(0.9, 1, 0.01)

gammas = np.array([0.9])
ice_world = Gridworld()

#  Initialize arrays for results
global V_tabs, policy_tabs, cycle_counts, convergence_times
V_tabs = [[], [], [], []]
policy_tabs = [[], [], [], []]
cycle_counts = [[], [], [], []]
convergence_times = [[], [], [], []]

# I did try to implement parameter tuning, but it seems useless. Algos break for most values.
for gamma in gammas:
    #  Policy Evaluation (MH-2)
    print_header(ALGORITHMS[0])
    with HiddenPrints():
        V, cycles = policy_evaluation(ice_world, gamma)
    print_results(V, None, cycles, 10)
    append_results(ALGORITHMS[0], V, None, cycles, 10)

    #  Value Iteration (MH-3)
    print_header(ALGORITHMS[1])
    with HiddenPrints():
        V, policy, cycles = value_iteration(ice_world, gamma, THETA)
    print(policy)
    print_results(V, policy, cycles, 10)
    append_results(ALGORITHMS[1], V, None, cycles, 10)

    #  Howard's Policy Iteration (MH-4)
    print_header(ALGORITHMS[2])
    with HiddenPrints():
        V, policy = howards_policy_iteration(ice_world, gamma, THETA)
    print_results(V, policy, 10, 10)
    append_results(ALGORITHMS[2], V, None, cycles, 10)

    #  Simple Policy Iteration (5a.)
    print_header(ALGORITHMS[3])
    with HiddenPrints():
        V, policy = simple_policy_iteration(ice_world, gamma, THETA)
    print_results(V, policy, 10, 10)
    append_results(ALGORITHMS[3], V, None, cycles, 10)

# V_tabs = np.array(V_tabs)


