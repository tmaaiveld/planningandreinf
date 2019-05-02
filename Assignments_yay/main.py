import numpy as np

from environment import Gridworld
from Policy_Evaluation import policy_evaluation
from Howards_Policy_Iteration import howards_policy_iteration
from Simple_Policy_Iteration import simple_policy_iteration
from Value_Iteration import value_iteration
from Assignments_yay.HiddenPrints import HiddenPrints

#  Constants
ALGORITHMS = ["Random Policy Evaluation", "Value Iteration", "Howard's Policy Iteration", "Simple Policy Iteration"]
GAMMA_RANGE = [0.9, 1]
GAMMA_INCREMENT = 0.01
THETA = 0.0001


def print_header(title):
    print("\n" + "-" * (len(title) + 6))
    print(" " * 3 + title)
    print("-" * (len(title) + 6))


def print_results(V, optimal_policy, cycles, time, gamma):
    """
    :param V: A 4x4 array of V values for the implemented policy.
    :param policy: An optimal policy converged upon by the algorithm.
    :param cycles: The amount of cycles before convergence.
    :param time: Time elapsed while the algorithm was running.
    """
    if optimal_policy is not None:
        print("\n A table with the final policy is shown below.")
        print(print_moves(optimal_policy))

    print("\nThe values of all 16 states are shown below (gamma = {}).".format(gamma))
    print(np.reshape(V, [4, 4]))

    print_cycles(cycles)
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


def print_cycles(cycles):
    print('Completed ' + str(cycles) + ' cycle(s)')


def append_results(algorithm, V, policy, cycles, convergence_time):
    """
    There's probably a much easier way to do this with Pandas or even NumPy, but this works fine.
    """
    V_tabs[ALGORITHMS.index(algorithm)].append(np.reshape(V,[4,4]))
    policy_tabs[ALGORITHMS.index(algorithm)].append(policy)
    cycle_counts[ALGORITHMS.index(algorithm)].append(cycles)
    convergence_times[ALGORITHMS.index(algorithm)].append(convergence_time)


#  Initialize environment and parameters (must-have 1)
ice_world = Gridworld()

gammas = np.arange(GAMMA_RANGE[0], GAMMA_RANGE[1], GAMMA_INCREMENT)
# gammas = np.array([0.9]) <- use if you only want to see a single value for gamma.


#  Initialize arrays for results
global V_tabs, policy_tabs, cycle_counts, convergence_times
V_tabs = [[], [], [], []]  # 4-D arrays. Dimensions: table.x, table.y, iteration, algorithm index.
policy_tabs = [[], [], [], []]

cycle_counts = [[], [], [], []]  # 3-D arrays. Dimensions: value, iteration, algorithm index.
convergence_times = [[], [], [], []]

# I did implement parameter tuning, but it seems useless. Algos break for all values over 0.91 (Down on cell [2,3]).
for gamma in np.flip(gammas):
    gamma = round(gamma, len(str(GAMMA_INCREMENT))-2)
    print("\n\n*** Running for gamma = {} ***".format(gamma))

    #  Policy Evaluation (MH-2)
    print_header(ALGORITHMS[0])
    V, policy, cycles, time = policy_evaluation(ice_world, gamma)
    print_results(V, policy, cycles, time, gamma)
    append_results(ALGORITHMS[0], V, policy, cycles, time)

    #  Value Iteration (MH-3)
    print_header(ALGORITHMS[1])
    V, policy, cycles, time = value_iteration(ice_world, gamma, THETA)
    print_results(V, policy, cycles, time, gamma)
    append_results(ALGORITHMS[1], V, policy, cycles, time)

    #  Howard's Policy Iteration (MH-4)
    print_header(ALGORITHMS[2])
    V, policy, cycles, time = howards_policy_iteration(ice_world, gamma, THETA)
    print_results(V, policy, cycles, time, gamma)
    append_results(ALGORITHMS[2], V, policy, cycles, time)

    #  Simple Policy Iteration (5a.)
    print_header(ALGORITHMS[3])
    V, policy, cycles, time = simple_policy_iteration(ice_world, gamma, THETA)
    print_results(V, policy, cycles, time, gamma)
    append_results(ALGORITHMS[3], V, policy, cycles, time)


#  Results
V_tabs = np.array(V_tabs)
