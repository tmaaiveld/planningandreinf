import numpy as np
import matplotlib.pyplot as plt

from environment import Gridworld
from Random_Policy import policy_evaluation
from Howards_Policy_Iteration import howards_policy_iteration
from Simple_Policy_Iteration import simple_policy_iteration
from Value_Iteration import value_iteration

#  Constants
ALGORITHMS = ["Random Policy Evaluation", "Value Iteration", "Howard's Policy Iteration", "Simple Policy Iteration"]
GAMMA_RANGE = [0.85, 1]
GAMMA_INCREMENT = 0.005
THETA = 0.0001 
NUMBER_OF_RUNS = 1


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
        print_policy(optimal_policy)

    print("\nThe values of all 16 states are shown below (gamma = {}).".format(gamma))
    print(np.round(np.reshape(V, [4, 4]), 2))

    print('\nCompleted ' + str(cycles) + ' cycle(s)')
    print("Time to convergence: " + str(round(time * 1000, 2)) + " ms")


def print_policy(policy):
    if type(policy) != list:
        print(np.array2string(print_moves(policy), separator=',', formatter={'str_kind': lambda x: x}))

    else:
        pols = []
        for pol in policy:
            pols.append(print_moves(pol))
        pols = np.array(pols)
        print(np.array2string(np.flip(pols,axis=0), separator=',', formatter={'str_kind': lambda x: x}))


def print_moves(policy):
    solution_matrix = []
    for row in policy:
        if row[0] == 0.25:
            solution_matrix.append(u"\u00D7")
        elif row[0] == 1:
            solution_matrix.append(u"\u2191")
        elif row[1] == 1:
            solution_matrix.append(u"\u2192")
        elif row[2] == 1:
            solution_matrix.append(u"\u2193")
        elif row[3] == 1:
            solution_matrix.append(u"\u2190")
    solution_matrix[3] = u"\u2691"
    return np.array(np.reshape(solution_matrix, [4,4]))


def append_results(algorithm, V, policy, cycles, convergence_time):
    V_tabs[ALGORITHMS.index(algorithm)].append(np.reshape(V,[4,4]))
    policy_tabs[ALGORITHMS.index(algorithm)].append(policy)
    cycle_counts[ALGORITHMS.index(algorithm)].append(cycles)
    convergence_times[ALGORITHMS.index(algorithm)].append(convergence_time)


def create_plot(name, x, x_title, y, y_title): 
    fig = plt.figure()
    for algorithm_index in range(4):    
        plt.plot(np.flip(x), y[algorithm_index], label= str(ALGORITHMS[algorithm_index])) 
        plt.legend()
    plt.xlabel(x_title)
    plt.ylabel(y_title)  
    plt.savefig(str(name)+'.eps')
    plt.show()


#  Initialize environment and parameters (must-have 1)
ice_world = Gridworld() 

gammas = np.arange(GAMMA_RANGE[0], (GAMMA_RANGE[1]-GAMMA_INCREMENT), GAMMA_INCREMENT)

#  Initialize arrays for results
global V_tabs, policy_tabs, cycle_counts, convergence_times
V_tabs = [[], [], [], []]  # 4-D arrays. Dimensions: table.x, table.y, iteration, algorithm index.
policy_tabs = [[], [], [], []]

cycle_counts = [[], [], [], []]  # 3-D arrays. Dimensions: value, iteration, algorithm index.
convergence_times = [[], [], [], []]

#  Create lists that stores averages of runtimes
running_averages = np.zeros([len(ALGORITHMS),int(round((GAMMA_RANGE[1]-GAMMA_RANGE[0])/GAMMA_INCREMENT))])

for run in range(NUMBER_OF_RUNS):

    print('\n ====================== This is the ' + str(run) + 'th run ====================== ')
     
    i = 0
    for gamma in np.flip(gammas):
        gamma = round(gamma, len(str(GAMMA_INCREMENT))-2)
        print("\n\n*** Running for gamma = {} ***".format(gamma))

        #  Policy Evaluation (MH-2)
        print_header(ALGORITHMS[0])
        V, policy, cycles, time = policy_evaluation(ice_world, gamma)
        print_results(V, policy, cycles, time, gamma)
        append_results(ALGORITHMS[0], V, policy, cycles, time)

        #  Value Iteration (MH-3)
        print_header(ALGORITHMS[2])
        V, policy, cycles, time = value_iteration(ice_world, gamma, THETA)
        print_results(V, policy, cycles, time, gamma)
        append_results(ALGORITHMS[2], V, policy, cycles, time)

        #  Howard's Policy Iteration (MH-4)
        print_header(ALGORITHMS[2])
        V, policy, cycles, time = howards_policy_iteration(ice_world, gamma, THETA)
        print_results(V, policy, cycles, time, gamma)
        append_results(ALGORITHMS[2], V, policy, cycles, time)

        #  Simple Policy Iteration (Opt. 5a.)
        print_header(ALGORITHMS[3])
        V, policy, cycles, time = simple_policy_iteration(ice_world, gamma, THETA)
        print_results(V, policy, cycles, time, gamma)
        append_results(ALGORITHMS[3], V, policy, cycles, time)

        # store runtimes for each gamma
        for algorithm in range(len(ALGORITHMS)):
            running_averages[algorithm][i] += convergence_times[algorithm][i] * (1 / NUMBER_OF_RUNS)
        i += 1

    #  Results
    V_tabs = np.array(V_tabs)

    #  Empty the results
    if not NUMBER_OF_RUNS - 1 == run:
        V_tabs = [[], [], [], []]
        policy_tabs = [[], [], [], []]
        cycle_counts = [[], [], [], []]
        convergence_times = [[], [], [], []]
 
create_plot('Iterations_gammas', gammas, 'gamma', cycle_counts, 'iterations')
create_plot('Runtime_gammas', gammas, 'gamma', running_averages, 'runtime (sec)')



