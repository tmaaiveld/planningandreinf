import numpy as np
import matplotlib.pyplot as plt

from environment import Gridworld
from Policy_Evaluation import policy_evaluation
from Howards_Policy_Iteration import howards_policy_iteration
from Simple_Policy_Iteration import simple_policy_iteration
from Value_Iteration import value_iteration
# from Assignments_yay.HiddenPrints import HiddenPrints
from HiddenPrints import HiddenPrints

#  Constants
ALGORITHMS = ["Random Policy Evaluation", "Value Iteration", "Howard's Policy Iteration", "Simple Policy Iteration"]
GAMMA_RANGE = [0.9, 1]
GAMMA_INCREMENT = 0.01
THETA = 0.0001 
NUMBER_OF_RUNS = 3

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
    """
    There's probably a much easier way to do this with Pandas or even NumPy, but this works fine.
    """
    V_tabs[ALGORITHMS.index(algorithm)].append(np.reshape(V,[4,4]))
    policy_tabs[ALGORITHMS.index(algorithm)].append(policy)
    cycle_counts[ALGORITHMS.index(algorithm)].append(cycles)
    convergence_times[ALGORITHMS.index(algorithm)].append(convergence_time)

def plot_iterations_gamma():
    cycles_gamma = plt.figure() 
    for algorithm_index in range(4):    
    #ax = plt.axes()
    #ax.plot(np.flip(gammas), cycle_counts[algorithm_index]) 
        plt.plot(np.flip(gammas), cycle_counts[algorithm_index], label= str(ALGORITHMS[algorithm_index])) 
        plt.legend()
    plt.xlabel("gamma")
    plt.ylabel("iterations")
    plt.savefig('Iterations_gamma.eps')
    plt.show()

def plot_runtime_gamma():
    runtime_gamma = plt.figure()
    for algorithm_index in range(4):    
    #ax = plt.axes()
    #ax.plot(np.flip(gammas), convergence_times[algorithm_index]) 
        plt.plot(np.flip(gammas), convergence_times[algorithm_index], label= str(ALGORITHMS[algorithm_index])) 
        plt.legend()
    plt.xlabel("gamma")
    plt.ylabel("run time (sec)")
    plt.savefig('runtimes_gamma.eps')
    plt.show()


#  Initialize environment and parameters (must-have 1)
ice_world = Gridworld() 

gammas = np.arange(GAMMA_RANGE[0], GAMMA_RANGE[1], GAMMA_INCREMENT)
# gammas = np.array([0.9]) # use if you only want to see a single value for gamma.


#  Initialize arrays for results
global V_tabs, policy_tabs, cycle_counts, convergence_times
V_tabs = [[], [], [], []]  # 4-D arrays. Dimensions: table.x, table.y, iteration, algorithm index.
policy_tabs = [[], [], [], []]

cycle_counts = [[], [], [], []]  # 3-D arrays. Dimensions: value, iteration, algorithm index.
convergence_times = [[], [], [], []]

### create lists that store sums of runtimes (and then averages are taken at the end)
sum_rand = [[], [], [], [], [], [], [], [], [], []]
sum_val = [[], [], [], [], [], [], [], [], [], []]
sum_how = [[], [], [], [], [], [], [], [], [], []]
sum_sim = [[], [], [], [], [], [], [], [], [], []] 

for run in range(NUMBER_OF_RUNS):

    print('\n _ __ _  This is the ' + str(run) + 'th run   _ __ _  ')
 
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

    ###store runtimes in empty nested-lists above the big loop
    for gamma_index in range(10):
        sum_rand[gamma_index].append(convergence_times[0][gamma_index])
    
    for gamma_index in range(10):
        sum_val[gamma_index].append(convergence_times[1][gamma_index])
        
    for gamma_index in range(10):
        sum_how[gamma_index].append(convergence_times[2][gamma_index])

    for gamma_index in range(10):
        sum_sim[gamma_index].append(convergence_times[3][gamma_index])

    ###empty the results of this run to start the next one fresh
    V_tabs = [[], [], [], []]  # 4-D arrays. Dimensions: table.x, table.y, iteration, algorithm index.
    policy_tabs = [[], [], [], []]

    cycle_counts = [[], [], [], []]  # 3-D arrays. Dimensions: value, iteration, algorithm index.
    convergence_times = [[], [], [], []]

###Take averages to get average run times for each algorithm for each gamma
avg_rand = []
for lst in sum_rand:
    avg_rand.append(sum(lst)/len(lst))
 
avg_val = []
for lst in sum_val:
    avg_val.append(sum(lst)/len(lst))

avg_how = []
for lst in sum_how:
    avg_how.append(sum(lst)/len(lst))

avg_sim = []
for lst in sum_sim:
    avg_sim.append(sum(lst)/len(lst))

averages_runtimes = [avg_rand, avg_val, avg_how, avg_sim]

###Print run times for all algos and all gammas (averages)
runtime_gamma_average = plt.figure()
for algorithm_index in range(4): 
    plt.plot(np.flip(gammas), averages_runtimes[algorithm_index], label= str(ALGORITHMS[algorithm_index])) 
    plt.legend()
plt.xlabel("gamma")
plt.ylabel('runtime (sec)')
plt.savefig('runtimes_gamma_multipleruns.eps')
plt.show()

# plot_iterations_gamma()
# plot_runtime_gamma()
