import numpy as np
import matplotlib.pyplot as plt

from Environment import Gridworld
from Qlearning_draft import qlearning

#  Constants
ALGORITHMS = ["qlearning"]
GAMMA_RANGE = [0.9]
GAMMA_INCREMENT = 0.005
EPSILON_RANGE = 0.05
ALPHA_RANGE = 0.01
THETA = 0.0001 
NUMBER_OF_RUNS = 1


def print_header(title):
    print("\n" + "-" * (len(title) + 6))
    print(" " * 3 + title)
    print("-" * (len(title) + 6))

def print_results2(Q, episode_count, time): # gamma, alpha, eps
    print(Q)
    print(episode_count)
    print(time)



def print_results1(V, optimal_policy, cycles, time, gamma): # Assignment 1 
    """
    :param V: A 4x4 array of V values for the implemented policy.
    :param policy: An optimal policy converged upon by the algorithm.
    :param cycles: The amount of cycles before convergence.
    :param time: Time elapsed while the algorithm was running.
    """
    if optimal_policy is not None:
        #print("\n A table with the final policy is shown below.")
        #print_policy(optimal_policy)
        print("\n An ugly table with the final policy is shown below.")
        print(optimal_policy) # 

    print("\nThe values of all 16 states are shown below (gamma = {}).".format(gamma))
    print(np.round(np.reshape(V, [4, 4]), 2))

    print('\nCompleted ' + str(cycles) + ' episode(s)')
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


# def print_moves(policy): # needs to be rewritten 
    # print('Print policy in an ugly way for now:')
    # print(policy)
    # solution_matrix = []
    # for row in policy:
    #     if row[0] == 0.25:
    #         solution_matrix.append(u"\u00D7")
    #     elif row[0] == 1:
    #         solution_matrix.append(u"\u2191")
    #     elif row[1] == 1:
    #         solution_matrix.append(u"\u2192")
    #     elif row[2] == 1:
    #         solution_matrix.append(u"\u2193")
    #     elif row[3] == 1:
    #         solution_matrix.append(u"\u2190")
    # print('Print solution_matrix:')
    # print(solution_matrix)
    # solution_matrix[3] = u"\u2691"
    # return np.array(np.reshape(solution_matrix, [4,4])) 


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


gammas = GAMMA_RANGE
# gammas = np.arange(GAMMA_RANGE[0], (GAMMA_RANGE[1]-GAMMA_INCREMENT), GAMMA_INCREMENT)
epsilon = EPSILON_RANGE
alpha = ALPHA_RANGE

#  Initialize arrays for results
global V_tabs, policy_tabs, cycle_counts, convergence_times
V_tabs = [[], [], [], []]  # 4-D arrays. Dimensions: table.x, table.y, iteration, algorithm index.
policy_tabs = [[], [], [], []]

cycle_counts = [[], [], [], []]  # 3-D arrays. Dimensions: value, iteration, algorithm index.
convergence_times = [[], [], [], []]

#  Create lists that stores averages of runtimes
#running_averages = np.zeros([len(ALGORITHMS),int(round((GAMMA_RANGE[1]-GAMMA_RANGE[0])/GAMMA_INCREMENT))])

for run in range(NUMBER_OF_RUNS):

    print('\n ====================== This is the ' + str(run) + 'th run ====================== ')
     
    i = 0
    for gamma in np.flip(gammas):
        gamma = round(gamma, len(str(GAMMA_INCREMENT))-2)
        print("\n\n*** Running for gamma = {} ***".format(gamma))


        #  Q-learning  (MH-8)
        print_header(ALGORITHMS[0])
        Q, cycles, time = qlearning(ice_world, epsilon, alpha, gamma) #, THETA)
        print_results2(Q, cycles, time)
        #print_results(V, policy, cycles, time, gamma)
        #append_results(ALGORITHMS[0], V, policy, cycles, time)



        # store runtimes for each gamma
        # for algorithm in range(len(ALGORITHMS)):
        #     running_averages[algorithm][i] += convergence_times[algorithm][i] * (1 / NUMBER_OF_RUNS)
        # i += 1

    #  Results
    V_tabs = np.array(V_tabs)

    #  Empty the results
    if not NUMBER_OF_RUNS - 1 == run:
        V_tabs = [[], [], [], []]
        policy_tabs = [[], [], [], []]
        cycle_counts = [[], [], [], []]
        convergence_times = [[], [], [], []]
 
#create_plot('Iterations_gammas', gammas, 'gamma', cycle_counts, 'iterations')
#create_plot('Runtime_gammas', gammas, 'gamma', running_averages, 'runtime (sec)')