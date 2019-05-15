import numpy as np
import matplotlib.pyplot as plt

from environment import Gridworld
from Qlearning import qlearning, qlearning_watkins
from SARSA import sarsa
from Value_Iteration import value_iteration

ALGORITHMS = ["Q-learning (\u03B5-greedy)", "Q-learning (Softmax)", "SARSA", "Q-learning (Eligibility Traces)"]
GAMMA_RANGE = [0.9]
GAMMA_INCREMENT = 0.005
EPSILON_RANGE = 0.1
TEMPERATURE_RANGE = 10
ALPHA_RANGE = 0.01
LABDA_RANGE = 0.1
THETA = 0.0001 
NUMBER_OF_RUNS = 1
NUMBER_OF_GAMES = 10000


def print_header(title):
    print("\n" + "-" * (len(title) + 6))
    print(" " * 3 + title)
    print("-" * (len(title) + 6))


def print_results(Q, episode_count, time, algorithm):
    """
    The state-values and optimal policies are extracted from the Qs and printed here
    """
    V = np.zeros([16, 1])
    for state in range(16):
        V[state] = np.amax(Q[state])
    print('\n The values of all 16 states are shown below')
    print(np.round(np.reshape(V, [4, 4]), 2))

    optimal_policy = np.empty([16,1])
    for state in range(16):
        optimal_policy[state] = np.argmax(Q[state]) 
    print('\n The optimal policy found by ' + algorithm + ' is as follows:')
    print_moves(optimal_policy) 
    
    print('\nCompleted ' + str(episode_count) + ' episode(s)')
    print("Time to convergence: " + str(round(time * 1000, 2)) + " ms")


def print_moves(policy):  
    """
    This prints the optimal policies in symbols (arrows and stuff), 
    quite different from Assignment 1's print_moves though
    """
    non_terminal_states = [2,1,0,6,4,12,10,9,8]
    solution_matrix = []
    state = 0
    for move in policy:
        if state not in non_terminal_states:
            solution_matrix.append(u"\u00D7")
        else:
            if move == [0]:
                solution_matrix.append(u"\u2191")
            elif move == [1]:
                solution_matrix.append(u"\u2192")
            elif move == [2]:
                solution_matrix.append(u"\u2193")
            elif move == [3]:
                solution_matrix.append(u"\u2190")
        state += 1 
    solution_matrix[3] = u"\u2691"
    print(np.array(np.reshape(solution_matrix, [4,4])))


def create_plot(name, x, x_title, y, y_title): 
    # fig = plt.figure()
    colors = ['r--', 'g--', 'b--', 'y--']
    for algorithm_index in range(len(ALGORITHMS)):
        plt.plot(x[algorithm_index], y[algorithm_index], colors[algorithm_index], label=ALGORITHMS[algorithm_index])
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend()
    plt.savefig(str(name)+'.eps')
    plt.show()


#  Initialize environment and parameters
ice_world = Gridworld() 


gammas = GAMMA_RANGE
epsilon = EPSILON_RANGE
alpha = ALPHA_RANGE
temperature = TEMPERATURE_RANGE
labda = LABDA_RANGE

#  Initialize arrays for results
{
# global V_tabs, policy_tabs, cycle_counts, convergence_times
# V_tabs = [[], [], [], []]  # 4-D arrays. Dimensions: table.x, table.y, iteration, algorithm index.
# policy_tabs = [[], [], [], []]
#
# cycle_counts = [[], [], [], []]  # 3-D arrays. Dimensions: value, iteration, algorithm index.
# convergence_times = [[], [], [], []]
#
#  Create lists that stores averages of runtimes
#running_averages = np.zeros([len(ALGORITHMS),int(round((GAMMA_RANGE[1]-GAMMA_RANGE[0])/GAMMA_INCREMENT))])
}

global RMSE_tabs
RMSE_tabs = [[], [], [], []]


for run in range(NUMBER_OF_RUNS):

    print('\n ====================== This is the ' + str(run) + 'th run ====================== ')
     
    i = 0
    for gamma in np.flip(gammas):
        gamma = round(gamma, len(str(GAMMA_INCREMENT))-2)
        print("\n\n*** Running for gamma = {} ***".format(gamma))

        # Value Iteration (optimal policy benchmark)
        V_pi = value_iteration(ice_world, gamma, THETA)

        #  Q-learning (MH-8)
        print_header(ALGORITHMS[0])
        Q, cycles, time, RMSE_QLG, times_QLG = qlearning(ice_world, gamma, epsilon, alpha, NUMBER_OF_GAMES, V_pi)
        print_results(Q, cycles, time, ALGORITHMS[0])
        RMSE_tabs[0].append(RMSE_QLG)

        # Softmax Exploration Strategy (MH-9)
        print_header(ALGORITHMS[1])
        Q, cycles, time, RMSE_QLS, times_QLS = qlearning(ice_world, gamma, -temperature, alpha, NUMBER_OF_GAMES, V_pi)
        print_results(Q, cycles, time, ALGORITHMS[1])
        RMSE_tabs[1].append(RMSE_QLS)

        #  SARSA (MH-10)
        print_header(ALGORITHMS[2])
        Q, cycles, time, RMSE_SARSA, times_SARSA = sarsa(ice_world, gamma, epsilon, alpha, NUMBER_OF_GAMES, V_pi)
        print_results(Q, cycles, time, ALGORITHMS[2])
        RMSE_tabs[2].append(RMSE_SARSA)

        # Q-Learning (Eligibility Traces) (O-12)
        print_header(ALGORITHMS[3])
        Q, cycles, time, RMSE_QLE, times_QLE = qlearning_watkins(ice_world, gamma, epsilon, alpha, labda, NUMBER_OF_GAMES, V_pi)
        print_results(Q, cycles, time, ALGORITHMS[3])
        RMSE_tabs[3].append(RMSE_QLE)

RMSE_means = np.mean(RMSE_tabs, axis=1)

# visualise results
create_plot("RMSE plot", 4 * [range(0, NUMBER_OF_GAMES)], 'Iterations',
            RMSE_means, 'RMSE, averaged over states')
create_plot("time plot", [times_QLG, times_QLS, times_SARSA, times_QLE], 'elapsed time (seconds)',
            RMSE_means, 'RMSE, averaged over states')
# the time plot looks nicer if you terminate after a given timespan.
# also, averaging over 100 runs will give much more consistent results.

#create_plot('Iterations_gammas', gammas, 'gamma', cycle_counts, 'iterations')
#create_plot('Runtime_gammas', gammas, 'gamma', running_averages, 'runtime (sec)')