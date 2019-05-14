import numpy as np
import matplotlib.pyplot as plt

from environment import Gridworld
from Qlearning_draft import qlearning
from SARSA_draft import sarsa

#  Constants
ALGORITHMS = ["Q-learning", "SARSA"]
GAMMA_RANGE = [0.9]
GAMMA_INCREMENT = 0.005
EPSILON_RANGE = 0.05
ALPHA_RANGE = 0.01
THETA = 0.0001 
NUMBER_OF_RUNS = 1
NUMBER_OF_GAMES =  500000


def print_header(title):
    print("\n" + "-" * (len(title) + 6))
    print(" " * 3 + title)
    print("-" * (len(title) + 6))


def print_results(Q, episode_count, time, algorithm):
    '''
    The state-values and optimal policies are extracted from the Qs and printed here
    '''
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
    '''
    This prints the optimal policies in symbols (arrows and stuff), 
    quite different from Assignment 1's print_moves though
    '''
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
        Q, cycles, time = qlearning(ice_world, gamma, epsilon, alpha, NUMBER_OF_GAMES)
        print_results(Q, cycles, time, ALGORITHMS[0])
        #print_results(V, policy, cycles, time, gamma)
        #append_results(ALGORITHMS[0], V, policy, cycles, time)

        # Softmax Exploration Strategy (MH-9)

        #  SARSA (parilla) (MH-10)
        print_header(ALGORITHMS[1])
        Q, cycles, time = sarsa(ice_world, gamma, epsilon, alpha, NUMBER_OF_GAMES)
        print_results(Q, cycles, time, ALGORITHMS[1])
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