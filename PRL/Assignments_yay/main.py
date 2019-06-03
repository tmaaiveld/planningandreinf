import numpy as np
import matplotlib.pyplot as plt

from environment import Gridworld
from Qlearning import qlearning, qlearning_watkins, dynaq
from SARSA import sarsa
from DoubleQLearning import doubleqlearning
from Value_Iteration import value_iteration


ALGORITHMS = ["Q-learning (\u03B5-greedy)", "Q-learning (Softmax)", "SARSA",
              "Q-learning (Eligibility Traces)", "Dyna-Q",
              "Double Q-learning (\u03B5-greedy)", "Double Q-learning (Softmax)"]

GAMMA_RANGE = [0.9]
GAMMA_INCREMENT = 0
EPSILON_RANGE = 0.12
TEMPERATURE_RANGE = 10
ALPHA_RANGE = 0.01
LABDA_RANGE = 0.1
THETA = 0.0001 
NUMBER_OF_RUNS = 1
NUMBER_OF_GAMES = 10000
PLANNING_STEPS = 10  # hyperparameter for Dyna-Q


def print_header(title):
    print("\n" + "-" * (len(title) + 6))
    print(" " * 3 + title)
    print("-" * (len(title) + 6))


def print_results(Q, episode_count, time, algorithm):
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


def create_plot(name, x, x_title, y, y_title, algorithms):
    colors = ['r--', 'g--', 'b--', 'y--', 'm--', 'k--', 'c--']
    for algorithm_index in range(len(algorithms)):
        plt.plot(x[algorithm_index], y[algorithm_index], colors[algorithm_index], label=algorithms[algorithm_index])
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend()
    plt.savefig(str(name)+'.eps')
    plt.show()


def create_time_plot(name, x, x_title, y, y_title, algorithms):
    colors = ['r--', 'g--', 'b--', 'y--', 'm--', 'k--', 'c--']
    for algorithm_index in range(len(algorithms)):
        plt.plot(x[algorithm_index], y[algorithm_index], colors[algorithm_index], label=algorithms[algorithm_index])
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.xlim(0,6)
    plt.legend()
    plt.savefig(str(name)+'.eps')
    plt.show()


def create_single_plot(name, x, x_title, y, y_title, algorithm_index):
    colors = ['r--', 'g--', 'b--', 'y--', 'm--', 'c--']
    plt.plot(x, y, colors[algorithm_index]) #, label=ALGORITHMS[algorithm_index])
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend()
    plt.savefig(str(name)+'.eps')
    plt.show()


def create_state_plot(name, V_tabs_QLG, NUMBER_OF_GAMES, V_pi):
    plt.plot(range(1, NUMBER_OF_GAMES + 1), V_tabs_QLG, 'b')
    plt.hlines(y=74.85948625, xmin=0, xmax=NUMBER_OF_GAMES, colors='r')
    plt.xlabel("Episodes")
    plt.ylabel("State value")
    plt.legend()
    plt.savefig(str(name)+'.eps')
    plt.show()

#  Initialize environment and parameters
ice_world = Gridworld()


gamma = 0.9
epsilon = EPSILON_RANGE
alpha = ALPHA_RANGE
temperature = TEMPERATURE_RANGE
labda = LABDA_RANGE

global RMSE_tabs, online_rewards_tabs
RMSE_tabs = [[], [], [], [], [], [], []]
CR_tabs = [[], [], []]

for run in range(NUMBER_OF_RUNS):

    print('\n ====================== This is the ' + str(run+1) + 'th run ====================== ')
    i = 0

    # Value Iteration (optimal policy benchmark)
    V_pi = value_iteration(ice_world, gamma, THETA)
    print("Optimal policy found by Value Iteration:")
    print(np.reshape(V_pi,[4,4]))

    #  Q-learning with \u03B5-greedy Exporation Strategy  (MH-8)
    print_header(ALGORITHMS[0])
    Q, cycles, time, RMSE_QLG, times_QLG, V_tabs_QLG, CR_QLG = \
        qlearning(ice_world, gamma, epsilon, alpha, NUMBER_OF_GAMES, V_pi)
    print_results(Q, cycles, time, ALGORITHMS[0])
    RMSE_tabs[0].append(RMSE_QLG)
    CR_tabs[0].append(CR_QLG)

    # Q-learning with Softmax Exploration Strategy (MH-9)
    print_header(ALGORITHMS[1])
    Q, cycles, time, RMSE_QLS, times_QLS, V_tabs_QLS, CR_QLS = \
        qlearning(ice_world, gamma, -temperature, alpha, NUMBER_OF_GAMES, V_pi)
    print_results(Q, cycles, time, ALGORITHMS[1])
    RMSE_tabs[1].append(RMSE_QLS)
    CR_tabs[1].append(CR_QLS)

    #  SARSA (MH-10)
    print_header(ALGORITHMS[2])
    Q, cycles, time, RMSE_SARSA, times_SARSA, CR_SARSA = \
        sarsa(ice_world, gamma, epsilon, alpha, NUMBER_OF_GAMES, V_pi)
    print_results(Q, cycles, time, ALGORITHMS[2])
    RMSE_tabs[2].append(RMSE_SARSA)
    CR_tabs[2].append(CR_SARSA)

    # Q-Learning (Eligibility Traces) (O-12)
    print_header(ALGORITHMS[3])
    Q, cycles, time, RMSE_QLE, times_QLE = \
        qlearning_watkins(ice_world, gamma, epsilon, alpha, labda, NUMBER_OF_GAMES, V_pi)
    print_results(Q, cycles, time, ALGORITHMS[3])
    RMSE_tabs[3].append(RMSE_QLE)

    # Q-Learning (Dyna-Q) (O-11)
    print_header(ALGORITHMS[4])
    Q, cycles, time, RMSE_QLD, times_QLD, CR_QLD = \
        dynaq(ice_world, gamma, epsilon, alpha, NUMBER_OF_GAMES, V_pi, PLANNING_STEPS)
    print_results(Q, cycles, time, ALGORITHMS[4])
    RMSE_tabs[4].append(RMSE_QLE)

    # Double Q-Learning with \u03B5-greedy Exploration Strategy (O-14)
    print_header(ALGORITHMS[5])
    Q, cycles, time, RMSE_DQLg, times_DQLg = doubleqlearning(ice_world, gamma, epsilon, alpha, NUMBER_OF_GAMES, V_pi)
    print_results(Q, cycles, time, ALGORITHMS[5])
    RMSE_tabs[5].append(RMSE_DQLg)

    # Double Q-Learning with \u03B5-greedy Exploration Strategy (O-14)
    print_header(ALGORITHMS[6])
    Q, cycles, time, RMSE_DQLs, times_DQLs = doubleqlearning(ice_world, gamma, -temperature, alpha, NUMBER_OF_GAMES, V_pi)
    print_results(Q, cycles, time, ALGORITHMS[6])
    RMSE_tabs[6].append(RMSE_DQLs)

RMSE_means = np.mean(RMSE_tabs, axis=1)
CR_means = np.mean(CR_tabs, axis=1)

print("-----")
print(len(ALGORITHMS))
print(len(RMSE_means))

# visualise results
create_plot("RMSE plot", len(ALGORITHMS) * [range(0, 10000)], 'Episode',
            RMSE_means[:,0:10000], 'RMSE, averaged over states', ALGORITHMS)
create_time_plot("time plot", [times_QLG[0:10000], times_QLS[0:10000],
                               times_SARSA[0:10000], times_QLE[0:10000], times_QLD[0:10000],
                               times_DQLg[0:10000], times_DQLs[0:10000]],
                 'elapsed time (seconds)', RMSE_means[:,0:10000], 'RMSE, averaged over states', ALGORITHMS)

# Make a plot of the state value for state [3,1]
create_state_plot("Value of the state above the start", V_tabs_QLG, NUMBER_OF_GAMES, V_pi)

# Make plots of the Cumulative Reward over time for the first three algorithms
create_plot("Cumulative Reward (1000 episodes)", [times_QLG[0:1000], times_QLS[0:1000], times_SARSA[0:1000]], "Time",
            CR_means[:,0:1000], "Cumulative Reward", ALGORITHMS[0:3])

create_plot("Cumulative Reward (10000 episodes)", [times_QLG[0:10000], times_QLS[0:10000], times_SARSA[0:10000]], "Time",
            CR_means[:,0:10000], "Cumulative Reward", ALGORITHMS[0:3])
