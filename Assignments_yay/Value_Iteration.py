import numpy as np
import time


def initialize():
    """
	Initializes Q and V
	"""
    V = np.zeros([16, 1])  # Initialize an empty value table
    Q = np.zeros([16, 4])  # Initialize an empty value table
    pi_star = np.ones([16, 4]) * 0.25  # Initialize an arbitrary policy,
    # doesn't matter because action probabilites aren't used here.I might want something else for policy matrix/table/array but focusing on getting the right state values right now
    return V, Q, pi_star



def value_iteration(env, gamma, theta):

	V, pi_star = env.initialize()
	Q = np.copy(pi_star*0)

	while True:
		env.amount_of_steps += 1
		delta = 0
		for state in env.non_terminal_states:
			v = np.copy(V[state])
			for action in range(4):
				Q_update = 0
				for possible_outcome in env.transition_function(state, action):
					trans_prob = possible_outcome[0]
					next_state = possible_outcome[1]
					Q_update += trans_prob * (env.R[next_state] + gamma * V[next_state])
				Q[state, action] = Q_update
			V[state] = np.amax(Q[state])
			pi_star[state] = np.argmax(Q[state])
			delta = max(delta, abs(v - V[state]))
		if delta < theta:
			break
	return V, pi_star, env.amount_of_steps
