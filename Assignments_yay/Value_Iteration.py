import numpy as np
import time


def value_iteration(env, gamma, theta):

	V, policy = env.initialize()

	while True:
		env.amount_of_steps += 1
		delta = 0
		for state in env.non_terminal_states:
			v = np.copy(V[state])

			Q = np.zeros([4, 1])
			for action in range(4):
				for possible_outcome in env.transition_function(state, action):
					trans_prob = possible_outcome[0]
					next_state = possible_outcome[1]
					Q[action] += trans_prob * (env.R[next_state] + gamma * V[next_state])

			V[state] = np.amax(Q)
			policy[state] = np.zeros([4])
			policy[state, np.argmax(Q)] = 1

			delta = max(delta, abs(v - V[state]))

		if delta < theta:
			return V, policy, env.amount_of_steps
