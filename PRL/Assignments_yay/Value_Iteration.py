import numpy as np
import time


def value_iteration(env, gamma, theta):
	t = time.perf_counter()
	unused, V, policy = env.initialize()
	loop_counter = 0

	while True:
		loop_counter += 1
		V, policy, converged = update(env, V, policy, gamma, theta)

		if converged:
			elapsed_time = time.perf_counter() - t
			return V


def update(env, V, policy, gamma, theta):
	delta = 0
	for state in env.non_terminal_states:
		v = np.copy(V[state])
		Q = np.zeros([4, 1])

		# generate new values and a new policy
		for action in range(4):
			for possible_outcome in env.transition_function(state, action):
				trans_prob = possible_outcome[0]
				next_state = possible_outcome[1]
				Q[action] += trans_prob * (env.R[next_state] + gamma * V[next_state])
		V[state] = np.amax(Q)
		policy[state] = np.zeros([4])
		policy[state, np.argmax(Q)] = 1

		# check for convergence of values
		delta = max(delta, abs(v - V[state]))
	return V, policy, (delta < theta)

