import numpy as np


def simple_policy_iteration(env, gamma, theta):
    """
	:param env: An implementation of the game rules.
	:param gamma: The discount factor.
	:param theta: An arbitrary small number to discontinue iteration. Do not adjust.
	:return: A table of value estimations (V), and a matrix of tables and actions representing the final (policy).
	"""
    #  Initialization
    V, policy = env.initialize()
    i = 0

    while True:
        #  Evaluation
        V, eval_counter = evaluate(env, V, policy, gamma, theta)

        #  Improvement
        policy, policy_stable = improve(env, V, policy, gamma)

        i += 1
        print("I've completed {} full loop(s).".format(i))

        if policy_stable is True:
            return V, policy
        # print("\n final policy")
        # print(solution_set(policy))
        #
        # print("\nValue table")
        # print(np.reshape(V1, [4,4]))
        # break


def evaluate(env, V, policy, gamma, theta):
    """
	:param env: An implementation of the game rules.
	:param V: A table of values under the current policy.
	:param policy: A 16x4 frame containing probabilities of action selection for each move under the current policy.
	:param gamma: The discount factor parameter.
	:param theta: An arbitrary small number to discontinue iteration. Do not adjust.
	:return: An updated value table (V); the number of loops completed (evaluation_counter).
	"""
    evaluation_counter = 0
    while True:
        delta = 0
        for state in env.non_terminal_states:
            v = np.copy(V[state])
            V[state] = 0
            for action in range(4):
                for possible_outcome in env.transition_function(state, action):
                    trans_prob = possible_outcome[0]
                    next_state = possible_outcome[1]

                    V[state] += policy[state, action] * trans_prob * (env.R1[next_state] + gamma * V[next_state])

            delta = max(delta, abs(v - V[state]))

        evaluation_counter += 1
        print('I have completed {} evaluation loop(s).'.format(evaluation_counter))

        if delta < theta:
            return V, evaluation_counter


def improve(env, V, policy, gamma):
    """
    :param env: An implementation of the game rules.
    :param V: A table of values under the current policy.
    :param policy: A 16x4 frame containing probabilities of action selection for each move under the current policy.
    :param gamma: The discount factor parameter.
    :return: The improved policy (policy), whether it is stable (policy_stable)
    """

    policy_stable = True
    for state in env.non_terminal_states:
        old_policy = np.copy(policy[state])
        action_values = np.zeros([4, 1])
        for action in range(4):
            for possible_outcome in env.transition_function(state, action):
                trans_prob = possible_outcome[0]
                next_state = possible_outcome[1]
                action_values[action] += trans_prob * (env.R1[next_state] + gamma * V[next_state])

        policy[state, np.argmax(action_values)] = 1
        policy[state, np.arange(4) != np.argmax(action_values)] = 0

        if not np.array_equal(old_policy, policy[state]):
            policy_stable = False
            break

    return policy, policy_stable
